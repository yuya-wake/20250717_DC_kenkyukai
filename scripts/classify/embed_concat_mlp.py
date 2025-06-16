#!/usr/bin/env python
"""
embed_concat_mlp.py
===================

Concatenate multimodal embeddings (title / description / transcript / OCR text /
OCR image) and classify videos as **fake** or **real** with a simple MLP.

Pipeline
--------
1. Load five pickle files created earlier in this project:

   * `roberta/title_emb.pkl`               (title, 1 × 768)
   * `roberta/desc_mean_emb.pkl`           (description-mean, 1 × 768)
   * `roberta/transcript_mean_emb.pkl`     (transcript-mean, 1 × 768)
   * `clip/ocr_text_image_emb.pkl`         (OCR text 1 × 512, OCR image 1 × 512)

2. Build a `VideoDataset` that concatenates the above vectors per `video_id`.
3. Train a 2-hidden-layer MLP with 5-fold cross-validation.
4. Log metrics to TensorBoard and save a JSON summary.

Reproducibility notes
--------------------
* Uses a fixed random seed (42) for NumPy / PyTorch / Lightning.
* Model weights are *not* checkpointed (set `--checkpoint` if desired).

"""

import os, json, pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score,
                             cohen_kappa_score)

# ────────────────────────────────────────────────────────────────────────
# 0. 乱数シード & 環境
# ────────────────────────────────────────────────────────────────────────
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")

# ────────────────────────────────────────────────────────────────────────
# 1. 埋め込み pickle の読み込み
# ────────────────────────────────────────────────────────────────────────
MNT       = os.getenv("MNT_PATH") or ""
PKL_DIR   = os.path.join(MNT, "networks", "pkl")

# pkl のパス
title_pkl       = os.path.join(PKL_DIR, "roberta", "title_emb.pkl")
desc_pkl        = os.path.join(PKL_DIR, "roberta", "desc_mean_emb.pkl")
transcript_pkl  = os.path.join(PKL_DIR, "roberta", "transcript_mean_emb.pkl")
ocr_pkl         = os.path.join(PKL_DIR, "clip",    "ocr_text_image_emb.pkl")

with open(title_pkl,      "rb") as f: title_dict      = pickle.load(f)
with open(desc_pkl,       "rb") as f: desc_dict       = pickle.load(f)
with open(transcript_pkl, "rb") as f: transcript_dict = pickle.load(f)
with open(ocr_pkl,        "rb") as f: ocr_dict        = pickle.load(f)

print("Loaded dict sizes — ",
      len(title_dict), len(desc_dict), len(transcript_dict), len(ocr_dict))

# ────────────────────────────────────────────────────────────────────────
# 2. 各埋め込みの次元数を取得 & 0 ベクトルを用意
# ────────────────────────────────────────────────────────────────────────
def _to_1d(vec, dim_target):
    """
    • list[Tensor/ndarray] なら 2-D に揃えて mean-pool
    • 0 長さや 0 行列なら dim_target の 0 ベクトルを返す
    • 最後に長さを強制的に dim_target に合わせる
    """
    if isinstance(vec, list):
        # 空リストならゼロを返してしまう
        if len(vec) == 0:
            return np.zeros(dim_target, dtype="float32")
        vec = np.stack([np.asarray(v) for v in vec], 0)

    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().numpy()

    vec = np.asarray(vec)

    # ------- 1-D / 2-D 正規化 -------
    if vec.ndim == 1:
        out = vec
    elif vec.ndim == 2:
        out = vec.mean(axis=0)
    else:
        raise ValueError(f"Unexpected ndim={vec.ndim} for embedding")

    # ------- 長さを dim_target に統一 -------
    if out.shape[0] != dim_target:
        if out.shape[0] > dim_target:        # truncate
            out = out[:dim_target]
        else:                                # pad with zeros
            pad = np.zeros(dim_target, dtype=out.dtype)
            pad[: out.shape[0]] = out
            out = pad

    return out.astype("float32")

def get_vec(d, vid, key, dim_target, default_zero):
    if vid in d and key in d[vid]:
        try:
            return _to_1d(d[vid][key], dim_target)
        except Exception as e:          # 予想外の壊れたデータも 0 埋め
            print(f"[WARN] {vid}:{key} -> {e}  ==> use zeros")
            return default_zero
    return default_zero

def _tensor_dim(x) -> int | None:
    """
    x が Tensor / np.ndarray / リスト(再帰的) なら最初の要素から次元を返す。
    それ以外は None
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.shape[-1]
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _tensor_dim(x[0])   # 先頭で判定
    return None

def first_tensor_dim(d: dict, key: str) -> int:
    """
    辞書 d から key に対応する埋め込みの次元を推定
    ・d[vid] = {'text': List[Tensor], 'image': Tensor, ...} などを想定
    """
    for v in d.values():
        if isinstance(v, dict) and key in v:
            dim = _tensor_dim(v[key])
            if dim is not None:
                return dim
    raise ValueError(f"{key} の埋め込みが見つかりませんでした。")

D_title = first_tensor_dim(title_dict, key='title')          # 768
D_desc  = first_tensor_dim(desc_dict, key='description_mean')           # 768
D_trans = first_tensor_dim(transcript_dict, key='transcript_mean')     # 768
D_ocr_t = first_tensor_dim(ocr_dict, key="text")          # 512
D_ocr_i = first_tensor_dim(ocr_dict, key="image")         # 512
D_total  = D_title + D_desc + D_trans + D_ocr_t + D_ocr_i

zero_title = np.zeros(D_title , dtype="float32")
zero_desc  = np.zeros(D_desc  , dtype="float32")
zero_trans = np.zeros(D_trans , dtype="float32")
zero_ocr_t = np.zeros(D_ocr_t , dtype="float32")
zero_ocr_i = np.zeros(D_ocr_i , dtype="float32")

# ────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ────────────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, label_csv: str):
        df = pd.read_csv(label_csv, usecols=["video_id", "label"])
        df = df[df["label"].isin(["real", "fake"])].reset_index(drop=True)

        self.vids   = df["video_id"].tolist()
        self.labels = torch.tensor(
            [0 if l == "real" else 1 for l in df["label"]], dtype=torch.long)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        v_title = get_vec(title_dict,      vid, "title",
                          D_title,  zero_title)
        v_desc  = get_vec(desc_dict,       vid, "description_mean",
                          D_desc,   zero_desc)
        v_trans = get_vec(transcript_dict, vid, "transcript_mean",
                          D_trans,  zero_trans)
        v_ocr_t = get_vec(ocr_dict,        vid, "text",
                          D_ocr_t,  zero_ocr_t)
        v_ocr_i = get_vec(ocr_dict,        vid, "image",
                          D_ocr_i,  zero_ocr_i)

        vec = np.concatenate(
            [v_title, v_desc, v_trans, v_ocr_t, v_ocr_i], axis=0
        ).astype("float32")                                      # [D_total]

        return torch.from_numpy(vec), self.labels[idx]


# ────────────────────────────────────────────────────────────────────────
# 4. LightningModule : 単純な 2 層 MLP
# ────────────────────────────────────────────────────────────────────────
class MLPClassifier(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim=512, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim//2, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.mlp(x)

    def _step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        preds  = logits.argmax(dim=1)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=y.size(0))
        acc = (preds == y).float().mean()
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True,
                 batch_size=y.size(0))
        return {"loss": loss, "preds": preds, "probs": F.softmax(logits,1)[:,1],
                "targets": y}

    def training_step(self, batch, _):  return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ────────────────────────────────────────────────────────────────────────
# 5. クロスバリデーション
# ────────────────────────────────────────────────────────────────────────
def run_cv():
    LABEL_CSV   = os.path.join(MNT, "label", "real_or_fake.csv")
    BATCH_SIZE  = 32
    NUM_WORKERS = 4
    EPOCHS      = 20
    PATIENCE    = 5
    N_SPLITS    = 5
    HIDDEN_DIM  = 512
    LR          = 1e-3

    out_dir = os.path.abspath("results_mlp_concat")
    os.makedirs(out_dir, exist_ok=True)

    dataset = VideoDataset(LABEL_CSV)
    kf      = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(dataset))), 1):
        print(f"\n==========  Fold {fold}/{N_SPLITS}  ==========")
        train_subset = torch.utils.data.Subset(dataset, tr_idx)
        val_subset   = torch.utils.data.Subset(dataset, va_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)

        model = MLPClassifier(D_total, HIDDEN_DIM, LR)

        logger = TensorBoardLogger(out_dir, name=f"tb_fold{fold}")
        es     = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")

        trainer = pl.Trainer(max_epochs=EPOCHS,
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1,
                             logger=logger,
                             callbacks=[es],
                             enable_checkpointing=False,
                             enable_model_summary=False)

        trainer.fit(model, train_loader, val_loader)

        # ── 検証データで最終評価 ─────────────────────────────
        model.eval()
        y_true, y_pred, y_prob, losses = [], [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(model.device))
                loss   = F.cross_entropy(logits, y.to(model.device))
                probs  = F.softmax(logits, 1)[:,1].cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()

                y_true.extend(y.numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)
                losses.append(loss.item())

        # 指標計算
        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred)
        rec   = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        auroc = roc_auc_score(y_true, y_prob)
        kappa = cohen_kappa_score(y_true, y_pred)

        m = dict(fold=fold, acc=acc, f1=f1, precision=prec, recall=rec,
                 tn=tn, fp=fp, fn=fn, tp=tp, auroc=auroc,
                 loss=np.mean(losses), kappa=kappa)
        fold_metrics.append(m)
        print(m)

    # ── 平均・分散を保存 ───────────────────────────────
    df = pd.DataFrame(fold_metrics)
    avg = df.mean(numeric_only=True).to_dict()
    std = df.std(numeric_only=True).to_dict()

    def numpy2py(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.generic, np.ndarray)):
                out[k] = v.item()      # → int or float
            else:
                out[k] = v
        return out

    avg = numpy2py(avg)
    std = numpy2py(std)
    fold_metrics = [numpy2py(m) for m in fold_metrics]

    with open(os.path.join(out_dir, "mlp_concat_results.json"), "w") as f:
        json.dump({"fold_metrics": fold_metrics,
                   "mean": avg, "std": std,
                   "hyperparams": {"batch":BATCH_SIZE,"epochs":EPOCHS,
                                   "hidden":HIDDEN_DIM,"lr":LR}},
                  f, indent=2)

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cv()
