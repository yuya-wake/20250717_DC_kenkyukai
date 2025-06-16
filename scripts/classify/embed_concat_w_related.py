#!/usr/bin/env python
"""
embed_concat_w_related.py
================================

Multimodal video classification that combines **root-video embeddings** with
aggregated **related-video embeddings**.

Architecture
------------
1. **Root vector** (fixed):
   • title  (768)
   • description (768)
   • transcript (768)
   • OCR text (512)
   • OCR image (512)  → total **3584** dims

2. **Related vector** (variable): for each related video
   ``title ⊕ description`` → 1536-D; mean-pooled across all related videos.

3. Small MLP (1536 → 256) compresses the related vector.

4. Concatenate root (3584) + reduced related (256) → 3840-D and feed into final
   MLP for binary classification (fake / real).

Training
--------
* 5-fold cross-validation
* Early stopping on `val_loss`
* TensorBoard logging (one folder per fold)

"""

import os, sys, json, pickle, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score,
                             cohen_kappa_score)

# ────────────────────────────────────────────────────────────────────────
# 0. seed & precision
# ────────────────────────────────────────────────────────────────────────
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('high')

# ────────────────────────────────────────────────────────────────────────
# 1. パスと埋め込み辞書のロード
# ────────────────────────────────────────────────────────────────────────
MNT = os.getenv("MNT_PATH") or ""
PKL_DIR = Path(MNT) / "networks" / "pkl"
REL_DIR = Path(MNT) / "related_videos" / "w_metadata"   # ← w_metadata_<root>.csv

title_dict      = pickle.load(open(PKL_DIR / "roberta/title_emb.pkl",            "rb"))
desc_dict       = pickle.load(open(PKL_DIR / "roberta/desc_mean_emb.pkl",        "rb"))
transcript_dict = pickle.load(open(PKL_DIR / "roberta/transcript_mean_emb.pkl",  "rb"))
ocr_dict        = pickle.load(open(PKL_DIR / "clip/ocr_text_image_emb.pkl",      "rb"))

print("Loaded dict sizes —",
      len(title_dict), len(desc_dict), len(transcript_dict), len(ocr_dict))

# ────────────────────────────────────────────────────────────────────────
# 2. ユーティリティ
# ────────────────────────────────────────────────────────────────────────
def _to_1d(vec) -> np.ndarray:
    """
    さまざまな形 (Tensor / ndarray / list[Tensor…]) を 1D ndarray に。
    * list or 2D → mean-pool
    * 空 list → ValueError → 呼び出し側で 0 ベクトルに
    """
    if isinstance(vec, list):
        if len(vec) == 0:
            raise ValueError("empty list")
        vec = np.stack([np.asarray(v) for v in vec], 0)   # -> 2D
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().numpy()
    vec = np.asarray(vec)

    if vec.ndim == 1:
        return vec
    if vec.ndim == 2:
        return vec.mean(axis=0)
    raise ValueError(f"unexpected ndim={vec.ndim}")

def get_vec(d:dict, vid:str, key:str, dim:int) -> np.ndarray:
    """
    dict が無い / key が無い / 空 list の場合は 0 ベクトルを返す
    """
    try:
        return _to_1d(d[vid][key])
    except Exception:
        return np.zeros(dim, dtype="float32")

def first_dim(d:dict, key:str) -> int:
    for v in d.values():
        if isinstance(v, dict) and key in v:
            try:
                return _to_1d(v[key]).shape[0]
            except Exception:
                pass
    raise RuntimeError(f"cannot infer dim for {key}")

# 次元を取得
D_TITLE = first_dim(title_dict,      "title")             # 768
D_DESC  = first_dim(desc_dict,       "description_mean")  # 768
D_TRANS = first_dim(transcript_dict, "transcript_mean")   # 768
D_OCR_T = first_dim(ocr_dict,        "text")              # 512
D_OCR_I = first_dim(ocr_dict,        "image")             # 512
ROOT_DIM = D_TITLE + D_DESC + D_TRANS + D_OCR_T + D_OCR_I

REL_SINGLE_DIM = D_TITLE + D_DESC       # 1536
RED_DIM        = 256                    # related 圧縮後の次元
TOTAL_DIM      = ROOT_DIM + RED_DIM

# ────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ────────────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    root video 向け:
      * root_vec  : 全 5 種を concat（欠損は 0）
      * rel_vec   : related 動画たちの (title⊕desc) を mean → [1536]
    """
    def __init__(self, label_csv:str):
        df = pd.read_csv(label_csv, usecols=["video_id","label"])
        df = df[df["label"].isin(["real","fake"])].reset_index(drop=True)
        self.vids   = df["video_id"].tolist()
        self.labels = torch.tensor(
            [0 if l=="real" else 1 for l in df["label"]], dtype=torch.long)

    # ――― helper ―――
    def _root_vector(self, vid:str) -> np.ndarray:
        v_title = get_vec(title_dict,      vid, "title",            D_TITLE)
        v_desc  = get_vec(desc_dict,       vid, "description_mean", D_DESC)
        v_trans = get_vec(transcript_dict, vid, "transcript_mean",  D_TRANS)
        v_ocr_t = get_vec(ocr_dict,        vid, "text",             D_OCR_T)
        v_ocr_i = get_vec(ocr_dict,        vid, "image",            D_OCR_I)
        return np.concatenate([v_title, v_desc, v_trans, v_ocr_t, v_ocr_i],
                              axis=0).astype("float32")             # [ROOT_DIM]

    def _related_vector(self, vid:str) -> np.ndarray:
        csv_path = REL_DIR / f"w_metadata_{vid}.csv"
        if not csv_path.exists():
            return np.zeros(REL_SINGLE_DIM, dtype="float32")

        try:
            rel_df = pd.read_csv(csv_path, usecols=["video_id"])
            rel_ids = rel_df["video_id"].unique().tolist()
        except Exception:
            return np.zeros(REL_SINGLE_DIM, dtype="float32")

        vecs = []
        for rid in rel_ids:
            r_title = get_vec(title_dict, rid, "title",            D_TITLE)
            r_desc  = get_vec(desc_dict,  rid, "description_mean", D_DESC)
            vecs.append(np.concatenate([r_title, r_desc], 0))          # [1536]

        if len(vecs) == 0:
            return np.zeros(REL_SINGLE_DIM, dtype="float32")
        return np.stack(vecs,0).mean(axis=0).astype("float32")         # mean-pool

    # ――― Dataset core ―――
    def __len__(self): return len(self.vids)

    def __getitem__(self, idx):
        vid   = self.vids[idx]
        label = self.labels[idx]

        root_vec = self._root_vector(vid)       # [ROOT_DIM]
        rel_vec  = self._related_vector(vid)    # [REL_SINGLE_DIM]

        return (torch.from_numpy(root_vec), torch.from_numpy(rel_vec)), label

# ────────────────────────────────────────────────────────────────────────
# 4. LightningModule
# ────────────────────────────────────────────────────────────────────────
class ConcatMLP(pl.LightningModule):
    """
    related ベクトルを小 MLP(1536→RED_DIM) で圧縮し、
    root_vec と結合して最終 MLP で 2 クラス分類
    """
    def __init__(self,
                 rel_in=REL_SINGLE_DIM, rel_out=RED_DIM,
                 final_in=TOTAL_DIM,  hidden=512, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.rel_mlp = nn.Sequential(
            nn.Linear(rel_in, rel_in//2), nn.ReLU(),
            nn.Linear(rel_in//2, rel_out), nn.ReLU()
        )
        self.cls_mlp = nn.Sequential(
            nn.Linear(final_in, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden//2, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, root_vec, rel_vec):
        rel_reduced = self.rel_mlp(rel_vec)
        concat      = torch.cat([root_vec, rel_reduced], dim=1)
        return self.cls_mlp(concat)

    # ----------- shared step ----------
    def _step(self, batch, stage):
        (root, rel), y = batch
        logits = self(root, rel)
        loss   = self.criterion(logits, y)
        preds  = logits.argmax(1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=y.size(0))
        self.log(f"{stage}_acc",
                 (preds==y).float().mean(), prog_bar=True, on_epoch=True,
                 batch_size=y.size(0))
        return {"loss":loss, "preds":preds,
                "probs":F.softmax(logits,1)[:,1], "targets":y}

    def training_step(self,b,i):  return self._step(b,"train")
    def validation_step(self,b,i):return self._step(b,"val")
    def test_step(self,b,i):      return self._step(b,"test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ────────────────────────────────────────────────────────────────────────
# 5. CV ループ
# ────────────────────────────────────────────────────────────────────────
def run_cv():
    LABEL_CSV   = Path(MNT)/"label"/"real_or_fake.csv"
    BATCH_SIZE  = 32
    WORKERS     = 4
    EPOCHS      = 20
    PATIENCE    = 5
    SPLITS      = 5
    HIDDEN      = 512
    LR          = 1e-3

    out_dir = Path("results_concat_B")
    out_dir.mkdir(exist_ok=True)

    ds = VideoDataset(LABEL_CSV)
    kf = KFold(n_splits=SPLITS, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(ds))),1):
        print(f"\n========== Fold {fold}/{SPLITS} ==========")
        tr_loader = DataLoader(torch.utils.data.Subset(ds,tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=WORKERS)
        va_loader = DataLoader(torch.utils.data.Subset(ds,va_idx),
                               batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=WORKERS)

        model = ConcatMLP(hidden=HIDDEN, lr=LR)

        logger = TensorBoardLogger(out_dir, name=f"tb_fold{fold}")
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")

        pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[es],
            enable_checkpointing=False,
            enable_model_summary=False
        ).fit(model, tr_loader, va_loader)

        # ---- 評価 ----
        model.eval()
        y_t,y_p,y_pb,lss = [],[],[],[]
        with torch.no_grad():
            for (r,rel),y in va_loader:
                logits = model(r.to(model.device),
                               rel.to(model.device))
                lss .append(F.cross_entropy(logits,y.to(model.device)).item())
                y_t.extend(y.numpy())
                y_pb.extend(F.softmax(logits,1)[:,1].cpu().numpy())
                y_p.extend(logits.argmax(1).cpu().numpy())

        acc   = accuracy_score(y_t,y_p)
        f1    = f1_score(y_t,y_p)
        prec  = precision_score(y_t,y_p)
        rec   = recall_score(y_t,y_p)
        tn,fp,fn,tp = confusion_matrix(y_t,y_p).ravel()
        auroc = roc_auc_score(y_t,y_pb)
        kappa = cohen_kappa_score(y_t,y_p)

        m = dict(fold=fold,acc=acc,f1=f1,precision=prec,recall=rec,
                 tn=tn,fp=fp,fn=fn,tp=tp,auroc=auroc,
                 loss=float(np.mean(lss)),kappa=kappa)
        fold_metrics.append(m)
        print(m)

    # ---- 平均 & 保存 ----
    df  = pd.DataFrame(fold_metrics)
    avg = {k:float(v) for k,v in df.mean(numeric_only=True).items()}
    std = {k:float(v) for k,v in df.std (numeric_only=True).items()}
    print("\n===== CV summary =====")
    for k in avg: print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

    with open(out_dir/"concat_w_related_results.json","w") as f:
        json.dump({"fold_metrics":[{k:(float(v) if isinstance(v,np.generic) else v)
                                   for k,v in d.items()}
                                   for d in fold_metrics],
                   "mean":avg,"std":std,
                   "hyperparams":{"batch":BATCH_SIZE,"epochs":EPOCHS,
                                  "hidden":HIDDEN,"lr":LR}},
                  f, indent=2)

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cv()
