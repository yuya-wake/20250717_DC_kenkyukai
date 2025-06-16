#!/usr/bin/env python

################################################################################
# gcn_2.py – Graph Neural Network cross‑validation
################################################################################

# The script trains a 2‑layer GCN on pre‑computed PyG graph files using
# 5‑fold cross‑validation and logs metrics to TensorBoard.

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Easier CUDA stack‑trace debugging

import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------
# Dataset: load .pt graphs and attach labels
# -----------------------------------------------------------------------------

class GraphDataset(Dataset):
    """PyG `Dataset` that pairs each graph .pt file with its label."""

    def __init__(self, pt_dir: str | Path, label_csv: str | Path):
        super().__init__(root=pt_dir)

        # 1) Build video‑id → label map (0=real, 1=fake) --------------------
        df = pd.read_csv(label_csv, usecols=['video_id', 'label'])
        invalid = set(df['label']) - {'real', 'fake'}
        if invalid:
            raise ValueError(f"Unexpected labels: {invalid}")
        self.label_map = {vid: 0 if lbl == 'real' else 1 for vid, lbl in df.values}

        # 2) Map video_id → .pt path ----------------------------------------
        pt_files = glob.glob(os.path.join(pt_dir, 'graph_*_bipartite_title_sim_common_channel.pt'))
        file_map = {Path(p).stem.split('_')[1]: p for p in pt_files}

        # 3) Keep only ids present in both label CSV and .pt directory -------
        valid_ids = set(self.label_map) & set(file_map)
        self.pt_paths = [file_map[i] for i in sorted(valid_ids)]

    # Required by PyG ----------------------------------------------------------
    def len(self):
        return len(self.pt_paths)

    def get(self, idx):
        path = self.pt_paths[idx]
        vid = Path(path).stem.split('_')[1]

        # Load graph and rebuild Data object with label ----------------------
        loaded: Data = torch.load(path, weights_only=False)
        data = Data(
            x=loaded.x,
            edge_index=loaded.edge_index,
            edge_weight=loaded.edge_weight,
            y=torch.tensor(self.label_map[vid], dtype=torch.long),
        )

        # Safety check: edge indices within bounds --------------------------
        if data.edge_index.numel() > 0:
            if data.edge_index.max() >= data.x.size(0) or data.edge_index.min() < 0:
                raise ValueError(f"Edge index out of bounds in {vid}")
        return data

# -----------------------------------------------------------------------------
# Model: 2‑layer GCN + linear classifier
# -----------------------------------------------------------------------------

class GCNClassifier(pl.LightningModule):
    def __init__(self, in_ch: int, hid: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(in_ch, hid)
        self.conv2 = GCNConv(hid, hid)
        self.cls   = nn.Linear(hid, 2)
        self.buffer: list[dict] = []  # Validation outputs per epoch

    # ---------- forward ----------------------------------------------------
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        return x

    # ---------- helper to locate root node per graph ----------------------
    def _root_indices(self, batch, n_nodes: int):
        idx = batch.ptr[:-1]  # Global index of node 0 in each graph
        if idx.max() >= n_nodes:
            raise IndexError("ptr index exceeds node tensor size")
        return idx

    # ---------- training ---------------------------------------------------
    def training_step(self, batch, _):
        node_emb = self(batch.x, batch.edge_index, getattr(batch, 'edge_weight', None))
        roots = self._root_indices(batch, node_emb.size(0))
        logits = self.cls(node_emb[roots])
        loss = F.cross_entropy(logits, batch.y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    # ---------- validation (store outputs) ---------------------------------
    def validation_step(self, batch, _):
        node_emb = self(batch.x, batch.edge_index, getattr(batch, 'edge_weight', None))
        roots = self._root_indices(batch, node_emb.size(0))
        logits = self.cls(node_emb[roots])
        out = {
            'loss':  F.cross_entropy(logits, batch.y).detach(),
            'preds': logits.argmax(1).detach(),
            'probs': F.softmax(logits, 1)[:, 1].detach(),
            'targs': batch.y.detach(),
        }
        self.buffer.append(out)
        self.log('val_loss_step', out['loss'], on_step=True)

    def on_validation_epoch_end(self):
        if not self.buffer:
            return
        preds = torch.cat([o['preds'] for o in self.buffer]).cpu().numpy()
        probs = torch.cat([o['probs'] for o in self.buffer]).cpu().numpy()
        targs = torch.cat([o['targs'] for o in self.buffer]).cpu().numpy()
        losses = torch.stack([o['loss'] for o in self.buffer]).cpu().numpy()
        self.buffer.clear()

        # Metric computation ------------------------------------------------
        metrics = {
            'val_loss_epoch': losses.mean(),
            'val_acc_epoch':  accuracy_score(targs, preds),
            'val_f1':        f1_score(targs, preds),
            'val_precision': precision_score(targs, preds),
            'val_recall':    recall_score(targs, preds),
            'val_auroc':     roc_auc_score(targs, probs),
            'val_kappa':     cohen_kappa_score(targs, preds),
        }
        self.log_dict(metrics, prog_bar=True)

    # ---------- optimiser --------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# -----------------------------------------------------------------------------
# Main cross‑validation routine
# -----------------------------------------------------------------------------

def main():
    set_global_seeds(42)

    # ---------------------- paths & hyper‑params ----------------------------
    mnt = Path(os.getenv('MNT_PATH', ''))
    pt_dir = mnt / 'networks' / 'pt_file' / 'title_desc_script_ocr' / 'bipartite' / 'comment_title_sim_and_common_channel'
    label_csv = mnt / 'label' / 'real_or_fake.csv'

    hid = 64
    lr = 1e-3
    epochs = 20
    patience = 5
    n_splits = 5
    batch_size = 4

    out_dir = Path.cwd() / 'results_gcn_title_desc_script_ocr'
    out_dir.mkdir(exist_ok=True)

    # ---------------------- dataset & split ---------------------------------
    ds = GraphDataset(pt_dir, label_csv)
    data_list = [ds.get(i) for i in range(len(ds))]
    in_ch = data_list[0].x.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(data_list), 1):
        print(f"\n########## Fold {fold}/{n_splits} ##########")
        tr_loader = DataLoader([data_list[i] for i in tr_idx], batch_size=batch_size, shuffle=True)
        va_loader = DataLoader([data_list[i] for i in va_idx], batch_size=batch_size, shuffle=False)

        model = GCNClassifier(in_ch, hid, lr)
        logger = TensorBoardLogger(out_dir, name=f'tb_fold{fold}')
        es = EarlyStopping(monitor='val_loss_epoch', patience=patience, mode='min')

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=logger,
            callbacks=[es],
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model, tr_loader, va_loader)
        res = trainer.validate(model, va_loader, verbose=False)[0]
        res['fold'] = fold
        cv_metrics.append(res)
        print(res)

    # ---------------- aggregate & save --------------------------------------
    avg = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0] if k != 'fold'}
    print("\n=== CV mean metrics ===")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    with (out_dir / 'gcn_cv_results.json').open('w') as f:
        json.dump({'fold_metrics': cv_metrics, 'average_metrics': avg}, f, indent=2)

# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
