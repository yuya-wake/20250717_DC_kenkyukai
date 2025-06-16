#!/usr/bin/env python
"""
Embed YouTube video metadata (title / description / transcript) with a Japanese
RoBERTa model and store them as pickle files.

Pipeline
--------
1. Load *original* video metadata (real_or_fake.csv)
2. Load *related* video metadata (w_metadata_{video_id}.csv)
3. Build vocabulary of all video IDs (pre‑computed all_video_ids.pkl)
4. Compute embeddings for three text fields  
   • title (single embedding)  
   • description (sentence‑level list of embeddings)  
   • transcript (sentence‑level list; original videos only)
5. Save dictionaries to
   ```${MNT_PATH}/networks/pkl/roberta/{title,desc,transcript}_emb.pkl```.

All heavy lifting (tokenisation + forward pass) is done on a single GPU.

"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def save_pickle(obj, path: Path, label: str) -> None:
    """Safely save *obj* to *path* with pickle and log the outcome."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)
        logging.info("[%s] Saved pickle → %s", label, path)
    except Exception as exc:  # pragma: no cover – informative by design
        logging.error("[%s] Failed to save pickle: %s", label, exc)


def embed_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """Return a *single* CLS embedding for *text* (shape = [D])."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu()


def embed_sentences(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Sentence-level embeddings for *text* split on "。"."""
    sentences = [s.strip() for s in str(text).split("。") if s.strip()]
    out: List[torch.Tensor] = []
    for sent in sentences:
        out.append(embed_text(sent, tokenizer, model, device, max_length))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_original(csv_path: Path) -> pd.DataFrame:
    cols = ["video_id", "title", "description", "transcript"]
    return pd.read_csv(csv_path, usecols=cols, encoding="utf-8", low_memory=False)


def load_related(dir_path: Path, base_id: str) -> pd.DataFrame | None:
    path = dir_path / f"w_metadata_{base_id}.csv"
    if not path.exists():
        return None
    cols = ["video_id", "title", "description"]
    return pd.read_csv(path, usecols=cols, encoding="utf-8", low_memory=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing routine
# ──────────────────────────────────────────────────────────────────────────────

def run(
    mnt_path: Path,
    model_name: str,
    gpu_id: int,
) -> None:
    logging.info("Mount path: %s", mnt_path)

    # Paths -------------------------------------------------------------------
    original_csv = mnt_path / "complete_data" / "real_and_fake.csv"
    related_dir = mnt_path / "related_videos" / "w_metadata"
    pkl_root = mnt_path / "networks" / "pkl" / "roberta"
    all_id_pkl = mnt_path / "networks" / "pkl" / "all_video_ids.pkl"

    # Model -------------------------------------------------------------------
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Load data ---------------------------------------------------------------
    logging.info("Reading original metadata …")
    orig_df = load_original(original_csv)
    orig_ids = set(orig_df["video_id"])  # For iteration over related metadata

    with all_id_pkl.open("rb") as f:
        all_ids: List[str] = pickle.load(f)
    logging.info("Loaded %d video IDs from %s", len(all_ids), all_id_pkl)

    # Containers --------------------------------------------------------------
    title_emb: Dict[str, torch.Tensor] = {}
    desc_emb: Dict[str, List[torch.Tensor]] = {}
    trans_emb: Dict[str, List[torch.Tensor]] = {}

    # Build lookup maps (avoid repeated DF filtering in the loops) ------------
    logging.info("Building title / description lookup tables …")
    title_map: Dict[str, str] = {
        **dict(orig_df[["video_id", "title"]].drop_duplicates().values.tolist())
    }
    desc_map: Dict[str, str] = {
        **dict(orig_df[["video_id", "description"]].drop_duplicates().values.tolist())
    }
    for vid in tqdm(orig_ids, desc="Read related CSVs"):
        rel_df = load_related(related_dir, vid)
        if rel_df is None:
            continue
        title_map.update(rel_df[["video_id", "title"]].drop_duplicates().values.tolist())
        desc_map.update(rel_df[["video_id", "description"]].drop_duplicates().values.tolist())

    # 1) Title embeddings -----------------------------------------------------
    logging.info("Embedding titles …")
    for vid in tqdm(all_ids, desc="Titles"):
        title_emb[vid] = embed_text(title_map.get(vid, ""), tokenizer, model, device)
    save_pickle(title_emb, pkl_root / "title_emb.pkl", "Title")

    # 2) Description embeddings ----------------------------------------------
    logging.info("Embedding descriptions …")
    for vid in tqdm(all_ids, desc="Descriptions"):
        desc_emb[vid] = embed_sentences(desc_map.get(vid, ""), tokenizer, model, device)
    save_pickle(desc_emb, pkl_root / "desc_emb.pkl", "Description")

    # 3) Transcript embeddings (original videos only) -------------------------
    logging.info("Embedding transcripts …")
    scripts = orig_df.dropna(subset=["transcript"]).set_index("video_id")["transcript"].to_dict()
    for vid, script in tqdm(scripts.items(), desc="Transcripts"):
        trans_emb[vid] = embed_sentences(script, tokenizer, model, device)
    save_pickle(trans_emb, pkl_root / "transcript_emb.pkl", "Transcript")

    logging.info("All embeddings saved under %s", pkl_root)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed YouTube text fields with RoBERTa")
    p.add_argument("--mnt_path", type=Path, default=os.getenv("MNT_PATH", ""),
                   help="Root mount path (default: $MNT_PATH env var)")
    p.add_argument("--model", default="nlp-waseda/roberta-large-japanese-seq512",
                   help="HF model name or local path")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format="%(levelname)s %(message)s")
    run(args.mnt_path.expanduser().resolve(), args.model, args.gpu)
