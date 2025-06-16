#!/usr/bin/env python
"""
Mean-pool sentence-level embeddings (e.g., transcript or description).

Input:
    Pickle file containing:
        emb_dict[video_id]["transcript"] = List[torch.Tensor[D]]

Output:
    emb_dict[video_id]["transcript_mean"] = torch.Tensor[D]
    (or np.ndarray[D] if post-processed)

Configuration:
    * Zero-vector if no sentence exists (optional)
    * Optionally save output to disk

Usage:
    python mean_pooling.py  # Or import and call `mean_pool()`

Notes:
    * Default input: transcript_emb.pkl
    * Default output: transcript_mean_emb.pkl
    * Assumes data was embedded with RoBERTa etc.

"""

import os
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional

def mean_pool(
    pkl_in: Union[str, Path],
    pkl_out: Optional[Union[str, Path]] = None,
    zero_when_empty: bool = True
) -> dict:
    """
    Apply mean pooling to sentence-level embeddings (e.g., transcript).

    Parameters
    ----------
    pkl_in : str or Path
        Path to input pickle containing sentence embeddings
    pkl_out : str or Path or None
        Where to save the new dict with mean-pooled vectors
    zero_when_empty : bool
        Whether to assign zero vector when no embeddings found

    Returns
    -------
    emb_dict : dict
        Same as input, but with ['transcript_mean'] added per video
    """
    pkl_in = Path(pkl_in)
    with pkl_in.open("rb") as f:
        emb_dict = pickle.load(f)

    D = None
    for d in emb_dict.values():
        if d.get("transcript"):
            D = d["transcript"][0].numel()
            break
    if D is None:
        raise ValueError("No transcript embeddings found to infer dimension.")

    for vid, d in tqdm(emb_dict.items(), desc="Mean pooling transcript"):
        sent_list = d.get("transcript", [])

        if not sent_list:
            if zero_when_empty:
                d["transcript_mean"] = torch.zeros(D)
            continue

        stacked = torch.stack(sent_list)  # [N, D]
        d["transcript_mean"] = stacked.mean(dim=0).cpu()

    for d in emb_dict.values():
        d.pop("transcript", None)

    if pkl_out:
        pkl_out = Path(pkl_out)
        with pkl_out.open("wb") as f:
            pickle.dump(emb_dict, f)
        print(f"Saved mean-pooled embeddings to {pkl_out}")

    return emb_dict

def main():
    mnt_path = Path(os.getenv("MNT_PATH", ""))
    pkl_in_path = mnt_path / "networks" / "pkl" / "roberta" / "transcript_emb.pkl"
    pkl_out_path = mnt_path / "networks" / "pkl" / "roberta" / "transcript_mean_emb.pkl"
    mean_pool(pkl_in_path, pkl_out_path)

if __name__ == "__main__":
    main()
