# YouTube Fake Video Detection using Multimodal GCN
This repository contains the official implementation of our paper:

**"医療系YouTube動画を対象とした動画間の関係性に着目したフェイク判定手法"**  
by Yuya Wake and Toshiyuki Amagasa, University of Tsukuba, 2025

# Directory Structure
```
.
├── scripts/
│   ├── create_embedding/
│   │   ├── roberta_embedding.py         # Create embeddings from title/description/transcript
│   │   ├── clip_embedding.py            # Extract embeddings from OCR text and image using CLIP
│   │   └── mean_pooling.py              # Mean-pool sentence-level embeddings
│   ├── classify/
│   │   ├── gcn_2.py                     # GCN model for classification
│   │   ├── embed_concat_mlp.py          # MLP-based classification (concatenated embeddings)
│   │   └── embed_concat_w_related.py    # MLP model with related video pooling
│   └── ocr/
│       ├── ocr.py                       # Run OCR on thumbnails using Google Cloud Vision
│       └── clean_ocr_text.py            # Clean texts from thumbnails 
├── pkl/                                 # Output of create_embedding/*
│   └── pkl.md                           # The .pkl files can be accessed from Google Drive
└── graph/
    └── build_graph.py                   # Build .pt graphs for GCN input
```

# Requirements

## For general (GNN-based) scripts
Use the environment defined in gnn_environment.yml and gnn_requirements.txt:
```
conda env create -f gnn_environment.yml
conda activate gnn_venv
pip install -r gnn_requirements.txt
```

## For CLIP embedding
Only clip_embedding.py requires a separate environment:
```
conda env create -f clip_environment.yml
conda activate clip_venv
pip install -r clip_requirements.txt
```

> **Note**  
> To use CLIP, please download `clip-japanese-base` with the following command:  
> `git clone https://huggingface.co/line-corporation/clip-japanese-base /home/user/model/clip_japanese_base`

# Usage
1. **Prepare embeddings**
   If you do not want to generate embeddings yourself, download .pkl files from the following URL:
   https://drive.google.com/drive/folders/1uJN1uG-Z6HFduPgDKIzFCj3gU8PcJ2oe?usp=sharing

2. **Create Graph (.pt) files**
   `python scripts/graph/build_graph.py`

3. **Run classification**
   - GCN model:
     `python scripts/classify/gcn_2.py`
   - MLP model:
     `python scripts/classify/embed_concat_mlp.py`
   - MLP with related videos model:
     `python scripts/classify/embed_concat_w_related.py'
