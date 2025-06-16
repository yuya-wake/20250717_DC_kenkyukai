import os
from rhoknp import Jumanpp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --- Configurations ---
MODEL       = "nlp-waseda/roberta-large-japanese-seq512"
mnt_path    = os.getenv('MNT_PATH') or ""
print(mnt_path)
original_csv= os.path.join(mnt_path, 'complete_data', 'real_and_fake.csv')
related_dir = os.path.join(mnt_path, 'related_videos', 'w_metadata')
output_dir  = os.path.join(mnt_path, 'networks', 'pt_file', 'title_desc_script_ocr', 'bipartite', 'comment_title_sim_and_common_channel')
pkl_dir    = os.path.join(mnt_path, 'networks', 'pkl')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pkl_dir, exist_ok=True)

threshold = 0.8

# --- Load Japanese text encoder ---
juman = Jumanpp(
    executable="/home/wake/local/bin/jumanpp",
    options=["--model=/home/wake/local/share/jumanpp/jumandic.jppmdl"]
)

sp_tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()
device = "cuda"
model = model.to(device)
max_length = 512

def embed_text(text):
    inputs = sp_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().cpu()

# --- CSV Loaders ---
def read_original():
    df_csv = pd.read_csv(original_csv,
                         low_memory=False,
                         encoding='utf-8',
                         on_bad_lines='warn',
                         lineterminator='\n',
                         delimiter=',',
                         quotechar='"',
                         usecols=['video_id', 'title', 'channel_id', 'comment_author_channel_id', 'reply_author_channel_id'])
    return df_csv

def read_related(file):
    try:
        df_csv = pd.read_csv(os.path.join(related_dir, f'w_metadata_{file}.csv'),
                            low_memory=False,
                            encoding='utf-8',
                            on_bad_lines='warn',
                            lineterminator='\n',
                            delimiter=',',
                            quotechar='"',
                            usecols=['video_id', 'title', 'channel_id', 'author_channel_id'])
        return df_csv
    except:
        pass

# --- 1) Load video IDs ---
orig_df = read_original()
orig_ids= set(orig_df['video_id'])
all_ids = set(orig_ids)
print(f'orig_ids: {len(orig_ids)}')

id_pkl_path = os.path.join(pkl_dir, 'all_video_ids.pkl')
with open(id_pkl_path, 'rb') as f:
    all_ids = pickle.load(f)
print(f"Loaded {len(all_ids)} video IDs.")

# --- 2) Load embeddings ---
title_pkl_path = os.path.join(pkl_dir, 'roberta', 'title_emb.pkl')
with open(title_pkl_path, 'rb') as f:
    title_dict = pickle.load(f)
print(f"Loaded {len(title_dict)} title embeddings.")

desc_pkl_path = os.path.join(pkl_dir, 'roberta', 'desc_mean_emb.pkl')
with open(desc_pkl_path, 'rb') as f:
    desc_dict = pickle.load(f)
print(f"Loaded {len(desc_dict)} desc embeddings.")

transcript_pkl_path = os.path.join(pkl_dir, 'roberta', 'transcript_mean_emb.pkl')
with open(transcript_pkl_path, 'rb') as f:
    transcript_dict = pickle.load(f)
print(f"Loaded {len(transcript_dict)} transcript embeddings.")

ocr_pkl_path = os.path.join(pkl_dir, 'clip', 'ocr_text_image_emb.pkl')
with open(ocr_pkl_path, 'rb') as f:
    ocr_dict = pickle.load(f)
print(f"Loaded {len(ocr_dict)} ocr embeddings.")

# --- 3) Get dimensions for each embedding type ---
def _tensor_dim(x) -> int | None:
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.shape[-1]
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _tensor_dim(x[0])
    return None

def first_tensor_dim(d: dict, key: str) -> int:
    for v in d.values():
        if isinstance(v, dict) and key in v:
            dim = _tensor_dim(v[key])
            if dim is not None:
                return dim
    raise ValueError(f"{key} embedding not found")

D_title = first_tensor_dim(title_dict, key='title')
D_desc  = first_tensor_dim(desc_dict, key='description_mean')
D_trans = first_tensor_dim(transcript_dict, key='transcript_mean')
D_ocr_t = first_tensor_dim(ocr_dict, key="text")
D_ocr_i = first_tensor_dim(ocr_dict, key="image")

z_title = torch.zeros(D_title)
z_desc  = torch.zeros(D_desc)
z_trans = torch.zeros(D_trans)
z_ocr_t = torch.zeros(D_ocr_t)
z_ocr_i = torch.zeros(D_ocr_i)

def _safe_tensor(x, zerovec: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return zerovec
        sub = [_safe_tensor(el, zerovec) for el in x]
        return torch.stack(sub, dim=0).mean(dim=0)
    return zerovec

def make_feature_vec(vid: str) -> torch.Tensor:
    title = _safe_tensor(title_dict.get(vid, {}).get('title'), z_title)
    desc  = _safe_tensor(desc_dict .get(vid, {}).get('description_mean'), z_desc)
    trans = _safe_tensor(transcript_dict.get(vid, {}).get('transcript_mean'), z_trans)

    if vid in ocr_dict:
        o_text = _safe_tensor(ocr_dict[vid].get('text'),  z_ocr_t)
        o_img  = _safe_tensor(ocr_dict[vid].get('image'), z_ocr_i)
    else:
        o_text, o_img = z_ocr_t, z_ocr_i

    return torch.cat([title, desc, trans, o_text, o_img], dim=-1).float()

# --- 4) Build graph for each original video ---
related_files = os.listdir(related_dir)
related_ids = [os.path.splitext(f.replace("w_metadata_", ""))[0] for f in related_files]
related_ids = set(related_ids)
print(f'related_ids: {len(related_ids)}')
orig_ids = orig_ids & related_ids

for orig_vid in tqdm(orig_ids, desc="Build graphs"):
    pt_file = os.path.join(output_dir, f"graph_{orig_vid}_bipartite_title_sim_common_channel.pt")
    if os.path.exists(pt_file):
        continue

    df_o = orig_df[orig_df['video_id']==orig_vid]
    df_r = read_related(orig_vid)
    all_video_ids = [orig_vid] + (df_r['video_id'].drop_duplicates().tolist() if df_r is not None else [])
    feature_tensors = [ make_feature_vec(vid) for vid in all_video_ids ]
    X = torch.stack(feature_tensors, dim=0)

    # Comment authors of original video
    orig_authors = set(df_o['comment_author_channel_id'].dropna()) | set(df_o['reply_author_channel_id'].dropna())
    related_map = {}
    if df_r is not None:
        for vid, grp in df_r.groupby('video_id'):
            related_map[vid] = set(grp['author_channel_id'].dropna())

    # 1. Edges based on shared authors
    orig_edges, orig_weights = [], []
    for vid, authors in related_map.items():
        w = len(orig_authors & authors)
        if w>0:
            idx = all_video_ids.index(vid)
            orig_edges.append((0, idx))
            orig_weights.append(w)
    rev_edges   = [(dst, src) for (src,dst) in orig_edges]
    rev_weights = orig_weights.copy()

    # 2. Edges based on title similarity
    all_emb = np.stack([title_dict[vid]['title'].numpy() for vid in all_video_ids], axis=0)
    sim_matrix = cosine_similarity(all_emb)

    sim_edges, sim_weights = [], []
    for i in range(len(all_video_ids)):
        for j in range(i+1, len(all_video_ids)):
            if sim_matrix[i, j] >= threshold:
                sim_edges.append((i, j))
                sim_edges.append((j, i))
                sim_weights.extend([sim_matrix[i, j]] * 2)

    # 3. Edges based on common channel ID
    orig_channel = df_o['channel_id'].iloc[0]
    channel_map = {orig_vid: orig_channel}
    for _, row in df_r[['video_id','channel_id']].drop_duplicates().iterrows():
        channel_map[row['video_id']] = row['channel_id']

    chan_edges, chan_weights = [], []
    for i in range(len(all_video_ids)):
        for j in range(i+1, len(all_video_ids)):
            vid_i, vid_j = all_video_ids[i], all_video_ids[j]
            if channel_map.get(vid_i) == channel_map.get(vid_j) and channel_map.get(vid_i) is not None:
                chan_edges.append((i, j))
                chan_edges.append((j, i))
                chan_weights.extend([1.0, 1.0])

    all_edges   = orig_edges + rev_edges + sim_edges + chan_edges
    all_weights = orig_weights + rev_weights + sim_weights + chan_weights

    edge_index  = torch.tensor(all_edges, dtype=torch.long).t()
    edge_weight = torch.tensor(all_weights, dtype=torch.float)

    if edge_index.numel() > 0:
        if edge_index.max() >= len(all_video_ids) or edge_index.min() < 0:
            print(f"ERROR in graph {orig_vid}: edge_index out of bounds!")
            raise ValueError(f"Edge index out of bounds for graph {orig_vid}")

    data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight)
    torch.save(data, pt_file)
