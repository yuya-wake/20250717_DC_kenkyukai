import sys
sys.path.append('/home/user/model')

from clip_japanese_base.configuration_clyp import CLYPConfig
from clip_japanese_base.model import create_vision_encoder
from clip_japanese_base.modeling_clyp import CLYPModel

import os
import pandas as pd
import pickle
from tqdm import tqdm_notebook as tqdm
from transformers import AutoTokenizer
from PIL import Image
from transformers import AutoImageProcessor
import torch

model_dir = "/home/user/model/clip_japanese_base"
config = CLYPConfig.from_pretrained(model_dir)
model = CLYPModel.from_pretrained(model_dir, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

mnt_path = os.getenv('MNT_PATH') or ""
file_path = os.path.join(mnt_path, 'complete_data', 'complete_data.csv')
df = pd.read_csv(file_path, encoding='utf-8', usecols=['video_id', 'ocr'])

thumb_dir = os.path.join(mnt_path, 'thumbnails')
df = df.drop_duplicates(subset=['video_id'])

save_dir = os.path.join(mnt_path, 'pkl', 'clip')
os.makedirs(save_dir, exist_ok=True)

def embed_text(text):
    text = str(text)
    sentences = [s.strip() for s in text.split("。") if s.strip()]  # 文分割
    embeddings = []

    base_tokenizer = tokenizer.tokenizer

    for sentence in sentences:
        inputs = base_tokenizer([sentence], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = [
            [base_tokenizer.cls_token_id] + ids.tolist() for ids in inputs["input_ids"]
        ]
        attention_mask = [[1] + am.tolist() for am in inputs["attention_mask"]]
        position_ids = [list(range(0, len(input_ids[0])))] * len(input_ids)

        # dict 構築
        model_inputs = {
            "input_ids": torch.tensor(input_ids).to(device),
            "attention_mask": torch.tensor(attention_mask).to(device),
            "position_ids": torch.tensor(position_ids).to(device),
        }

        with torch.no_grad():
            text_feat = model.text_encoder(model_inputs)
        
        embeddings.append(text_feat.squeeze().cpu())
    
    return embeddings  

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    with torch.no_grad():
        image_feat = model.get_image_features(**image_inputs)
    
    return image_feat.squeeze().cpu()  # [D]

embedding_dict = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding text+image"):
    vid = row['video_id']
    ocr_text = row['ocr']
    image_path = os.path.join(thumb_dir, f"{vid}.jpg")
    # 初期化
    embedding_dict.setdefault(vid, {})

    # テキスト embedding
    embedding_dict[vid]['text'] = embed_text(ocr_text)

    # 画像 embedding（ファイルが存在する場合のみ）
    if os.path.isfile(image_path):
        embedding_dict[vid]['image'] = embed_image(image_path)
    else:
        print(f"Image not found: {image_path}")
        embedding_dict[vid]['image'] = None

save_path = os.path.join(save_dir, 'ocr_text_image_emb.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(embedding_dict, f)
