import json
import torch
from sentence_transformers import SentenceTransformer

data = []
with open("data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

captions = [item["caption"] for item in data]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode(
    captions,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)
torch.save(embeddings, f"data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M_caption_all-mpnet-base-v2.pt")
