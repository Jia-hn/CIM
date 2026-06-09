import json
import torch
from sentence_transformers import SentenceTransformer

with open("data/SC-Captioner-data/llamafactory_json/train_coco6k_sft.json", "r", encoding="utf-8") as f:
    data = json.load(f)

captions = [item["conversations"][1]["value"] for item in data]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode(
    captions,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)
torch.save(embeddings, "data/SC-Captioner-data/llamafactory_json/train_coco6k_caption_all-mpnet-base-v2.pt")
