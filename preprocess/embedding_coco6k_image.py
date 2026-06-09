import json
import open_clip
from PIL import Image
import torch
from tqdm import tqdm

with open("data/SC-Captioner-data/llamafactory_json/train_coco6k_sft.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14",
    pretrained="laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
)
model = model.to("cuda").eval()

images = []
for item in data:
    image_path = item["images"][0]
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    images.append(image)

images = torch.stack(images)
embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(images), 128)):
        batch_embeddings = model.encode_image(images[i:i+128].to("cuda"))
        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
        embeddings.append(batch_embeddings)

embeddings = torch.cat(embeddings, dim=0)
torch.save(embeddings, "data/SC-Captioner-data/llamafactory_json/train_coco6k_image_ViT-H-14.pt")
