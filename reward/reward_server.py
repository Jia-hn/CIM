import argparse
from typing import List
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

DEVICE = torch.device("cuda:0")
print(f"DEVICE:{DEVICE}")

TEXT_EMB = None
IMAGE_EMB = None

class RewardData(BaseModel):
    response: str
    index: int
    k: int

def init_db(text_emb_paths: List[str], image_emb_paths: List[str]):
    global TEXT_EMB, IMAGE_EMB

    text_emb_all = []
    image_emb_all = []
    for text_emb_path in text_emb_paths:
        text_emb_all.append(torch.load(text_emb_path, map_location="cpu").float())
    for image_emb_path in image_emb_paths:
        image_emb_all.append(torch.load(image_emb_path, map_location="cpu").float())

    text_emb_all = torch.cat(text_emb_all, dim=0)
    image_emb_all = torch.cat(image_emb_all, dim=0)

    assert text_emb_all.shape[0] == image_emb_all.shape[0], "Mismatch in number of rows between text and image embeddings"
    print(f"text_emb_all.shape: {text_emb_all.shape}, image_emb_all.shape: {image_emb_all.shape}")

    text_emb_all = F.normalize(text_emb_all, p=2, dim=1)
    image_emb_all = F.normalize(image_emb_all, p=2, dim=1)

    TEXT_EMB = text_emb_all.to(DEVICE, non_blocking=True)
    IMAGE_EMB = image_emb_all.to(DEVICE, non_blocking=True)


@torch.no_grad()
def topk_indices(query_emb, emb_matrix, k=5):
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)
    sims = torch.matmul(query_emb, emb_matrix.T).squeeze(0)
    _, topk_idx = torch.topk(sims, k=k)
    return topk_idx


def build_app(model_path: str) -> FastAPI:
    app = FastAPI()

    model = SentenceTransformer(model_path, device=str(DEVICE))

    @app.post("/cim")
    @torch.no_grad()
    def cim(data: RewardData):
        k = data.k

        q = model.encode([data.response], convert_to_tensor=True, normalize_embeddings=True).float()

        top_inds = topk_indices(q, TEXT_EMB, k)

        target_vec = IMAGE_EMB[data.index].to(dtype=torch.float32)
        cand_vecs = IMAGE_EMB.index_select(0, top_inds).to(dtype=torch.float32)

        sims = cand_vecs @ target_vec
        weights = 0.5 ** torch.arange(sims.shape[0], device=DEVICE, dtype=torch.float32)
        sim_score = float((weights * sims).sum().item())

        mrl_score = float(torch.linalg.norm(cand_vecs.mean(dim=0), ord=2).item())

        return sim_score + mrl_score

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--text_emb_paths", type=str, nargs='+',
                        default=[
                            "data/SC-Captioner-data/llamafactory_json/train_coco6k_caption_all-mpnet-base-v2.pt",
                            "data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M_caption_all-mpnet-base-v2.pt"
                        ])
    parser.add_argument("--image_emb_paths", type=str, nargs='+',
                        default=[
                            "data/SC-Captioner-data/llamafactory_json/train_coco6k_image_ViT-H-14.pt",
                            "data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M_image_ViT-H-14.pt"
                        ])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3545)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    init_db(args.text_emb_paths, args.image_emb_paths)

    app = build_app(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
