import os, json
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import open_clip
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, jsonl_path, preprocess):
        self.paths = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.paths.append(json.loads(line)["image_path"])
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img), idx
        
def collate(batch):
    imgs, idxs = zip(*batch)
    return torch.stack(imgs, 0), torch.tensor(idxs, dtype=torch.long)

def main():
    # --- DDP init ---
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # --- model & preprocess ---
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    model = model.to(device).eval()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- data ---
    ds = ImageDataset("data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M.jsonl", preprocess)
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    dl = DataLoader(
        ds, batch_size=128, sampler=sampler,
        num_workers=max(1, (os.cpu_count() or 4)-2), pin_memory=True, collate_fn=collate
    )

    # --- extract per-rank ---
    local_idx, local_emb = [], []
    with torch.no_grad():
        for imgs, idxs in tqdm(dl):
            imgs = imgs.to(device, non_blocking=True)
            feats = model.module.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            local_idx.append(idxs.cpu())
            local_emb.append(feats.cpu())

    local_idx = torch.cat(local_idx, 0)
    local_emb = torch.cat(local_emb, 0)

    # --- gather to rank0 ---
    part = {"idx": local_idx, "emb": local_emb}
    world = dist.get_world_size()
    gather = [None] * world if rank == 0 else None
    dist.gather_object(part, gather, dst=0)

    # --- assemble & save on rank0 ---
    if rank == 0:
        N = len(ds)
        emb_dim = gather[0]["emb"].shape[1]
        full = torch.empty((N, emb_dim), dtype=torch.float32)
        for g in gather:
            full[g["idx"]] = g["emb"]
        torch.save(full, "data/DenseFusion-1M/DenseFusion-1M/DenseFusion-1M_image_ViT-H-14.pt")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
