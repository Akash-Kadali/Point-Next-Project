import sys
from pathlib import Path
import argparse
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import load_config
from datasets.loader import build_dataloader
from models.factory import build_model
from training.checkpoint import save_checkpoint
from self_supervised.autoencoder import PointNeXtAutoEncoder, chamfer_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cls_pointnext_s.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/ssl_pointnext_autoencoder/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config).to_dict()
    loader, _ = build_dataloader(cfg["dataset"], split="train")

    backbone = build_model(dict(cfg["model"]))

    model = PointNeXtAutoEncoder(
        backbone=backbone,
        embedding_dim=512,
        num_points=cfg["dataset"]["num_points"],
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            xyz = batch["xyz"].to(args.device)
            features = batch["features"].to(args.device)

            recon, emb = model(xyz, features)
            loss = chamfer_distance(recon, xyz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: reconstruction loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(args.out, model.backbone, config=cfg)
            print(f"Saved encoder to {args.out}")


if __name__ == "__main__":
    main()