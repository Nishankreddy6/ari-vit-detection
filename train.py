"""Train the ViT.   Run:  python train.py --data-dir data --epochs 5"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from core import ARIDataset, build_model


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def auc_score(probs, labels):
    if probs.shape[1] == 2:
        return roc_auc_score(labels, probs[:, 1])
    return roc_auc_score(labels, probs, multi_class="ovr")


def limit_per_class(ds, n):
    """Keep only the first n images of each class (used for quick test runs)."""
    kept, seen = [], {}
    for path, label in ds.samples:
        if seen.get(label, 0) < n:
            kept.append((path, label))
            seen[label] = seen.get(label, 0) + 1
    ds.samples = kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--workers", type=int, default=0)  # 0 is safest on Windows
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny test run (60 train + 20 val images per class)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds = ARIDataset(a.data_dir, "train", train=True)
    val_ds = ARIDataset(a.data_dir, "val")
    if a.quick:
        limit_per_class(train_ds, 60)
        limit_per_class(val_ds, 20)
    classes, counts = train_ds.classes, train_ds.class_counts()
    print("Classes:", classes, "| train images per class:", counts.tolist(), "| val images:", len(val_ds))

    train_loader = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers)
    val_loader = DataLoader(val_ds, a.batch_size, num_workers=a.workers)

    model = build_model(len(classes)).to(device)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    Path("checkpoints").mkdir(exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        sched.step()

        probs, labels = get_probs(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / n,
               "val_acc": float((probs.argmax(1) == labels).mean()), "val_auc": float(auc_score(probs, labels))}
        history.append(row); print(row)
        if row["val_auc"] > best:
            best = row["val_auc"]
            torch.save({"model_state": model.state_dict(), "classes": classes}, "checkpoints/best.pt")
            print("  saved checkpoints/best.pt")

    Path("checkpoints/history.json").write_text(json.dumps(history, indent=2))
    print("Finished. Best validation AUC:", round(best, 4))


if __name__ == "__main__":
    main()
