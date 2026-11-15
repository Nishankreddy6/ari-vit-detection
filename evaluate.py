"""Test the trained model.   Run:  python evaluate.py"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core import ARIDataset, load_checkpoint
from train import auc_score, get_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_checkpoint(a.checkpoint, device)
    ds = ARIDataset(a.data_dir, a.split)
    if ds.classes != classes:
        raise ValueError(f"Class folders {ds.classes} do not match the model's classes {classes}")
    probs, labels = get_probs(model, DataLoader(ds, 8), device)
    preds = probs.argmax(1)

    print(classification_report(labels, preds, target_names=classes, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    print("ROC-AUC:", round(float(auc_score(probs, labels)), 4))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/metrics.json").write_text(json.dumps(
        {"accuracy": float((preds == labels).mean()), "roc_auc": float(auc_score(probs, labels)),
         "confusion_matrix": confusion_matrix(labels, preds).tolist()}, indent=2))
    print("Saved outputs/metrics.json")


if __name__ == "__main__":
    main()
