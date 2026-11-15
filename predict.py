"""Classify one image.   Run:  python predict.py "C:\\path\\to\\xray.jpeg" """
import sys

import torch

from core import load_checkpoint, load_rgb, predict_rgb

if len(sys.argv) < 2:
    sys.exit('Usage: python predict.py "path/to/image.jpeg"')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, classes = load_checkpoint("checkpoints/best.pt", device)
print(predict_rgb(model, classes, load_rgb(sys.argv[1]), device))
