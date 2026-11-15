"""Shared code: image preprocessing (OpenCV + NumPy), dataset, ViT model, prediction."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------- preprocessing ----------
def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img_rgb, size=IMG_SIZE):
    """Resize -> scale to 0-1 -> normalize -> channels-first."""
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def augment(img_rgb):
    """Small rotation/scale/shift + brightness/contrast. No flips (chest anatomy is not symmetric)."""
    rng = np.random.default_rng()
    h, w = img_rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    m[:, 2] += (rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h)
    out = cv2.warpAffine(img_rgb, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return cv2.convertScaleAbs(out, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))


# ---------- dataset ----------
class ARIDataset(Dataset):
    """Folder layout: data/<split>/<class_name>/<image files>"""

    def __init__(self, root, split, train=False):
        folder = Path(root) / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder.resolve()}")
        self.classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        self.samples = [
            (p, i)
            for i, c in enumerate(self.classes)
            for p in sorted((folder / c).iterdir())
            if p.suffix.lower() in EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"No images found in {folder.resolve()}")
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_rgb(path)
        if self.train:
            img = augment(img)
        return torch.from_numpy(preprocess(img)), label

    def class_counts(self):
        return np.bincount([l for _, l in self.samples], minlength=len(self.classes))


# ---------- model ----------
def build_model(num_classes=2, pretrained=True):
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt["classes"]


@torch.no_grad()
def predict_rgb(model, classes, img_rgb, device="cpu"):
    x = torch.from_numpy(preprocess(img_rgb)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top = int(probs.argmax())
    return {
        "label": classes[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }
