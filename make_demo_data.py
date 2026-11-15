"""Creates SYNTHETIC demo images so the project can run end to end.

These are drawings made by code, NOT real X-rays. Results on them mean nothing medically.
Run:  python make_demo_data.py
"""
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
COUNTS = {"train": 60, "val": 15, "test": 15}      # images per class
CLASSES = ["demo_clear", "demo_hazy"]              # hazy = bright cloudy patches in the lungs


def make_image(hazy, rng):
    img = np.full((SIZE, SIZE), 20, np.float32)
    cv2.ellipse(img, (128, 128), (100 + int(rng.integers(-5, 6)), 120), 0, 0, 360, 70, -1)   # body
    lungs = []
    for cx in (88, 168):                                                                       # two lungs
        cx += int(rng.integers(-6, 7)); cy = 125 + int(rng.integers(-6, 7))
        rx, ry = 40 + int(rng.integers(-5, 6)), 70 + int(rng.integers(-6, 7))
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 35, -1)
        lungs.append((cx, cy, rx, ry))
    cv2.line(img, (128, 30), (128, 235), 110, 10)                                              # spine
    for y in range(55, 205, 22):                                                               # ribs
        cv2.line(img, (128, y), (45, y + 18), 85, 3)
        cv2.line(img, (128, y), (211, y + 18), 85, 3)
    if hazy:
        layer = np.zeros_like(img)
        for cx, cy, rx, ry in lungs:
            for _ in range(int(rng.integers(2, 5))):
                x = cx + int(rng.integers(-rx // 2, rx // 2 + 1)); y = cy + int(rng.integers(-ry // 2, ry // 2 + 1))
                cv2.circle(layer, (x, y), int(rng.integers(10, 22)), float(rng.uniform(40, 80)), -1)
        img += cv2.GaussianBlur(layer, (0, 0), 10)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(0)
    for split, n in COUNTS.items():
        for cls in CLASSES:
            folder = Path("data") / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(1, n + 1):
                cv2.imwrite(str(folder / f"img_{i:03d}.jpg"), make_image(cls == "demo_hazy", rng))
    print("Done. Demo images are in the data folder.")


if __name__ == "__main__":
    main()
