"""
Dataset sanity check: render image + road-mask pairs for a few tiles per sensor.

Saves montages to outputs/check_dataset/<sensor>_sample_*.png so we can eyeball:
  - image/mask alignment per sensor
  - normalization quality (not too dark / not clipped)
  - mask quality, esp. the known Sentinel-2/Bengaluru node-blob risk

Usage:
    python experiments/check_dataset.py [--n 3] [--out outputs/check_dataset]
"""

import os
import sys

import numpy as np
import torch
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from segmentation.dataset import MultiSensorDataset  # noqa: E402


def render_rgb(img, mask):
    """img: [C,H,W] normalized [0,1] with bands (NIR,R,G,B). RGB = R,G,B."""
    rgb = img[[1, 2, 3]].clamp(0, 1)           # (3,H,W)
    # red overlay for roads
    road = mask[0] > 0.5
    out = rgb.clone()
    out[0, road] = 1.0
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "check_dataset"))
    args = parser.parse_args()

    ds = MultiSensorDataset(split="train", augment=False)   # train split includes liss+c3
    ds_all = MultiSensorDataset(split="none", augment=False)  # all sensors incl. sentinel2

    os.makedirs(args.out, exist_ok=True)

    # per-sensor: pick a few non-empty tiles directly from the index
    from collections import defaultdict
    picked = defaultdict(list)
    for sensor, inds in ds_all.per_sensor.items():
        for idx in inds:
            if len(picked[sensor]) >= args.n:
                break
            img, mask, _ = ds_all[idx]
            if mask.sum() > 0.5:
                picked[sensor].append((idx, img, mask))

    for sensor, samples in picked.items():
        grids = []
        for k, (idx, img, mask) in enumerate(samples[:args.n]):
            rgb = render_rgb(img, mask)                       # overlay
            grids.append(rgb)
            tile = os.path.basename(ds_all.items[idx][1])
            print(f"{sensor:10s} tile {tile} road_px={int(mask.sum())}")
        grid = torch.stack(grids).float()
        grid_img = make_grid(grid, nrow=args.n, padding=4, normalize=False).cpu()
        save = os.path.join(args.out, f"{sensor}_check.png")
        from torchvision.utils import save_image
        save_image(grid_img, save)
        print(f"  saved -> {save}")

    # quick S2 mask-quality stat: mean road fraction across all S2 tiles
    s2 = ds_all.per_sensor.get("sentinel2", [])
    if s2:
        fracs = []
        for i in s2:
            _, mask, _ = ds_all[i]
            fracs.append(mask.mean().item())
        print(f"\nSentinel-2 masks: n={len(fracs)} mean_road_frac={np.mean(fracs):.4f}")
        low = sum(1 for f in fracs if f < 0.005)
        print(f"tiles with road_frac<0.5% : {low} ({100*low/len(fracs):.1f}%)")


if __name__ == "__main__":
    main()