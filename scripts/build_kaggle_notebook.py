#!/usr/bin/env python3
"""Generate the fine-tune Kaggle notebook.

Writes the same notebook to:
    - <repo>/kaggle_finetune.ipynb            (deliverable, task-specified location)
    - <repo>/notebooks/kaggle_finetune.ipynb   (canonical location)

Run:  python3 scripts/build_kaggle_notebook.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO, "kaggle_finetune.ipynb")
OUT_NB = os.path.join(REPO, "notebooks", "kaggle_finetune.ipynb")

MD_INTRO = """\
# TERRAIN — multi-sensor SwinUNet fine-tune (Kaggle)

Private repo, so the code rides in as a Kaggle input (`terrain-code`), and the
processed tiles ship as **two separate inputs** — `terrain-tiles-cartosat3` and
`terrain-tiles-liss4-s2`. Cell 4 symlinks both into `/kaggle/working/data/processed/`
so the repo's `MultiSensorDataset` (which takes a single root) works unchanged.

**Inputs to add first (three):** `terrain-code`, `terrain-tiles-cartosat3`,
`terrain-tiles-liss4-s2` (mounts as `/kaggle/input/datasets/mayanktiwari18/<slug>`).
**Split:** per-sensor 80/20 tile-index split — LISS-IV now **trains** too (the old
full-sensor holdout hid the cross-GSD gap). Every sensor contributes train AND val.
**Training stack (finetune.py):**
  - fine-tunes from `outputs/models/terrain_best.pth` if present (the existing
    5-epoch checkpoint), else starts from a random decoder + pretrained encoder
  - reloads the Swin encoder from timm ImageNet weights (resets a degraded backbone)
  - freezes the encoder for the first `FREEZE_ENCODER_EPOCHS` epochs
  - two AdamW groups (ENCODER_LR / DECODER_LR) + OneCycleLR (warmup 10%)
  - Focal(+pos_weight) + Dice composite loss, sparse-tile oversampling
  - AMP (FP16) + grad accumulation (GRAD_ACCUM) + early stopping (patience 8)
**Accelerator:** GPU T4 x2 (or T4). Single CUDA device (device 0).
**Internet = ON — required.** Cell 6 downloads timm's ImageNet-pretrained Swin
weights (Hugging Face). If that download fails, the encoder reload falls back to
random init.
"""

CELL_ENV = """\
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, f"{p.total_memory/1e9:.1f} GB")
"""

CELL_DEPS = """\
!pip install -q --no-input timm albumentations segmentation-models-pytorch
"""

CELL_CODE = """\
!cp -r /kaggle/input/datasets/mayanktiwari18/terrain-code/. /kaggle/working/
%cd /kaggle/working
!ls
"""

CELL_CONFIG = '''\
import os
import shutil

import config

# Tiles arrive as two separate Kaggle inputs. Map every sensor to its input.
# (Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>.)
TILE_INPUTS = {
    "cartosat3": "/kaggle/input/datasets/mayanktiwari18/terrain-tiles-cartosat3",
    "liss4":     "/kaggle/input/datasets/mayanktiwari18/terrain-tiles-liss4-s2",
    "sentinel2": "/kaggle/input/datasets/mayanktiwari18/terrain-tiles-liss4-s2",
}


def _sensor_root(base, sensor):
    """Return the <sensor> dir (containing image/ + mask/) inside `base`, probing
    the likely archive layouts (data/processed, processed, or bare)."""
    for r in (f"{base}/data/processed", f"{base}/processed", base):
        cand = os.path.join(r, sensor)
        if os.path.isdir(os.path.join(cand, "image")):
            return cand
    raise FileNotFoundError(
        f"no '{sensor}/image' found under {base} "
        f"(tried data/processed, processed, bare layouts)")


def _find_norm_stats():
    """_norm_stats.json is ONE combined file for all three sensors — find it."""
    for base in set(TILE_INPUTS.values()):
        for rel in ("data/processed/_norm_stats.json",
                    "processed/_norm_stats.json",
                    "_norm_stats.json"):
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        "no _norm_stats.json in the tile inputs (expected one combined file "
        "with keys cartosat3/liss4/sentinel2)")


# Unify the two tile inputs under /kaggle/working/data/processed via symlinks so
# MultiSensorDataset's single-root logic, per-sensor percentile norm, LISS zero-fill,
# multiscale RandomScale and the per-sensor 80/20 split all work unchanged.
config.DATA_DIR = "/kaggle/working/data/"
processed = os.path.join(config.DATA_DIR, "processed")
os.makedirs(processed, exist_ok=True)
for sensor, base in TILE_INPUTS.items():
    src = _sensor_root(base, sensor)
    link = os.path.join(processed, sensor)
    if not os.path.lexists(link):
        os.symlink(src, link)
    print(f"linked {sensor} -> {src}")

stats_src = _find_norm_stats()
config.NORM_STATS_PATH = os.path.join(processed, "_norm_stats.json")
if not os.path.isfile(config.NORM_STATS_PATH):
    shutil.copy(stats_src, config.NORM_STATS_PATH)
print("NORM_STATS_PATH ->", config.NORM_STATS_PATH)

# Kaggle-side overrides: cache decoded tiles (4 CPUs = the bottleneck, not the GPU)
# so AMP + batch 32 stop re-reading GeoTIFFs every epoch. Everything else (EPOCHS,
# BATCH_SIZE, AMP, GRAD_ACCUM, OneCycleLR, POS_WEIGHT, sparse oversampling) now
# lives in config.py.
config.CACHE_SIZE = 10000

print("DATA_DIR =", config.DATA_DIR, "| EPOCHS =", config.EPOCHS,
      "| BATCH_SIZE =", config.BATCH_SIZE, "| AMP =", config.AMP,
      "| VAL_SPLIT_FRACTION =", config.VAL_SPLIT_FRACTION,
      "| POS_WEIGHT =", config.POS_WEIGHT, "| SCALE_AUG =", config.SCALE_AUG,
      "| GSD =", config.GSD_METERS)
'''

CELL_DATASET = """\
from segmentation.dataset import MultiSensorDataset

# Multiscale handled end-to-end by the repo dataset:
#   - per-sensor percentile normalize (p2-p98) -> [0,1]
#   - LISS-IV 3-band -> 4 channels (pad band normalized via config.PAD_VALUE)
#   - cross-GSD RandomScale (0.75x-1.5x) then Resize(256) -> multiscale aug
#   - sensor-balanced batches so C3 (47k tiles) doesn't drown S2 / LISS
# Split (domain-gap fix): per-sensor 80/20 by tile index — train = first 80% of
# EVERY sensor, val = last 20% of EVERY sensor. LISS-IV is no longer held out.
tr = MultiSensorDataset(split="train")
va = MultiSensorDataset(split="val")

# Real internal state: sensor->id map + per-sensor tile index lengths.
print("train:", len(tr), "| sensors:", list(tr.sensor_ids),
      "| per-sensor tiles:", {s: len(v) for s, v in tr.per_sensor.items()})
print("val  :", len(va), "| sensors:", list(va.sensor_ids),
      "| per-sensor tiles:", {s: len(v) for s, v in va.per_sensor.items()})

img, mask, meta = tr[0]
print("sample:", img.shape, mask.shape, "meta[sensor_id, gsd_m]", meta,
      f"| img range [{float(img.min()):.2f}, {float(img.max()):.2f}]")

assert img.shape == (4, 256, 256) and mask.shape == (1, 256, 256)
assert set(tr.sensor_ids) == set(va.sensor_ids), "both splits must cover all sensors"
tr_keys = {s: set(v) for s, v in tr.per_sensor.items()}
va_keys = {s: set(v) for s, v in va.per_sensor.items()}
for s in tr_keys:
    assert tr_keys[s].isdisjoint(va_keys[s]), f"leak: {s} train/val overlap"
print("split OK: per-sensor 80/20, no train/val leak, all sensors train")
"""

CELL_CONFIG_SUMMARY = """\
import config

# Fine-tune knobs in effect for this run.
print("EPOCHS", config.EPOCHS, "| GRAD_ACCUM", config.GRAD_ACCUM,
      "| FREEZE_ENCODER_EPOCHS", config.FREEZE_ENCODER_EPOCHS)
print("ENCODER_LR", config.ENCODER_LR, "| DECODER_LR", config.DECODER_LR,
      "| WARMUP_FRACTION", config.WARMUP_FRACTION)
print("POS_WEIGHT", config.POS_WEIGHT, "| FINETUNE_PATIENCE", config.FINETUNE_PATIENCE)
print("INIT_CKPT", config.FINETUNE_INIT_CKPT, "| SAVE", config.FINETUNE_SAVE_PATH)
"""

CELL_FINETUNE = """\
import torch
import finetune

torch.backends.cudnn.benchmark = True

# Optional: if you also uploaded the existing 5-epoch checkpoint as
# /kaggle/working/outputs/models/terrain_best.pth it is used as the init; otherwise
# finetune.py starts from a random decoder + pretrained encoder (documented fallback).
model, history = finetune.run_finetune(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    save_path="/kaggle/working/terrain_finetuned_best.pth",
    device=torch.device("cuda"),
)

# history_finetune.json is written next to the ckpt by run_finetune.
"""

CELL_PLOT = """\
import json
import matplotlib.pyplot as plt

json.dump(history, open("/kaggle/working/history_finetune.json", "w"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history])
axes[0].set_title("train loss")
axes[1].plot([h["epoch"] for h in history], [h["val_iou"] for h in history])
axes[1].set_title("val mean IoU")
plt.show()

print("best val IoU:", max(h["val_iou"] for h in history))
"""

CELL_EVAL = """\
import os
import random

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from segmentation.evaluate import compute_ious, sparse_vs_dense_ious
from segmentation.inference import run_segmentation, save_mask
from segmentation.dataset import _normalize, _load_norm_stats

device = torch.device("cuda")
ckpt = "/kaggle/working/terrain_finetuned_best.pth"

# 1) threshold-swept metrics on the val split (all three sensors)
res = compute_ious(model, va, device=device, name="val(all sensors)")
svd = sparse_vs_dense_ious(model, va, device=device)
print("best threshold:", res["best_threshold"], "| pooled IoU:", res["best"]["iou"])

# 2) sample overlays (image | GT | pred | prob) + georeferenced masks
os.makedirs("/kaggle/working/preds", exist_ok=True)
random.seed(0)
sample = random.sample(va.items, min(8, len(va.items)))
stats = _load_norm_stats()
fig, axes = plt.subplots(len(sample), 4, figsize=(16, 4 * len(sample)))
axes = np.atleast_2d(axes)
for r, (sensor, img_path, mask_path) in enumerate(sample):
    with rasterio.open(img_path) as ds:
        arr = ds.read().astype(np.float32)
    rgb = np.transpose(np.clip(_normalize(arr, sensor, stats, config.IN_CHANNELS)[:3], 0, 1), (1, 2, 0))
    with rasterio.open(mask_path) as ds:
        gt = ds.read(1)
    mask, prob = run_segmentation(img_path, checkpoint_path=ckpt, device=device)
    axes[r, 0].imshow(rgb);        axes[r, 0].set_title("image"); axes[r, 0].axis("off")
    axes[r, 1].imshow(gt, cmap="gray");  axes[r, 1].set_title("GT");   axes[r, 1].axis("off")
    axes[r, 2].imshow(mask, cmap="gray"); axes[r, 2].set_title("pred"); axes[r, 2].axis("off")
    axes[r, 3].imshow(prob, cmap="jet", vmin=0, vmax=1); axes[r, 3].set_title("prob"); axes[r, 3].axis("off")
    save_mask(mask, img_path, f"/kaggle/working/preds/{os.path.basename(mask_path)}")
fig.tight_layout()
fig.savefig("/kaggle/working/preds/sample_overlays.png", dpi=90)
plt.show()
print("saved /kaggle/working/preds/* (georeferenced masks + overlay grid)")
"""

CELL_INFER = """\
# Full-coverage mask inference: writes
#   /kaggle/working/masks/<sensor>/<tile_id>_mask.png   (0/255 PNG)
#   /kaggle/working/masks/<sensor>/<tile_id>_mask.tif   (georeferenced uint8)
#   /kaggle/working/masks/<sensor>_mosaic.tif           (rasterio.merge of the sensor)
# Threshold = config.INFER_THRESHOLD (0.4). Uses terrain_finetuned_best.pth.
import inference

counts = inference.run_inference(
    input_dir="/kaggle/working/data/processed",
    output_dir="/kaggle/working/masks",
    checkpoint="/kaggle/working/terrain_finetuned_best.pth",
    device=torch.device("cuda"),
)
"""

CELL_DOWNLOAD = """\
# Persist artifacts BEFORE the session ends (download from the Output/File tabs).
!zip -q -r /kaggle/working/outputs.zip \
    /kaggle/working/terrain_finetuned_best.pth \
    /kaggle/working/terrain_finetuned_best_best.json \
    /kaggle/working/history_finetune.json \
    /kaggle/working/preds \
    /kaggle/working/masks \
    -x "*.ipynb_checkpoints*" -x ".*"
!ls -la /kaggle/working/outputs.zip
print("DOWNLOAD /kaggle/working/outputs.zip from the output panel now.")
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(MD_INTRO),
    md("## Cell 1 — environment check"),
    code(CELL_ENV),
    md("## Cell 2 — deps (torch/rasterio are preinstalled on Kaggle)"),
    code(CELL_DEPS),
    md("## Cell 3 — pull the repo code into the workspace"),
    code(CELL_CODE),
    md("## Cell 4 — config + unify the two tile inputs (symlink farm)"),
    code(CELL_CONFIG),
    md("## Cell 5 — dataset / multiscale / per-sensor 80-20 split smoke test"),
    code(CELL_DATASET),
    md("## Cell 6 — fine-tune knobs in effect"),
    code(CELL_CONFIG_SUMMARY),
    md("## Cell 7 — fine-tune (the long cell; encoder reload + freeze + OneCycleLR + "
       "AMP; best ckpt auto-saved to working)"),
    code(CELL_FINETUNE),
    md("## Cell 8 — history + curve"),
    code(CELL_PLOT),
    md("## Cell 9 — evaluate on the val split (all sensors) + sample binary masks"),
    code(CELL_EVAL),
    md("## Cell 10 — full-coverage masks + per-sensor mosaics"),
    code(CELL_INFER),
    md("## Cell 11 — download artifacts (ckpt + history + masks + mosaics)"),
    code(CELL_DOWNLOAD),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for path in (OUT_ROOT, OUT_NB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print("wrote", path)
