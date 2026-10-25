"""
TERRAIN inference entrypoint (standalone — runs on Kaggle without finetune.py).

Loads the fine-tuned checkpoint (falls back to the pre-finetune one), converts
every raw tile under <input_dir> into a binary road mask, and writes:

    <output_dir>/<sensor>/<tile_id>_mask.png   0/255 PNG preview
    <output_dir>/<sensor>/<tile_id>_mask.tif   georeferenced single-band uint8 GeoTIFF
    <output_dir>/<sensor>_mosaic.tif           rasterio.merge of that sensor's tiles

Mask generation reuses segmentation/inference.run_segmentation (same normalizer,
speckle removal, morphological close). Threshold defaults to config.INFER_THRESHOLD
(0.4 — the val sweep's best). Nothing here imports finetune.py.

CLI:
    python inference.py --input-dir /kaggle/input/terrain-tiles-liss4-s2 \
                        --output-dir /kaggle/working/masks
    python inference.py --input-dir <dir> --output-dir <dir> --ckpt path --threshold 0.4

The notebook calls run_inference(input_dir, output_dir, ...) in the final cell.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from segmentation.inference import run_segmentation, save_mask, load_model  # noqa: E402


def find_tiles(input_dir):
    """Return [(sensor, tile_path), ...] for any '<sensor>/image/tile_*.tif' layout.

    Sensor is taken from the path component just above 'image'; bare
    'tile_*.tif' files at the input root fall back to a '<basename>_mask' sensor
    so output stays grouped by sensor directory.
    """
    tiles = []
    pattern = os.path.join(input_dir, "**", "image", "tile_*.tif")
    for path in glob.glob(pattern, recursive=True):
        parts = path.replace("\\", "/").split("/")
        sensor = parts[parts.index("image") - 1] if "image" in parts else "tiles"
        tiles.append((sensor, path))
    if not tiles:  # flat tile_*.tif directly in input_dir
        for path in sorted(glob.glob(os.path.join(input_dir, "tile_*.tif"))):
            tiles.append(("tiles", path))
    # deterministic order
    tiles.sort(key=lambda t: (t[0], t[1]))
    return tiles


def build_mosaics(output_dir, sensors):
    """rasterio.merge each sensor's *_mask.tif into one real-CRS uint8 mosaic."""
    import rasterio
    from rasterio.merge import merge
    for sensor in sensors:
        srcs = sorted(glob.glob(
            os.path.join(output_dir, sensor, "*_mask.tif")))
        if not srcs:
            continue
        with rasterio.open(srcs[0]) as ref:
            kwargs = ref.profile
        mosaic, out_transform = merge(
            srcs, method="first", nodata=0,
            res=ref.transform.a if ref.transform else None)
        mosaic = mosaic.astype(np.uint8)
        while mosaic.ndim > 2:  # merge may return (1,1,H,W) / (1,H,W)
            mosaic = mosaic[0]
        kwargs.update({
            "driver": "GTiff", "dtype": "uint8", "count": 1,
            "height": mosaic.shape[0], "width": mosaic.shape[1],
            "transform": out_transform, "nodata": 0,
            "compress": "deflate",
        })
        out_path = os.path.join(output_dir, f"{sensor}_mosaic.tif")
        with rasterio.open(out_path, "w", **kwargs) as dst:
            dst.write(mosaic, 1)
        print(f"[mosaic] {sensor}: {len(srcs)} tiles -> {out_path}")


def run_inference(input_dir, output_dir, checkpoint=None, threshold=None,
                  device=None, cleanup=True):
    """Mask every tile under input_dir; return dict of per-sensor counts."""
    import PIL.Image as Image
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    threshold = config.INFER_THRESHOLD if threshold is None else threshold

    # Resolve checkpoint: prefer fine-tuned, fall back to pre-finetune.
    if checkpoint and os.path.isfile(checkpoint):
        ckpt = checkpoint
    else:
        ckpt = config.FINETUNE_SAVE_PATH
        if not os.path.isfile(ckpt):
            ckpt = config.INFER_CHECKPOINT
    print(f"[inference] device={device} threshold={threshold} ckpt={ckpt}")

    tiles = find_tiles(input_dir)
    if not tiles:
        print(f"[inference] no tiles found under {input_dir}")
        return {}

    model = load_model(ckpt, device)   # build once, reuse for every tile

    counts = {}
    for sensor, path in tiles:
        sensor_dir = os.path.join(output_dir, sensor)
        os.makedirs(sensor_dir, exist_ok=True)
        tile_id = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(sensor_dir, f"{tile_id}_mask")

        mask_uint8, _ = run_segmentation(
            path, device=device,
            threshold=threshold, cleanup=cleanup, model=model)
        # 0/255 binary from the helper
        Image.fromarray(mask_uint8).save(base + ".png")
        save_mask(mask_uint8, path, base + ".tif")
        counts[sensor] = counts.get(sensor, 0) + 1
        if counts[sensor] % 200 == 1:
            print(f"[inference] {sensor}: tile {counts[sensor]} ({tile_id})")

    build_mosaics(output_dir, sorted(counts))
    print("[inference] SUMMARY")
    for sensor, n in sorted(counts.items()):
        print(f"    {sensor}: {n} masks -> {output_dir}/{sensor}/")
    total = sum(counts.values())
    print(f"    TOTAL: {total} tiles, {len(counts)} sensors")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TERRAIN mask inference.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/masks")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_inference(
        args.input_dir, args.output_dir,
        checkpoint=args.ckpt, threshold=args.threshold,
        device=torch.device(args.device) if args.device else None)
