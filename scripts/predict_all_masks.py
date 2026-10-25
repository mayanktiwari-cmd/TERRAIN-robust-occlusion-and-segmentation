"""
Batch road-mask prediction for the graph pipeline.

Runs the trained segmentation model over every (or a subset of) satellite tile
in data/processed/<sensor>/image/ and writes georeferenced binary road masks to
data/masked/<sensor>/mask_tile_*.tif (0/255 uint8, same geotransform as the tile).
The friend's graph pipeline consumes exactly these masks:
    skeletonize_mask(mask) -> build_graph(skeleton, nodes, edges) -> heal_graph.

The model is loaded once and reused across tiles (run_segmentation reloads it
every call, which would be wasteful over 51k tiles). Resume-friendly: tiles
whose mask already exists are skipped unless --overwrite.

Usage:
    python scripts/predict_all_masks.py                     # all sensors
    python scripts/predict_all_masks.py --sensors liss4 sentinel2
    python scripts/predict_all_masks.py --limit 100         # first 100 tiles/sensor
    python scripts/predict_all_masks.py --device cpu
    python scripts/predict_all_masks.py --overwrite
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from segmentation.inference import load_model, _normalize_tile, _cleanup, _infer_sensor  # noqa: E402


def predict_tiles(sensor, tiles, model, out_root, device, threshold, overwrite):
    mask_dir = os.path.join(out_root, sensor, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    t0 = time.time()
    done = 0
    for i, tile_path in enumerate(tiles, 1):
        name = os.path.basename(tile_path).replace("tile_", "mask_tile_")
        out_path = os.path.join(mask_dir, name)
        if os.path.isfile(out_path) and not overwrite:
            continue
        with rasterio.open(tile_path) as ds:
            arr = ds.read()
            profile = ds.profile
        norm = _normalize_tile(arr, sensor)              # per-sensor percentile norm
        inp = torch.from_numpy(norm).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)
        mask = _cleanup(mask)
        profile.update(dtype="uint8", count=1, compress="deflate", nodata=0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask, 1)
        done += 1
        if i % 200 == 0:
            rate = (time.time() - t0) / max(1, i)
            print(f"  {sensor}: {i}/{len(tiles)} | {rate:.3f}s/tile | "
                  f"ETA {rate*(len(tiles)-i)/60:.1f}min", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description="Batch road-mask prediction.")
    parser.add_argument("--sensors", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="max tiles per sensor (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=config.INFER_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=config.INFER_THRESHOLD)
    parser.add_argument("--out", default="data/masked")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    sensors = args.sensors or list(config.SENSORS)

    grand = 0
    for sensor in sensors:
        tiles = sorted(glob.glob(os.path.join(config.DATA_DIR, "processed", sensor,
                                              "image", "tile_*.tif")))
        if args.limit > 0:
            tiles = tiles[:args.limit]
        if not tiles:
            print(f"[skip] {sensor}: no tiles")
            continue
        print(f"== {sensor}: {len(tiles)} tiles -> {os.path.join(args.out, sensor)} ==")
        n = predict_tiles(sensor, tiles, model, args.out, device, args.threshold,
                          args.overwrite)
        print(f"  {sensor}: wrote {n} masks")
        grand += n
    print(f"done: {grand} masks total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
