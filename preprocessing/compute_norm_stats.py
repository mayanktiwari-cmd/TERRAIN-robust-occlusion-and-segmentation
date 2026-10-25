"""
Compute per-sensor, per-band percentile normalization stats and cache them.

For each sensor, images live under data/processed/<sensor>/image/. We pool a bounded,
stratified sample of pixels per band (spread across many tiles) and compute genuine
np.percentile(lo)/np.percentile(hi) for the config.NORM_PERCENTILE range. Results are
written to config.NORM_STATS_PATH so the dataset/scaling only needs a JSON load.

One-time CPU pass (a few minutes). Re-run after adding/removing tiles.

Usage:
    python preprocessing/compute_norm_stats.py
"""

import glob
import json
import os
import sys

import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402

MAX_PIXELS_PER_SENSOR = 8_000_000   # cap pooled samples to bound memory/runtime
MAX_TILES_PER_SENSOR = 4000         # upper bound on tiles touched per sensor


def compute_sensor_stats(sensor, lo_pct, hi_pct):
    image_dir = os.path.join(config.DATA_DIR, "processed", sensor, "image")
    tiles = sorted(glob.glob(os.path.join(image_dir, "tile_*.tif")))
    if not tiles:
        return None

    # Stratified subsample of tiles, spaced across the sensor footprint.
    step = max(1, len(tiles) // MAX_TILES_PER_SENSOR)
    chosen = tiles[::step]

    per_band = {}   # band -> list of pooled pixel arrays

    for path in tqdm(chosen, desc=f"{sensor} norm stats"):
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32)   # (C,H,W)
        n_bands = arr.shape[0]
        # Stratified downsampling across the tile to bound memory.
        sub = arr[:, ::4, ::4]
        for b in range(n_bands):
            vals = sub[b].ravel()
            # Nodata/black regions (large fractions of edge tiles) must not skew the
            # percentile bounds; keep positive pixels only, nodata -> 0 after scaling.
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            if b not in per_band:
                per_band[b] = []
            per_band[b].append(vals)
            # Cap memory: drop oldest pool contributions once exceeded.
            while sum(x.nbytes for x in per_band[b]) > MAX_PIXELS_PER_SENSOR * 4:
                per_band[b].pop(0)

    if not per_band:
        return None

    n_bands = max(per_band) + 1 if per_band else 0
    # Align every sensor to config.IN_CHANNELS bands so downstream code can always
    # assume 4-wide stats. Missing (padded) bands normalize to [0,1] and map the
    # pad value (config.PAD_VALUE) to 0 after clamping (see segmentation.dataset._normalize).
    lo_b = []
    hi_b = []
    for b in range(config.IN_CHANNELS):
        if b in per_band and per_band[b]:
            pooled = np.concatenate(per_band[b])
            lo_b.append(float(np.percentile(pooled, lo_pct)))
            hi_b.append(float(np.percentile(pooled, hi_pct)))
        else:
            lo_b.append(0.0)
            hi_b.append(1.0)

    return {
        "sensor": sensor,
        "bands": config.IN_CHANNELS,
        "n_actual_bands": n_bands,
        "padded": n_bands < config.IN_CHANNELS,
        "pct_lo": lo_b,
        "pct_hi": hi_b,
        "count": len(tiles),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute sensor normalization percentiles.")
    parser.add_argument("--out", default=config.NORM_STATS_PATH, help="Output json path.")
    args = parser.parse_args()

    lo_pct, hi_pct = config.NORM_PERCENTILE
    print(f"Percentiles: {lo_pct}-{hi_pct} | sensors: {config.SENSORS}")

    stats = {}
    for sensor in config.SENSORS:
        s = compute_sensor_stats(sensor, lo_pct, hi_pct)
        stats[sensor] = s
        if s is None:
            print(f"  {sensor}: NO TILES")
        else:
            lo = [round(v, 1) for v in s["pct_lo"]]
            hi = [round(v, 1) for v in s["pct_hi"]]
            print(f"  {sensor}: bands={s['bands']} tiles={s['count']} pct_lo={lo} pct_hi={hi}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats -> {args.out}")


if __name__ == "__main__":
    main()