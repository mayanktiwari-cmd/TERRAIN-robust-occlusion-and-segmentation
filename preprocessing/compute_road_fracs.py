"""
Compute per-tile road fractions for all sensors -> config.ROAD_FRACS_PATH.

One pass over every mask in data/processed/<sensor>/mask/, records the binary
road-pixel fraction keyed by "<sensor>/<tile_basename>". The dataset's
MIN_ROAD_FRAC filter and (optional) road-frac sample weighting, plus
evaluate.sparse_vs_dense_ious, all read this file.

Usage:
    python preprocessing/compute_road_fracs.py               # all sensors
    python preprocessing/compute_road_fracs.py --sensors liss4 --out data/processed/road_fracs.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


def compute_sensor(sensor, out_fracs):
    import rasterio
    mask_dir = os.path.join(config.DATA_DIR, "processed", sensor, "mask")
    masks = sorted(glob.glob(os.path.join(mask_dir, "tile_*.tif")))
    fracs = {}
    for i, mask_path in enumerate(masks, 1):
        with rasterio.open(mask_path) as ds:
            fracs[f"{sensor}/{os.path.basename(mask_path)}"] = float((ds.read(1) > 0).mean())
        if i % 5000 == 0:
            print(f"  {sensor}: {i}/{len(masks)} tiles", flush=True)
    if fracs:
        vals = np.array(list(fracs.values()))
        print(f"{sensor}: n={len(vals)} | road% mean={vals.mean()*100:.2f} "
              f"p50={np.median(vals)*100:.2f} p90={np.percentile(vals,90)*100:.2f} "
              f"min={vals.min()*100:.3f}")
    else:
        print(f"[skip] {sensor}: no masks at {mask_dir}")
    return fracs


def main():
    parser = argparse.ArgumentParser(description="One-pass per-tile road fractions.")
    parser.add_argument("--sensors", nargs="*", default=None)
    parser.add_argument("--out", default=config.ROAD_FRACS_PATH)
    args = parser.parse_args()

    sensors = args.sensors or list(config.SENSORS)
    # ROAD_FRACS_PATH ("data/processed/road_fracs.json") is already rooted at CWD,
    # matching how dataset.py/_load_road_fracs and evaluate.py open it.
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    all_fracs = {}
    for s in sensors:
        all_fracs.update(compute_sensor(s, out))

    with open(out, "w") as f:
        json.dump(all_fracs, f, indent=0, sort_keys=True)
    print(f"wrote {len(all_fracs)} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
