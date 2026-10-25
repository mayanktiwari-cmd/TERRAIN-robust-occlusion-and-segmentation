"""
Filter out image/mask tile pairs whose mask contains no road pixels (all-zero).

Processes each sensor folder under data/processed/<sensor>/:
  - image/tile_*.tif
  - mask/tile_*.tif

A pair is deleted (both image and mask) if the mask band has no non-zero value.
Supports --dry-run to preview without deleting anything.

Usage:
  python preprocessing/filter_empty_tiles.py
  python preprocessing/filter_empty_tiles.py --dry-run
  python preprocessing/filter_empty_tiles.py --sensors liss4 sentinel2
"""

import argparse
import glob
import os
import sys

import numpy as np
import rasterio

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")


def mask_has_roads(mask_path):
    with rasterio.open(mask_path) as ds:
        arr = ds.read(1)
    return bool(arr.any())


def filter_sensor(sensor_dir, dry_run=False):
    mask_dir = os.path.join(sensor_dir, "mask")
    image_dir = os.path.join(sensor_dir, "image")
    if not os.path.isdir(mask_dir) or not os.path.isdir(image_dir):
        print(f"[skip] missing image/mask dirs: {sensor_dir}")
        return 0, 0

    masks = sorted(glob.glob(os.path.join(mask_dir, "tile_*.tif")))
    deleted = 0
    missing_imgs = 0

    for mask_path in masks:
        if not mask_has_roads(mask_path):
            tile_name = os.path.basename(mask_path)
            img_path = os.path.join(image_dir, tile_name)
            if not os.path.isfile(img_path):
                missing_imgs += 1
                continue
            if dry_run:
                deleted += 1
                continue
            os.remove(mask_path)
            os.remove(img_path)
            deleted += 1

    return deleted, missing_imgs


def main():
    parser = argparse.ArgumentParser(description="Delete tiles with empty road masks.")
    parser.add_argument("--sensors", nargs="*", default=None,
                        help="Sensor subfolder names, e.g. cartosat3 liss4 sentinel2 (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be deleted, without deleting.")
    args = parser.parse_args()

    sensors = args.sensors or [
        d for d in sorted(os.listdir(PROCESSED_DIR))
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ]

    total_deleted = 0
    for sensor in sensors:
        sensor_dir = os.path.join(PROCESSED_DIR, sensor)
        if not os.path.isdir(sensor_dir):
            print(f"[skip] not a dir: {sensor_dir}")
            continue
        deleted, missing = filter_sensor(sensor_dir, dry_run=args.dry_run)
        total_deleted += deleted
        action = "WOULD DELETE" if args.dry_run else "deleted"
        print(f"{sensor}: {action} {deleted} image/mask pairs" + (f" ({missing} masks had no matching image)" if missing else ""))

    mode = "DRY-RUN (no changes made)" if args.dry_run else "done"
    print(f"\nTotal {mode}: {total_deleted} pairs removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
