"""
Physically remove tiles with fewer than 0.5% road pixels.

For every mask in data/processed/<sensor>/mask/tile_*.tif whose binary road
fraction is below config.MIN_ROAD_FRAC (0.005), delete BOTH the mask tile and
its matching image tile. Then regenerate config.ROAD_FRACS_PATH.

Usage:
    python preprocessing/prune_low_road_tiles.py            # delete below-threshold tiles
    python preprocessing/prune_low_road_tiles.py --dry-run # only report what would be deleted
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


def prune_sensor(sensor, dry_run):
    import rasterio
    import numpy as np

    root = os.path.join(config.DATA_DIR, "processed", sensor)
    mask_dir = os.path.join(root, "mask")
    image_dir = os.path.join(root, "image")
    masks = sorted(glob.glob(os.path.join(mask_dir, "tile_*.tif")))
    thresh = config.MIN_ROAD_FRAC

    kept, removed_mask, removed_image = 0, 0, 0
    for i, mask_path in enumerate(masks, 1):
        name = os.path.basename(mask_path)
        with rasterio.open(mask_path) as ds:
            frac = float((ds.read(1) > 0).mean())
        image_path = os.path.join(image_dir, name)
        has_image = os.path.isfile(image_path)
        if frac < thresh:
            if not dry_run:
                os.remove(mask_path)
                if has_image:
                    os.remove(image_path)
            removed_mask += 1
            removed_image += 1 if has_image else 0
        else:
            kept += 1
        if i % 5000 == 0:
            print(f"  {sensor}: scanned {i}/{len(masks)}", flush=True)

    print(f"[{sensor}] {'[dry-run] ' if dry_run else ''}kept={kept} "
          f"removed_mask={removed_mask} removed_image={removed_image}")
    return removed_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=config.ROAD_FRACS_PATH)
    args = parser.parse_args()

    sensors = list(config.SENSORS)
    for s in sensors:
        prune_sensor(s, args.dry_run)

    if not args.dry_run:
        # Rebuild road_fracs.json so it no longer references deleted tiles.
        from preprocessing.compute_road_fracs import compute_sensor
        all_fracs = {}
        for s in sensors:
            all_fracs.update(compute_sensor(s, args.out))
        with open(args.out, "w") as f:
            json.dump(all_fracs, f)
        print(f"wrote {len(all_fracs)} fractions -> {args.out}")


if __name__ == "__main__":
    main()