"""
Package the processed tiles into the two Kaggle datasets the notebook mounts:

    terrain-tiles-cartosat3    -> processed/cartosat3/{image,mask}  (N sampled tiles)
    terrain-tiles-liss4-s21    -> processed/{liss4,sentinel2}/{image,mask}

Each zip stores the tiles under a top-level `processed/` dir (with _norm_stats.json
and a road_fracs.json restricted to the tiles actually present) so the notebook's
_sensor_root()/{base}/processed/<sensor> symlinks and _find_norm_stats() work.

Cartosat-3 tiles are capped at N (default 5000), sampled deterministically
(random.Random(seed)), plus their matching masks.
"""

import argparse
import io
import json
import os
import random
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(REPO, "data", "processed")
OUT = os.path.join(REPO, "kaggle_upload")


def _sortkeys(keys):
    return sorted(keys, key=lambda k: (os.path.dirname(k), os.path.basename(k)))


def _sample(keys, n, seed):
    sorted_keys = sorted(keys)
    if n is None or n >= len(sorted_keys):
        return sorted_keys
    return sorted(random.Random(seed).sample(sorted_keys, n))


def write_zip(path, names):
    """names: list of 'sensor/basename' keys (must exist in road_fracs.json)."""
    proj_all = json.load(open(os.path.join(PROC, "road_fracs.json")))
    progress = [0]

    def tick():
        progress[0] += 1
        if progress[0] % 2000 == 0:
            print(f"  {progress[0]}/{len(names)}", flush=True)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        # per-tile image + mask
        for key in names:
            sensor, base = key.split("/", 1)
            for sub in ("image", "mask"):
                src = os.path.join(PROC, sensor, sub, base)
                if os.path.isfile(src):
                    try:
                        z.write(src, os.path.join("processed", sensor, sub, base))
                    except OSError as e:
                        print(f"  !! skip unreadable {sub} {key}: {e}", flush=True)
            tick()
        # shared metadata
        norm = os.path.join(PROC, "_norm_stats.json")
        if os.path.isfile(norm):
            z.write(norm, "processed/_norm_stats.json")
        fracs = {k: proj_all[k] for k in names}
        z.writestr("processed/road_fracs.json", json.dumps(fracs, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartosat-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-liss", action="store_true", default=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fracs = json.load(open(os.path.join(PROC, "road_fracs.json")))
    cartos3 = [k for k in fracs if k.startswith("cartosat3/")]
    liss = [k for k in fracs if k.startswith("liss4/")]
    sent = [k for k in fracs if k.startswith("sentinel2/")]

    # --- cartosat-3: sampled subset ---
    chosen = _sample(cartos3, args.cartosat_n, args.seed)
    p = os.path.join(args.out, "terrain-tiles-cartosat3.zip")
    print(f"creating {p} ({len(chosen)} sampled cartosat3 tiles)")
    write_zip(p, chosen)
    sz = os.path.getsize(p) / 1e6
    print(f"  -> {os.path.basename(p)}: {sz:.0f} MB, {len(chosen)} tiles")

    # --- liss4 + sentinel2: full ---
    if args.include_liss:
        both = _sortkeys(liss + sent)
        p = os.path.join(args.out, "terrain-tiles-liss4-s21.zip")
        print(f"creating {p} ({len(both)} tiles: liss4={len(liss)} sentinel2={len(sent)})")
        write_zip(p, both)
        sz = os.path.getsize(p) / 1e6
        print(f"  -> {os.path.basename(p)}: {sz:.0f} MB, {len(both)} tiles")

    print("done. Files in", args.out)


if __name__ == "__main__":
    main()