"""
Re-rasterize over-thick road masks for coarse-GSD sensors (Phase-0 fix).

Phase-0 mask audit showed inconsistent road-buffer widths across cities:
    Hyderabad/C3  ~6m   (1m GSD)
    Delhi/LISS    ~17m  (5m GSD)   <- over-thick
    Bengaluru/S2  ~15m  (10m GSD)  <- over-thick

This script re-rasterizes the mask tiles for the coarse sensors from their source
buffered vector at a uniform physical road width (~target meters), in place under
data/processed/<sensor>/mask/. Cartosat-3 is untouched (its ~6m width is the
reference and the big zip stays frozen).

Per-sensor source mapping (city -> buffered gpkg):
    liss4      <- data/buffered/Delhi_buffered.gpkg
    sentinel2  <- data/buffered/Bengaluru_buffered.gpkg

Usage:
    python preprocessing/rerasterize_thick_masks.py --dry-run
    python preprocessing/rerasterize_thick_masks.py --target 10.0
"""

import argparse
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box

SENSOR_VEC = {
    "liss4": "data/buffered/Delhi_buffered.gpkg",
    "sentinel2": "data/buffered/Bengaluru_buffered.gpkg",
}
TARGET_W = 10.0  # meters of physical road width to normalize toward


def vector_p50_width(gdf):
    a, L = gdf.geometry.area.values, gdf.geometry.length.values
    w = np.where(L > 0, 2 * a / np.maximum(L, 1e-6), np.nan)
    return float(np.nanmedian(w))


def process_sensor(sensor, vec_path, target, dry_run):
    import config
    mask_dir = os.path.join(config.DATA_DIR, "processed", sensor, "mask")
    tiles = sorted(glob.glob(os.path.join(mask_dir, "tile_*.tif")))
    if not tiles:
        print(f"[skip] {sensor}: no tiles")
        return

    gdf = gpd.read_file(vec_path)
    if gdf.crs is None:
        raise ValueError(f"{vec_path} has no CRS")
    cur = vector_p50_width(gdf)
    neg_buffer = -max(0.0, (cur - target) / 2.0)
    print(f"{sensor}: vector p50 width {cur:.1f}m -> target {target:.1f}m | "
          f"negative buffer {neg_buffer:.2f}m | {len(tiles)} tiles")
    if dry_run:
        return

    geom = gdf.geometry.buffer(neg_buffer)
    keep = (~geom.is_empty) & (geom.length > 1e-6) & (geom.area > 1e-6)
    gdf = gpd.GeoDataFrame(geometry=geom[keep], crs=gdf.crs).reset_index(drop=True)
    tree = gdf.sindex

    n_rewritten = 0
    for path in tiles:
        with rasterio.open(path) as ds:
            profile = ds.profile.copy()
            bounds = ds.bounds
            transform = ds.transform
            shape = (ds.height, ds.width)
        candidates = list(tree.intersection(bounds))
        clip = box(*bounds)
        geoms = [g.intersection(clip) for g in gdf.geometry.iloc[candidates]]
        geoms = [g for g in geoms if not g.is_empty and g.area > 1e-9]
        arr = rasterio.features.rasterize(
            ((g, 1) for g in geoms),
            out_shape=shape, transform=transform, fill=0,
            all_touched=True, dtype=rasterio.uint8)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        n_rewritten += 1
        if n_rewritten % 500 == 0:
            print(f"  {sensor}: {n_rewritten}/{len(tiles)}", flush=True)
    print(f"{sensor}: rewrote {n_rewritten} masks in place")


def main():
    parser = argparse.ArgumentParser(description="Re-rasterize over-thick road masks.")
    parser.add_argument("--sensors", nargs="*", default=list(SENSOR_VEC))
    parser.add_argument("--target", type=float, default=TARGET_W)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for s in args.sensors:
        if s not in SENSOR_VEC:
            print(f"[skip] {s}: no source vector mapping")
            continue
        process_sensor(s, SENSOR_VEC[s], args.target, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
