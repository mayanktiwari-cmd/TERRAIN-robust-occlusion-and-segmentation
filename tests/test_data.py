import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize

BASE = r"C:\Users\isumi\Desktop\TERRAIN\TERRAIN---ISRO-BAH-26\PS4\SENTINEL2"
TIF = os.path.join(BASE, "Bengaluru.tif")
GPKG = os.path.join(BASE, "Bengaluru_mask.gpkg")
OUT = os.path.join(_HERE, "graph-tests", "data", "bengaluru_mask_window.npy")

GRID_H, GRID_W = 512, 512

with rasterio.open(TIF) as src:
    transform = src.transform
    crs = src.crs
    print("crs", crs, "size", src.width, "x", src.height, "bands", src.count)

    # Pick a central window of grid size, clipped to image bounds.
    c0 = max(0, src.width // 2 - GRID_W // 2)
    r0 = max(0, src.height // 2 - GRID_H // 2)
    w = min(GRID_W, src.width - c0)
    h = min(GRID_H, src.height - r0)

    win = Window(c0, r0, w, h)

    # Ground window bbox in projected coords.
    bounds = rasterio.windows.bounds(win, transform)
    print("window bbox", bounds)

    subtransform = rasterio.windows.transform(win, transform)

    # Rasterize the vector mask over the same window.
    gdf = gpd.read_file(GPKG, bbox=bounds)
    print("gdf n_feat", len(gdf), "crs", gdf.crs)
    mask = np.zeros((h, w), dtype=np.uint8)
    shapes = ((geom, 255) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    rasterize(shapes, out_shape=(h, w), transform=subtransform, fill=0, out=mask)
    print("mask foreground px", int((mask > 0).sum()), "fraction", float((mask > 0).mean()))

np.save(OUT, mask)
print("saved", OUT)