"""
Rasterize a vector mask (.gpkg/.shp) to match a reference raster's grid.

Features:
- Takes vector path, reference raster path, and output raster path as user input.
- Repairs invalid/self-intersecting geometries.
- Dissolves overlapping polygons.
- Reprojects vector to match raster CRS.
- Creates a binary mask (0 = background, 1 = feature).
- Uses all_touched=True to avoid gaps in thin features.

Example Inputs:
---------------
Enter vector file path (.gpkg/.shp):
C:\Data\roads_buffered.gpkg

Enter reference raster path (.tif):
C:\Data\satellite_image.tif

Enter output raster path (.tif):
C:\Data\road_mask.tif
"""

import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from shapely.ops import unary_union
from shapely.validation import make_valid

# -------------------- User Inputs --------------------
vector_path = input(
    "Enter vector file path (.gpkg/.shp):\n"
).strip().strip('"')

reference_raster_path = input(
    "\nEnter reference raster path (.tif):\n"
).strip().strip('"')

output_raster_path = input(
    "\nEnter output raster path (.tif):\n"
).strip().strip('"')

burn_input = input(
    "\nEnter burn value (default = 1): "
).strip()

burn_value = int(burn_input) if burn_input else 1

# -------------------- Load Reference Raster --------------------
with rasterio.open(reference_raster_path) as ref:
    meta = ref.meta.copy()
    out_shape = (ref.height, ref.width)
    transform = ref.transform
    crs = ref.crs

# -------------------- Load Vector --------------------
gdf = gpd.read_file(vector_path)

# Reproject if required
if gdf.crs != crs:
    print("Reprojecting vector to match raster CRS...")
    gdf = gdf.to_crs(crs)

# Remove empty geometries
gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

# Repair invalid geometries
print("Repairing invalid geometries...")
gdf["geometry"] = gdf.geometry.apply(
    lambda g: make_valid(g) if not g.is_valid else g
)

# -------------------- Merge Overlapping Polygons --------------------
print("Merging overlapping polygons...")
merged_geom = unary_union(gdf.geometry.values)

if merged_geom.geom_type == "MultiPolygon":
    geoms = list(merged_geom.geoms)
else:
    geoms = [merged_geom]

shapes = ((geom, burn_value) for geom in geoms)

# -------------------- Rasterize --------------------
print("Rasterizing...")

mask_arr = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,
    all_touched=True,
    dtype=rasterio.uint8,
)

# -------------------- Save Output --------------------
meta.update(
    {
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 255,
    }
)

with rasterio.open(output_raster_path, "w", **meta) as dst:
    dst.write(mask_arr, 1)

print("\nRasterization completed successfully!")
print(f"Output saved to: {output_raster_path}")
print(f"Raster Shape : {mask_arr.shape}")
print(f"Unique Values: {np.unique(mask_arr)}")