import os
import math
import rasterio
from rasterio.windows import Window


def patchify_dataset(image_path, mask_path, output_dir, tile_size, overlap_percent):
    """
    Patchify a satellite image and its corresponding road mask.

    Parameters
    ----------
    image_path : str
        Path to input satellite image.
    mask_path : str
        Path to corresponding road mask.
    output_dir : str
        Output directory.
    tile_size : int
        Tile size (pixels).
    overlap_percent : float
        Overlap percentage (0-99).
    """

    if overlap_percent < 0 or overlap_percent >= 100:
        raise ValueError("Overlap percentage must be between 0 and 99.")

    overlap = int(tile_size * overlap_percent / 100)
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError("Invalid overlap percentage.")

    image_out = os.path.join(output_dir, "image")
    mask_out = os.path.join(output_dir, "mask")

    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:

        # -------------------------
        # Validation
        # -------------------------
        if (
            img_src.width != mask_src.width or
            img_src.height != mask_src.height
        ):
            raise ValueError("Image and mask dimensions do not match.")

        if img_src.crs != mask_src.crs:
            raise ValueError("Image and mask CRS do not match.")

        print("\nInput Image")
        print("----------------------------")
        print(f"Width       : {img_src.width}")
        print(f"Height      : {img_src.height}")
        print(f"Bands       : {img_src.count}")
        print(f"CRS         : {img_src.crs}")
        print(f"Resolution  : {img_src.res}")

        print("\nInput Mask")
        print("----------------------------")
        print(f"Width       : {mask_src.width}")
        print(f"Height      : {mask_src.height}")
        print(f"Bands       : {mask_src.count}")

        width = img_src.width
        height = img_src.height

        cols = math.ceil((width - overlap) / stride)
        rows = math.ceil((height - overlap) / stride)

        image_profile = img_src.profile.copy()
        mask_profile = mask_src.profile.copy()

        tile_count = 0

        for row in range(rows):
            for col in range(cols):

                x = col * stride
                y = row * stride

                # Shift last tiles to image boundary
                if x + tile_size > width:
                    x = max(width - tile_size, 0)

                if y + tile_size > height:
                    y = max(height - tile_size, 0)

                window = Window(x, y, tile_size, tile_size)

                img_tile = img_src.read(window=window)
                mask_tile = mask_src.read(window=window)

                transform = rasterio.windows.transform(window, img_src.transform)

                image_profile.update(
                    width=img_tile.shape[2],
                    height=img_tile.shape[1],
                    transform=transform
                )

                mask_profile.update(
                    width=mask_tile.shape[2],
                    height=mask_tile.shape[1],
                    transform=transform
                )

                filename = f"tile_r{row:04d}_c{col:04d}.tif"

                image_file = os.path.join(image_out, filename)
                mask_file = os.path.join(mask_out, filename)

                with rasterio.open(image_file, "w", **image_profile) as dst:
                    dst.write(img_tile)

                with rasterio.open(mask_file, "w", **mask_profile) as dst:
                    dst.write(mask_tile)

                tile_count += 1

    print("\n========================================")
    print("Patchification Completed Successfully")
    print("========================================")
    print(f"Tile Size           : {tile_size}")
    print(f"Overlap Percentage  : {overlap_percent}%")
    print(f"Overlap Pixels      : {overlap}")
    print(f"Stride              : {stride}")
    print(f"Total Tiles         : {tile_count}")
    print(f"\nSatellite Tiles : {image_out}")
    print(f"Mask Tiles      : {mask_out}")


if __name__ == "__main__":

    print("\nEnter Satellite Image Path")
    print(r"Example: C:\Dataset\Sentinel2\image.tif")
    image_path = input("Satellite Image: ").strip().strip('"')

    print("\nEnter Road Mask Path")
    print(r"Example: C:\Dataset\RoadMask\road_mask.tif")
    mask_path = input("Road Mask: ").strip().strip('"')

    print("\nEnter Output Folder")
    print(r"Example: C:\Dataset\Patches")
    output_dir = input("Output Folder: ").strip().strip('"')

    print("\nEnter Tile Size (pixels)")
    print("Example: 256 or 512")
    tile_size = int(input("Tile Size: "))

    print("\nEnter Overlap Percentage")
    print("Example: 0, 10, 20, 25, 50")
    overlap_percent = float(input("Overlap (%): "))

    patchify_dataset(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap_percent=overlap_percent
    )