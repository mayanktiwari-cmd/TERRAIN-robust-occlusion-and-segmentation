import geopandas as gpd

def create_buffer(input_file, output_file, buffer_distance):
    # Read vector file using pyogrio
    gdf = gpd.read_file(input_file, engine="pyogrio")

    print(f"\nLoaded {len(gdf)} features")
    print(f"CRS: {gdf.crs}")

    if gdf.crs is None:
        raise ValueError("Input file has no CRS defined.")

    # Check if CRS is geographic (lat/lon)
    if gdf.crs.is_geographic:
        print("\nWARNING:")
        print("Input CRS is Geographic (Latitude/Longitude).")
        print("Buffer distance will be in degrees, not meters.")
        print("Please reproject the layer to a projected CRS (e.g., UTM).")
        return

    # Create buffer
    buffered = gdf.copy()
    buffered["geometry"] = gdf.geometry.buffer(buffer_distance)

    # Save output using pyogrio
    buffered.to_file(output_file, engine="pyogrio")

    print("\nBuffer created successfully!")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":

    input_file = input("Enter input vector file path: ").strip().strip('"')
    output_file = input("Enter output buffered file path: ").strip().strip('"')
    buffer_distance = float(input("Enter buffer distance (meters): "))

    create_buffer(input_file, output_file, buffer_distance)