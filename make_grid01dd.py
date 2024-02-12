import rasterio
from rasterio.transform import from_origin

import numpy as np

# Define the output file path
output_file = "/home/luca/Documents/CIMA_code/EDO-GDO/grids/grid_01dd.tif"

# Define the bounding box and resolution
bbox = (-180, -90, 180, 90)
resolution = 0.01

# Calculate the number of pixels in x and y directions
width = int((bbox[2] - bbox[0]) / resolution)
height = int((bbox[3] - bbox[1]) / resolution)

# create an array of alternating 0s and 1s
data = np.zeros((height, width), dtype=np.int8)
data[::2, ::2] = 1

# Create the output dataset
with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=rasterio.int8,
    crs="EPSG:4326",
    transform=from_origin(bbox[0], bbox[3], resolution, resolution),
    compress="LZW",
) as dst:
    # Write the pixel id, x, and y bands
    dst.write(data, 1)
