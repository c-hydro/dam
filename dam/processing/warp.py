from osgeo import gdal, gdalconst
import tempfile

from typing import Optional
import numpy as np
import os

from ..utils.io_geotiff import read_geotiff, write_geotiff
from ..utils.rm import remove_file

def match_grid(input: str,
               grid: str,
               resampling_method: str = 'NearestNeighbour',
               nodata_value: Optional[float] = None,
               nodata_threshold: Optional[float] = None,
               output: Optional[str] = None,
               rm_input: bool = False) -> str:
    
    _resampling_methods = ['NearestNeighbour', 'Bilinear',
                           'Cubic', 'CubicSpline',
                           'Lanczos',
                           'Average', 'Mode',
                           'Max', 'Min',
                           'Med', 'Q1', 'Q3']
    
    for method in _resampling_methods:
        if method.lower() == resampling_method.lower():
            resampling_method = method
            break
    else:
        raise ValueError(f'resampling_method must be one of {_resampling_methods}')
    
    if output is None:
        output = input.replace('.tif', '_regridded.tif')

    # Open the input and reference raster files
    input_ds = read_geotiff(input, out='gdal')
    input_transform = input_ds.GetGeoTransform()
    input_projection = input_ds.GetProjection()

    if nodata_value is not None:
        input_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

    input_ds = None

    # Open the reference raster file
    input_grid = read_geotiff(grid, out='gdal')
    grid_transform = input_grid.GetGeoTransform()
    grid_projection = input_grid.GetProjection()

    # Get the resampling method
    resampling = getattr(gdalconst, f'GRA_{resampling_method}')

    # get the output bounds = the grid bounds
    # input_bounds = [input_transform[0], input_transform[3], input_transform[0] + input_transform[1] * input_ds.RasterXSize,
    #                 input_transform[3] + input_transform[5] * input_ds.RasterYSize]
    output_bounds = [grid_transform[0], grid_transform[3], grid_transform[0] + grid_transform[1] * input_grid.RasterXSize,
                     grid_transform[3] + grid_transform[5] * input_grid.RasterYSize]
    
    # set the type of the output to the type of the input if resampling is nearest neighbour, otherwise to float32
    if resampling == gdalconst.GRA_NearestNeighbour:
        output_type = input_ds.GetRasterBand(1).DataType
    else:
        output_type = gdalconst.GDT_Float32

    os.makedirs(os.path.dirname(output), exist_ok=True)
    gdal.Warp(output, input, outputBounds=output_bounds, #outputBoundsSRS = input_projection,
              srcSRS=input_projection, dstSRS=grid_projection,
              xRes=grid_transform[1], yRes=grid_transform[5], resampleAlg=resampling,
              options=['NUM_THREADS=ALL_CPUS'],
              outputType=output_type,
              format='GTiff', creationOptions=['COMPRESS=LZW'], multithread=True)
    
    if nodata_threshold is not None:
        # make a mask of the nodata values in the original input
        if np.isnan(nodata_value):
            mask = np.isnan(read_geotiff(input, out='array'))
        else:
            mask = read_geotiff(input, out = 'array') == nodata_value

        with tempfile.TemporaryDirectory() as tempdir:
            maskfile = os.path.join(tempdir, 'nan_mask.tif')

            mask = mask.astype(np.uint8)
            write_geotiff(mask, filename = maskfile, template = input)
            mask = None

            avg_nan = match_grid(maskfile, grid, 'Average')

            # set the output to nodata where the value of mask is > nodata_threshold
            mask = read_geotiff(avg_nan, out = 'array')
            mask = mask > nodata_threshold

            output_array = read_geotiff(output, out = 'array')
            metadata = read_geotiff(output, out = 'xarray').attrs

            output_array[mask == 1] = nodata_value
            write_geotiff(data = output_array, filename = output, template = output, metadata=metadata, nodata_value = nodata_value)
    
    if rm_input:
        remove_file(input)
    
    return output