from osgeo import gdal, gdalconst

from typing import Optional
import os

from .io_geotiff import read_geotiff_asGDAL, write_geotiff_fromGDAL

def match_grid(input: str,
               grid: str,
               resampling_method: str = 'NearestNeighbour',
               nodata_value: Optional[float] = None,
               output: Optional[str] = None) -> str:
    
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
    input_ds = read_geotiff_asGDAL(input)
    input_transform = input_ds.GetGeoTransform()
    input_projection = input_ds.GetProjection()

    if nodata_value is not None:
        input_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

    # Open the reference raster file
    input_grid = read_geotiff_asGDAL(grid)
    grid_transform = input_grid.GetGeoTransform()
    grid_projection = input_grid.GetProjection()

    # Get the resampling method
    resampling = getattr(gdalconst, f'GRA_{resampling_method}')

    # get the output bounds
    input_bounds = [input_transform[0], input_transform[3], input_transform[0] + input_transform[1] * input_ds.RasterXSize,
                    input_transform[3] + input_transform[5] * input_ds.RasterYSize]
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    gdal.Warp(output, input, outputBounds=input_bounds, outputBoundsSRS = input_projection,
              srcSRS=input_projection, dstSRS=grid_projection,
              xRes=grid_transform[1], yRes=grid_transform[5], resampleAlg=resampling,
              options=['NUM_THREADS=ALL_CPUS'],
              format='GTiff', creationOptions=['COMPRESS=LZW'], multithread=True)
    
    return output