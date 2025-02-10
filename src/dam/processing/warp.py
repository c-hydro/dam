import xarray as xr
import rioxarray as rxr
import rasterio
import geopandas as gdp

from d3tools.spatial.space_utils import clip_xarray
from d3tools.errors import GDAL_ImportError

from typing import Optional
import numpy as np
import tempfile
import os

from ..utils.register_process import as_DAM_process

_resampling_methods = {
    'NearestNeighbour': 0,
    'Bilinear': 1,
    'Cubic': 2,
    'CubicSpline': 3,
    'Lanczos': 4,
    'Average': 5,
    'Mode': 6,
    #'Gauss': 7,
    'Max': 8,
    'Min': 9,
    'Med': 10,
    'Q1': 11,
    'Q3': 12,
    'Sum': 13,
    'RMS': 14
}

@as_DAM_process(input_type='xarray', output_type='xarray',continuous_space = False)
def match_grid(input: xr.DataArray,
               grid: xr.DataArray,
               resampling_method: str|int = 'NearestNeighbour',
               nodata_threshold: float = 1,
               nodata_value: Optional[float] = None,
               engine = 'xarray'
               ) -> xr.DataArray:

    if engine == 'xarray':
        regridded =_match_grid_xarray(input, grid, resampling_method, nodata_threshold, nodata_value)
    elif engine == 'gdal': # this is for compatibility with a previous version. It is not recommended to use gdal
        regridded = _match_grid_gdal(input, grid, resampling_method, nodata_threshold, nodata_value)
    else:
        raise ValueError('engine must be one of [xarray, gdal]')
    
    # ensure that the y coordinates are descending
    y_dim = regridded.rio.y_dim
    regridded = regridded.sortby(y_dim, ascending=False)

    # and order the dimentions to match the grid
    dim_names = grid.dims
    regridded = regridded.transpose(*dim_names)

    return regridded

@as_DAM_process(input_type='xarray', output_type='xarray', continuous_space = False)
def clip_to_bounds(input: xr.DataArray,
                   bounds: tuple[float, float, float, float]|xr.DataArray|gdp.GeoDataFrame,
                   ) -> xr.DataArray:

    if isinstance(bounds, xr.DataArray):
        bounds_da = bounds
        bounds = bounds_da.rio.bounds()
    elif isinstance(bounds, gdp.GeoDataFrame):
        bounds_gdf = bounds
        bounds = bounds_gdf.total_bounds

    input_clipped = clip_xarray(input, bounds)
    
    return input_clipped

def _match_grid_xarray(input: xr.DataArray,
                          grid: xr.DataArray,
                          resampling_method: str|int,
                          nodata_threshold: float,
                          nodata_value: Optional[float]
                          ) -> xr.DataArray:
    input_da = input
    mask_da = grid

    if isinstance(resampling_method, str):
        resampling_method = resampling_method.replace(' ', '')
        for method in _resampling_methods:
            if method.lower() == resampling_method.lower():
                resampling = rasterio.enums.Resampling(_resampling_methods[method])
                break
        else:
            raise ValueError(f'resampling_method must be one of {_resampling_methods}')
    elif resampling_method == int(resampling_method) and 0<=resampling_method<=14 and resampling_method != 7:
        resampling = rasterio.enums.Resampling(resampling_method)
    else:
        raise ValueError(f'resampling_method must be one of {_resampling_methods}')
    
    if resampling_method not in ['NearestNeighbour', 'Mode', 0, 6]:
        input_da = input_da.astype(np.float32)

    input_reprojected = input_da.rio.reproject_match(mask_da, resampling=resampling)
    if nodata_threshold < 0:
        nodata_threshold = 0
    elif nodata_threshold >= 1:
        return input_reprojected

    nodata_value = nodata_value or input.rio.nodata

    def process_chunk(chunk, nodata_value):
        return chunk.copy(data = np.isclose(chunk, nodata_value, equal_nan=True).astype(np.int8))
    
    chunk_sizes = [np.ceil(s/10) for s in input_da.shape]

    chunked_input = input_da.chunk(chunk_sizes)
    result_da     = chunked_input.map_blocks(process_chunk, args = (nodata_value,))
    nan_mask = result_da.compute()

    nan_mask = nan_mask * 100
    regridded_mask = nan_mask.rio.reproject_match(mask_da, resampling=rasterio.enums.Resampling(5))

    return input_reprojected.where(regridded_mask < nodata_threshold*100, nodata_value)

def _match_grid_gdal(input: xr.DataArray,
                        grid: xr.DataArray,
                        resampling_method: str|int,
                        nodata_threshold: float,
                        nodata_value: Optional[float]) -> xr.DataArray:
    try:
        from osgeo import gdal, gdalconst
    except ImportError:
        raise GDAL_ImportError('match_grid')

    _resampling_methods_gdal = ['NearestNeighbour', 'Bilinear',
                                'Cubic', 'CubicSpline',
                                'Lanczos',
                                'Average', 'Mode',
                                'Max', 'Min',
                                'Med', 'Q1', 'Q3']
    
    if isinstance(resampling_method, int):
        for method in _resampling_methods:
            if _resampling_methods[method] == resampling_method:
                resampling_method = method
                break
    
    for method in _resampling_methods_gdal:
        if method.lower() == resampling_method.lower():
            resampling_method = method
            break
    else:
        raise ValueError(f'resampling_method must be one of {_resampling_methods_gdal}')

    # save input and grid to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f'{tmpdir}/input.tif'
        grid_path = f'{tmpdir}/grid.tif'
        input.rio.to_raster(input_path)
        grid.rio.to_raster(grid_path)

        output_path = input_path.replace('.tif', '_regridded.tif')

        # Open the input and reference raster files
        input_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
        input_transform = input_ds.GetGeoTransform()
        input_projection = input_ds.GetProjection()

        if nodata_value is not None:
            input_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

        in_type = input_ds.GetRasterBand(1).DataType
        input_ds = None

        # Open the reference raster file
        input_grid = gdal.Open(grid_path, gdalconst.GA_ReadOnly)
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
            output_type = in_type
        else:
            output_type = gdalconst.GDT_Float32

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdal.Warp(output_path, input_path, outputBounds=output_bounds, #outputBoundsSRS = input_projection,
                srcSRS=input_projection, dstSRS=grid_projection,
                xRes=grid_transform[1], yRes=grid_transform[5], resampleAlg=resampling,
                outputType=output_type,
                format='GTiff', creationOptions=['COMPRESS=LZW'], multithread=True)
        
        output = rxr.open_rasterio(output_path)
    
    return output