import xarray as xr
import rasterio
import dask

from typing import Optional
import numpy as np

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

@as_DAM_process(continuous_space = False)
def match_grid(input: xr.DataArray,
               grid: xr.DataArray,
               resampling_method: str|int = 'NearestNeighbour',
               nodata_threshold: float = 1,
               nodata_value: Optional[float] = None,
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