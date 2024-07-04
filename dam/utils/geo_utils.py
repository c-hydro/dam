import xarray as xr
import numpy as np


# -------------------------------------------------------------------------------------
# Method to extract values from xr.DataArray based on lat and lon
def ltln2val_from_2dDataArray(input_map: xr.DataArray,
                              lat: np.array,
                              lon: np.array,
                              method: str):

    # if dims of input_map are not x and y, we need to change them
    if input_map.dims[0] != 'y' or input_map.dims[1] != 'x':
        old_dim_names = input_map.dims
        new_dim_names = ['y', 'x']
        rename_dict = dict(zip(old_dim_names, new_dim_names))
        input_map = input_map.rename(rename_dict)

    lon_query = xr.DataArray(lon, dims="points")
    lat_query = xr.DataArray(lat, dims="points")
    values = input_map.sel(x=lon_query, y=lat_query, method=method)

    return values
# -------------------------------------------------------------------------------------