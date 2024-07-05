import xarray as xr
import numpy as np
from dam.utils.io_csv import read_csv, save_csv
from dam.utils.io_geotiff import read_geotiff_asXarray
from typing import Optional

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

# -------------------------------------------------------------------------------------
# Method to extract residuals from xr.DataArray based on lat-lon and values
def compute_residuals(input_map: str,
                      input_data: str,
                      name_columns_csv: list[str],
                      method: Optional[str] = 'nearest',
                      destination: Optional[str] = None,
                      method_residuals: Optional[str] = 'data_minus_map'):
    """
    Compute residuals between data and map. The input is a csv file with columns for latitude, longitude, and data.
    Note that the csv file must have five columns in this order: station_id', 'station_name', 'lat', 'lon', 'data'.
    Names of this csv file can change, but the order of the columns must be the same.
    Another input is a raster map.
    The residuals are computed with a given method, whether as data minus map or map minus data.
    The residuals are saved to a new csv file.
    """

    if destination is None:
        destination = input_data.replace('.csv', '_residuals.csv')

    # load data
    data = read_csv(input_data)
    lat_points = data[name_columns_csv[2]].to_numpy()
    lon_points = data[name_columns_csv[3]].to_numpy()
    data_points = data[name_columns_csv[4]].to_numpy()

    # load map
    map = read_geotiff_asXarray(input_map)
    map = map.squeeze()  # remove single dimensions, usually time

    # extract values in map based on lat and lon
    values_map = ltln2val_from_2dDataArray(input_map=map, lat=lat_points, lon=lon_points, method=method)

    # compute residuals
    if method_residuals == 'data_minus_map':
        residuals = data_points - values_map    # data minus map
    elif method_residuals == 'map_minus_data':
        residuals = values_map - data_points
    else:
        raise NotImplementedError('Method method_residuals not implemented')

    # append residuals to data as a new column, then save
    data['residuals'] = residuals.values
    save_csv(data, destination)

    return destination
# -------------------------------------------------------------------------------------

