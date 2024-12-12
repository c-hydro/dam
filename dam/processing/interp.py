import numpy as np
import xarray as xr
from sklearn import linear_model
from typing import Optional
from ..utils.geo_utils import ltln2val_from_2dDataArray
from ..utils.io_csv import read_csv, save_csv
from ..utils.io_geotiff import write_geotiff_fromXarray, read_geotiff
from ..utils.random_string import random_string
from ..utils.io_vrt import create_point_vrt
from ..utils.exec_process import exec_process
from ..utils.rm import remove_file
from ..utils.register_process import as_DAM_process

from .filter import apply_raster_mask
import os
import pandas as pd
import rioxarray

@as_DAM_process()
def interp_with_elevation(input: pd.DataFrame,
                          DEM:xr.DataArray,
                          homogeneous_regions: xr.DataArray,
                          name_lat_lon_data_csv: list[str],
                          minimum_number_sensors_in_region: Optional[int] = 10,
                          minimum_r2: Optional[float] = 0.25,
                          crs:str = 'EPSG:4326') -> xr.DataArray:
    """
    Interpolate data using elevation. The input is a dataframe with columns for latitude, longitude, and data.
    Note that the dataframe may have any number of columns, but MUST have latitude, longitude, and data.
    The ORDER of those columns in name_lat_lon_data_in must be the same.
    The homogeneous_regions is a raster map with the same shape as the DEM, where each value corresponds to a region.
    The DEM is a raster map with elevation values.
    The data is interpolated using a linear regression with elevation for each homogeneous region.
    The interpolated data is saved to a xr.DataArray.
    """

    # load data
    data = input
    lat_points = data[name_lat_lon_data_csv[0]].to_numpy()
    lon_points = data[name_lat_lon_data_csv[1]].to_numpy()
    data = data[name_lat_lon_data_csv[2]].to_numpy()

    #load dem and homogeneous regions
    dem = DEM
    dem = np.squeeze(dem)
    homogeneous_regions = np.squeeze(homogeneous_regions)

    # get homogeneous regions and elevation for each station
    homogeneous_regions_stations = ltln2val_from_2dDataArray(input_map=homogeneous_regions, lat=lat_points, lon=lon_points, method="nearest")
    homogeneous_regions_stations = homogeneous_regions_stations.values
    elevation_stations = ltln2val_from_2dDataArray(input_map=dem, lat=lat_points, lon=lon_points, method="nearest")
    elevation_stations = elevation_stations.values

    # get list of homogeneous regions
    list_regions = np.unique(np.ravel(homogeneous_regions_stations))
    list_regions = list_regions[list_regions > 0]

    # loop on this list and do the work for each region; if data points are insufficient, skip computations!
    map_target = np.empty([dem.shape[0], dem.shape[1]]) * np.nan
    for i, region_id in enumerate(list_regions):

         # determine available stations in this region
         data_this_region = data[homogeneous_regions_stations == region_id]

         if data_this_region.shape[0] >= minimum_number_sensors_in_region:

             elevations_this_region = elevation_stations[homogeneous_regions_stations == region_id]
             elevations_this_region_filtered = elevations_this_region[(~np.isnan(elevations_this_region)) & (~np.isnan(data_this_region))]
             data_this_region_filtered = data_this_region[(~np.isnan(elevations_this_region)) & (~np.isnan(data_this_region))]

             # compute linear regression
             elevations_this_region_filtered = elevations_this_region_filtered.reshape((-1, 1))  # this is needed to use .fit in LinearReg
             regr = linear_model.LinearRegression(fit_intercept=True)
             regr.fit(elevations_this_region_filtered, data_this_region_filtered)
             r2 = regr.score(elevations_this_region_filtered, data_this_region_filtered)

             if r2 >= minimum_r2:
                 map_target[homogeneous_regions.values == region_id] = \
                      regr.coef_[0] * dem.values[homogeneous_regions.values == region_id] + regr.intercept_

             else:
                 print(' ---> WARNING: r2 was LOWER than threshold, data NOT spatialized for region ... ' + str(region_id))

         else:
             print(' ---> WARNING: Homogeneous region: ' + str(region_id) + ', number of stations: ' + str(data_this_region.shape[0]) + ' insufficient, data NOT spatialized')


    # now we must fill NaNs in maps using a national lapse rate
    elevations_all_filtered = elevation_stations[
         (~np.isnan(elevation_stations)) & (~np.isnan(data))]
    data_all_filtered = data[(~np.isnan(elevation_stations)) & (~np.isnan(data))]

    # we compute regression
    elevations_all_filtered = elevations_all_filtered.reshape((-1, 1))  # this is needed to use .fit in LinearReg
    regr_all = linear_model.LinearRegression(fit_intercept=True)
    regr_all.fit(elevations_all_filtered, data_all_filtered)

    # we apply in nan areas
    map_target[np.isnan(map_target)] = \
          regr_all.coef_[0] * dem.values[np.isnan(map_target)] + regr_all.intercept_

    # create xarray and save
    map_2d = xr.DataArray(map_target,
                          coords=[dem.coords[dem.coords.dims[0]], dem.coords[dem.coords.dims[1]]],
                          dims=['y', 'x'])
    map_2d = map_2d.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.set_crs(crs)

    return map_2d


@as_DAM_process(input_type = 'csv', output_type = 'xarray')
def interp_idw(input:pd.DataFrame,
               name_lat_lon_data_csv: list[str],
               dem:xr.DataArray,
               tmp_dir:str,
               exponent_idw:Optional[int] = 2,
               interp_radius_x:Optional[float] = 1,
               interp_radius_y:Optional[float] = 1,
               interp_no_data:Optional[float] = np.nan,
               epsg_code:Optional[str] = '4326',
               n_cpu:Optional[int]=1) -> xr.DataArray:
    """
    Interpolate data using IDW.
    """

    # load data
    data = input
    lat_points = data[name_lat_lon_data_csv[0]].to_numpy()
    lon_points = data[name_lat_lon_data_csv[1]].to_numpy()
    data_points = data[name_lat_lon_data_csv[2]].to_numpy()

    # load grid
    dem = np.squeeze(dem)

    # create random tags for temp files
    tag = random_string()

    # define paths
    path_output = tmp_dir
    os.makedirs(path_output, exist_ok=True)
    file_name_csv = os.path.join(path_output, tag + '.csv')
    file_name_vrt = os.path.join(path_output, tag + '.vrt')

    # Define geographical information
    geox_out_min = np.min(dem.coords[dem.coords.dims[1]]).values
    geox_out_max = np.max(dem.coords[dem.coords.dims[1]]).values
    geoy_out_min = np.min(dem.coords[dem.coords.dims[0]]).values
    geoy_out_max = np.max(dem.coords[dem.coords.dims[0]]).values
    geo_out_cols = dem.shape[0]
    geo_out_rows = dem.shape[1]

    # create and save csv file
    pd_data = pd.DataFrame({'x': lon_points, 'y': lat_points, 'values': data_points})
    pd_data.set_index('x', inplace=True)
    save_csv(pd_data, file_name_csv)

    # Create vrt file
    create_point_vrt(file_name_vrt, file_name_csv, tag)

    # build command
    output = os.path.join(path_output, tag + '.tif')
    interp_option = ('-a invdist:power=' + str(exponent_idw) +':smoothing=0.0:radius1=' +
                     str(interp_radius_x) + ':radius2=' +
                     str(interp_radius_y) + ':angle=0.0:nodata=' +
                     str(interp_no_data))
    line_command = ('gdal_grid -zfield "values"  -txe ' +
                    str(geox_out_min) + ' ' + str(geox_out_max) + ' -tye ' +
                    str(geoy_out_min) + ' ' + str(geoy_out_max) + ' -a_srs EPSG:' + epsg_code + ' ' +
                    interp_option + ' -outsize ' + str(geo_out_rows) + ' ' + str(geo_out_cols) +
                    ' -of GTiff -ot Float32 -l ' + tag + ' ' +
                    file_name_vrt + ' ' + output + ' --config GDAL_NUM_THREADS ' + str(n_cpu))
    exec_process(command_line=line_command)

    # read output
    map_2d = read_geotiff(output)
    return map_2d


