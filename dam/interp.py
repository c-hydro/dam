import numpy as np
import xarray as xr
from sklearn import linear_model
from typing import Optional
from dam.utils.geo_utils import ltln2val_from_2dDataArray
from dam.utils.io_csv import read_csv
from dam.utils.io_geotiff import read_geotiff_asXarray, write_geotiff_fromXarray

def interp_with_elevation(input: str,
                          homogeneous_regions: str,
                          dem: str,
                          name_columns_csv: list[str],
                          destination: str,
                          rm_input: bool = False,
                          minimum_number_sensors_in_region: Optional[int] = 10,
                          minimum_r2: Optional[float] = 0.25) -> str:

    # load data
    data = read_csv(input)
    lat_points = data[name_columns_csv[2]].to_numpy()
    lon_points = data[name_columns_csv[3]].to_numpy()
    data = data[name_columns_csv[4]].to_numpy()

    #load dem and homogeneous regions
    dem = read_geotiff_asXarray(dem)
    dem = np.squeeze(dem)
    homogeneous_regions = read_geotiff_asXarray(homogeneous_regions)
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
             elevations_this_region_filtered = elevations_this_region[(~np.isnan(elevations_this_region)) | (~np.isnan(data_this_region))]
             data_this_region_filtered = data_this_region[(~np.isnan(elevations_this_region)) | (~np.isnan(data_this_region))]

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
         (~np.isnan(elevation_stations)) | (~np.isnan(data))]
    data_all_filtered = data[(~np.isnan(elevation_stations)) | (~np.isnan(data))]

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
    write_geotiff_fromXarray(map_2d, destination)

