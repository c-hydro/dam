import numpy as np
import xarray as xr
from dam.utils.io_geotiff import read_geotiff_asXarray
from sklearn import linear_model

# pivot to choose the right interp method
def interpolator(data = np.array, lat=np.array, lon=np.array, method = str, grids = dict):

    if method == 'linear_with_elevation':
        map_2d = interp_linear_with_elevation(data, lat, lon, grids)
    elif method == 'idw':
        pass
    else:
        raise ValueError('Interpolation method not recognized')

    return map_2d

def interp_linear_with_elevation(data = np.array, lat=np.array, lon=np.array, grids = dict,
                                 minimum_number_sensors_in_region = 10, minimum_r2 = 0.25):

    # get homogeneous regions and elevation for each station
    xr_homogeneous_regions = np.squeeze(grids['homogeneous_regions'])
    xr_DEM = np.squeeze(grids['DEM'])
    lon_query = xr.DataArray(lon, dims="points")
    lat_query = xr.DataArray(lat, dims="points")
    homogeneous_regions_stations = ltln2val_from_2dDataArray(input_map=xr_homogeneous_regions, lat=lat_query, lon=lon_query, method="nearest")
    elevation_stations = ltln2val_from_2dDataArray(input_map=xr_DEM, lat=lat_query, lon=lon_query, method="nearest")

    # get list of homogeneous regions
    list_regions = np.unique(np.ravel(homogeneous_regions_stations.values))
    list_regions = list_regions[list_regions > 0]

    # loop on this list and do the work for each region; if data points are insufficient, skip computations!
    temp_map_target = np.empty([xr_DEM.shape[0], xr_DEM.shape[1]]) * np.nan
    for i, region_id in enumerate(list_regions):

        # determine available stations in this region
        data_this_region = data[homogeneous_regions_stations.values == region_id]

        if data_this_region.shape[0] >= minimum_number_sensors_in_region:

            elevations_this_region = elevation_stations.values[homogeneous_regions_stations.values == region_id]
            elevations_this_region_filtered = elevations_this_region[(~np.isnan(elevations_this_region)) | (~np.isnan(data_this_region))]
            data_this_region_filtered = data_this_region[(~np.isnan(elevations_this_region)) | (~np.isnan(data_this_region))]

            # compute linear regression
            elevations_this_region_filtered = elevations_this_region_filtered.reshape((-1, 1))  # this is needed to use .fit in LinearReg
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(elevations_this_region_filtered, data_this_region_filtered)
            r2 = regr.score(elevations_this_region_filtered, data_this_region_filtered)

            if r2 >= minimum_r2:
                temp_map_target[xr_homogeneous_regions.values == region_id] = \
                     regr.coef_[0] * xr_DEM.values[xr_homogeneous_regions.values == region_id] + regr.intercept_

            else:
                print(' ---> WARNING: r2 was LOWER than threshold, data NOT spatialized for region ... ' + str(region_id))

        else:
            print(' ---> WARNING: Homogeneous region: ' + str(region_id) + ', number of stations: ' + str(data_this_region.shape[0]) + ' insufficient, data NOT spatialized')


    # # now we must fill NaNs in maps using a national lapse rate
    elevations_all = elevation_stations.values
    elevations_all_filtered = elevations_all[
        (~np.isnan(elevations_all)) | (~np.isnan(data))]
    data_all_filtered = data[(~np.isnan(elevations_all)) | (~np.isnan(data))]

    # we compute regression
    elevations_all_filtered = elevations_all_filtered.reshape((-1, 1))  # this is needed to use .fit in LinearReg
    regr_all = linear_model.LinearRegression(fit_intercept=True)
    regr_all.fit(elevations_all_filtered, data_all_filtered)
    r2_all = regr_all.score(elevations_all_filtered, data_all_filtered)

    # we apply in nan areas
    temp_map_target[np.isnan(temp_map_target)] = \
         regr_all.coef_[0] * xr_DEM.values[np.isnan(temp_map_target)] + regr_all.intercept_

    # create xarray
    map_2d = xr.DataArray(temp_map_target,
                            coords=[xr_DEM.coords[xr_DEM.coords.dims[0]], xr_DEM.coords[xr_DEM.coords.dims[1]]],
                          dims=['y', 'x'])

    return map_2d

# QAQC method based on climatology
def QAQC_climatology(dati = np.array, lat=np.array, lon=np.array, path_climatology = str, thresholds = np.array):

    # load climatology as xarray
    xr_climatology = np.squeeze(read_geotiff_asXarray(path_climatology))
    lon_query = xr.DataArray(lon, dims="points")
    lat_query = xr.DataArray(lat, dims="points")
    climatology_points = (
        ltln2val_from_2dDataArray(input_map=xr_climatology, lat=lat_query, lon=lon_query, method="nearest"))

    # compute min e max range
    climatology_points_min = climatology_points - thresholds[0]
    climatology_points_max = climatology_points + thresholds[1]

    # # apply threshold
    dati[dati < climatology_points_min.values] = np.nan
    dati[dati > climatology_points_max.values] = np.nan

    number_stations_before = dati.shape[0]
    lat = lat[~np.isnan(dati)]
    lon = lon[~np.isnan(dati)]
    dati = dati[~np.isnan(dati)]
    number_stations_after = dati.shape[0]
    filtered_stations = number_stations_before - number_stations_after

    return dati, lat, lon, filtered_stations


# Method to extract values from xr.DataArray based on lat and lon
def ltln2val_from_2dDataArray(input_map = xr.DataArray, lat=np.array, lon=np.array, method = str):

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