import pandas as pd
from dam.processing.interp import interp_with_elevation, interp_idw
from dam.processing.filter import filter_csv_with_climatology, apply_raster_mask
from dam.processing.calc import combine_raster_data, compute_residuals
from dam.processing.smoothing import gaussian_smoothing

def main():

    # ------------------------------------------
    # parameters
    path_data_in = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_drops2/%Y/%m/%d/DROPS2_TERMOMETRO_%Y%m%d%H%M.csv'
    name_lat_lon_data_in = ['lat', 'lon', 'data']
    path_climatology_QAQC = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/BIGBANG/td_ltaa_%mm_WGS84.tif'
    path_out_elevation = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/maps/%Y/%m/%d/AirT_%Y%m%d%H%M.tif'
    period = pd.date_range(start='2024-04-01', end='2024-04-02 00:00:00', freq='H')
    path_DEM = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/DEM_Italy_grid_MCM_v2.tif'
    path_MASK = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/MCM_mask_0nan.tif'
    path_homogeneous_regions = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/Zone_Vigilanza_01_2021_WGS84_v2.tif'
    method_residuals = 'data_minus_map'
    # ------------------------------------------

    # ------------------------------------------
    # loop on timestamps
    for timestamp in period:

        # apply QAQC on original data
        path_climatology_QAQC_this_timestamp = timestamp.strftime(path_climatology_QAQC)
        path_data_in_this_timestamp = timestamp.strftime(path_data_in)
        path_filtered = filter_csv_with_climatology(
            input=path_data_in_this_timestamp, climatology=path_climatology_QAQC_this_timestamp,
            thresholds = [25, 25], name_lat_lon_data_csv=name_lat_lon_data_in)

        # interpolate with elevation
        path_out_elevation_this_timestamp = timestamp.strftime(path_out_elevation)
        interp_with_elevation(input=path_filtered,
                              name_lat_lon_data_csv=name_lat_lon_data_in,
                              homogeneous_regions=path_homogeneous_regions,
                              dem=path_DEM,
                              output=path_out_elevation_this_timestamp)

        # compute residuals
        path_residuals = compute_residuals(input=[path_filtered, path_out_elevation_this_timestamp],
                                           name_lat_lon_data_csv=name_lat_lon_data_in,
                                           method_residuals=method_residuals)

        # compute idw of residuals
        path_out_residuals_this_timestamp = path_out_elevation_this_timestamp.replace('.tif', '_idw_residuals.tif')
        path_idw_residuals = interp_idw(input=path_residuals, name_lat_lon_data_csv=name_lat_lon_data_in,
                                        output=path_out_residuals_this_timestamp,
                                        grid=path_DEM, n_cpu=6,
                                        exponent_idw=2,
                                        interp_radius_x=1,
                                        interp_radius_y=1, rm_input=True)

        # apply residuals
        path_map_with_residuals = path_out_elevation_this_timestamp.replace('.tif', '_with_residuals.tif')
        combine_raster_data(input=[path_out_elevation_this_timestamp, path_out_residuals_this_timestamp],
                            statistic='sum', output= path_map_with_residuals,
                            na_ignore=True, rm_input=True)
        # we sum them as we computed the residuals as data - map, thus data = map + residuals

        # smoothing
        path_map_with_residuals_smoothed = gaussian_smoothing(input=path_map_with_residuals, stddev_kernel=8, rm_input=True)

        # apply mask
        apply_raster_mask(path_map_with_residuals_smoothed, path_MASK,
                          output=path_map_with_residuals_smoothed.replace('.tif', '_masked.tif'), rm_input=True)
    # ------------------------------------------

if __name__ == '__main__':
    main()