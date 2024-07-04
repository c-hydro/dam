import pandas as pd
import logging
from dam.utils.logging import set_logging
from dam.utils.io_geotiff import read_geotiff_asXarray, write_geotiff_fromXarray
from dam.interp import interp_with_elevation
from dam.filter import filter_csv_with_climatology

def main():

    # ------------------------------------------
    # parameters
    path_data_in = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_drops2/%Y/%m/%d/DROPS2_TERMOMETRO_%Y%m%d%H%M.csv'
    name_columns_data_in = ['station_id', 'station_name', 'lat', 'lon', 'data']
    path_climatology_QAQC = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/BIGBANG/td_ltaa_%mm_WGS84.tif'
    path_out_elevation = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/maps/%Y/%m/%d/AirT_%Y%m%d%H%M.tif'
    path_out_with_residuals = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/maps/%Y/%m/%d/AirT_with_residuals_%Y%m%d%H%M.tif'
    path_out_smoothed = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/maps/%Y/%m/%d/AirT_smoothed_%Y%m%d%H%M.tif'
    period = pd.date_range(start='2024-04-01', end='2024-04-02 00:00:00', freq='H')
    path_DEM = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/DEM_Italy_grid_MCM_v2.tif'
    path_homogeneous_regions = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/Zone_Vigilanza_01_2021_WGS84_v2.tif'
    #note: methods below require the grids to be in the same projection as the data (EPSG:4326) AND SAME GRID SIZE
    #also, dictionary keys must be the same as above
    interpolation_method = 'linear_with_elevation'
    interpolation_method_residuals = 'idw'
    smoothing = True
    # ------------------------------------------

    # ------------------------------------------
    # loop on timestamps
    for timestamp in period:

        # apply QAQC on original data
        path_climatology_QAQC_this_timestamp = timestamp.strftime(path_climatology_QAQC)
        path_data_in_this_timestamp = timestamp.strftime(path_data_in)
        path_filtered = filter_csv_with_climatology(
            input=path_data_in_this_timestamp, climatology=path_climatology_QAQC_this_timestamp,
            thresholds = [25, 25], name_columns_csv=name_columns_data_in)

        # interpolate with elevation
        path_out_elevation_this_timestamp = timestamp.strftime(path_out_elevation)
        interp_with_elevation(input=path_filtered,
                              name_columns_csv=name_columns_data_in,
                              homogeneous_regions=path_homogeneous_regions,
                              dem=path_DEM,
                              destination=path_out_elevation_this_timestamp)

        # compute and apply residuals


        # smoothing

        # mask on domain

        # save

    # ------------------------------------------










    print()






    # loop on timestamps

    # do the work





    
if __name__ == '__main__':
    main()