import pandas as pd
import logging
from dam.utils.logging import set_logging
from dam.utils.io_geotiff import read_geotiff_asXarray, write_geotiff_fromXarray
from dam.utils.io_csv import read_csv
from dam.interp import QAQC_climatology, interpolator

def main():

    # ------------------------------------------
    # parameters
    path_data_in = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_drops2/%Y/%m/%d/DROPS2_TERMOMETRO_%Y%m%d%H%M.csv'
    name_columns_data_in = ['station_id', 'station_name', 'lat', 'lon', 'data']
    path_climatology_QAQC = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/BIGBANG/td_ltaa_%mm_WGS84.tif'
    path_data_out = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/maps/%Y/%m/%d/{variable}_%Y%m%d%H%M.tif'
    period = pd.date_range(start='2024-04-01', end='2024-04-02 00:00:00', freq='H')
    grids = {'DEM': '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/DEM_Italy_grid_MCM_v2.tif',
                'homogeneous_regions': '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/test_interp/static/Zone_Vigilanza_01_2021_WGS84_v2.tif'}
    #note: methods below require the grids to be in the same projection as the data (EPSG:4326) AND SAME GRID SIZE
    #also, dictionary keys must be the same as above
    interpolation_method = 'linear_with_elevation'
    interpolation_residuals = 'idw'
    smoothing = True
    # ------------------------------------------

    # ------------------------------------------
    # load grids from grids
    for grid_name in grids:
        temp_grid = read_geotiff_asXarray(grids[grid_name])
        grids[grid_name] = temp_grid # we replace the path with the actual grid
    # ------------------------------------------

    # ------------------------------------------
    # loop on timestamps
    for timestamp in period:

        # load data
        path_data_in = timestamp.strftime(path_data_in)
        data = read_csv(path_data_in)
        lat = data[name_columns_data_in[2]].to_numpy()
        lon = data[name_columns_data_in[3]].to_numpy()
        data = data[name_columns_data_in[4]].to_numpy()

        # apply QAQC
        path_climatology_QAQC = timestamp.strftime(path_climatology_QAQC)
        data_filtered, lat_filtered, lon_filtered, filtered = (
            QAQC_climatology(data, lat, lon, path_climatology_QAQC, thresholds = [25, 25]))

        # interpolate
        map_2d = interpolator(data_filtered, lat_filtered, lon_filtered,
                              method = interpolation_method, grids = grids)

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