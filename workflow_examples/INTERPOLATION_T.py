from dam.tools.data import LocalDataset
from dam.tools.timestepping import TimeRange
from dam import DAMWorkflow

from dam.processing.filter import filter_csv_with_climatology
from dam.processing.interp import interp_with_elevation, interp_idw
from dam.processing.calc import compute_residuals

import os

DATA_PATH = '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/july_2024/test_drops2'

def main():

    data = LocalDataset(path=os.path.join(DATA_PATH, '%Y', '%m', '%d'),
                        filename='DROPS2_TERMOMETRO_%Y%m%d%H%M.csv',
                        time_signature='start')
    climatologia = LocalDataset(
        path='/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/july_2024/test_interp/static/BIGBANG/',
        filename='td_ltaa_%mm_WGS84.tif')
    homogeneous_regions = LocalDataset(
        path='/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/july_2024/test_interp/static/',
        filename='Zone_Vigilanza_01_2021_WGS84_v2.tif')
    dem = LocalDataset(
        path='/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/july_2024/test_interp/static/',
        filename='DEM_Italy_grid_MCM_v2.tif')
    grid = LocalDataset(
        path='/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/july_2024/test_interp/static/',
        filename='MCM_mask_0nan.tif')
    output = LocalDataset(
        path='/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/december_2024/maps_T/',
        filename='AirT_%Y%m%d%H%M.tif', time_signature='start',
        name='AirT')

    wf = DAMWorkflow(input=data, output=output,  # data.update(variable = 'FAPAR'), output = output,
                     options={'intermediate_output': 'Tmp',
                              'tmp_dir': '/home/francesco/Documents/Projects/Drought/IT_DROUGHT/DEV_procedure/december_2024/tmp/'})

    wf.add_process(filter_csv_with_climatology, climatology=climatologia,
                   thresholds=[25, 25], name_lat_lon_data_csv=['lat', 'lon', 'data'])
    wf.add_process(interp_with_elevation, DEM=dem, homogeneous_regions=homogeneous_regions,
                   name_lat_lon_data_csv=['lat', 'lon', 'data'])
    wf.add_process(compute_residuals, data=data,
                   name_lat_lon_data_csv=['lat', 'lon', 'data'])
    wf.add_process(interp_idw, name_lat_lon_data_csv=['lat', 'lon', 'data'],
                   dem=dem, tmp_dir=wf.tmp_dir)

    wf.run(time=TimeRange('2024-04-01 00:00', '2024-04-02 00:00'))

if __name__ == '__main__':
    main()