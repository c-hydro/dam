from dam import DAMWorkflow
from dam.tools.data import LocalDataset
from dam.tools.timestepping import TimeRange

from dam.processing.filter import keep_valid_range, apply_binary_mask
from dam.processing.calc import apply_scale_factor
from dam.processing.tile import combine_tiles, split_in_tiles
from dam.processing.warp import match_grid

import os

DATA_PATH = '/home/luca/Documents/CIMA_code/tests/VIIRS_processing'

def main():

    data = LocalDataset(path     = os.path.join(DATA_PATH, 'raw', '%Y', '%m', '%d'),
                        filename = 'VIIRS-{variable}_%Y%m%d_tile{tile}.tif',
                        tile_names = os.path.join(DATA_PATH, 'global_tile_list.txt'))

    keep_dict = {'012': [0,1],             # main algorithm used
                 '3':   [0],               # both sensors working
                 '4567':[1,2,3,4,5,6,7,8]} # remove non-vegetated areas (incl. water (0))

    tile_data = LocalDataset(path = os.path.join(DATA_PATH, 'tile_data'),
                             filename = 'VIIRS-FAPAR_%Y%m%d_tile{tile}.tif',
                             tile_names = os.path.join(DATA_PATH, 'global_tile_list.txt'))
    output = LocalDataset(path = os.path.join(DATA_PATH, 'processed'),
                          filename = 'VIIRS-FAPAR_%Y%m%d_tile{tile}.tif')

    grid = LocalDataset(path = DATA_PATH, filename='grid_01dd.tif')

    wf = DAMWorkflow(input = tile_data, output = output, #data.update(variable = 'FAPAR'), output = output,
                     options = {'intermediate_output' : 'Tmp',
                                'tmp_dir' : os.path.join(DATA_PATH, 'tmp'),
                                'break_on_missing_tiles' : False})
    
    #wf.add_process(keep_valid_range, valid_range = (0, 100))
    #wf.add_process(apply_binary_mask, mask = data.update(variable = 'FAPAR_QC'), keep = keep_dict)
    wf.add_process(combine_tiles)
    wf.add_process(match_grid, grid = grid, resampling_method = 'NearestNeighbour')
    wf.add_process(split_in_tiles, n_tiles = (6,4))
    wf.add_process(apply_scale_factor, scale_factor = 0.01)
    
    wf.run(time = TimeRange('2024-07-11', '2024-07-11'))
if __name__ == '__main__':
    main()