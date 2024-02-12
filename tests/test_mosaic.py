from dam.tile import combine_tiles, split_in_tiles
from dam.warp import match_grid

import glob

def main():
    path = '/home/luca/Documents/CIMA_code/tests/VIIRS_dwl/viirs_IT_test/'
    pattern = 'VIIRS-FAPAR_20120117_ita-tile*.tif'

    # get all the files
    files = glob.glob(path + pattern)

    # make the mosaic
    mosaic = combine_tiles(files)
    print(mosaic)

    GRID = '/home/luca/Documents/CIMA_code/EDO-GDO/grids/grid_01dd.tif'
    # match the grid
    regridded = match_grid(mosaic, GRID)
    print(regridded)

    # split the mosaic
    tiles = split_in_tiles(regridded, tile_size=(1000, 1000))
    print(tiles)
    
    
if __name__ == '__main__':
    main()