import xarray as xr
import numpy as np
import tempfile
from typing import Optional, Generator
import os

from ..utils.register_process import as_DAM_process
from d3tools.errors import GDAL_ImportError
from d3tools.exit import rm_at_exit

@as_DAM_process(input_type="file", output_type='file', input_tiles = True)
def combine_tiles(inputs: list[str],
                  num_cpus: Optional[int] = None
                  )-> str:
    """
    Mosaic a set of input rasters.
    """

    try:
        from osgeo import gdal
    except ImportError:
        raise GDAL_ImportError(function = 'dam.processing.combine_tiles')

    if isinstance(inputs[0], str):
        ds0 = gdal.Open(inputs[0])
        input_dataType = ds0.GetRasterBand(1).DataType
    else:
        input_dataType = inputs[0].GetRasterBand(1).DataType

    if num_cpus is None:
        num_cpus = 'ALL_CPUS'

    tmp_path = tempfile.mkdtemp()
    rm_at_exit(tmp_path)

    output_file = os.path.join(tmp_path, 'mosaic.tif')

    warp_options = gdal.WarpOptions(
        format='GTiff',
        multithread=True,
        warpOptions=[f'NUM_THREADS={num_cpus}']
    )
 
    gdal.Warp(output_file, inputs, options=warp_options, 
              outputType=input_dataType, 
              creationOptions=['COMPRESS=LZW'])

    return output_file

@as_DAM_process(input_type = 'xarray', output_type = 'xarray', output_tiles = True, input_as_is = True)
def split_in_tiles(input: str, n_tiles: int | tuple[int, int],
                   tile_names = '{i}', start = 'bl', dir = 'vh') -> Generator[xr.DataArray, None, None]:
    """
    Split a raster into tiles.

    tiles can be defined by either the tile_size or the n_tiles parameter.
    If both are provided, the tile_size parameter will be used.
    They both accept a single integer or a tuple of two integers (i.e. xsize, ysize or nx, ny).

    The tile_names parameter is a string that will be used to name the output tiles.
    It can contain the following placeholders:
    i      : the index of the tile (0 to n_tiles-1)
    hi, vi : index of the tile in horizontal and vertical direction (0 to nx-1, 0 to ny-1)
    leading zeros can be added to the index using common string formatting (e.g. {i:03}).

    start can be 'tl' (top-left), 'br' (bottom-right), 'tr' (top-right) or 'bl' (bottom-left).
    It defines the corner of the raster where the tiling starts.
    
    dir can be 'vh' (vertical-horizontal) or 'hv' (horizontal-vertical).
    It defines the order in which the tiles are generated from the starting corner.
    """
    
    # get the input raster
    data = input
    
    # rename the dimensions to x and y
    ydim = data.rio.y_dim
    xdim = data.rio.x_dim
    data = data.rename({ydim: 'y', xdim: 'x'})

    # get the size of the raster
    band, ysize, xsize = data.values.shape

    # figure out the input parameters
    if isinstance(n_tiles, int):
        n_tiles = (n_tiles, n_tiles)
    nx, ny = n_tiles
    tile_xsize = xsize // nx
    tile_ysize = ysize // ny

    xsizes = optimal_sizes(xsize, tile_xsize)
    ysizes = optimal_sizes(ysize, tile_ysize)

    nx = len(xsizes)
    ny = len(ysizes)

    # Reorder the dimensions of the raster to match the starting corner
    if start.startswith('t'):
        data = data.sortby('y', ascending=False)
    elif start.startswith('b'):
        data = data.sortby('y', ascending=True)

    if start.endswith('l'):
        data = data.sortby('x', ascending=True)
    elif start.endswith('r'):
        data = data.sortby('x', ascending=False)

    def make_tile(hi, vi):
        xoff = sum(xsizes[:hi])
        yoff = sum(ysizes[:vi])
        tile_xsize = xsizes[hi]
        tile_ysize = ysizes[vi]

        tile_data = data.isel(x=slice(xoff, xoff+tile_xsize), y=slice(yoff, yoff+tile_ysize))
        
        # reorder the dimensions of the tile correctly
        tile_data = tile_data.sortby('y', ascending=False)
        tile_data = tile_data.sortby('x', ascending=True)

        # rename the dimensions back to the original names
        tile_data = tile_data.rename({'y': ydim, 'x': xdim})
        
        return tile_data
            
    i = 0
    if dir == 'hv':
        for vi in range(ny):
            for hi in range(nx):
                tile_data = make_tile(hi, vi)
                
                i += 1
                yield tile_data
    elif dir == 'vh':
        for hi in range(nx):
            for vi in range(ny):
                tile_data = make_tile(hi, vi)
                
                i += 1
                yield tile_data
            
def optimal_sizes(N, n):
    pieces = N // n
    remainder = N % n

    if remainder > n / 2:
        pieces += 1

    sizes = [np.round(N / pieces)] * (pieces-1)
    remainder = N - sum(sizes)
    sizes.append(remainder)

    return [int(s) for s in sizes]