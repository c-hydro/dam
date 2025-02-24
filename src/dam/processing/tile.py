import xarray as xr
import numpy as np
from collections import defaultdict
from typing import Optional, Generator

from ..utils.register_process import as_DAM_process
from d3tools.errors import GDAL_ImportError

@as_DAM_process(input_type = 'file', output_type = 'gdal', input_tiles = True)
def combine_tiles(inputs: list[str]|list['gdal.Dataset'],
                  num_cpus: Optional[int] = None
                  )-> 'gdal.Dataset':
    """
    Mosaic a set of input rasters.
    """

    try:
        from osgeo import gdal
    except ImportError:
        raise GDAL_ImportError(function = 'dam.processing.combine_tiles')

    if num_cpus is None:
        num_cpus = 'ALL_CPUS'
 
    warp_options = gdal.WarpOptions(
        format='MEM',
        multithread=True,
        warpOptions=[f'NUM_THREADS={num_cpus}']
    )
 
    out_ds = gdal.Warp('', inputs, options=warp_options)

    return out_ds

@as_DAM_process(input_type = 'xarray', output_type = 'xarray', output_tiles = True, tile_name_attr = 'tile_name')
def split_in_tiles(input: str,
                   tile_size: Optional[int | tuple[int, int]] = None,
                   n_tiles: Optional[int | tuple[int, int]] = None,
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
    if tile_size is not None:
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        tile_xsize, tile_ysize = tile_size
    elif n_tiles is not None:
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

    def get_tile_name(hi, vi):
        sub_values = defaultdict(str, i=i, hi=hi, vi=vi)
        tile_name = tile_names.format_map(sub_values)
        return tile_name
            
    i = 0
    if dir == 'hv':
        for vi in range(ny):
            for hi in range(nx):
                tile_data = make_tile(hi, vi)
                tile_name = get_tile_name(hi, vi)
                tile_data.attrs['tile_name'] = tile_name
                
                i += 1
                yield tile_data
    elif dir == 'vh':
        for hi in range(nx):
            for vi in range(ny):
                tile_data = make_tile(hi, vi)
                tile_name = get_tile_name(hi, vi)
                tile_data.attrs['tile_name'] = tile_name
                
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