from osgeo import gdal
import re
import numpy as np

from typing import Optional

from .utils.io_geotiff import read_geotiff_asGDAL, write_geotiff_fromGDAL
from .utils.rm import remove_file

def combine_tiles(inputs: list[str],
                  output: Optional[str] = None,
                  rm_input: bool = False) -> str:
    """
    Mosaic a set of input rasters.
    """
    inputs.sort()
    if output is None:
        # replace any combination of '-tile-0' with ''
        # where - is either - or _ or nothing
        s = inputs[0]
        output = re.sub('[-_]?tile[-_]?\d{1,3}', '', s)
    
    out_ds = gdal.Warp('', inputs, format = 'MEM', options=['NUM_THREADS=ALL_CPUS'])
    write_geotiff_fromGDAL(out_ds, output)

    if rm_input:
        for input in inputs:
            remove_file(input)

    return output

def split_in_tiles(input: str,
                   tile_size: int|tuple[int, int] = 1024,
                   output: Optional[str] = None,
                   rm_input: bool = False) -> list[str]:
    """
    Split a raster into tiles.
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    tile_xsize, tile_ysize = tile_size

    if output is None:
        output = input.replace('.tif', '_tile{tile}.tif')
    
    #get the input raster
    ds = gdal.Open(input)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    xsizes = optimal_sizes(xsize, tile_xsize)
    ysizes = optimal_sizes(ysize, tile_ysize)
    
    nx = len(xsizes)
    ny = len(ysizes)

    outfiles = []
    for it in range(nx):
        for jt in range(ny):
            xoff = sum(xsizes[:it])
            yoff = sum(ysizes[:jt])
            tile_xsize = xsizes[it]
            tile_ysize = ysizes[jt]
            tile_file = output.format(tile=it+jt*nx)
            tile_ds = gdal.Translate('', input, format='MEM', srcWin=[xoff, yoff, tile_xsize, tile_ysize])
            write_geotiff_fromGDAL(tile_ds, tile_file)
            outfiles.append(tile_file)

    if rm_input:
        remove_file(input)
    
    return outfiles

def optimal_sizes(N, n):
    pieces = N // n
    remainder = N % n

    if remainder > n / 2:
        pieces += 1

    sizes = [np.round(N / pieces)] * (pieces-1)
    remainder = N - sum(sizes)
    sizes.append(remainder)

    return sizes