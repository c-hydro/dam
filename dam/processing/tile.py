from osgeo import gdal
import re
import numpy as np
import tempfile

from typing import Optional

from ..utils.io_geotiff import write_geotiff, read_geotiff
from ..utils.rm import remove_file

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
    write_geotiff(out_ds, output)

    if rm_input:
        for input in inputs:
            remove_file(input)

    return output


def split_in_tiles(input: str,
                   tile_size: int | tuple[int, int] = 1024,
                   mask: Optional[str] = None,
                   mask_nodata: Optional[int] = None,
                   output: Optional[str] = None,
                   only_numtiles: bool = False,
                   rm_input: bool = False) -> list[str]:
    """
    Split a raster into tiles.
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    tile_xsize, tile_ysize = tile_size

    if output is None:
        output = input.replace('.tif', '_tile{tile}.tif')

    # get the input raster
    ds = gdal.Open(input)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    xsizes = optimal_sizes(xsize, tile_xsize)
    ysizes = optimal_sizes(ysize, tile_ysize)

    nx = len(xsizes)
    ny = len(ysizes)

    if only_numtiles:
        return nx * ny

    outfiles = []
    id_tile = 0
    for it in range(nx):
        for jt in range(ny):
            xoff = sum(xsizes[:it])
            yoff = sum(ysizes[:jt])
            tile_xsize = xsizes[it]
            tile_ysize = ysizes[jt]
            
            # if we have a mask, we need to check if the tile has any valid data
            if mask is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mask_ds = gdal.Translate('', mask, format='MEM', srcWin=[xoff, yoff, tile_xsize, tile_ysize])
                    write_geotiff(mask_ds, f'{tmpdir}/mask.tif')
                    mask_array = read_geotiff(f'{tmpdir}/mask.tif', out='array')
                    mask_ds = None
                if mask_nodata is not None:
                    mask_array = mask_array != mask_nodata
                else:
                    mask_array = ~np.isnan(mask_array)
                if not np.any(mask_array):
                    continue
                else:
                    # get the minimal extent of the non-nan values
                    x_min, x_max = np.where(mask_array.any(axis=0))[0][[0, -1]]
                    y_min, y_max = np.where(mask_array.any(axis=1))[0][[0, -1]]
                xoff += x_min
                yoff += y_min
                tile_xsize = x_max - x_min + 1
                tile_ysize = y_max - y_min + 1
            
            tile_file = output.format(tile=id_tile)
            tile_ds = gdal.Translate('', input, format='MEM', srcWin=[xoff, yoff, tile_xsize, tile_ysize])
            write_geotiff(tile_ds, tile_file)
            outfiles.append(tile_file)
            id_tile += 1

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