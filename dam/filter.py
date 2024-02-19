from typing import Optional

import numpy as np
import xarray as xr
import rioxarray
import os

import unpackqa

from .utils.io_geotiff import read_geotiff_asXarray, write_geotiff_fromXarray
from .utils.rm import remove_file

### functions useful for filtering data
# all these functions should be applied to a single band GeoTIFF file (that is the first imput of the function)
# and they all save the output to a single band GeoTIFF file

def keep_valid_range(input: str,
                     valid_range: tuple[float],
                     nodata_value: float|int = np.nan,
                     destination: Optional[str] = None,
                     rm_input: bool = False) -> str:
    """
    Keep only the values in the valid range.
    """
    if destination is None:
        destination = input.replace('.tif', '_validrange.tif')

    data = read_geotiff_asXarray(input)
    data = data.where((data >= valid_range[0]) & (data <= valid_range[1]), other=nodata_value)

    data = data.rio.write_nodata(nodata_value)
    write_geotiff_fromXarray(data, destination)

    if rm_input:
        remove_file(input)

    return destination

def apply_binary_mask(input: str,
                      mask: str,
                      keep: dict[str, list[int]],
                      nodata_value: float|int = np.nan,
                      get_masks: bool = False,
                      destination: Optional[str] = None,
                      rm_input: bool = False) -> str:
    """
    Apply a bitwise mask to the input. These are for example the QA flags of MODIS and VIIRS products.
    The rules are defined in a dictionary called keep, where the key is bit numbers in string format and the value is a list of values that should be kept.
    e.g. keep = {'0':[0,1], '1':[0]} means that the first bit should be 0 or 1 and the second bit should be 0.
         keep = {'01':[2,3], '2':[0]} means that the first two bits should be 2 or 3 and the third bit should be 0.
    
    note that the bits are counted from right to left, starting from 0.
    for an 8-bit number, the bits are numbered from 0 to 7: 76543210
    """
    if destination is None:
        destination = input.replace('.tif', '_filtered.tif')
    
    data = read_geotiff_asXarray(input)
    mask = read_geotiff_asXarray(mask)

    # get the flag info and the values to keep from the keep dictionary
    flag_info = {}
    keep_values = {}
    for i, key in enumerate(keep.keys()):
        flag_info[f'f{i}'] = [int(v) for v in key]
        keep_values[f'f{i}'] = keep[key]

    # get the number of bits from the data type of the mask
    num_bits = np.iinfo(mask.dtype).bits

    # QC flags, i.e. the meaning of each bit in the QC file
    QC_binary_flags = {'flag_info': flag_info,
                       'max_value' : 2**num_bits - 1,
                       'num_bits'  : num_bits}
    
    # unpack the QC file to a dictionary of masks
    unpacked_mask = unpackqa.unpack_to_dict(mask.values, QC_binary_flags)
    for flag_n, mask in unpacked_mask.items():
        data = data.where(np.isin(mask, keep_values[flag_n]), other=nodata_value)
        if get_masks:
            mask_ds = data.copy(data=mask)
            mask_dest = input.replace(".tif", f"_mask{flag_n}.tif")
            write_geotiff_fromXarray(mask_ds, mask_dest)

    data = data.rio.write_nodata(nodata_value)
    write_geotiff_fromXarray(data, destination)

    if rm_input:
        remove_file(input)

    return destination
    
def apply_raster_mask(input: str,
                      mask: str,
                      filter_values: list[float|int] = [np.nan],
                      nodata_value: float|int = np.nan,
                      destination: Optional[str] = None,
                      rm_input: bool = False) -> xr.DataArray:
    """
    Apply a raster map to the input. The raster map is a DataArray with the same shape as the input, where each value corresponds to a category.
    The input is masked where the raster map has the nodata value.
    """
    if destination is None:
        destination = input.replace('.tif', '_masked.tif')
    
    data = read_geotiff_asXarray(input)
    mask = read_geotiff_asXarray(mask)

    for value in filter_values:
        data = data.where(mask != value, other=nodata_value)

    data = data.rio.write_nodata(nodata_value)
    write_geotiff_fromXarray(data, destination)

    if rm_input:
        remove_file(input)
    
    return destination