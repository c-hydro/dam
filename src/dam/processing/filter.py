from typing import Optional

import pandas as pd
import numpy as np
import xarray as xr
import re

import unpackqa

from ..utils.geo_utils import ltln2val_from_2dDataArray
from ..utils.register_process import as_DAM_process

### functions useful for filtering data
@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def keep_valid_range(input: xr.DataArray,
                     valid_range: Optional[tuple[float]] = None,
                     nodata_value: Optional[float|int] = None,
                     ) -> xr.DataArray:
    """
    Keep only the values in the valid range.
    """

    data = input

    if valid_range is None:
        valid_range_str = data.attrs.pop('valid_range')
        try:
            valid_range = tuple(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", valid_range_str)))
        except:
            raise ValueError("Could not parse valid range from metadata")
        if valid_range is None:
            raise ValueError("No valid range provided")

    if nodata_value is None:
        nodata_value = data.attrs.get('_FillValue')

    data = data.where((data >= valid_range[0]) & (data <= valid_range[1]), other=nodata_value)
    data = data.rio.write_nodata(nodata_value)

    return data

@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def apply_binary_mask(input: xr.DataArray,
                      mask: xr.DataArray,
                      keep: list[tuple[list[int]]],
                      nodata_value: Optional[float|int] = None,
                      get_masks: bool = False
                      ) -> xr.DataArray|tuple[xr.DataArray, list[xr.DataArray]]:
    """
    Apply a bitwise mask to the input. These are for example the QA flags of MODIS and VIIRS products.
    The rules are defined in a list a tuples (or lists) called keep, in each tuple, the first element is a list of bit numbers, the second is the values to keep for those bits.
    e.g. keep = [([0],  [0,1]), ([1],[0])] means that the first bit should be 0 or 1 and the second bit should be 0.
         keep = [([0,1],[2,3]), ([2],[0])} means that the first two bits should be 2 or 3 and the third bit should be 0.
    
    note that the bits are counted from right to left, starting from 0.
    for an 8-bit number, the bits are numbered from 0 to 7: 76543210
    """
    
    data = input
    mask = mask

    if nodata_value is None:
        nodata_value = data.attrs.get('_FillValue')

    # get the flag info and the values to keep from the keep dictionary
    flag_info = {}
    keep_values = {}
    for i, rule in enumerate(keep):
        flag_info[f'f{i}'] = rule[0]
        keep_values[f'f{i}'] = rule[1]

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

    data = data.rio.write_nodata(nodata_value)

    if get_masks:
        return data, [m for m in unpacked_mask.values()]
    else:
        return data

@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def apply_raster_mask(input: xr.DataArray,
                      mask: xr.DataArray,
                      filter_values: list[float|int]|str = None,
                      nodata_value: Optional[float|int] = None,
                      ) -> xr.DataArray:
    """
    Apply a raster map to the input. The raster map is a DataArray with the same shape as the input, where each value corresponds to a category.
    The input is masked (set to nodata_value) where the raster map has the filter_values.
    """
    if isinstance(filter_values, str) and filter_values.lower() == 'none':
        return input
    elif filter_values is None:
        filter_values = []

    data = input
    mask_data = mask

    if nodata_value is None:
        nodata_value = data.attrs.get('_FillValue')

    # make sure coordinates of the mask and the data are in the same order
    mask_nodata = mask_data.attrs.get('_FillValue')
    mask_data = mask_data.rio.reproject_match(data)

    for value in filter_values + [mask_nodata]:
        data = data.where(~np.isclose(mask_data, value, equal_nan=True), other=nodata_value)

    data = data.rio.write_nodata(nodata_value)
    return data

@as_DAM_process(input_type = 'csv', output_type = 'csv')
def filter_csv_with_climatology(input: pd.DataFrame,
                     climatology: xr.DataArray,
                     thresholds: list[float],
                     name_lat_lon_data_csv: list[str]) -> pd.DataFrame:
    """
        Filter a dataframe based on a climatology +- thresholds.
    """

    data = input
    climatology = climatology.squeeze()
    lat = data[name_lat_lon_data_csv[0]].to_numpy()
    lon = data[name_lat_lon_data_csv[1]].to_numpy()

    #now get climatology values for the same lat and lon
    climatology_points = (ltln2val_from_2dDataArray(input_map=climatology, lat=lat, lon=lon, method="nearest"))

    # compute min e max range
    climatology_points_min = climatology_points - thresholds[0]
    climatology_points_max = climatology_points + thresholds[1]

    # apply threshold tp dataframe
    data[data[name_lat_lon_data_csv[2]] < climatology_points_min.values] = np.nan
    data[data[name_lat_lon_data_csv[2]] > climatology_points_max.values] = np.nan
    data.set_index(data.columns[0], inplace=True)
    # remove columns with all NaNs
    data = data.dropna(axis='rows', how='all')

    return data
