from ..utils.register_agg import as_agg_function

from d3tools.timestepping import TimeRange

import xarray as xr
import datetime as dt
import numpy as np
from typing import Sequence

@as_agg_function()
def sum(input: list[xr.DataArray],
        input_agg: list[TimeRange],
        this_agg: TimeRange,
        na_ignore = True,
        **kwargs) -> xr.DataArray:

     all_inputs = np.stack(input, axis = 0)
     if na_ignore:
          sum_data = np.nansum(all_inputs, axis = 0)
     else:
          sum_data = np.sum(all_inputs, axis = 0)
     sum_da = xr.DataArray(sum_data, coords = input[0].coords, dims = input[0].dims)

     return sum_da

@as_agg_function()
def mean(input: list[xr.DataArray],
         input_agg: list[TimeRange],
         this_agg: TimeRange,
         na_ignore = True,
         **kwargs) -> xr.DataArray:
     
     all_inputs = np.stack(input, axis = 0)

     if na_ignore:
          sum_data = np.nansum(all_inputs, axis = 0)
          count_data = np.sum(~np.isnan(all_inputs), axis = 0)
          mean_data = sum_data / count_data
     else:
          sum_data = np.sum(all_inputs, axis = 0)
          mean_data = sum_data / len(input)
     
     mean_da = xr.DataArray(mean_data, coords = input[0].coords, dims = input[0].dims)

     return mean_da

def calc_overlap(this_tr: TimeRange, other_tr: TimeRange|list[TimeRange]) -> TimeRange:
     if isinstance(other_tr, Sequence):
          return [calc_overlap(this_tr, tr) for tr in other_tr]
     
     start = max(this_tr.start, other_tr.start)
     end = min(this_tr.end, other_tr.end)
     size = TimeRange(start, end).length()
     original_size = this_tr.length()
     if size < 1:
          size = TimeRange(start, end).length(unit = 'hours')
          original_size = this_tr.length(unit = 'hours')

     return size / original_size