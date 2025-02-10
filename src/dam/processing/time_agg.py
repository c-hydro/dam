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
        **kwargs) -> xr.DataArray:
     
     overlaps = calc_overlap(this_agg, input_agg)
     total_overlap = np.sum(overlaps)
     rel_overlaps = [ov / total_overlap for ov in overlaps]

     multiplied_data = [rel_ov * da.values * len(input) for rel_ov, da in zip(rel_overlaps, input)]
     all_inputs = np.stack(multiplied_data, axis = 0)
     
     sum_data = np.sum(all_inputs, axis = 0)
     sum_da = xr.DataArray(sum_data, coords = input[0].coords, dims = input[0].dims)

     return sum_da

@as_agg_function()
def mean(input: list[xr.DataArray],
         input_agg: list[TimeRange],
         this_agg: TimeRange,
         **kwargs) -> xr.DataArray:
     
     overlaps = calc_overlap(this_agg, input_agg)
     total_overlap = np.sum(overlaps)
     rel_overlaps = [ov / total_overlap for ov in overlaps]

     multiplied_data = [rel_ov * da.values for rel_ov, da in zip(rel_overlaps, input)]
     all_inputs = np.stack(multiplied_data, axis = 0)

     sum_data = np.sum(all_inputs, axis = 0)
     sum_da = xr.DataArray(sum_data, coords = input[0].coords, dims = input[0].dims)

     return sum_da

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

     