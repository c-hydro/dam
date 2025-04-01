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
        na_ignore = False,
        **kwargs) -> xr.DataArray:

     all_inputs = np.stack(input, axis = 0)
     if na_ignore:
          sum_data = np.nansum(all_inputs, axis = 0)
     else:
          sum_data = np.sum(all_inputs, axis = 0)
     sum_da = xr.DataArray(sum_data, coords = input[0].coords, dims = input[0].dims)

     return sum_da

@as_agg_function()
def sum_weighted(input: list[xr.DataArray],
                 input_agg: list[TimeRange],
                 this_agg: TimeRange,
                 na_ignore = False,
                 nan_threshold = 0.5,
                 **kwargs) -> xr.DataArray:

     mean_da = mean(input, input_agg, this_agg, na_ignore = na_ignore, nan_threshold = nan_threshold)
     return mean_da * this_agg.length()

@as_agg_function()
def mean(input: list[xr.DataArray],
         input_agg: list[TimeRange],
         this_agg: TimeRange,
         na_ignore = False,
         nan_threshold = 0.5,
         **kwargs) -> xr.DataArray:

     weights       = np.array(calc_overlap(this_agg, input_agg))
     all_inputs    = np.stack(input, axis = 0)

     # multiply each data in the brick by the weight, b is the band of the raster (which is always 1)
     weighted_data = np.einsum('i,ibjk->ibjk', weights, all_inputs)
     
     # sum all the weighted data and divide by the total weight, protecting nans
     mask = np.isfinite(weighted_data) # <- this is a mask of all the nans in the weighted data
     weights_3d = np.broadcast_to(weights[:, np.newaxis, np.newaxis, np.newaxis], weighted_data.shape)
     selected_weights = np.where(mask, weights_3d, 0)
     total_weights = np.sum(selected_weights, axis=0)

     # sum all the weighted data
     if na_ignore:
          weighted_sum = np.nansum(weighted_data, axis = 0)
     else:
          weighted_sum = np.sum(weighted_data, axis = 0)
     
     # divide by the total weights
     # to avoid division by zero, we set the zero weights to nan right away
     weighted_sum  = np.where(total_weights < 1e-6, np.nan, weighted_sum)
     total_weights = np.where(total_weights < 1e-6, 1e-6, total_weights)
     weighted_mean = weighted_sum / total_weights

     # Remove where the total sum of weights is less than nan_threshold x 100% the total weights
     weighted_mean = np.where(total_weights/np.sum(weights) <= nan_threshold, np.nan, weighted_mean)
     
     mean_da = xr.DataArray(weighted_mean, coords = input[0].coords, dims = input[0].dims)
     mean_da.attrs['_FillValue'] = np.nan

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