from ..utils.register_agg import as_agg_function

from d3tools.timestepping import TimeRange

import xarray as xr
import numpy as np
from typing import Sequence

@as_agg_function()
def sum(input: list[xr.DataArray|np.ndarray],
        input_agg: list[TimeRange],
        this_agg: TimeRange,
        na_ignore = False,
        nan_threshold = 0.5,
        **kwargs) -> xr.DataArray|np.ndarray:
     
     weights       = np.array(calc_overlap(input_agg, this_agg)) # this looks how much of each input overlaps with the current aggregation (opposite of the mean)
     all_inputs    = np.stack(input, axis = 0)

     # multiply each data in the brick by the weight, b is the band of the raster (which is always 1)
     weighted_data = np.einsum('i,ibjk->ibjk', weights, all_inputs)
     
     # sum the weighted data
     if na_ignore:  

          # calculate the weighted sum
          weighted_sum = np.nansum(weighted_data, axis = 0)

          ## we do all this to fill-in the nans
          # get a mask of all the nans in the weighted data
          mask = np.isfinite(weighted_data) # <- 
          # calculte the weights for the mean (i.e. percentage of the final aggregation in each input)
          mean_weights = np.array(calc_overlap(this_agg, input_agg))
          # broadcast the mean weights to the shape of the weighted data
          weights_3d = np.broadcast_to(mean_weights[:, np.newaxis, np.newaxis, np.newaxis], weighted_data.shape)
          # apply the mask to the weights to set the weights to 0 where the data is nan
          selected_weights = np.where(mask, weights_3d, 0)
          # get the total weights for each pixel
          total_weights = np.sum(selected_weights, axis=0)

          # to avoid division by zero, we set the zero weights to nan right away
          weighted_sum  = np.where(total_weights < 1e-6, np.nan, weighted_sum)
          total_weights = np.where(total_weights < 1e-6, 1e-6, total_weights)

          # get the missing weights (i.e. percentage of the final aggregation that is not in any input)
          missing_weights = (np.sum(mean_weights) - total_weights) / np.sum(mean_weights)

          # add the component that is not in any input, by multiplying the weighted sum / total_weights (i.e. the mean) by the missing weights
          weighted_sum += weighted_sum / total_weights * missing_weights

          # Remove where the total sum of weights is at least (nan_threshold x 100)% of the total weights
          weighted_sum = np.where(total_weights/np.sum(mean_weights) <= (1-nan_threshold), np.nan, weighted_sum)
     else:
          weighted_sum = np.sum(weighted_data, axis = 0)

     if isinstance(input[0], np.ndarray):
          return weighted_sum

     sum_da = xr.DataArray(weighted_sum, coords = input[0].coords, dims = input[0].dims)
     sum_da.attrs['_FillValue'] = np.nan
     
     return sum_da

@as_agg_function()
def mean(input: list[xr.DataArray|np.ndarray],
         input_agg: list[TimeRange],
         this_agg: TimeRange,
         na_ignore = False,
         nan_threshold = 0.5,
         **kwargs) -> xr.DataArray|np.ndarray:

     weights       = np.array(calc_overlap(this_agg, input_agg)) # this looks how much of the current aggregation overlaps with each inputs (opposite of the sum)
     all_inputs    = np.stack(input, axis = 0)

     # multiply each data in the brick by the weight, b is the band of the raster (which is always 1)
     weighted_data = np.einsum('i,ibjk->ibjk', weights, all_inputs)
     # sum all the weighted data
     if na_ignore:    
          ## protecting nans
          # get a mask of all the nans in the weighted data
          mask = np.isfinite(weighted_data)
          # broadcast the weights to the shape of the weighted data
          weights_3d = np.broadcast_to(weights[:, np.newaxis, np.newaxis, np.newaxis], weighted_data.shape)
          # apply the mask to the weights to set the weights to 0 where the data is nan
          selected_weights = np.where(mask, weights_3d, 0)
          # get the total weights for each pixel
          total_weights = np.sum(selected_weights, axis=0)

          weighted_sum = np.nansum(weighted_data, axis = 0)

          # Remove where the total sum of weights is at least (nan_threshold x 100)% of the total weights
          weighted_sum = np.where(total_weights/np.sum(weights) <= (1-nan_threshold), np.nan, weighted_sum)
     else:
          weighted_sum  = np.sum(weighted_data, axis = 0)
          total_weights = np.full_like(weighted_sum, np.sum(weights))
     
     # divide by the total weights
     # to avoid division by zero, we set the zero weights to nan right away
     weighted_sum  = np.where(total_weights < 1e-6, np.nan, weighted_sum)
     total_weights = np.where(total_weights < 1e-6, 1e-6, total_weights)
     weighted_mean = weighted_sum / total_weights
     
     if isinstance(input[0], np.ndarray):
          return weighted_mean

     mean_da = xr.DataArray(weighted_mean, coords = input[0].coords, dims = input[0].dims)
     mean_da.attrs['_FillValue'] = np.nan

     return mean_da

def calc_overlap(this_tr: TimeRange, other_tr: TimeRange|list[TimeRange]) -> TimeRange:
     if isinstance(other_tr, Sequence):
          return [calc_overlap(this_tr, tr) for tr in other_tr]
     elif isinstance(this_tr, Sequence):
          return [calc_overlap(tr, other_tr) for tr in this_tr]
     
     if this_tr.start > other_tr.end or this_tr.end < other_tr.start:
          return 0

     start = max(this_tr.start, other_tr.start)
     end = min(this_tr.end, other_tr.end)
     size = TimeRange(start, end).length()
     original_size = this_tr.length()
     if size < 1:
          size = TimeRange(start, end).length(unit = 'hours')
          original_size = this_tr.length(unit = 'hours')

     return size / original_size