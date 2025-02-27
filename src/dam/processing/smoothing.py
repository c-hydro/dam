from astropy.convolution import convolve, Gaussian2DKernel
from typing import Optional
import numpy as np
import xarray as xr
import warnings

from ..utils.register_process import as_DAM_process

# -------------------------------------------------------------------------------------
# Method to apply gaussian smoothing to a raster map
@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def gaussian_smoothing(input: xr.DataArray,
                       stddev_kernel: float = 2,
                       nodata_value: Optional[float|int] = None,
                       ) -> xr.DataArray:
    """
    Applies gaussian smoothing to a raster map. The input map is convolved with a 2D gaussian kernel. The standard
    deviation of the kernel can be specified with the parameter stddev_kernel. The output is written to a new file. The
    input file can be removed with the parameter rm_input.
    """

    if stddev_kernel == 0:
        return input

    # read map
    data = input

    if nodata_value is None:
        nodata_value = data.attrs.get('_FillValue')

    kernel = Gaussian2DKernel(x_stddev=stddev_kernel)
    # execute smoothing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.squeeze(data)
        original_shape = data.shape
        smooth_data = convolve(data, kernel, nan_treatment = 'interpolate', preserve_nan = True)
        
    # write output
    reshaped = np.reshape(smooth_data, original_shape)
    data.values = reshaped
    data = data.rio.write_nodata(nodata_value)

    return data

# -------------------------------------------------------------------------------------