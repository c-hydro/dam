from astropy.convolution import convolve, Gaussian2DKernel
from typing import Optional
import numpy as np
import xarray as xr

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

    # read map
    data = input

    if nodata_value is None:
        nodata_value = data.attrs.get('_FillValue')

    # execute smoothing
    data = np.squeeze(data)
    kernel = Gaussian2DKernel(x_stddev=stddev_kernel)
    data.values = convolve(data.values, kernel)

    # write output
    data = data.rio.write_nodata(nodata_value)
    return data

# -------------------------------------------------------------------------------------