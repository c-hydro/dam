from astropy.convolution import convolve, Gaussian2DKernel
from typing import Optional
import numpy as np
from dam.utils.io_geotiff import read_geotiff, write_geotiff
from dam.utils.rm import remove_file

# -------------------------------------------------------------------------------------
# Method to apply gaussian smoothing to a raster map
def gaussian_smoothing(input: str,
                      stddev_kernel: float = 2,
                      output: Optional[str] = None,
                      nodata_value: float = np.nan,
                      rm_input: bool = False,
                      ) -> str:
    """
    Applies gaussian smoothing to a raster map. The input map is convolved with a 2D gaussian kernel. The standard
    deviation of the kernel can be specified with the parameter stddev_kernel. The output is written to a new file. The
    input file can be removed with the parameter rm_input.
    """

    if output is None:
        output = input.replace('.tif', '_smoothed.tif')

    # read map
    data = read_geotiff(input, out='xarray')
    data = np.squeeze(data)

    # execute smoothing
    kernel = Gaussian2DKernel(x_stddev=stddev_kernel)
    data.values = convolve(data.values, kernel)

    # write output
    data = data.rio.write_nodata(nodata_value)
    write_geotiff(data, output)

    if rm_input:
        remove_file(input)

    return output

# -------------------------------------------------------------------------------------