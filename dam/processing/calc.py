import numpy as np
import rioxarray

from typing import Optional

from ..utils.io_geotiff import read_geotiff, write_geotiff
from ..utils.rm import remove_file

def apply_scale_factor(input: str,
                       scale_factor: float,
                       nodata_value: float = np.nan,
                       output: Optional[str] = None,
                       rm_input: bool = False,
                       destination: Optional[str] = None # destination is kept for backward compatibility
                       ) -> str:
    """
    Applies a scale factor to a raster.
    """

    if output is None:
        if destination is not None:
            output = destination
        else:
            output = input.replace('.tif', '_scaled.tif')

    data = read_geotiff(input, out = 'xarray')
    current_nodata = data.rio.nodata
    metadata = data.attrs

    # apply the scale factor
    data = data * scale_factor

    # replace the current nodata value (which was scaled) with the new one
    if current_nodata is not None:
        rescaled_nodata = current_nodata * scale_factor
        data = data.where(data != rescaled_nodata, other = nodata_value)

    data = data.rio.write_nodata(nodata_value)
    data.attrs.update(metadata)
    write_geotiff(data, output)

    if rm_input:
        remove_file(input)
    
    return output