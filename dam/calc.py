import numpy as np
import rioxarray

from typing import Optional

from .utils.io_geotiff import read_geotiff_asXarray, write_geotiff_fromXarray
from .utils.rm import remove_file

def apply_scale_factor(input: str,
                       scale_factor: float,
                       nodata_value: float = np.nan,
                       destination: Optional[str] = None,
                       rm_input: bool = False) -> str:
    """
    Applies a scale factor to a raster.
    """

    if destination is None:
        destination = input.replace('.tif', '_scaled.tif')

    data = read_geotiff_asXarray(input)
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
    write_geotiff_fromXarray(data, destination)

    if rm_input:
        remove_file(input)
    
    return destination