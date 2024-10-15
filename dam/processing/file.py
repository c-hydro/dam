from ..utils.register_process import as_DAM_process
import xarray as xr

@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def copy(input: xr.DataArray) -> xr.DataArray:
    """
    Copy the input data.
    """
    return input.copy()