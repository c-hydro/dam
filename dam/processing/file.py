from ..utils.register_process import as_DAM_process
import xarray as xr

@as_DAM_process()
def copy(input):
    """
    Copy the input data.
    """
    return input.copy()