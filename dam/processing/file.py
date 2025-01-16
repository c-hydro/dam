from ..utils.register_process import as_DAM_process

@as_DAM_process()
def copy(input):
    """
    Copy the input data.
    """
    return input.copy()