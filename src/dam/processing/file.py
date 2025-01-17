from ..utils.register_process import as_DAM_process
import geopandas as gdp

@as_DAM_process()
def copy(input):
    """
    Copy the input data.
    """
    return input.copy()

@as_DAM_process(input_type='shape', output_type='table')
def extract_table(input: gdp.GeoDataFrame, cols_to_keep = None):
    """
    Extract the table from the shapefile.
    """

    input = input.drop(columns='geometry')

    if cols_to_keep is not None:
        input = input[cols_to_keep]
    return input