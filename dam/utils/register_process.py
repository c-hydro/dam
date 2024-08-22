import xarray as xr
import rioxarray as rxr
from osgeo import gdal
import tempfile
import os

from typing import Iterable

global DAM_PROCESSES
DAM_PROCESSES = {}

def as_DAM_process(input_type: str = 'xarray', output_type: str = 'xarray', **kwargs):
    def decorator(func):
        def wrapper(input, *args, **kwargs):
            # Ensure the input is in the correct format
            if input_type == 'gdal':
                # convert xr.DataArray to gdal.Dataset
                input = xarray_to_gdal(input)
            elif input_type == 'file':
                # convert filename to xr.DataArray
                input = xarray_to_file(input)

            # Call the original function
            result = func(input, *args, **kwargs)

            #remove the teporary file if input is a file
            if input_type == 'file':
                remove(input)

            # Ensure the output is in the correct format
            if output_type == 'gdal':
                # Add your output validation logic here
                result = gdal_to_xarray(result)
            elif output_type == 'file':
                # Add your output validation logic here
                result = file_to_xarray(result)
            return result
        
        wrapper.__name__ = func.__name__
        for key, value in kwargs.items():
            setattr(wrapper, key, value)
        # Add the wrapped function to the global list of processes
        DAM_PROCESSES[func.__name__] = {'function': wrapper}

        return wrapper
    return decorator

def with_list_input(func):
    def wrapper(input, *args, **kwargs):
        if isinstance(input, Iterable):
            return [func(i, *args, **kwargs) for i in input]
        else:
            return func(input, *args, **kwargs)
    return wrapper

@with_list_input
def remove(filename: str):
    os.remove(filename)

@with_list_input
def xarray_to_file(data_array: xr.DataArray) -> str:
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_file.close()
    
    # Save the DataArray to the temporary file
    data_array.rio.to_raster(temp_file.name, compress='LZW')
    
    # Move the temporary file to the desired filename
    return temp_file.name

@with_list_input
def file_to_xarray(filename: str) -> xr.DataArray:
    # Open the file with xarray
    return rxr.open_rasterio(filename)

@with_list_input
def xarray_to_gdal(data_array: xr.DataArray) -> gdal.Dataset:
    temp_file = xarray_to_file(data_array)
    
    # Open the temporary file with GDAL
    gdal_dataset = gdal.Open(temp_file)
    
    # Optionally, delete the temporary file after opening it with GDAL
    os.remove(temp_file)
    
    return gdal_dataset

@with_list_input
def gdal_to_xarray(dataset: gdal.Dataset) -> xr.DataArray:
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_file.close()
    
    # Save the Dataset to the temporary file
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(temp_file.name, dataset, options=['COMPRESS=LZW'])

    # Open the temporary file with xarray
    data_array = rxr.open_rasterio(temp_file.name)

    # Optionally, delete the temporary file after opening it with GDAL
    os.remove(temp_file.name)

    return data_array