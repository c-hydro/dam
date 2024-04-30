import os
import numpy as np
import rioxarray as rxr
import xarray as xr

from osgeo import gdal, gdalconst

def read_geotiff_asGDAL(filename):
    ds = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return ds

def read_geotiff_asXarray(filename):
    data = rxr.open_rasterio(filename)

    # ensure that the data has descending latitudes
    y_dim = data.rio.y_dim
    if y_dim is None:
        for dim in data.dims:
            if 'lat' in dim.lower() | 'y' in dim.lower():
                y_dim = dim
                break
    if data[y_dim][0] < data[y_dim][-1]:
        data = data.sortby(y_dim, ascending = False)

    return data

def read_multiple_geotiffs_asXarray(tiff_file_paths):
    data_arrays = [rxr.open_rasterio(file_path) for file_path in tiff_file_paths]
    stacked_data = xr.concat(data_arrays, dim='band')
    return stacked_data

def write_geotiff_fromXarray(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data.rio.to_raster(filename, compress='LZW')

def write_geotiff_fromGDAL(ds, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gdal.Translate(filename, ds, creationOptions=['COMPRESS=LZW'])

def read_geotiff_as_array(filename):
    data = read_geotiff_asXarray(filename)
    values = data.values.squeeze()
    return values

def write_geotiff_singleband(filename,geotransform,geoprojection,data,metadata = None,nodata_value = np.nan):
    (x,y) = data.shape
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_datatype = gdal.GDT_Float32

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dst_ds = driver.Create(filename,y,x,1,dst_datatype)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    if metadata:
        dst_ds.SetMetadata(metadata)
    dst_ds = None