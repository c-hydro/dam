import os
import numpy as np
import rioxarray as rxr
import xarray as xr

from osgeo import gdal, gdalconst

def read_geotiff_asGDAL(filename):
    ds = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return ds

def read_geotiff_asXarray(filename):
    return rxr.open_rasterio(filename)

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
    filehandle = gdal.Open(filename)
    band1 = filehandle.GetRasterBand(1)
    band1data = band1.ReadAsArray()
    filehandle = None
    return band1data

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