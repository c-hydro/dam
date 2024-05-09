import pytest
import os

import numpy as np
import xarray as xr
import rioxarray
from osgeo import gdal, gdalconst
from dam.utils.io_geotiff import read_geotiff, write_geotiff

class TestIOGeotiff:
    def setup_method(self):
        self.data_array  = np.random.rand(1, 10, 10)
        self.data_xarray = xr.DataArray(self.data_array, coords={'band': [0], 'y' : np.arange(10), 'x' : np.arange(10)})
        self.data_xarray = self.data_xarray.rio.write_crs('EPSG:4326')

        self.test_raster = 'tests/test_raster.tif'
        self.data_xarray.rio.to_raster(self.test_raster)

    def test_read_geotriff_as_xarray(self):
        data = read_geotiff(self.test_raster, out='xarray')
        assert isinstance(data, xr.DataArray)
        assert data.shape == self.data_xarray.shape
        assert data.dims == self.data_xarray.dims
        
    def test_read_geotiff_as_gdal(self):
        data = read_geotiff(self.test_raster, out='gdal')
        assert isinstance(data, gdal.Dataset)

    def test_read_geotiff_as_array(self):
        data = read_geotiff(self.test_raster, out='array')
        assert isinstance(data, np.ndarray)
        # due to the fact that we ensure that the data has descending latitudes, we need to flip the array vertically
        assert np.allclose( np.flipud(data), self.data_array)

    def test_write_geotiff_from_xarray(self):
        test_file = self.test_raster.replace('.tif', '_xr.tif')
        test_data = self.data_xarray
        write_geotiff(test_data, test_file)
        assert os.path.exists(test_file)
        # as a quick check, we read the file back and compare it to the original data
        data = read_geotiff(test_file, out='array')
        assert np.allclose(np.flipud(data), self.data_array)
        os.system(f'rm {test_file}')

    def test_write_geotiff_from_gdal(self):
        test_file = self.test_raster.replace('.tif', '_gdal.tif')
        test_data = gdal.Open(self.test_raster, gdalconst.GA_ReadOnly)
        write_geotiff(test_data, test_file)
        assert os.path.exists(test_file)
        # as a quick check, we read the file back and compare it to the original data
        data = read_geotiff(test_file, out='array')
        assert np.allclose(np.flipud(data), self.data_array)
        os.system(f'rm {test_file}')

    def test_write_geotiff_from_array(self):
        test_file = self.test_raster.replace('.tif', '_np.tif')
        test_data = np.flipud(self.data_array)
        print(test_data)
        write_geotiff(test_data, test_file, template=self.test_raster)
        assert os.path.exists(test_file)
        # as a quick check, we read the file back and compare it to the original data
        data = read_geotiff(test_file, out='array')
        print(data)
        assert np.allclose(data, self.data_array)
        os.system(f'rm {test_file}')

    def teardown_method(self):
        os.system(f'rm {self.test_raster}')