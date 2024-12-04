from setuptools import setup, find_packages
from subprocess import check_output

# get the version of gdal-config to use as a requirement (from bash)
gdalconfig_version = check_output('gdal-config --version', shell=True).decode('utf-8').strip()

setup(
    name='dam',
    version='1.1.4',
    packages=find_packages(),
    description='A package for raster data processing developed at the CIMA Research Foundation',
    author='Luca Trotter',
    author_email='luca.trotter@cimafoundation.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    keywords='meteorological data, satellite data, climatological data, environmental data, raster data,\
        xarray, netcdf, grib, hdf-eos, hdf-eos5, geotiff',
    install_requires=[
        f'gdal[numpy]=={gdalconfig_version}',
        'numpy>=1.24.0',
        'xarray>=2023.9.0',
        'rioxarray>=0.7.1',
        'unpackqa>=0.1.0',
        'geopandas',
        'pandas',
        'astropy>=6.1.1',
        'dask',
        'boto3',
        'paramiko',
        'scipy'
    ],
    python_requires='>=3.10',
    test_suite='tests',
)