# Installation

When installing DAM, there might be issues with the package gdal and its gdal_array functionality. This happens if pip trys to cache a "bad" version of gdal instead of creating new wheels with the version required for this package (see [here](https://stackoverflow.com/questions/75372275/importerror-cannot-import-name-gdal-array-from-osgeo)). This issue really only comes into play when using the warp.match_grid function with nodata_threshold != None at this point, but in the future, it might become a bigger issue.

To solve it. Either install gdal separately _before_ installing DAM, ensure its wheels are build and not taken from cache.

```bash
# Create a new virtual environment
venv_dir=".venv"
python3 -m venv $venv_dir

# Activate the virtual environment
source $venv_dir/bin/activate

# Install the version of GDAL's python bindings matching the one on your machine (must be >=3.4.3)
gdalv=$(gdal-config --version)
pip install --no-cache-dir --force-reinstall "GDAL[numpy]==$gdalv"

# Install DAM
pip install https://github.com/c-hydro/dam/archive/dev.zip
```

Otherwise, pass the same options directly to the DAM installation. This will, however apply the settings to the installation of every required package and be therefore slightly slower.

```bash
# Create a new virtual environment
venv_dir=".venv"
python3 -m venv $venv_dir

# Activate the virtual environment
source $venv_dir/bin/activate

# Install DAM
pip install --no-cache-dir --force-reinstall https://github.com/c-hydro/dam/archive/dev.zip
```
