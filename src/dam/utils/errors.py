class GDAL_ImportError(ImportError):

    message = """
    The package 'gdal' is not installed. Please install 'gdal' to use this function.
    To install gdal:
    1. ensure gdal and libgdal-dev are installed on your system (version 3.6.0 or higher)
    2. find the version of gdal that is running your system: `gdalinfo --version`
    3. install the corresponding version of gdal: `pip install gdal==<version>`
    """

    def __init__(self, message=None):
        if message is None:
            message = self.message
        super().__init__(message)