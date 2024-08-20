from dam import tile, calc

shape = '/home/luca/Documents/CIMA_code/tests/Volta_SHP/VOLTA_ADM2_CDI.shp'
input = '/home/luca/Documents/CIMA_code/tests/Volta_SHP/CDI_20240420.tif'

# summarise the mosaic by the shape
output = calc.summarise_by_shape(input, shape, statistic='mode', nodata_value=8)
