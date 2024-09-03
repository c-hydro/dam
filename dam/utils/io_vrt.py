# -------------------------------------------------------------------------------------
# Method to create vrt ancillary file
def create_point_vrt(file_name_vrt, file_name_csv, var_name_layer):

    var_name_data = 'values'
    var_name_geox = 'x'
    var_name_geoy = 'y'

    with open(file_name_vrt, 'w') as file_handle:
        file_handle.write('<OGRVRTDataSource>\n')
        file_handle.write('    <OGRVRTLayer name="' + var_name_layer + '">\n')
        file_handle.write('        <SrcDataSource>' + file_name_csv + '</SrcDataSource>\n')
        file_handle.write('    <GeometryType>wkbPoint</GeometryType>\n')
        file_handle.write('    <LayerSRS>WGS84</LayerSRS>\n')
        file_handle.write(
            '    <GeometryField encoding="PointFromColumns" x="' +
            var_name_geox + '" y="' + var_name_geoy + '" z="' + var_name_data + '"/>\n')
        file_handle.write('    </OGRVRTLayer>\n')
        file_handle.write('</OGRVRTDataSource>\n')
        file_handle.close()

# -------------------------------------------------------------------------------------