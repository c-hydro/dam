import numpy as np
import geopandas as gpd
import rasterio

from typing import Optional
import os

from ..utils.io_geotiff import read_geotiff, write_geotiff
from ..utils.rm import remove_file

def apply_scale_factor(input: str,
                       scale_factor: float,
                       nodata_value: float = np.nan,
                       output: Optional[str] = None,
                       rm_input: bool = False,
                       destination: Optional[str] = None # destination is kept for backward compatibility
                       ) -> str:
    """
    Applies a scale factor to a raster.
    """

    if output is None:
        if destination is not None:
            output = destination
        else:
            output = input.replace('.tif', '_scaled.tif')

    data = read_geotiff(input, out = 'xarray')
    current_nodata = data.rio.nodata
    metadata = data.attrs

    # apply the scale factor
    data = data * scale_factor

    # replace the current nodata value (which was scaled) with the new one
    if current_nodata is not None:
        rescaled_nodata = current_nodata * scale_factor
        data = data.where(data != rescaled_nodata, other = nodata_value)

    # data = data.rio.write_nodata(nodata_value)
    # data.attrs.update(metadata)
    write_geotiff(data, output, metadata = metadata, nodata_value = nodata_value)

    if rm_input:
        remove_file(input)
    
    return output

def summarise_by_shape(input: str,
                       shapes: str,
                       statistic: str = 'mean',
                       breaks: Optional[list[float]] = None,
                       name: Optional[str] = None,
                       nodata_value: float = np.nan,
                       output: Optional[str] = None,
                       rm_input: bool = False,
                       rm_shapes: bool = False
                       ) -> str:
    """
    Summarise a raster by a shapefile.
    """

    if output is None:
        output = input.replace('.tif', f'_{statistic}.csv')

    # Open the shapefile
    gdf = gpd.read_file(shapes)

    # Open the raster file
    with rasterio.open(input) as src:
        # Initialize an empty list to store the statistics
        stats = []

        # Loop over each geometry in the GeoDataFrame
        for geom in gdf.geometry:
            # Mask the raster with the current geometry
            out_image, out_transform = rasterio.mask.mask(src,
                                                          [geom],
                                                          crop=True,
                                                          all_touched=True,
                                                          nodata=nodata_value)
            
            #TODO: use the portions of the pixels inside the geometry to weight the statistics

            # we only care about the data, not the shape
            out_data = out_image.flatten()

            # remove the nodata values
            out_data = out_data[~np.isclose(out_data, nodata_value, equal_nan=True)]

            if len(out_data) == 0:
                stats.append(nodata_value)
                continue

            # if we want the mode, we assume that the data is either integer or should be classified
            if statistic == 'mode':
                if breaks is not None:
                    out_data = np.digitize(out_data, breaks)
                    
                # get the most frequent value
                stat = np.bincount(out_data).argmax()
            
            elif statistic == 'mean':
                stat = np.mean(out_data)

            # Append the statistic to the list
            stats.append(stat)


    # set the name of the new field
    if name is None:
        name = statistic + '_' + os.path.basename(input).split('.')[0]

    # option 1: add the statistics to the GeoDataFrame and save as a shapefile
    # note that the legend will be the same as the breaks, but it won't show in the file.
    if output.endswith('.shp'):
        gdf[name] = stats
        gdf.to_file(output)

    # option 2: save the statistics as a csv
    elif output.endswith('.csv'):
        # remove the geometry column
        gdf = gdf.drop(columns='geometry')

        gdf[name] = stats

        # create a legend from the breaks
        if breaks is not None:
            legend = ["-inf : " + str(breaks[0])]
            legend.extend([f'{breaks[i]} : {breaks[i+1]}' for i in range(len(breaks)-1)])
            legend.append(str(breaks[-1]) + ' : inf')
            
            stats_legend = []
            for i in range(len(stats)):
                if np.isnan(stats[i]) or stats[i] == nodata_value:
                    stats_legend.append('nodata')
                else:
                    stats_legend.append(legend[int(stats[i])])

            gdf[name + '_legend'] = stats_legend
        gdf.to_csv(output)

    if rm_input:
        remove_file(input)
    
    if rm_shapes:
        remove_file(shapes)

    return output

def combine_raster_data(input: list[str],
                        statistic: str = 'mean',
                        weights: Optional[list[float]] = None,
                        nodata_value: float = np.nan,
                        na_ignore: bool = False,
                        output: Optional[str] = None,
                        rm_input: bool = False,
                        ) -> str:
    """
    Combine multiple rasters into a single raster.
    """

    if output is None:
        output = input[0].replace('.tif', f'_{statistic}.tif')

    # check that the weitghts are the correct length
    if weights is None:
        weights = [1] * len(input)
    elif len(weights) != len(input):
        raise ValueError('The number of weights must be the same as the number of rasters.')

    match statistic:
        case 'sum':
            weights = weights
        case 'mean':
            weights = [w / sum(weights) for w in weights]
        case _:
            raise ValueError('The statistic must be either "sum" or "mean".')

    databrick = np.stack([read_geotiff(i, out = 'array') for i in input])
    if na_ignore:
        databrick_nan = np.where(np.isclose(databrick, nodata_value, equal_nan=True), 0,      databrick)
    else:
        databrick_nan = np.where(np.isclose(databrick, nodata_value, equal_nan=True), np.nan, databrick)

    weighted_data = np.einsum('i,ijk->ijk', weights, databrick_nan)
    mask = np.isfinite(weighted_data)
    mask = np.isfinite(weighted_data) # <- this is a mask of all the nans in the weighted data
    weights = np.array(weights)
    weights_3d = np.broadcast_to(weights[:, np.newaxis, np.newaxis], weighted_data.shape)
    selected_weights = np.where(mask, weights_3d, 0)
    total_weights = np.sum(selected_weights, axis=0)

    weighted_sum = np.nansum(weighted_data, axis = 0)

    if statistic == 'sum':
        result = weighted_sum
    elif statistic == 'mean':
        total_weights = np.where(total_weights < 1e-3, 1e-3, total_weights)
        weighted_mean = weighted_sum / total_weights
        weighted_mean = np.where(total_weights == 1e-3, np.nan, weighted_mean)
        result = weighted_mean

    result = np.where(np.isnan(result), nodata_value, result)

    write_geotiff(result, output, template = input[0], nodata_value = nodata_value)

    if rm_input:
        for i in input:
            remove_file(i)

    return output
