import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio

from typing import Optional
import os

from ..utils.io_geotiff import read_geotiff, write_geotiff
from ..utils.io_csv import read_csv, save_csv
from ..utils.geo_utils import ltln2val_from_2dDataArray
from ..utils.rm import remove_file

def apply_scale_factor(input: xr.DataArray,
                       scale_factor: Optional[float] = None,
                       nodata_value: float = np.nan,
                       ) -> xr.DataArray:
    """
    Applies a scale factor to a raster.
    """

    data = input

    scale_factor_metadata = data.attrs.get('scale_factor')
    if scale_factor_metadata is not None:
        try:
            scale_factor_metadata = float(scale_factor_metadata)
        except ValueError:
            scale_factor_metadata = None
    
    if scale_factor is None:
        if scale_factor_metadata is None:
            raise ValueError("No scale factor provided or found in metadata")
        else:
            scale_factor = scale_factor_metadata
            data.attrs.pop('scale_factor')
    else:
        if scale_factor_metadata is not None:
            if scale_factor_metadata == scale_factor:
                data.attrs.pop('scale_factor')
            else:
                scale_factor_metadata = scale_factor / scale_factor_metadata
                data.attrs['scale_factor'] = str(scale_factor_metadata)
               
    current_nodata = data.attrs.get('_FillValue')

    # apply the scale factor
    data = data.copy(data = data.values * scale_factor)

    # replace the current nodata value (which was scaled) with the new one
    if current_nodata is not None:
        rescaled_nodata = current_nodata * scale_factor
        data = data.where(data != rescaled_nodata, other = nodata_value)

    return data

def summarise_by_shape(input: str,
                       shapes: str,
                       statistic: str = 'mean',
                       breaks: Optional[list[float]] = None,
                       percentages: Optional[bool] = False,
                       threshold: Optional[int] = None,
                       round: Optional[int] = None,
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
        if breaks is not None and percentages:
            # Initialize an empty list to store the percentages
            df_percentages = []

            if threshold is not None:
                thresholds = []

        # Loop over each geometry in the GeoDataFrame
        for geom in gdf.geometry:
            # Mask the raster with the current geometry
            out_image, out_transform = rasterio.mask.mask(src,
                                                          [geom],
                                                          crop=True,
                                                          all_touched=True,
                                                          nodata=nodata_value)

            # we only care about the data, not the shape
            out_data = out_image.flatten()

            # remove the nodata values
            out_data = out_data[~np.isclose(out_data, nodata_value, equal_nan=True)]

            if len(out_data) == 0:
                stats.append(nodata_value)
                if breaks is not None and percentages:
                    percentages.append([nodata_value] * (len(breaks) + 1))
                continue

            # if we want the mode, we assume that the data is either integer or should be classified
            if statistic == 'mode':
                if breaks is not None:
                    out_data = np.digitize(out_data, breaks)

                # get the most frequent value
                stat = np.bincount(out_data).argmax()

            elif statistic == 'mean':
                stat = np.mean(out_data)

            elif statistic == 'median':
                stat = np.median(out_data)

                if breaks is not None:
                    stat = np.digitize(stat, breaks)

            else:
                raise ValueError('The statistic must be either "mean", "median" or "mode".')

            if round is not None:
                stat = np.round(stat, round)

            # Append the statistic to the list
            stats.append(stat)

            if breaks is not None and percentages:
                # Get the histogram of the values
                hist, _ = np.histogram(out_data, bins=[-np.inf] + breaks + [np.inf])

                # Turn the histogram into percentages of the total rounded to 0 decimals
                hist_percentages = np.round(hist / hist.sum() * 100, 0)

                if threshold is not None:
                     # Start from the highest class and move to the lower classes
                    total_percentage = 0
                    for i in range(len(hist_percentages) - 1, -1, -1):
                        total_percentage += hist_percentages[i]
                        if total_percentage > threshold:
                            # Record the current class and break the loop
                            thresholds.append(i)
                            break

                # Add the histogram percentages to the list
                df_percentages.append(hist_percentages)

    # set the name of the new field
    if name is None:
        name = statistic + '_' + os.path.basename(input).split('.')[0]

    # option 1: add the statistics to the GeoDataFrame and save as a shapefile
    # note that the legend will be the same as the breaks, but it won't show in the file.
    if output.endswith('.shp'):
        if percentages:
            # turn the list of lists into a numpy array
            df_percentages = np.array(df_percentages)
            for i in range(len(breaks) + 1):
                gdf[f'class_{i}'] = df_percentages[:, i]

            if threshold is not None:
                gdf[name] = thresholds
        else:
            if threshold is not None:
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

        if breaks is not None and percentages:
            # turn the list of lists into a numpy array
            df_percentages = np.array(df_percentages)
            for i in range(len(breaks) + 1):
                gdf[f'class_{i}'] = df_percentages[:, i]

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

    weighted_sum = np.sum(weighted_data, axis = 0)

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

# -------------------------------------------------------------------------------------
# Method to extract residuals from xr.DataArray based on lat-lon and values
def compute_residuals(input: list[str],
                      name_lat_lon_data_csv: list[str],
                      method: Optional[str] = 'nearest',
                      output: Optional[str] = None,
                      method_residuals: Optional[str] = 'data_minus_map',
                      rm_input: bool = False):
    """
    Compute residuals between data and map. The input is a csv file with columns for latitude, longitude, and data.
    Note that the csv file may have any column, but also needs to have latitude, longitude, and data.
    Names of this csv file can change, but the ORDER of those columns in name_lat_lon_data_in must be the same.
    Another input is a raster map.
    Inputs are given as a LIST of two elements: [input_data, input_map].
    The residuals are computed with a given method, whether as data minus map or map minus data.
    The residuals are saved to a new csv file.
    """

    input_data = input[0]
    input_map = input[1]

    if output is None:
        output = input_data.replace('.csv', '_residuals.csv')

    # load data
    data = read_csv(input_data)
    lat_points = data[name_lat_lon_data_csv[0]].to_numpy()
    lon_points = data[name_lat_lon_data_csv[1]].to_numpy()
    data_points = data[name_lat_lon_data_csv[2]].to_numpy()

    # load map
    map = read_geotiff(input_map)
    map = map.squeeze()  # remove single dimensions, usually time

    # extract values in map based on lat and lon
    values_map = ltln2val_from_2dDataArray(input_map=map, lat=lat_points, lon=lon_points, method=method)

    # compute residuals
    if method_residuals == 'data_minus_map':
        residuals = data_points - values_map    # data minus map
    elif method_residuals == 'map_minus_data':
        residuals = values_map - data_points
    else:
        raise NotImplementedError('Method method_residuals not implemented')

    # create new dataframe from data and replace data with residuals
    data = data.drop(columns=['data'])
    data['data'] = residuals
    data.set_index(data.columns[0], inplace=True)
    save_csv(data, output)

    if rm_input:
        remove_file(input)

    return output
# -------------------------------------------------------------------------------------