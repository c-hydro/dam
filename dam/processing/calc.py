import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd

from typing import Optional

from ..utils.geo_utils import ltln2val_from_2dDataArray
from ..utils.register_process import as_DAM_process

@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
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
    current_nodata = data.attrs.get('_FillValue')

    # apply the scale factor
    data = data.copy(data = data.values * scale_factor)
    data = data.astype(np.float32)

    # replace the current nodata value (which was scaled) with the new one
    if current_nodata is not None:
        rescaled_nodata = current_nodata * scale_factor
        data = data.where(~np.isclose(data, rescaled_nodata, equal_nan=True), other = np.nan)
        data.attrs['_FillValue'] = nodata_value

    return data

@as_DAM_process(input_type = 'xarray', output_type = 'shape')
def summarise_by_shape(input: xr.DataArray,
                       shapes: gpd.GeoDataFrame,
                       statistic: str = 'mean',
                       thr_quantile: Optional[float] = None,
                       thr_value: Optional[float] = None,
                       thr_side: str = 'above',
                       all_touched: bool = False,
                       column_name: Optional[str] = None,
                       ) -> gpd.GeoDataFrame:
    """
    Summarise a raster by a shapefile.
    """

    # get no_data value
    nodata_value = input.attrs.get('_FillValue', np.nan)

    # Loop over each geometry in the GeoDataFrame
    for geom in shapes.geometry:
        # Mask the raster with the current geometry
        out_image = input.rio.clip([geom],
                                   all_touched=all_touched)

        # we only care about the data, not the shape
        out_data = out_image.values.flatten()

        # check if all values are nodata
        if np.all(np.isclose(out_data, nodata_value, equal_nan=True)):
            # if statistic is mode set to 0 else set to nan
            if statistic == 'mode':
                stat = 0
            else:
                stat = np.nan
        else:
            # remove the nodata values
            out_data = out_data[~np.isclose(out_data, nodata_value, equal_nan=True)]

            if thr_quantile is not None:
                # Sort the out_data array
                sorted_data = np.sort(out_data)
                # Calculate the index for the quantile threshold
                quantile_position = int(thr_quantile * len(sorted_data))
                if thr_side.lower() == 'above':
                    # filter the data above the threshold
                    data = sorted_data[quantile_position:]
                elif thr_side.lower() == 'below':
                    # filter the data below the threshold
                    data = sorted_data[:quantile_position]
                else:
                    raise ValueError('The threshold side must be either "above" or "below".')
            elif thr_value is not None:
                if thr_side.lower() == 'above':
                    # filter the data above the threshold
                    data = out_data[out_data > thr_value]
                elif thr_side.lower() == 'below':
                    # filter the data below the threshold
                    data = out_data[out_data < thr_value]
                else:
                    raise ValueError('The threshold side must be either "above" or "below".')
            else:
                data = out_data

            if statistic == 'mean':
                stat = np.mean(data)
            elif statistic == 'median':
                stat = np.median(data)
            elif statistic == 'mode':
                # check if data values are integers
                if np.all(np.equal(np.mod(data, 1), 0)):
                    stat = np.bincount(data.astype(int)).argmax()
                else:
                    raise ValueError('The "mode" statistic can only be calculated for integer data.')
            elif statistic == 'sum':
                stat = np.sum(data)
            elif statistic == 'quantile':
                if thr_quantile is None:
                    raise ValueError('The threshold quantile must be provided for the "quantile" statistic.')
                stat = np.quantile(data, thr_quantile)
            else:
                raise ValueError('The statistic must be either "mean", "median", "mode", "sum" or "quantile".')

        if column_name is None:
            # if both threshold quantile and value are not provided
            if thr_quantile is None and thr_value is None:
                column_name = f'{statistic}'
            else:
                column_name = f'{statistic}_q{thr_quantile}' if thr_quantile is not None else f'{statistic}_val{thr_value}'

        shapes.loc[shapes.geometry == geom, column_name] = stat

    return shapes

@as_DAM_process(input_type='xarray', output_type='shape')
def get_percentages_by_shape(input: xr.DataArray,
                      shapes: gpd.GeoDataFrame,
                      classes: list[int],
                      ) -> gpd.GeoDataFrame:
    """
    Classify a raster by a shapefile.
    """
    # get no_data value
    nodata_value = input.attrs.get('_FillValue', np.nan)

    # Initialize an empty list to store the results
    results = []

    # Calculate the bin edges dynamically
    bin_edges = np.append(classes, classes[-1] + 1)

    # Loop over each geometry in the GeoDataFrame
    for geom in shapes.geometry:
        # Mask the raster with the current geometry
        out_image = input.rio.clip([geom])

        # we only care about the data, not the shape
        out_data = out_image.values.flatten()
        
        # check if all values are nodata
        if np.all(np.isclose(out_data, nodata_value, equal_nan=True)):
            # Add a row of zeros to the results
            results.append(np.zeros(len(classes)))
        else:
            # remove the nodata values
            out_data = out_data[~np.isclose(out_data, nodata_value, equal_nan=True)]

            # get the values in the classes
            hist, _ = np.histogram(out_data, bins=bin_edges)

            # Turn the histogram into percentages of the total rounded to 0 decimals
            hist_percentages = np.round(hist / hist.sum() * 100, 0)

            # Add the histogram percentages to the list
            results.append(hist_percentages)


    # Turn the list of lists into a numpy array
    results = np.array(results)
    for i in range(len(classes)):
        # Define the column name
        column_name = f'class_{classes[i]}'
        shapes[column_name] = results[:, i]

    return shapes

@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def combine_raster_data(input: xr.DataArray,
                        statistic: str = 'mean',
                        weights: Optional[list[float]] = None,
                        nodata_value: float = np.nan,
                        na_ignore: bool = False,
                        **kwargs) -> xr.DataArray:
    """
    Combine multiple rasters into a single raster.
    """

    match statistic:
        case 'sum':
            weights = weights
        case 'mean':
            weights = [w / sum(weights) for w in weights]
        case _:
            raise ValueError('The statistic must be either "sum" or "mean".')

    input_data = [input.values]
    for k, v in kwargs.items():
        if isinstance(v, xr.DataArray):
            input_data.append(v.values)

    if len(input_data) == 1:
        raise ValueError('No additional rasters were provided for combine_raster_data.')

    # check that the weitghts are the correct length
    if weights is None:
        weights = [1] * len(input_data)
    elif len(weights) != len(input_data):
        raise ValueError('The number of weights must be the same as the number of rasters.')

    # remove 1d dimensions
    databrick = np.stack([d.squeeze() for d in input_data])
    if na_ignore:
        databrick_nan = np.where(np.isclose(databrick, nodata_value, equal_nan=True), 0, databrick)
    else:
        databrick_nan = np.where(np.isclose(databrick, nodata_value, equal_nan=True), np.nan, databrick)

    weighted_data = np.einsum('i,ijk->ijk', weights, databrick_nan)
    mask = np.isfinite(weighted_data)  # <- this is a mask of all the nans in the weighted data
    weights = np.array(weights)
    weights_3d = np.broadcast_to(weights[:, np.newaxis, np.newaxis], weighted_data.shape)
    selected_weights = np.where(mask, weights_3d, 0)
    total_weights = np.sum(selected_weights, axis=0)

    weighted_sum = np.sum(weighted_data, axis=0)

    if statistic == 'sum':
        result = weighted_sum
    elif statistic == 'mean':
        total_weights = np.where(total_weights < 1e-3, 1e-3, total_weights)
        weighted_mean = weighted_sum / total_weights
        weighted_mean = np.where(total_weights == 1e-3, np.nan, weighted_mean)
        result = weighted_mean

    result = np.where(np.isnan(result), nodata_value, result)

    # reshape the result to the original shape
    result = result.reshape(input.shape)
    return input.copy(data=result)

# -------------------------------------------------------------------------------------
# Method to extract residuals from xr.DataArray based on lat-lon and values

@as_DAM_process(output_type = 'csv')
def compute_residuals(input: xr.DataArray,
                      data: pd.DataFrame,
                      name_lat_lon_data_csv: list[str],
                      method_ltln2val: Optional[str] = 'nearest',
                      method_residuals: Optional[str] = 'data_minus_map') -> pd.DataFrame:
    """
    Compute residuals between data and map
    """

    # load map
    map = input
    map = map.squeeze()  # remove single dimensions, usually time

    # load data
    lat_points = data[name_lat_lon_data_csv[0]].to_numpy()
    lon_points = data[name_lat_lon_data_csv[1]].to_numpy()
    data_points = data[name_lat_lon_data_csv[2]].to_numpy()

    # extract values in map based on lat and lon
    values_map = ltln2val_from_2dDataArray(input_map=map, lat=lat_points, lon=lon_points, method=method_ltln2val)

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

    return data

# -------------------------------------------------------------------------------------
# Method to classifies a raster file based on specified thresholds and assigns class values
@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def classify_raster(input: xr.DataArray,
                    thresholds: list[float],
                    classes: Optional[list[int]] = None,
                    side: str = "left",
                    normalize: bool = False) -> xr.DataArray:
    """
    Classifies a raster based on thresholds using np.digitize. Classes are optional.

    Parameters:
    - input (xr.DataArray): Input raster to classify.
    - thresholds (list): List of threshold values (breaks).
    - classes (list, optional): List of class values corresponding to thresholds. Defaults to [0, 1, ..., n].
    - side (str): "left" (default) to classify using `>= threshold`, "right" to use `< threshold`.

    Returns:
    - xr.DataArray: Classified raster.
    """
    if side.lower() not in ["left", "right"]:
        raise ValueError("Parameter 'side' must be either 'left' or 'right'.")

    # Get no_data value
    nodata_value = input.attrs.get('_FillValue', np.nan)

    # Default classes to [0, 1, 2, ..., len(thresholds)]
    if classes is None:
        classes = list(range(len(thresholds) + 1))

    if len(classes) != len(thresholds) + 1:
        raise ValueError("The number of classes must be equal to the number of thresholds + 1.")

    # Perform classification using np.digitize
    classified_values = np.digitize(input.values, thresholds, right=(side.lower() == "right"))

    # Map digitized values to the provided or default classes
    classified_map = np.array(classes, dtype=int)[classified_values]

    # Preserve NaN values
    classified_map = np.where(np.isnan(input.values), np.nan, classified_map)

    # Convert back to xarray.DataArray with the same coordinates and dimensions
    output = xr.DataArray(classified_map,
                          dims=input.dims,
                          coords=input.coords,
                          name="classified_map")
    return output

# -------------------------------------------------------------------------------------
# Method to normalize a raster file
@as_DAM_process(input_type = 'xarray', output_type = 'xarray')
def normalize_raster(input: xr.DataArray,
                     method: str = "minmax",
                     min_value: Optional[float] = None,
                     max_value: Optional[float] = None,
                     ) -> xr.DataArray:
    """
    Normalizes a raster using specified method.

    Parameters:
    - input (xr.DataArray): Input raster to normalize.
    - method (str): Normalization method to use. Options: "minmax", "meanstd".
    - min_value (float, optional): Minimum value for normalization. Defaults to None.
    - max_value (float, optional): Maximum value for normalization. Defaults to None.

    Returns:
    - xr.DataArray: Normalized raster.
    """
    if method.lower() not in ["minmax", "meanstd"]:
        raise ValueError("Parameter 'method' must be either 'minmax' or 'meanstd'.")

    # Normalize using Min-Max scaling
    if method.lower() == "minmax":
        if min_value is None:
            min_value = input.min().item()
        if max_value is None:
            max_value = input.max().item()
        if min_value == max_value:
            raise ValueError("min_value and max_value cannot be equal.")
        normalized_values = (input - min_value) / (max_value - min_value)


    # Normalize using Mean-Std scaling
    elif method == "meanstd":
        normalized_values = (input - input.mean()) / input.std()

    # Convert back to xarray.DataArray with the same coordinates and dimensions
    output = xr.DataArray(normalized_values,
                          dims=input.dims,
                          coords=input.coords,
                          name="normalized_map")
    return output
