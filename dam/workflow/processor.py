from ..tools.data import Dataset
from ..tools.timestepping import TimeRange
from ..tools.timestepping.time_utils import get_date_from_str

import datetime as dt
from typing import Optional, Callable, Generator
from functools import partial
import xarray as xr

class DAMProcessor:
    def __init__(self,
                 function: Callable,
                 input: Dataset,
                 args: dict = {},
                 output: Dataset = None) -> None:
        
        ds_args = {}
        static_args = {}
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, Dataset):
                ds_args[arg_name] = arg_value
            else:
                static_args[arg_name] = arg_value

        self.function = partial(function, **static_args)
        self.ds_args = ds_args
        self.input = input

        if output is not None:
            output._template = input._template
            
        self.output = output

    def run(self, time: dt.datetime|str|TimeRange, **kwargs) -> xr.DataArray|Generator[xr.DataArray, None, None]:

        if 'tile' not in kwargs:
            tiles = self.input.tile_names
            for tile in tiles:
                self.run(time, tile = tile, **kwargs)

        # return a generator if the input is a TimeRange
        if isinstance(time, TimeRange):
            timesteps = self.input.get_times(time, **kwargs)
            return (self.run(timestep, **kwargs) for timestep in timesteps)
        
        # otherwise, return the output of the function
        if isinstance(time, str):
            time = get_date_from_str(time)
        
        input_data = self.input.get_data(time, **kwargs)
        ds_args = {arg_name: arg.get_data(time, **kwargs) for arg_name, arg in self.ds_args.items()}

        output_data = self.function(input = input_data, **ds_args)
        if self.output is not None:
            self.output.write_data(output_data, time, **kwargs)

        return output_data