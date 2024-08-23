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
        self.break_point = False

        ds_args = {}
        static_args = {}
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, Dataset):
                if not arg_value.is_static:
                    ds_args[arg_name] = arg_value
                else:
                    static_args[arg_name] = arg_value.get_data()
            else:
                static_args[arg_name] = arg_value

        self.funcname = function.__name__
        self.function = partial(function, **static_args)
        self.ds_args = ds_args
        self.input = input

        input_tiles  = function.__dict__.get('input_tiles',False)
        output_tiles = function.__dict__.get('output_tiles',False)
        self.input_options = {'all_tiles' : input_tiles}
        self.output_options = {'all_tiles' : output_tiles}

        if output is not None and not input_tiles and not output_tiles:
            output._template = input._template
        else:
            self.break_point = True
            
        self.output = output

    def __repr__(self):
        return f'DAMProcessor({self.funcname})'

    def run(self, time: dt.datetime|str|TimeRange, **kwargs) -> xr.DataArray|list[xr.DataArray]:

        input_options = self.input_options
        if 'tile' not in kwargs and not input_options['all_tiles']:
                tiles = self.input.tile_names
                for tile in tiles:
                    self.run(time, tile = tile, **kwargs)

        # return a list if the input is a TimeRange
        if isinstance(time, TimeRange):
            timesteps = self.input.get_times(time, **kwargs)
            return [self.run(timestep, **kwargs) for timestep in timesteps]
        
        # otherwise, return the output of the function
        if isinstance(time, str):
            time = get_date_from_str(time)

        if input_options['all_tiles']:
            input_data = (self.input.get_data(time, tile = tile, **kwargs) for tile in self.input.tile_names)
        else:
            input_data = self.input.get_data(time, **kwargs)
        ds_args = {arg_name: arg.get_data(time, **kwargs) for arg_name, arg in self.ds_args.items()}

        output_data = self.function(input = input_data, **ds_args)
        if self.output is not None:
            self.output.write_data(output_data, time, **kwargs)

        return output_data