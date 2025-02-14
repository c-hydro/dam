from d3tools.data import Dataset
from d3tools.timestepping.time_utils import get_date_from_str

import datetime as dt
from typing import Callable
from functools import partial

class DAMProcessor:
    def __init__(self,
                 function: Callable,
                 input: Dataset,
                 args: dict = {},
                 output: Dataset = None,
                 wf_options: dict = {}) -> None:
        
        self.S_break_point = False
        self.funcname = function.__name__

        ds_args, static_args = self.get_args(args)

        self.function = partial(function, **static_args)
        self.ds_args = ds_args
        self.input = input

        input_tiles  = function.__dict__.get('input_tiles',False)
        output_tiles = function.__dict__.get('output_tiles',False)
        self.input_options  = {'tiles' : input_tiles, 'break_on_missing_tiles' : wf_options.get('break_on_missing_tiles', False)}
        self.output_options = {'tiles' : output_tiles, 'tile_name_attr' : function.__dict__.get('tile_name_attr', 'tile_name')}

        tiling = input_tiles or output_tiles
        continuous_space = function.__dict__.get('continuous_space', True)
        if tiling:
            continuous_space = False
            self.S_break_point = True

        if output is not None and continuous_space:
            output._template = input._template
            if input.tile_names is not None:
                output.tile_names = input.tile_names
        elif input_tiles and not output_tiles:
            output.tile_names = ['__tile__']
            output.key_pattern = output.key_pattern.replace('{tile}', '').replace('tile', '')
        elif not input_tiles and output_tiles:
            if '{tile}' not in output.key_pattern:
                output.key_pattern = output.key_pattern.replace('.tif', 'tile{tile}.tif')
            
        self.output = output

    def __repr__(self):
        return f'DAMProcessor({self.funcname})'

    def run(self, time: dt.datetime|str, **kwargs) -> None:
        if isinstance(time, str):
            time = get_date_from_str(time)

        input_options = self.input_options
        if 'tile' not in kwargs:
            if input_options['break_on_missing_tiles']:
                all_tiles = self.input.tile_names
            else:
                all_tiles = self.input.find_tiles(time, **kwargs)
                if len(all_tiles) == 0:
                    all_tiles = ['__tile__']
            if not input_options['tiles']:
                for tile in all_tiles:
                    self.run(time, tile = tile, **kwargs)
                return
            else:
                input_data = (self.input.get_data(time, tile = tile, **kwargs) for tile in all_tiles)
                metadata = {'tiles': all_tiles}
        else:
            input_data = self.input.get_data(time, **kwargs)
            metadata = {}

        ds_args = {arg_name: arg.get_data(time, **kwargs) for arg_name, arg in self.ds_args.items()}
        output = self.function(input = input_data, **ds_args)

        output_options = self.output_options
        print(f'{self.funcname} - {time}, {kwargs}')
        if output_options['tiles']:
            self.output.tile_names = []
            for i, output_data in enumerate(output):
                this_tile_name = output_data.attrs.get(output_options['tile_name_attr'], f'{i}')
                kwargs['tile'] = this_tile_name
                self.output.write_data(output_data, time, metadata = metadata, **kwargs)
                self.output.tile_names.append(this_tile_name)
        else:
            self.output.write_data(output, time, metadata = metadata, **kwargs)

    @staticmethod
    def get_args(args: dict) -> tuple:
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
        
        return ds_args, static_args