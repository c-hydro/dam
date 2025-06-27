from d3tools.data import Dataset
from d3tools.timestepping import TimeStep

import datetime as dt
from typing import Callable

from ..utils.register_process import DAM_PROCESSES

class Processor:
    """
    This is a class for simple linear processes.
    1 input = 1 output at the same timestep.
    """

    T_break_point = False

    pid = None

    propagate_metadata = []
    make_past = True

    timestep_settings = None

    def __init__(self,
                 function: Callable,
                 args: dict = None) -> None:
        
        self.function = function
        self.output = args.pop('output', None) if args else None
        self.set_args(args)

        # this flag is used to determine if we need to inherit the template from the input.
        self.continuous_space = function.__dict__.get('continuous_space', True)
        self.output_ext = function.__dict__.get('output_ext', None)

        self.tile_output = function.__dict__.get('output_tiles', False)
        self.tile_input  = function.__dict__.get('input_tiles', False)

        self.input_as_is = function.__dict__.get('input_as_is', False)

    def run(self, time: dt.datetime|TimeStep, args: dict, tags: dict) -> None:
        
        arg_str = {k:str(args.get(k, tags[k])) for k in tags}
        if self.input.check_data(time, **tags):
            input_data = self.input.get_data(time, **tags, as_is=self.input_as_is)
        elif self.input.check_data(time, **arg_str):
            input_data = self.input.get_data(time, **arg_str, as_is=self.input_as_is)
        else:
            return ##TODO: add a warning or something

        these_args = {}
        ts_shifts = self.timestep_settings or {}
        for arg_name in self.args:
            arg_value = args.get(f'{self.pid}.{arg_name}', self.args[arg_name])
            if isinstance(arg_value, Dataset):
                these_args[arg_name] = arg_value.get_data(time + ts_shifts.get(arg_name, 0), **tags, as_is=self.input_as_is)
            else:
                these_args[arg_name] = arg_value

        output = self.function(input_data, **these_args)

        str_tags = {k.replace(f'{self.pid}.', ''): v for k, v in tags.items()}
        if 'tile' in tags: str_tags['tile'] = tags['tile']
        tag_str = ', '.join([f'{k}={v}' for k, v in str_tags.items()])
        print(f'{self.pid} - {time}, {tag_str}')

        metadata = {}
        for key in self.propagate_metadata:
            if key in input_data.attrs:
                metadata[key] = input_data.attrs[key]

        self.output.write_data(output, time, metadata = metadata, **tags)

    def set_args(self, args: None|dict) -> tuple:

        if args is None:
            args = {}

        def set_arg_type(value):
            if not isinstance(value, Dataset):
                return value
            elif not value.is_static:
                return value
            else:
                return value.get_data()

        for key, value in args.items():
            if isinstance(value, dict):
                args[key] = {k: set_arg_type(v) for k, v in value.items()}
            else:
                args[key] = set_arg_type(value)

        self.args = args

def make_process(function_name: str|Callable,
                 args: None|dict = None,
                 output: Dataset|None = None,
                 **kwargs) -> Processor:
    
    from .time_aggregator import TimeAggregator
    from .tile_processor import TileMerger, TileSplitter

    if ".timesteps" in args:
        timestep_settings = args.pop(".timesteps")
    else:
        timestep_settings = None

    if isinstance(function_name, str):
        if function_name == 'aggregate_times':
            this_process = TimeAggregator(args)
        elif function_name == 'combine_tiles':
            this_process = TileMerger(args)
        elif function_name == 'split_in_tiles':
            this_process = TileSplitter(args)
        else:
            function = DAM_PROCESSES[function_name]
            this_process = Processor(function, args)
    
    this_process.output = output
    this_process.timestep_settings = timestep_settings
    return this_process