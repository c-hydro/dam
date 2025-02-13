import xarray as xr
from functools import partial
import datetime as dt

from typing import Callable, Sequence

from d3tools.data import Dataset
from d3tools.timestepping.time_utils import get_date_from_str, find_unit_of_time
from d3tools.timestepping import TimeRange, TimeStep, TimeWindow

from .processor import DAMProcessor
from ..utils.register_process import as_DAM_process
from ..utils.register_agg import AGG_FUNCTIONS

class DAMAggregator(DAMProcessor):

    def __init__(self,
                 function: Callable,
                 agg_args: dict,
                 input: Dataset,
                 args: dict = {},
                 output: Dataset = None,
                 wf_options: dict = {}) -> None:
        
        self.S_break_point = False
        self.funcname = function.__name__

        ds_args, static_args = self.get_args(args)
        self.agg_args = agg_args

        self.function = partial(function, **static_args)
        self.ds_args = ds_args
        self.input = input
        self.output = output

        self.input_options  = {'break_on_missing_tiles' : wf_options.get('break_on_missing_tiles', False)}

    def run(self, time: TimeRange|TimeStep|dt.datetime|str, **kwargs) -> None:

        raw_input  = self.input.copy()
        raw_output = self.output.copy()

        window_names, windows = zip(*self.agg_args['windows'].items())
        max_window = max(windows)

        for i in range(len(window_names)):
            this_window = windows[i]
            run_window = max_window - this_window
            run_time_start = run_window.apply(time.start).start
            run_time = TimeRange(run_time_start - dt.timedelta(1), time.end)

            mult_windows = [w for w in windows[0:i] if this_window.is_multiple(w)]
            if len(mult_windows) > 0:
                input_window = max(mult_windows)
                input_window_name = window_names[windows.index(input_window)]
                self.input = raw_output.update(agg_window = input_window_name)
                self.input.agg = input_window
            else:
                self.input = raw_input
            self.run_singlewindow(run_time, (window_names[i], this_window), **kwargs)
            
    def run_singlewindow(self,
                         time: TimeRange|TimeStep|dt.datetime|str,
                         window_tuple: tuple[str,TimeWindow],
                         **kwargs) -> None:
        
        window_name, window = window_tuple

        step = self.agg_args['step']
        if step is None:
            step = self.input.estimate_timestep(**kwargs).unit

        if isinstance(time, str):
            time = get_date_from_str(time)
        
        if isinstance(time, Sequence):
            time = TimeRange(min(time), max(time))

        if isinstance(time, dt.datetime):
            ts = TimeStep.from_unit(step)
            ts.agg_window = window
            time = ts.from_date(time)
        elif isinstance(time, TimeStep):
            if time.agg_window != window:
                time.agg_window = window
        elif isinstance(time, TimeRange):
            tss = time.get_timesteps(freq = step, agg = window)
            for time in tss:
                self.run_singlewindow(time, window_tuple, **kwargs)

        # from here on time is a TimeStep with the correct agg_window

        input_options = self.input_options
        if 'tile' not in kwargs:
            if input_options['break_on_missing_tiles']:
                all_tiles = self.input.tile_names
            else:
                all_tiles = self.input.find_tiles(time, **kwargs)
                if len(all_tiles) == 0:
                    all_tiles = ['__tile__']
            for tile in all_tiles:
                self.run_singlewindow(time, window_tuple,  tile = tile, **kwargs)
            return

        agg_range = time.agg_range
        relevant_ts = self.input.get_timesteps(agg_range, **kwargs)
        if len(relevant_ts) == 0:
            return

        if relevant_ts[0].agg_range.start > time.agg_range.start:
            return

        input_data = [self.input.get_data(t, **kwargs) for t in relevant_ts]
        ds_args    = {arg_name: [arg.get_data(t, **kwargs) for t in relevant_ts] for arg_name, arg in self.ds_args.items()}
        
        #relevant_times = [(t.agg_range.start, t.agg_range.end) for t in relevant_ts]
        output = self.function(input     = input_data,
                               input_agg = [ts.agg_range for ts in relevant_ts],
                               this_agg  = agg_range,
                               **ds_args)

        print(f'{self.funcname}, {window_name} - {time}, {kwargs}')
        metadata = {'agg_method' : f'{self.funcname}, {window_name}'}
        self.output.timestep = TimeStep.from_unit(step).with_agg(window)
        self.output.write_data(output, time, agg_window = window_name, metadata = metadata, **kwargs)

def get_agg_args(args: dict):

    # func
    # {step, window}
    # other

    agg_func   = 'mean'
    agg_args   = {'step' : None, 'windows' : '1m'}
    other_args = {}

    for key, value in args.items():
        if key in ['agg_func', 'agg_function', 'agg_fn']:
            agg_func = value
        elif key == 'agg_step':
            agg_args['step'] = find_unit_of_time(value)
        elif key == 'agg_window':
            if isinstance(value, str):
                raw_dict = {value: TimeWindow.from_str(value)}
            elif isinstance(value, dict):
                raw_dict = {k:TimeWindow.from_str(v) for k, v in value.items()}
            elif isinstance(value, Sequence):
                raw_dict = {v:TimeWindow.from_str(v) for v in value}
            agg_args['windows'] = {k:v for k, v in sorted(raw_dict.items(), key = lambda item: item[1])}
        else:
            other_args[key] = value

    return AGG_FUNCTIONS[agg_func], agg_args, other_args

@as_DAM_process()
def aggregate_times(input):
    pass