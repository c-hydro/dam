from typing import Sequence

from d3tools.data import Dataset
from d3tools.timestepping.time_utils import find_unit_of_time
from d3tools.timestepping import TimeRange, TimeStep, TimeWindow

from .processor import Processor
from ..utils.register_process import as_DAM_process
from ..utils.register_agg import AGG_FUNCTIONS

class TimeAggregator(Processor):

    S_break_point = False
    T_break_point = True
    continuous_space = True
    output_ext = None

    def __init__(self, args: dict) -> None:

        self.output = args.pop('output', None)
        self.set_args(args)

    def run(self, timerange: TimeRange, args: dict, tags: dict) -> None:
        
        raw_input  = self.input.copy()
        raw_output = self.output.copy()

        window_names, windows = zip(*self.agg_windows.items())
        max_window = max(windows)
        for i in range(len(window_names)):
            this_window = windows[i]
            run_window = max_window - this_window
            run_time = timerange.extend(run_window, before = True)

            this_window_tags = tags | {f'{self.pid}.agg_window': window_names[i]}
            this_window_args = args | {'agg_window' : window_names[i]}

            mult_windows = [w for w in windows[0:i] if this_window.is_multiple(w)]
            if len(mult_windows) > 0:
                input_window = max(mult_windows)
                input_window_name = window_names[windows.index(input_window)]
                input_window_tags = {f'{self.pid}.agg_window': input_window_name}
                self.input = raw_output.update(**input_window_tags, inplace = False)
                self.input.agg = input_window
            else:
                self.input = raw_input

            self.run_singlewindow(run_time, this_window_args, this_window_tags)

    def run_singlewindow(self, timerange: TimeRange, args: dict, tags: dict) -> None:

        step = self.agg_step
        if step is None:
            step = self.input.estimate_timestep(**tags).unit

        window  = args.pop(f'agg_window')
        self.output.timestep = TimeStep.from_unit(step).with_agg(window)
        timesteps = timerange.get_timesteps(freq = step, agg = window)
        
        for ts in timesteps:

            agg_range = ts.agg_range
            relevant_ts = self.input.get_timesteps(agg_range, **tags)

            if len(relevant_ts) == 0 or relevant_ts[0].agg_range.start > ts.agg_range.start:
                continue
            
            input_data = [self.input.get_data(t, **tags) for t in relevant_ts]
            input_agg = [ts.agg_range for ts in relevant_ts]

            these_args = {}
            for arg_name in self.args:
                arg_value = args.get(f'{self.pid}.{arg_name}', self.args[arg_name])
                if isinstance(arg_value, Dataset):
                    these_args[arg_name] = [arg_value.get_data(t, **tags) for t in relevant_ts]
                else:
                    these_args[arg_name] = arg_value
        
            output = self.agg_function(input_data, input_agg = input_agg, this_agg  = agg_range, **these_args)
            
            str_tags = {k.replace(f'{self.pid}.', ''): v for k, v in tags.items() if self.pid in k}
            tag_str = ', '.join([f'{k}={v}' for k, v in str_tags.items()])
            print(f'{self.pid} - {ts}, {tag_str}')

            metadata = {'agg_method' : f'{self.agg_function_name}, {tags[f"{self.pid}.agg_window"]}'}
            self.output.write_data(output, ts, metadata = metadata, **tags)

    def set_args(self, args: dict):
        nonagg_args = {}
        agg_args = {}

        for key, value in args.items():
            if key in ['agg_func', 'agg_function', 'agg_fn']:
                if isinstance(value, str):
                    self.agg_function = AGG_FUNCTIONS.get(value)
                    self.agg_function_name = value
                else:
                    raise ValueError('agg_func must be a string with the name of an aggregation function, only one function can be specified per aggrergator')
            elif key == 'agg_step':
                self.agg_step = find_unit_of_time(value)
            elif key == 'agg_window':
                if isinstance(value, str):
                    raw_dict = {value: TimeWindow.from_str(value)}
                elif isinstance(value, dict):
                    raw_dict = {k:TimeWindow.from_str(v) for k, v in value.items()}
                elif isinstance(value, Sequence):
                    raw_dict = {v:TimeWindow.from_str(v) for v in value}
                self.agg_windows = {k:v for k, v in sorted(raw_dict.items(), key = lambda item: item[1])}
                self.max_window = max(self.agg_windows.values())

            else:
                nonagg_args[key] = value

        super().set_args(nonagg_args)
        self.args.update(agg_args)

@as_DAM_process()
def aggregate_times(input):
    pass