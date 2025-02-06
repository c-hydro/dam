from .processor import DAMProcessor
from .time_aggregator import DAMAggregator, get_agg_args

from ..utils.register_process import DAM_PROCESSES
from ..utils.register_agg import AGG_FUNCTIONS

from d3tools.data import Dataset
from d3tools.data.memory_dataset import MemoryDataset
from d3tools.data.local_dataset import LocalDataset
from d3tools.timestepping import TimeRange, get_date_from_str, TimeStep
from d3tools.config.options import Options

import datetime as dt
from typing import Optional, Sequence
import tempfile
import os
import shutil

class DAMWorkflow:

    default_options = {
        'intermediate_output'   : 'Mem', # 'Mem' or 'Tmp'
        'break_on_missing_tiles': False,
        'tmp_dir'               : None
    }

    def __init__(self,
                 input: Dataset,
                 output: Optional[Dataset] = None,
                 options: Optional[dict] = None) -> None:
        
        self.input = input
        self.output = output
        self.processes = []
        self.S_break_points = []   #spatial break points
        self.T_break_point  = None #temporal break point

        self.lookback_window = None

        self.options = self.default_options
        if options is not None:
            self.options.update(options)

        if self.options['intermediate_output'] == 'Tmp':
            tmp_dir = self.options.get('tmp_dir', tempfile.gettempdir())
            os.makedirs(tmp_dir, exist_ok = True)
            self.tmp_dir = tempfile.mkdtemp(dir = tmp_dir)

    @classmethod
    def from_options(cls, options: Options|dict) -> 'DAMWorkflow':
        if isinstance(options, dict): options = Options(options)
        input = options.get('input',ignore_case=True)
        if isinstance(input, dict):
            input = Dataset.from_options(input)
        output = options.get('output', None, ignore_case=True)
        if isinstance(output, dict):
            output = Dataset.from_options(output)
        wf_options = options.get('options', None, ignore_case=True)

        wf = cls(input, output, wf_options)
        processes = options.get(['processes','process_list'], [], ignore_case=True)
        for process in processes:
            function_str = process.pop('function')
            function = DAM_PROCESSES[function_str]
            output = process.pop('output', None)
            wf.add_process(function, output, **process)

        return wf

    def clean_up(self):
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception as e:
                print(f'Error cleaning up temporary directory: {e}')

    def make_output(self, input: Dataset,
                    output: Optional[Dataset|dict] = None,
                    function = None) -> Dataset:
        if isinstance(output, Dataset):
            return output
        
        input_pattern = input.key_pattern
        input_name = input.name
        if function is not None:
            name = f'_{function.__name__}'
            ext_in = os.path.splitext(input_pattern)[1][1:]
            ext_out = function.__getattribute__('output_ext') or ext_in
            output_pattern = input_pattern.replace(f'.{ext_in}', f'_{name}.{ext_out}')
            output_name = f'{input_name}_{name}'

        if output is None:
            output_pattern = output_pattern
        elif isinstance(output, dict):
            output_pattern = output.get('key_pattern', output_pattern)
        else:
            raise ValueError('Output must be a Dataset or a dictionary.')
        
        output_type = self.options['intermediate_output']
        if output_type == 'Mem':
            output_ds = MemoryDataset(key_pattern = output_pattern)
        elif output_type == 'Tmp':
            filename = os.path.basename(output_pattern)
            output_ds = LocalDataset(path = self.tmp_dir, filename = filename)

        output_ds.name = output_name
        return output_ds

    def add_process(self, function, output: Optional[Dataset|dict] = None, **kwargs) -> None:
        if len(self.processes) == 0:
            previous = None
            this_input = self.input
        else:
            previous = self.processes[-1]
            this_input = previous.output

        this_output = self.make_output(this_input, output, function)
        if function.__name__ == 'aggregate_times':
            agg_func, agg_args, other_args = get_agg_args(kwargs)
            this_process = DAMAggregator(function = agg_func,
                                         agg_args = agg_args,
                                         input = this_input,
                                         args = other_args,
                                         output = this_output,
                                         wf_options = self.options)
            
            if self.T_break_point is not None:
                raise ValueError('Only one time aggregation is supported per workflow.')
            else:
                max_window = max([w for w in agg_args['windows'].values()])
                self.T_break_point = (len(self.processes), max_window)
            
        else:

            this_process = DAMProcessor(function = function,
                                        input = this_input,
                                        args = kwargs,
                                        output = this_output,
                                        wf_options = self.options)

            if this_process.S_break_point:
                self.S_break_points.append(len(self.processes))

        self.processes.append(this_process)

    def run(self, time: dt.datetime|str|TimeRange|Sequence[dt.datetime], **kwargs) -> None:

        if isinstance(time, Sequence):
            time.sort()
            time = TimeRange(time[0], time[-1])

        if len(self.processes) == 0:
            raise ValueError('No processes have been added to the workflow.')
        elif isinstance(self.processes[-1].output, MemoryDataset) or\
            (isinstance(self.processes[-1].output, LocalDataset) and hasattr(self, 'tmp_dir') and self.tmp_dir in self.processes[-1].output.dir):
            if self.output is not None:
                self.processes[-1].output = self.output.copy()
            else:
                raise ValueError('No output dataset has been set.')

        if self.T_break_point is not None:
            for wf, lb in self.split_workflow(self.T_break_point):
                if lb is not None:
                    run_start = lb.apply(time.start - dt.timedelta(1)).start
                    run_time  = TimeRange(run_start, time.end)
                else:
                    run_time = time
                wf.run(run_time, **kwargs)
            return
        
        if isinstance(time, TimeRange):
            timestamps = self.input.get_times(time, **kwargs)

            if self.input.time_signature == 'end+1':
                timestamps = [t - dt.timedelta(days = 1) for t in timestamps]

            timestep = self.input.estimate_timestep(timestamps)

            if timestep is not None:
                timesteps = [timestep.from_date(t) for t in timestamps]
            else:
                timesteps = timestamps
            
            if len(timesteps) == 0:
                return
            else:
                for timestep in timesteps:
                    self.run_single_ts(timestep, **kwargs)
                return
        else:
            self.run_single_ts(time, **kwargs)
    
    def run_single_ts(self, time: dt.datetime|str|TimeRange, **kwargs) -> None:
        
        if isinstance(time, str):
            time = get_date_from_str(time)

        if len(self.S_break_points) == 0:
            self._run_processes(self.processes, time, **kwargs)
        else:
            # proceed in chuncks: run until the breakpoint, then stop
            i = 0
            processes_to_run = []
            while i < len(self.processes):
                #collect all processes until the breakpoint
                if i not in self.S_break_points:
                    processes_to_run.append(self.processes[i])
                else:
                    # run the processes until the breakpoint
                    self._run_processes(processes_to_run, time, **kwargs)
                    # then run the breakpoint by itself
                    self.processes[i].run(time, **kwargs)
                    # if this process outputs tiles and this isn't the last process, 
                    if self.processes[i].output_options['tiles'] and i < len(self.processes) - 1:
                        # make sure the tile names are passed to the next process
                        self.processes[i+1].input.tile_names = self.processes[i].output.tile_names
                    # reset the list of processes
                    processes_to_run = []
                i += 1
            # run the remaining processes
            self._run_processes(processes_to_run, time, **kwargs)
        
        # clean up the temporary directory
        self.clean_up()

    def _run_processes(self, processes, time: dt.datetime, **kwargs) -> None:
        if len(processes) == 0:
            return

        input = processes[0].input
        if 'tile' not in kwargs:
            all_tiles = input.tile_names if self.options['break_on_missing_tiles'] else input.find_tiles(time, **kwargs)
            if len(all_tiles) == 0:
                all_tiles = ['__tile__']
            for tile in all_tiles:
                self._run_processes(processes, time, tile = tile, **kwargs)

        else:
            for process in processes:
                process.run(time, **kwargs)

    def split_workflow(self, breakpoint):
        options = self.options
        subworkflows = []

        breakpoint_id, breakpoint_lookback = breakpoint

        if breakpoint_id > 0:
            # this is the definition of the subworkflow
            this_input = self.processes[0].input
            this_output = self.processes[breakpoint_id-1].output
            this_wf = DAMWorkflow(this_input, this_output, options)
            these_processes = self.processes[0:breakpoint_id]
            this_wf.processes = these_processes

            subworkflows.append((this_wf, breakpoint_lookback))
                
        # here is the definition of the breakpoint itself
        subworkflows.append((self.processes[breakpoint_id], None))

        if breakpoint_id < len(self.processes)-1:
            # and the rest of the workflow
            this_input = self.processes[breakpoint_id+1].input
            this_output = self.processes[-1].output
            this_wf = DAMWorkflow(this_input, this_output, options)
            these_processes = self.processes[breakpoint_id+1:]
            this_wf.processes = these_processes
            subworkflows.append((this_wf, None))
        
        return subworkflows