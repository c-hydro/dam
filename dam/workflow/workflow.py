from .processor import DAMProcessor

from ..tools.data import Dataset
from ..tools.data.memory_dataset import MemoryDataset
from ..tools.data.local_dataset import LocalDataset

from ..tools.timestepping import TimeRange
from ..tools.timestepping.time_utils import get_date_from_str

import datetime as dt
from typing import Optional
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
        self.break_points = []

        self.options = self.default_options
        if options is not None:
            self.options.update(options)

        if self.options['intermediate_output'] == 'Tmp':
            tmp_dir = self.options.get('tmp_dir', tempfile.gettempdir())
            os.makedirs(tmp_dir, exist_ok = True)
            self.tmp_dir = tempfile.mkdtemp(dir = tmp_dir)

    def clean_up(self):
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def make_output(self, input: Dataset, output: Optional[Dataset|dict] = None) -> Dataset:
        if isinstance(output, Dataset):
            return output
        
        if output is None:
            key_pattern = input.key_pattern
        elif isinstance(output, dict):
            key_pattern = output.get('key_pattern', input.key_pattern)
        else:
            raise ValueError('Output must be a Dataset or a dictionary.')
        
        output_type = self.options['intermediate_output']
        if output_type == 'Mem':
            return MemoryDataset(key_pattern = input.key_pattern)
        elif output_type == 'Tmp':
            filename = os.path.basename(key_pattern)
            return LocalDataset(path = self.tmp_dir, filename = filename)

    def add_process(self, function, output: Optional[Dataset|dict] = None, **kwargs) -> None:
        if len(self.processes) == 0:
            previous = None
            this_input = self.input
        else:
            previous = self.processes[-1]
            this_input = previous.output

        this_output = self.make_output(this_input, output)
        this_output.name = f'{this_output.name}_{function.__name__}'
        this_process = DAMProcessor(function = function,
                                    input = this_input,
                                    args = kwargs,
                                    output = this_output,
                                    wf_options = self.options)

        if this_process.break_point:
            self.break_points.append(len(self.processes))

        self.processes.append(this_process)

    def run(self, time: dt.datetime|str|TimeRange, **kwargs) -> None:
        if len(self.processes) == 0:
            raise ValueError('No processes have been added to the workflow.')
        elif isinstance(self.processes[-1].output, MemoryDataset) or\
            (isinstance(self.processes[-1].output, LocalDataset) and self.tmp_dir in self.processes[-1].output.dir):
            if self.output is not None:
                self.processes[-1].output = self.output.copy()
            else:
                raise ValueError('No output dataset has been set.')

        if isinstance(time, TimeRange):
            timesteps = self.input.get_times(time, **kwargs)
            for timestep in timesteps:
                self.run(timestep, **kwargs)
            return
        elif isinstance(time, str):
            time = get_date_from_str(time)

        if len(self.break_points) == 0:
            self._run_processes(self.processes, time, **kwargs)
        else:
            # proceed in chuncks: run until the breakpoint, then stop
            i = 0
            processes_to_run = []
            while i < len(self.processes):
                #collect all processes until the breakpoint
                if i not in self.break_points:
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
        
        if isinstance(time, TimeRange):
            timesteps = input.get_times(time, **kwargs)
            for timestep in timesteps:
                self._run_processes(processes, timestep, **kwargs)

        elif 'tile' not in kwargs:
            all_tiles = input.tile_names if self.options['break_on_missing_tiles'] else input.find_tiles(time, **kwargs)
            for tile in all_tiles:
                self._run_processes(processes, time, tile = tile, **kwargs)

        else:
            for process in processes:
                process.run(time, **kwargs)