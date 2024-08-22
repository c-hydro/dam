from .processor import DAMProcessor

from ..tools.data import Dataset
from ..tools.data.memory_dataset import MemoryDataset

from ..tools.timestepping import TimeRange

import datetime as dt
from typing import Optional

class DAMWorkflow:
    def __init__(self, input: Dataset, output: Optional[Dataset] = None) -> None:
        self.input = input
        self.output = output
        self.processes = []
        self.break_points = []

    def add_process(self, function, output: Optional[Dataset|dict] = None, **kwargs) -> None:
        if len(self.processes) == 0:
            previous = None
            this_input = self.input
        else:
            previous = self.processes[-1]
            this_input = previous.output

        if output is None:
            this_output = MemoryDataset(key_pattern = this_input.key_pattern)
        elif isinstance(output, dict):
            key_pattern = output.pop('key_pattern', this_input.key_pattern)
            this_output = MemoryDataset(key_pattern, **output)

        this_output._template = this_input._template

        this_process = DAMProcessor(function = function,
                                    input = this_input,
                                    args = kwargs,
                                    output = this_output)

        if this_process.break_point:
            self.break_points.append(len(self.processes))

        self.processes.append(this_process)

    def run(self, time: dt.datetime|str|TimeRange, **kwargs) -> None:
        if len(self.processes) == 0:
            raise ValueError('No processes have been added to the workflow.')
        elif isinstance(self.processes[-1].output, MemoryDataset):
            if self.output is not None:
                self.processes[-1].output = self.output
            else:
                raise ValueError('No output dataset has been set.')

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
                    # reset the list of processes
                    processes_to_run = []
                i += 1
            self._run_processes(processes_to_run, time, **kwargs)

    def _run_processes(self, processes, time: dt.datetime, **kwargs) -> None:
        if len(processes) == 0:
            return

        input = processes[0].input
        
        if isinstance(time, TimeRange):
            timesteps = input.get_times(time, **kwargs)
            for timestep in timesteps:
                self._run_processes(processes, timestep, **kwargs)

        elif 'tile' not in kwargs:
            tiles = input.tile_names
            for tile in tiles:
                self._run_processes(processes, time, tile = tile, **kwargs)
        else:
            for process in processes:
                process.run(time, **kwargs)