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

        self.processes.append(this_process)

    def get_times(self, time: TimeRange, **kwargs) -> list[dt.datetime]:
        return self.input.get_times(time, **kwargs)

    def run(self, time: dt.datetime|str|TimeRange, **kwargs) -> None:

        if len(self.processes) == 0:
            raise ValueError('No processes have been added to the workflow.')
        elif isinstance(self.processes[-1].output, MemoryDataset):
            if self.output is not None:
                self.processes[-1].output = self.output
            else:
                raise ValueError('No output dataset has been set.')
        
        if isinstance(time, TimeRange):
            timesteps = self.get_times(time, **kwargs)
            for timestep in timesteps:
                self.run(timestep, **kwargs)
        elif 'tile' not in kwargs:
            tiles = self.input.tile_names
            for tile in tiles:
                self.run(time, tile = tile, **kwargs)
        else:
            for process in self.processes:
                process.run(time, **kwargs)