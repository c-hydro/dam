from .processor import make_process, Processor

from d3tools.data import Dataset
from d3tools.data.memory_dataset import MemoryDataset
from d3tools.data.local_dataset import LocalDataset
from d3tools.timestepping import TimeRange, TimeWindow
from d3tools.config.options import Options
from d3tools.cases import CaseManager
from d3tools.exit import rm_at_exit

import datetime as dt
from typing import Optional, Sequence
import tempfile
import os
import shutil
import copy
import re

class DAMWorkflow:

    default_options = {
        'intermediate_output'   : 'Tmp', # 'Mem' or 'Tmp'
        'break_on_missing_tiles': False,
        'tmp_dir'               : os.getenv('TMP', '/tmp'),
        'propagate_metadata'    : [],
        'make_past'             : True
    }

    def __init__(self,
                 input: Dataset,
                 output: Dataset = None,
                 options: Optional[dict] = None) -> None:
        
        self.input = input
        self.output = output
        
        self.processes = {
            'processes' : [],
            'pids'      : []
        }

        self.run_instructions = {
            'process_groups'  : [],
            'group_types'     : [],
            'lookback_windows': []
        }

        all_options = copy.deepcopy(self.default_options)
        if options is not None:
            all_options.update(options)

        # all the options that are not in the self.default_options will be used to set the input
        self.input_options = {k: v for k, v in all_options.items() if k not in self.default_options}
        self.options  = {k: v for k, v in all_options.items() if k not in self.input_options}

        # ensure that the "propagate_metadata" option is a list
        if self.options.get('propagate_metadata') is not None and isinstance(self.options['propagate_metadata'], str):
            self.options['propagate_metadata'] = [self.options['propagate_metadata']]

        # if the input data has tiles, we need to add the tile name to the input options
        if self.input.has_tiles:
            tile_names = self.input.tile_names
            self.input_options['tile'] = dict(zip(tile_names, tile_names))

        # set the input_options as the root of the case_tree
        self.case_tree = CaseManager(self.input_options.copy())

        if self.options['intermediate_output'] == 'Tmp':
            tmp_dir = self.options.get('tmp_dir', tempfile.gettempdir())
            os.makedirs(tmp_dir, exist_ok = True)
            self.tmp_dir = tempfile.mkdtemp(dir = tmp_dir)
            rm_at_exit(self.tmp_dir)
            
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
            output = process.pop('output', None)
            pid = process.pop('pid', None)
            wf.add_process(function_str, output, pid, **process)

        return wf
    
    def add_process(self, function: str, output: Optional[Dataset|dict] = None, pid : Optional[str] = None, **kwargs) -> None:
        
        process_list = self.processes['processes']
        pids = self.processes['pids']

        if pid is None:
            pid = function

        same_processes = [i for i,p in enumerate(pids) if p.split('#')[0] == pid]

        if len(same_processes) == 1:
            pids[same_processes[0]] = f'{pid}#1'
            pid = f'{pid}#2'
        elif len(same_processes) > 1:
            pid = f'{pid}#{len(same_processes)+1}'

        this_process = make_process(function, kwargs)
        this_process.output = output
        this_process.propagate_metadata = self.options['propagate_metadata']
        this_process.make_past          = self.options['make_past']

        process_list.append(this_process)
        pids.append(pid)
        i = len(process_list)-1

        self.update_tree(this_process, pid)

        if not this_process.T_break_point:
            if len(self.run_instructions['group_types']) > 0 and self.run_instructions['group_types'][-1] == 'linear':
                self.run_instructions['process_groups'][-1].append(i)
            else:
                self.run_instructions['process_groups'].append([i])
                self.run_instructions['group_types'].append('linear')
                self.run_instructions['lookback_windows'].append(TimeWindow(0, 'd'))
        elif this_process.T_break_point:
            self.run_instructions['process_groups'].append(i)
            self.run_instructions['group_types'].append('aggregator')
            this_window = this_process.max_window
            self.run_instructions['lookback_windows'] = [max(w, this_window) for w in self.run_instructions['lookback_windows']]
            self.run_instructions['lookback_windows'].append(TimeWindow(0, 'd'))

    def update_tree(self, process: Processor, pid: str) -> None:
        _args = {f'{pid}.{k}': v for k, v in process.args.items()}
        
        merge = None

        if process.tile_input:
            merge = 'tile'
            process.input_tiles = [t for t in self.case_tree.options[-1]['tile'].values()]
        
        self.case_tree.add_layer(_args, name = pid, merge = merge)
        if process.tile_output:
            _args = {'tile': {t:t for t in process.tile_names}}
            self.case_tree.add_layer(_args, name = '')
            self.processes['processes'].append(None)
            self.processes['pids'].append('')
        elif process.T_break_point:
            if len(process.agg_windows) > 1:
                _args = {f'{pid}.agg_window': process.agg_windows}
            else:
                _args = {}
            self.case_tree.add_layer(_args, name = '')
            self.processes['processes'].append(None)
            self.processes['pids'].append('')

    def run(self, time: TimeRange|Sequence[dt.datetime|str]) -> None:

        # get the time as a TimeRange
        time = TimeRange.from_any(time)

        # add the pids to each process now that they are "final", as well as the inputs and outpus
        self._processes = []
        for i in range(len(self.processes['processes'])):
            process = self.processes['processes'][i]
            
            if process is None:
                self._processes.append(None)
                continue
            
            pid = self.processes['pids'][i]
            process.pid = pid
            
            # set the input:
            k = i
            while k>0:
                if self._processes[k-1] is not None:
                    process.input = self._processes[k-1].output
                    break
                k -= 1
            else:
                process.input = self.input

            # set the output:
            if process.output is None:
                tags = self.case_tree.tags[i+1] if not (process.tile_output or process.T_break_point) else self.case_tree.tags[i+2]
                process.output = self.make_output(process, tags)

            self._processes.append(process)

        # set the output of the last process to be the output of the workflow
        k = len(self._processes)
        while k > 0:
            p = self._processes[k-1]
            if p is not None:
                p.output = self.output.copy()
                break
            k -= 1

        # for p in self._processes:
        #     if p is None: continue
        #     print("----------------")
        #     print(p.pid)
        #     print(p.input.key_pattern)
        #     print(p.output.key_pattern)
        #     print("----------------")

        # loop through the process groups and run them
        for group, type, window in zip(self.run_instructions['process_groups'],
                                       self.run_instructions['group_types'],
                                       self.run_instructions['lookback_windows']):

            if not self.options['make_past']:
                time = time.extend(window, before = True)

            if type == 'linear':
                self._run_linear_group(group, time)
            elif type == 'aggregator':
                self._run_aggregator(group, time)

    def _run_linear_group(self, process_n, time, window = None) -> None:

        layer_n   = [p+1 for p in process_n]
        
        first_process = self._processes[min(process_n)]
        timesteps = set()
        for case in self.case_tree[min(process_n)].values():
            these_tss = first_process.input.get_timesteps(time, **case.tags)
            if len(these_tss) == 0:
                these_tss = first_process.input.get_timesteps(time, **case.options)
            timesteps.update(these_tss)

        timesteps = list(timesteps)
        timesteps.sort()
        for ts in timesteps:
            for case, layer in self.case_tree.iterate_tree(max(layer_n), get_layer = True):
                if layer not in layer_n: continue
                process = self._processes[layer-1]
                process.run(ts, case.options, case.tags)

    def _run_aggregator(self, process_n, time) -> dict:

        layer_n   = process_n + 1 
        process = self._processes[process_n]

        for case in self.case_tree[layer_n].values():
            process.run(time, case.options, case.tags)

    def make_output(self,
                    process: Processor,
                    tags: Sequence) -> Dataset:

        input = process.input
        input_pattern = input.key_pattern
        root_in, ext_in = os.path.splitext(input_pattern)
        ext_out  = process.output_ext or ext_in[1:]
        root_out = '%Y%m%d' + '%H' * ('%H' in root_in)

        output_pattern = f'{process.pid}/{process.pid}' + "_".join([f'{{{t}}}' for t in tags]) + f'_{root_out}.{ext_out}'
        
        output_type = self.options['intermediate_output']
        if output_type == 'Mem':
            output_ds = MemoryDataset(key_pattern = output_pattern)
        elif output_type == 'Tmp':
            output_ds = LocalDataset(path = self.tmp_dir, filename = output_pattern)

        if process.continuous_space:
            output_ds._template = input._template
        else:
            output_ds._template = {}

        return output_ds
    
    def get_last_ts(self, **kwargs) -> tuple[TimeRange]:
        """
        Get the last available timesteps of the input and output datasets, based on input and output cases
        """

        input_ds = [self.input]
        all_processes = self.processes.get('processes', [])

        for p in filter(lambda p: p is not None, all_processes):
            for a in p.args.values():
                if isinstance(a, Dataset) and '%Y' in re.sub(r'\{[^}]*\}', '', a.key_pattern):
                    input_ds.append(a)

        input_dates = []
        input_cases = self.case_tree[0]
        for ds in input_ds:
            last_ts_input = None
            for case in input_cases.values():
                now = None if last_ts_input is None else last_ts_input.end + dt.timedelta(days = 1)
                input = ds.get_last_ts(now = now, **case.tags, **kwargs)
                if input is None:
                    input = ds.get_last_ts(now = now, **case.options, **kwargs)
                if input is not None:
                    last_ts_input = input if last_ts_input is None else min(input, last_ts_input)
                else:
                    last_ts_input = None
                    break
            input_dates.append(last_ts_input)
        last_ts_input = min(filter(None, input_dates), default = None)

        output_cases = self.case_tree[-1]
        last_ts_output = None
        for case in output_cases.values():
            now = None if last_ts_output is None else last_ts_output.end + dt.timedelta(days = 1)
            output = self.output.get_last_ts(now = now, **case.tags, **kwargs)
            if output is not None:
                last_ts_output = output if last_ts_output is None else min(output, last_ts_output)
            else:
                last_ts_output = None
                break

        return last_ts_input, last_ts_output