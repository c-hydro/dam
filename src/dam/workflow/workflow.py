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

class DAMWorkflow:

    default_options = {
        'intermediate_output'   : 'Tmp', # 'Mem' or 'Tmp'
        'break_on_missing_tiles': False,
        'tmp_dir'               : os.getenv('TMP', '/tmp')
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

        # if the input data has tiles, we need to add the tile name to the input options
        if self.input.has_tiles:
            tile_names = self.input.tile_names
            self.input_options['tile'] = dict(zip(tile_names, tile_names))

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
            pid = f'{pid}_{len(same_processes)+1}'

        this_process = make_process(function, kwargs)
        this_process.output = output
        process_list.append(this_process)
        pids.append(pid)

        if not this_process.T_break_point:
            if len(self.run_instructions['group_types']) > 0 and self.run_instructions['group_types'][-1] == 'linear':
                self.run_instructions['process_groups'][-1].append(len(process_list)-1)
            else:
                self.run_instructions['process_groups'].append([len(process_list)-1])
                self.run_instructions['group_types'].append('linear')
                self.run_instructions['lookback_windows'].append(TimeWindow(0, 'd'))
        # elif this_process.S_break_point:
        #     self.run_instructions['process_groups'].append(len(process_list)-1)
        #     self.run_instructions['group_types'].append('tiling')
        #     self.run_instructions['lookback_windows'].append(TimeWindow(0, 'd'))
        elif this_process.T_break_point:
            self.run_instructions['process_groups'].append(len(process_list)-1)
            self.run_instructions['group_types'].append('aggregator')
            this_window = this_process.max_window
            self.run_instructions['lookback_windows'] = [max(w, this_window) for w in self.run_instructions['lookback_windows']]
            self.run_instructions['lookback_windows'].append(TimeWindow(0, 'd'))

    def run(self, time: TimeRange|Sequence[dt.datetime|str]) -> None:

        # set the output of the last process to the output of the workflow 
        if self.output is not None:
            self.processes['processes'][-1].output = self.output.copy()
        elif not hasattr(self.processes['processes'][-1], 'output') or self.processes['processes'][-1].output is None:
            raise ValueError('No output dataset has been set.')

        # get the time as a TimeRange
        time = TimeRange.from_any(time)

        # add the pids to each process now that they are "final"
        self._processes = []
        for process, pid in zip(self.processes['processes'], self.processes['pids']):
            process.pid = pid
            self._processes.append(process)

        # get the input and the input_options
        input_options = self.input_options.copy()
        input         = {'ds': self.input, 'options' : input_options}

        # loop through the process groups and run them
        for group, type, window in zip(self.run_instructions['process_groups'],
                                       self.run_instructions['group_types'],
                                       self.run_instructions['lookback_windows']):

            this_time = time.extend(window, before = True)
            if type == 'linear':
                processes = [self._processes[i] for i in group]
                output = self._run_linear(processes, this_time, input)
            elif type == 'aggregator':
                process = self._processes[group]
                output = self._run_aggregator(process, this_time, input)

            input = output

    def _run_linear(self, processes, time, input) -> dict:

        input_options = input['options']
        case_trees = []#CaseManager(input_options)
        tree_processes = []
        next_args = {}

        input = input['ds']
        for i, process in enumerate(processes):
            pid = process.pid
            args = {f'{pid}.{k}': v for k, v in process.args.items()}
            args.update(next_args)
            if process.tile_input:
                if len(case_trees) == 0:
                    opts = input_options.copy()
                else:
                    opts = case_trees[-1].options[-1]

                process_opts = {k: v for k, v in opts.items() if k != 'tile'}
                tiles = input_options.get('tile', {})

                new_tree = CaseManager(process_opts)
                new_tree.add_layer(args)
                case_trees.append(new_tree)
                process.input_tiles = list(tiles.values())
                tree_processes.append([process])
            else:
                if len(case_trees) == 0:
                    case_trees.append(CaseManager(input_options))
                    tree_processes.append([])
                    
                case_trees[-1].add_layer(args)
                tree_processes[-1].append(process)

            if process.tile_output:
                next_args = {'tile' : {t:t for t in process.tile_names}}
                output_tags = set(['tile'])
            else:
                next_args = {}
                output_tags = set()

            for case in case_trees[-1][-1].values():
                for tag in case.tags:
                    if tag in args:
                        output_tags.add(tag)

            process.input = input if i == 0 else processes[i-1].output
            if process.output is None:
                process.output = self.make_output(process, output_tags)

        timesteps = input.get_timesteps(time)
        for ts in timesteps:
            for case_tree, t_processes in zip(case_trees, tree_processes):
                for case_id, case in case_tree[0].items():
                    cases = case_tree.iterate_subtree(case_id, layer = True)
                    for c, l in cases:
                        t_processes[l-1].run(ts, c.options, c.tags)

        output = processes[-1].output
        output_options = case_trees[-1].options[-1]

        return {'ds': output, 'options': output_options}

    def _run_aggregator(self, process, time, input) -> dict:

        input_options = input['options']
        case_tree = CaseManager(input_options)

        input = input['ds']
        windows = process.agg_windows

        pid = process.pid
        output_tags = set() if len(windows) <= 1 else set([f'{pid}.agg_window'])

        args = {f'{pid}.{k}': v for k, v in process.args.items()}
        case_tree.add_layer(args)
        for case in case_tree[-1].values():
            for tag in case.tags:
                if tag in args:
                    output_tags.add(tag)

        process.input = input
        if process.output is None:
            process.output = self.make_output(process, output_tags)

        output = process.output.copy()

        for case_id, case in case_tree[0].items():
            cases = case_tree.iterate_subtree(case_id, 1, layer = False)
            for c in cases:
                process.input = input.copy()
                process.run(time, c.options, c.tags)                    

        output_options = case_tree.options[-1] | {f'{process.pid}.agg_window': windows}

        return {'ds': output, 'options': output_options}

    def clean_up(self):
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception as e:
                print(f'Error cleaning up temporary directory: {e}')

    def make_output(self,
                    process: Processor,
                    tags: Sequence) -> Dataset:

        input = process.input
        input_pattern = input.key_pattern
        root_in, ext_in = os.path.splitext(input_pattern)
        ext_out = process.output_ext or ext_in[1:]
        output_pattern = "_".join([f'{root_in}_{process.pid}'] + [f'{{{t}}}' for t in tags]) + f'.{ext_out}'
        
        output_type = self.options['intermediate_output']
        if output_type == 'Mem':
            output_ds = MemoryDataset(key_pattern = output_pattern)
        elif output_type == 'Tmp':
            filename = os.path.basename(output_pattern)
            output_ds = LocalDataset(path = self.tmp_dir, filename = filename)

        if process.continuous_space:
            output_ds._template = input._template
        else:
            output_ds._template = {}

        return output_ds