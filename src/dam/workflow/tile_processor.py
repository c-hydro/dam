import datetime as dt
from collections import defaultdict

from d3tools.data import Dataset
from d3tools.timestepping import TimeStep

from .processor import Processor
from ..utils.register_process import DAM_PROCESSES

class TileMerger(Processor):

    T_break_point = False

    def __init__(self, args: dict = None) -> None:

        function = DAM_PROCESSES['combine_tiles']
        super().__init__(function, args)

        self.continuous_space = False

    def run(self, time: dt.datetime|TimeStep, args: dict, tags: dict) -> None:

        arg_str = {k:str(args.get(k, tags[k])) for k in tags}
        if self.input.check_data(time, tile = self.input_tiles[0], **tags):
            input_data = [self.input.get_data(time, tile = t, **tags) for t in self.input_tiles]
        elif self.input.check_data(time, tile = self.input_tiles[0], *arg_str):
            input_data = [self.input.get_data(time, tile = t, **arg_str) for t in self.input_tiles]
        else:
            return ##TODO: add a warning or something

        these_args = {}
        for arg_name in self.args:
            arg_value = args.get(f'{self.pid}.{arg_name}', self.args[arg_name])
            if isinstance(arg_value, Dataset):
                these_args[arg_name] = arg_value.get_data(time, **tags)
            else:
                these_args[arg_name] = arg_value

        output = self.function(input_data, **these_args)

        str_tags = {k.replace(f'{self.pid}.', ''): v for k, v in tags.items()}
        tag_str = ', '.join([f'{k}={v}' for k, v in str_tags.items()])
        print(f'{self.pid} - {time}, {tag_str}')

        self.output.write_data(output, time, **tags)

class TileSplitter(Processor):

    T_break_point = False

    def __init__(self, args: dict = None) -> None:

        function = DAM_PROCESSES['split_in_tiles']
        super().__init__(function, args)

        self.continuous_space = False

        n_tiles = args.get('n_tiles', None)
        tile_name_format = args.get('tile_names', '{i}')
        dir = args.get('dir', 'vh')

        self.tile_names = self.get_tile_names(n_tiles, tile_name_format, dir)
    
    def run(self, time: dt.datetime|TimeStep, args: dict, tags: dict) -> None:

        arg_str = {k:str(args.get(k, tags[k])) for k in tags}
        if self.input.check_data(time, **tags):
            input_data = self.input.get_data(time, **tags)
        elif self.input.check_data(time, **arg_str):
            input_data = self.input.get_data(time, **arg_str)
        else:
            return ##TODO: add a warning or something
            
        these_args = {}
        for arg_name in self.args:
            arg_value = args.get(f'{self.pid}.{arg_name}', self.args[arg_name])
            if isinstance(arg_value, Dataset):
                these_args[arg_name] = arg_value.get_data(time, **tags)
            else:
                these_args[arg_name] = arg_value
            
        output = self.function(input_data, **these_args)

        str_tags = {k.replace(f'{self.pid}.', ''): v for k, v in tags.items()}
        tag_str = ', '.join([f'{k}={v}' for k, v in str_tags.items()])
        print(f'{self.pid} - {time}, {tag_str}')
        for this_output, tile_name in zip(output, self.tile_names):
            self.output.write_data(this_output, time, tile = tile_name, **tags)
    
    @staticmethod
    def get_tile_names(n_tiles, name_format, dir):
        
        def get_tile_name(i, hi, vi):
            sub_values = defaultdict(str, i=i, hi=hi, vi=vi)
            tile_name = name_format.format_map(sub_values)
            return tile_name

        if isinstance(n_tiles, int):
            n_tiles = (n_tiles, n_tiles)
        
        nx, ny = n_tiles
        tile_names = []
        i=0
        if dir == 'hv':
            for vi in range(ny):
                for hi in range(nx):
                    tile_name = get_tile_name(i, hi, vi)
                    tile_names.append(tile_name)
                    i += 1
        elif dir == 'vh':
            for hi in range(nx):
                for vi in range(ny):
                    tile_name = get_tile_name(i, hi, vi)
                    tile_names.append(tile_name)
                    i += 1

        return tile_names