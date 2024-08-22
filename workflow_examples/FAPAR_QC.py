from dam import DAMWorkflow
from dam.tools.data import LocalDataset
from dam.tools.timestepping import TimeRange
from dam.tools.config.parse import extract_date_and_tags

from dam.processing.filter import keep_valid_range, apply_binary_mask
from dam.processing.calc import apply_scale_factor

import os
import datetime as dt

DATA_PATH = '/home/luca/Documents/CIMA_code/tests/VIIRS_processing'

def main():

    data = LocalDataset(path     = os.path.join(DATA_PATH, '8d', '%Y', '%m', '%d'),
                        filename = 'VIIRS_{variable}_%Y%m%d_tile{tile}.tif')

    keep_dict = {'012': [0,1],             # main algorithm used
                 '3':   [0],               # both sensors working
                 '4567':[1,2,3,4,5,6,7,8]} # remove non-vegetated areas (incl. water (0))

    wf = DAMWorkflow(input  = data.update(variable = 'raw_FAPAR'),
                     output = data.update(variable = 'FAPAR_filtered'))
    wf.add_process(keep_valid_range)
    wf.add_process(apply_binary_mask, mask = data.update(variable = 'raw_FAPAR_QC'), keep = keep_dict)
    wf.add_process(apply_scale_factor)

    wf.run(time = TimeRange('2020-01-01', '2020-02-10'))

if __name__ == '__main__':
    main()