from dam import DAMWorkflow, DAMProcessor
from dam.tools.data import LocalDataset
from dam.tools.timestepping import TimeRange

from dam.processing.filter import keep_valid_range, apply_binary_mask
from dam.processing.calc import apply_scale_factor

import os

DATA_PATH = '/home/luca/Documents/CIMA_code/tests/VIIRS_dwl'

def main():

    input = LocalDataset(path     = os.path.join(DATA_PATH, '%Y', '%m', '%d'),
                         filename = 'VIIRS-JPSS1_FAPAR_%Y%m%d.tif')
    
    test_process = DAMProcessor(function = keep_valid_range,
                                input    = input)
    
    output = test_process.run(time = TimeRange('2020-01-01', '2020-02-02'))
    #output = test_process.run(time = '2020-01-01')

    breakpoint()
    qc = LocalDataset(path     = os.path.join(DATA_PATH, '%Y', '%m', '%d'),
                      filename = 'VIIRS-JPSS1_FAPAR_QC_%Y%m%d.tif')
    
    filtered = LocalDataset(path     = os.path.join(DATA_PATH, 'filtered'),
                            filename = 'VIIRS-JPSS1_FAPAR_filtered_%Y%m%d.tif')



    wf = DAMWorkflow(config = {'input': input, 'qc': qc})

    bound = keep_valid_range(data, (0, 100), nodata_value = 255)

    qc = '/home/luca/Documents/CIMA_code/tests/VIIRS_dwl/global_raw/VIIRS-FAPAR_QC_wld-tile32_20170202.tif'
    keep = {'012': [0,1],             # main algorithm used
            '3':   [0],               # both sensors working
            '4567':[1,2,3,4,5,6,7,8]} # remove non-vegetated areas (incl. water (0))

    
    filtered = apply_binary_mask(bound, qc, keep, nodata_value = 255)
    rescaled = apply_scale_factor(filtered, 0.01)


if __name__ == '__main__':
    main()