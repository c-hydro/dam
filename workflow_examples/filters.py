from dam.filter import apply_binary_mask, keep_valid_range
from dam.calc import apply_scale_factor

def main():
    data = '/home/luca/Documents/CIMA_code/tests/VIIRS_dwl/global_raw/VIIRS-FAPAR_wld-tile32_20170202.tif'

    bound = keep_valid_range(data, (0, 100), nodata_value = 255)

    qc = '/home/luca/Documents/CIMA_code/tests/VIIRS_dwl/global_raw/VIIRS-FAPAR_QC_wld-tile32_20170202.tif'
    keep = {'012': [0,1],             # main algorithm used
            '3':   [0],               # both sensors working
            '4567':[1,2,3,4,5,6,7,8]} # remove non-vegetated areas (incl. water (0))

    
    filtered = apply_binary_mask(bound, qc, keep, nodata_value = 255)
    rescaled = apply_scale_factor(filtered, 0.01)


if __name__ == '__main__':
    main()