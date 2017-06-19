import matplotlib.pyplot as plt

from maps.core.z_stamping import z_stamping_step
from maps.helpers.logging_config import logger_config
from maps.settings import reload_current_settings, setting

import logging

logging.config.dictConfig(logger_config)
logger = logging.getLogger('MaPS')

# Path to settings json for preloading settings
settings_json = 'D:\\Scripts\\MaPS\\MaPS scripts\\maps\\current_inputs.json'

# Initialize the settings object
reload_current_settings(settings_json)

# Number of zooks to skip from the start#
setting['ignore_zooks_at_start'] = 1
# Number of frames to ignore at the start of every zook
setting['ignore_startzook'] = 7
# Number of frames to ignore at the end of every zook
setting['ignore_endzook'] = 3
# Physical resolution of brightfield images in um
setting['BF_resolution'] = 0.6296
# Prefix of image file names
setting['image_prefix'] = 'Phase_Bidi1_'
# Location of raw data dump. This includes pickled objects, csv files and plots
setting['data_dump'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
# Upsampling factor while finding optimal z stamp
setting['resampling_factor'] = 5
# Horizontal width of window within which best correlation is to be found (in
# low res domain)
setting['slide_limit'] = 5
# Number of frames in a single zook
setting['ZookZikPeriod'] = 192
# First index in the filename of images
setting['index_start_at'] = 1
# Number of image digits to be used in the file name
setting['num_digits'] = 5
# Location of first minima in the kymograph
setting['first_minima'] = 0

# Path to single kymograph image. Kymograph needs to be binarized
kymograph_path = 'D:\Scripts\MaPS\Data sets\Kymographs\KM-XZ_0.tif'

# Number of frames to process
frame_count = 9999

# Folder containing brightfield images for phase stamping. The images must be
# named sequentially
phase_image_folder = 'D:\\Scripts\\MaPS\\Data sets\\Phase_Bidi\\'

# Whether to use existing z stamp values from "data dump" folder, or compute
# from raw images over again
use_existing_datadump_vals = False

# STEP 1: Z stamping step
# This function takes a single kymograph, the number of frames to process,
# the location of phase stamping images and an optional flag (whether to
# recompute z stamp values or used pickled values)

z_stamp_opt, z_stamp_cf, res, bad_zooks = z_stamping_step(
    kymo_path=kymograph_path,
    frame_count=frame_count,
    phase_img_path=phase_image_folder,
    use_old=use_existing_datadump_vals
)

# Example of display values
plt.plot(z_stamp_opt)
plt.show()

for bad_zook in bad_zooks:
    print '=' * 80
    print 'Zook #%d' % bad_zook[0]
    print 'Fault locations:'
    print bad_zook[2]
    print 'Fault values:'
    print bad_zook[1]

plt.plot(res)
plt.show()
