import matplotlib.pyplot as plt

from maps.core.z_stamping import z_stamping_step,\
    z_stamping_step_yz,\
    shift_frames_and_store,\
    shift_frames_and_store_yz
from maps.helpers.logging_config import logger_config
from maps.settings import BASE_DIR, read_setting_from_json, setting
from maps.helpers.misc import pickle_object
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame

import logging
import os
import glob

logging.config.dictConfig(logger_config)
logger = logging.getLogger('MaPS')

# Path to settings json for preloading settings
settings_json = os.path.join(BASE_DIR, 'current_inputs.json')

# Initialize the settings object
read_setting_from_json('job1', settings_json)

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
setting['frame_count'] = 1000

# Folder containing brightfield images for phase stamping. The images must be
# named sequentially
phase_image_folder = 'D:\\Scripts\\MaPS\\Data sets\\Phase_Bidi\\'

# Whether to use existing z stamp values from "data dump" folder, or compute
# from raw images over again
use_existing_datadump_vals = False

# STEP 1: Correlation window selection
# This step selects the small rectangular region to be used when calculating
# shift in pixels of each frame from a reference frame. This is used for
# z stamping the frames. The window a=parameters are dudmped as a csv file.
USE_GUI_CORR_WINDOW = False
USE_GUI_CROP_WINDOW = False

# Frame number of the frame with the largest heart size.
# This frame will be used as reference frame
frame_no = 21

img_path = glob.glob(
    os.path.join(phase_image_folder, '*.tif')
)

if USE_GUI_CORR_WINDOW and not use_existing_datadump_vals:
    # Uses GUI to select rectangular window in specified frame.
    # Click and drag with mouse tto select region. Clos window to finalize
    img_seq = load_image_sequence([img_path[frame_no]])
    max_heartsize_frame(img_seq[0])

    params = get_rect_params()

else:
    params = {}
    # Bottom edge of window
    params['x_end'] = 114
    # Height of window
    params['height'] = 29
    # Right edge of window
    params['y_end'] = 248
    # Width of window
    params['width'] = 33

x_end = params['x_end']
height = params['height']
y_end = params['y_end']
width = params['width']

data = [
    ('frame', frame_no),
    ('x_end', x_end),
    ('height', height),
    ('y_end', y_end),
    ('width', width),
]

print 'Using reference window as-'
print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

pickle_object(data, file_name='corr_window.csv', dumptype='csv')

raw_input('Press enter to continue...')

# STEP 2: Z stamping step
# This function takes a single kymograph, the number of frames to process,
# the location of phase stamping images and an optional flag (whether to
# recompute z stamp values or used pickled values)

z_stamp_opt, x_stamp_opt, z_stamp_cf, res, bad_zooks, minp = z_stamping_step_yz(
    kymo_path=setting['km_path'],
    frame_count=setting['frame_count'],
    phase_img_path=setting['bf_images'],
    use_old=setting['usePickled'],
    datafile_name='z_stamp_opt_KM-XZ_0.pkl',
    datafile_name_x='y_stamp_opt_KM-XZ_0.pkl'
)
# z_stamp_opt, z_stamp_cf, res, bad_zooks, minp = z_stamping_step(
#     kymo_path=kymograph_path,
#     frame_count=frame_count,
#     phase_img_path=phase_image_folder,
#     use_old=use_existing_datadump_vals,
#     datafile_name='z_stamp_opt_KM-XZ_0.pkl'
# )

# # Example of display values
# plt.figure(111)
# plt.plot(z_stamp_opt)
# plt.figure(112)
# plt.plot(x_stamp_opt)
# plt.show()
#
# for bad_zook in bad_zooks:
#     print '=' * 80
#     print 'Zook #%d' % bad_zook[0]
#     print 'Fault locations:'
#     print bad_zook[2]
#     print 'Fault values:'
#     print bad_zook[1]
#
# plt.plot(res)
# plt.show()

# STEP 3: Cropping window selection and masking
# This step selects the rectangular window to be used when cropping the
# frame. It also creates the free form mask around the heart image and saves
# the mask image

if USE_GUI_CROP_WINDOW:
    img_seq = load_image_sequence([img_path[frame_no]])
    masking_window_frame(img_seq[0])
    crop_params = get_rect_params()
else:
    crop_params = {}
    # Bottom edge of window
    crop_params['x_end'] = 307
    # Height of window
    crop_params['height'] = 173
    # Right edge of window
    crop_params['y_end'] = 303
    # Width of window
    crop_params['width'] = 181


x_end = crop_params['x_end']
height = crop_params['height']
y_end = crop_params['y_end']
width = crop_params['width']

data = [
    ('frame', frame_no),
    ('x_end', x_end),
    ('height', height),
    ('y_end', y_end),
    ('width', width),
]

print 'Using cropping window as-'
print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

pickle_object(data, file_name='crop_window.csv', dumptype='csv')

raw_input('Press enter to continue...')

# STEP 4: Shift and crop the frames
# Shift and store the frames in the good zooks

shift_frames_and_store_yz(phase_image_folder, z_stamp_opt, x_stamp_opt, [], minp, x_end, height, y_end, width, frame_no)
# shift_frames_and_store(phase_image_folder, z_stamp_opt, [], minp, x_end, height, y_end, width, frame_no)
