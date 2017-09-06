from maps.core.z_stamping import z_stamping_step_yz
from maps.core import phase_stamping

from maps.settings import setting, read_setting_from_json, \
    read_setting_from_csv
from maps.helpers.misc import pickle_object
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame

import glob
import os

read_setting_from_json('job1')

ref_frame_no = 21

USE_GUI_CORR_WINDOW = True
USE_GUI_CROP_WINDOW = True

img_path = glob.glob(
    os.path.join(setting['bf_images'], '*.tif')
)

if USE_GUI_CORR_WINDOW:
    # Uses GUI to select rectangular window in specified frame.
    # Click and drag with mouse tto select region. Clos window to finalize
    img_seq = load_image_sequence([img_path[setting['index_start_at'] + ref_frame_no]])
    max_heartsize_frame(img_seq[0])

    params = get_rect_params()

else:
    params = {}
    # Bottom edge of window
    params['x_end'] = 118
    # Height of window
    params['height'] = 38
    # Right edge of window
    params['y_end'] = 479
    # Width of window
    params['width'] = 41

x_end = params['x_end']
height = params['height']
y_end = params['y_end']
width = params['width']

data = [
    ('frame', ref_frame_no),
    ('x_end', x_end),
    ('height', height),
    ('y_end', y_end),
    ('width', width),
]

print 'Using reference window as-'
print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

pickle_object(data, file_name='corr_window.csv', dumptype='csv')

raw_input('Press enter to continue...')

z_stamp_opt, x_stamp_opt, z_stamp_cf, res, bad_zooks, minp = z_stamping_step_yz(
    kymo_path=glob.glob(os.path.join(setting['km_path'], '*.tif'))[0],
    frame_count=setting['frame_count'],
    phase_img_path=setting['bf_images'],
    use_old=setting['usePickled'],
    datafile_name='z_stamp_opt_KM-XZ_0.pkl',
    datafile_name_x='y_stamp_opt_KM-XZ_0.pkl'
)
