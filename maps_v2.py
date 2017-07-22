from maps.core.z_stamping import z_stamping_step_yz, shift_frames_and_store_yz
from maps.core.phase_stamping import crop_and_compute_dwt_matrix, compute_dwt_matrix

from maps.settings import setting, read_setting_from_json, \
    display_current_settings, JOBS_DIR
from maps.helpers.misc import pickle_object
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame

import glob
import os
import time

job_list = next(os.walk('./jobs/'))[1]

job_choice = 0

print '\n'.join(
    map(
        lambda x, y: '%d-%s' % (x, y),
        range(1, len(job_list) + 1),
        job_list
    )
)

while job_choice not in range(1, len(job_list) + 1):
    job_choice = int(raw_input('Select job:'))

    curr_job = job_list[job_choice - 1]
    read_setting_from_json(curr_job)


USE_GUI_CORR_WINDOW = False
USE_GUI_CROP_WINDOW = False
Y_STAGE_CORRECTION = True
WRITE_IMAGES_TO_DISK = True
USE_PICKLED_ZSTAMPS = False
USE_PICKLED_DWT = False
OVERRIDE_SETTINGS = True

if OVERRIDE_SETTINGS:
    setting['bf_images'] = os.path.join(JOBS_DIR, curr_job, "Phase_Bidi")
    setting['km_path'] = os.path.join(JOBS_DIR, curr_job, "Kymographs")
    setting['fm_images'] = "Fluorescence_Bidi_1"
    setting['cropped_bf_images'] = os.path.join(JOBS_DIR, curr_job, "Phase_images_cropped")
    setting['bf_images_dwt'] = os.path.join(JOBS_DIR, curr_job, "Phase_images_DWT")
    setting['bf_images_dwt_upsampled'] = os.path.join(JOBS_DIR, curr_job, "Phase_images_DWT_upsampled")
    setting['stat_images_cropped'] = os.path.join(JOBS_DIR, curr_job, "Stationary_images_cropped")
    setting['stat_images_dwt_upsampled'] = os.path.join(JOBS_DIR, curr_job, "Stationary_images_DWT_upsampled")
    setting['canon_frames'] = os.path.join(JOBS_DIR, curr_job, "Canonical_heartbeat_frames")
    setting['final_images'] = os.path.join(JOBS_DIR, curr_job, "Final_imageset")
    setting['workspace'] = os.path.join(JOBS_DIR, curr_job, "Workspace")
    setting['approx_fphb'] = 1
    setting['ignore_zooks_at_start'] = 1
    setting['ignore_startzook'] = 7
    setting['ignore_endzook'] = 3
    setting['first_minima'] = 0
    setting['ZookZikPeriod'] = 192
    setting['index_start_at'] = 4001
    setting['frame_count'] = 60000
    setting['stat_index_start_at'] = 1
    setting['stat_frame_count'] = 4000
    setting['image_prefix'] = "Phase_Bidi1_"
    setting['num_digits'] = 5
    setting['canon_frac'] = 10
    setting['bf_fps'] = 100
    setting['slide_limit'] = 5
    setting['y_slide_limit'] = 10
    setting['resampling_factor'] = 5
    setting['time_resampling_factor'] = 4
    setting['BF_resolution'] = 0.6296
    setting['validTiff'] = True
    setting['usePickled'] = False

display_current_settings()
ref_frame_no = 21

if USE_GUI_CORR_WINDOW:
    # Uses GUI to select rectangular window in specified frame.
    # Click and drag with mouse tto select region. Clos window to finalize
    img_path = glob.glob(
        os.path.join(setting['bf_images'], '*.tif')
    )
    img_seq = load_image_sequence([img_path[setting['index_start_at'] + ref_frame_no]])
    max_heartsize_frame(img_seq[0])

    params = get_rect_params()

else:
    params = {}
    # Bottom edge of window
    params['x_end'] = 115
    # Height of window
    params['height'] = 38
    # Right edge of window
    params['y_end'] = 253
    # Width of window
    params['width'] = 37

data = [
    ('frame', ref_frame_no),
    ('x_end', params['x_end']),
    ('height', params['height']),
    ('y_end', params['y_end']),
    ('width', params['width']),
]

print 'Using reference window as-'
print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

pickle_object(data, file_name='corr_window.csv', dumptype='csv')

raw_input('Press enter to continue...')

kymos = glob.glob(os.path.join(setting['km_path'], '*.tif'))
if not USE_PICKLED_ZSTAMPS:
    print '\n'.join(
        map(
            lambda x, y: '%d-%s' % (x, y),
            range(1, len(kymos) + 1),
            kymos
        )
    )
    choice = 0
    while choice not in range(1, len(kymos) + 1):
        choice = int(raw_input('Select kymograph:'))
else:
    choice = 1
z_stamp_opt, y_stamp_opt, z_stamp_cf, res, bad_zooks, minp = z_stamping_step_yz(
    kymo_path=kymos[choice - 1],
    frame_count=setting['frame_count'],
    phase_img_path=setting['bf_images'],
    use_old=USE_PICKLED_ZSTAMPS or setting['usePickled'],
    datafile_name='z_stamp_optimal.pkl',
    datafile_name_x='y_stamp_optimal.pkl'
)

if USE_GUI_CROP_WINDOW:
    img_seq = load_image_sequence([img_path[setting['index_start_at'] + ref_frame_no]])
    masking_window_frame(img_seq[0])
    crop_params = get_rect_params()
else:
    crop_params = {}
    # Bottom edge of window
    crop_params['x_end'] = 319
    # Height of window
    crop_params['height'] = 194
    # Right edge of window
    crop_params['y_end'] = 317
    # Width of window
    crop_params['width'] = 213


data = [
    ('frame', ref_frame_no),
    ('x_end', crop_params['x_end']),
    ('height', crop_params['height']),
    ('y_end', crop_params['y_end']),
    ('width', crop_params['width']),
]

print 'Using cropping window as-'
print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

pickle_object(data, file_name='crop_window.csv', dumptype='csv')

raw_input('Press enter to continue...')

if WRITE_IMAGES_TO_DISK:
    if Y_STAGE_CORRECTION:
        processed_frame_list = shift_frames_and_store_yz(
            img_path=setting['bf_images'],
            z_stamps=z_stamp_opt,
            y_stamps=y_stamp_opt,
            discarded_zooks_list=[],
            minima_points=minp,
            x_end=crop_params['x_end'],
            height=crop_params['height'],
            y_end=crop_params['y_end'],
            width=crop_params['width'],
            ref_frame_no=ref_frame_no
        )
    else:
        processed_frame_list = shift_frames_and_store(
            img_path=setting['bf_images'],
            z_stamp=z_stamp_opt,
            discarded_zooks_list=[],
            minima_points=minp,
            x_end=crop_params['x_end'],
            height=crop_params['height'],
            y_end=crop_params['y_end'],
            width=crop_params['width'],
            ref_frame_no=ref_frame_no
        )

if USE_PICKLED_DWT:
    unpickle_object(dwt_array)
else:
    tic = time.time()
    if WRITE_IMAGES_TO_DISK:
        # Use written images
        moving_dwt_array = compute_dwt_matrix(
            img_path=setting['cropped_bf_images'],
            frame_indices=processed_frame_list,
            write_to=setting['bf_images_dwt']
        )
    else:
        if Y_STAGE_CORRECTION:
            moving_dwt_array = crop_and_compute_dwt_matrix(
                img_path=setting['bf_images'],
                z_stamps=z_stamp_opt,
                y_stamps=y_stamp_opt,
                discarded_zooks_list=[],
                minima_points=minp,
                x_end=crop_params['x_end'],
                height=crop_params['height'],
                y_end=crop_params['y_end'],
                width=crop_params['width'],
                write_to=setting['bf_images_dwt']
            )
        else:
            moving_dwt_array = crop_and_compute_dwt_matrix(
                img_path=setting['bf_images'],
                z_stamps=z_stamp_opt,
                y_stamps=np.zeros(z_stamp_opt.shape),
                discarded_zooks_list=[],
                minima_points=minp,
                x_end=crop_params['x_end'],
                height=crop_params['height'],
                y_end=crop_params['y_end'],
                width=crop_params['width'],
                write_to=setting['bf_images_dwt']
            )
    time_taken = time.time() - tic

    print 'Average DWT computation time:', time_taken / moving_dwt_array.shape[2]

print moving_dwt_array.shape
