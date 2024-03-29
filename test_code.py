from maps.core.z_stamping_v2 import compute_raw_zstage, compute_zooks,\
    compute_zookzik_stats, drop_bad_zooks_zstage, compute_zstamp,\
    compute_deterministic_zstamp, compute_optimal_yz_stamp,\
    compute_zstamp_curvefit, detect_bad_zooks, shift_frames_and_store_yz,\
    Zook, Zooks
# from maps.core.z_stamping import z_stamping_step_yz, shift_frames_and_store_yz
from maps.core.phase_stamping_v2 import crop_and_compute_dwt_matrix,\
    compute_dwt_matrix, load_dwt_matrix, interchelate_dwt_array,\
    compute_heartbeat_length, compute_canonical_heartbeat,\
    phase_stamp_images, compile_phase_z_matrix,\
    write_phase_stamped_fluorescent_images

from maps import settings
from maps.settings import setting, read_setting_from_json, \
    read_setting_from_csv
from maps.helpers.misc import pickle_object, unpickle_object
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame

import glob
import os
import code
import matplotlib.pyplot as plt

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

OVERRIDE_SETTINGS = False

if OVERRIDE_SETTINGS:
    setting['bf_images'] = os.path.join(JOBS_DIR, curr_job, "Phase_Bidi")
    setting['km_path'] = os.path.join(JOBS_DIR, curr_job, "Kymographs")
    setting['fm_images'] = "Fluorescence_Bidi_1"
    setting['cropped_bf_images'] = os.path.join(
        JOBS_DIR, curr_job, "Phase_images_cropped")
    setting['bf_images_dwt'] = os.path.join(
        JOBS_DIR, curr_job, "Phase_images_DWT")
    setting['bf_images_dwt_upsampled'] = os.path.join(
        JOBS_DIR, curr_job, "Phase_images_DWT_upsampled")
    setting['stat_images_cropped'] = os.path.join(
        JOBS_DIR, curr_job, "Stationary_images_cropped")
    setting['stat_images_dwt_upsampled'] = os.path.join(
        JOBS_DIR, curr_job, "Stationary_images_DWT_upsampled")
    setting['canon_frames'] = os.path.join(
        JOBS_DIR, curr_job, "Canonical_heartbeat_frames")
    setting['final_images'] = os.path.join(
        JOBS_DIR, curr_job, "Final_imageset")
    setting['workspace'] = os.path.join(JOBS_DIR, curr_job, "Workspace")
    setting['approx_fphb'] = 1
    setting['ignore_zooks_at_start'] = 1
    setting['ignore_startzook'] = 7
    setting['ignore_endzook'] = 3
    setting['first_minima'] = 0
    setting['ZookZikPeriod'] = 192
    setting['index_start_at'] = 4001
    setting['frame_count'] = 10000
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

pkl_files = glob.glob(os.path.join(setting['workspace'], '*.pkl'))
pkl_files = [os.path.basename(fn).split(
    '/')[-1].replace('.pkl', '') for fn in pkl_files]

print 'Available variables'
print '-' * 80
print '\n'.join(pkl_files)

skip_list = [
    'dwt_array'
]

loaded_variables = []

for pkl_file in pkl_files:
    load = True
    for skip in skip_list:
        if skip in pkl_file:
            load = False
            break
    if load:
        print 'Loading %s' % pkl_file
        exec("%s=unpickle_object('%s')" % (pkl_file, pkl_file))
        print 'Loaded %s' % pkl_file
        loaded_variables.append(pkl_file)
# Start interactive shell
code.interact(local=locals())
# import cv2



# corr_val_cv_mask

# corr_val_cv_nomask

# corr_val_ip

# corr_val_cv_multmask