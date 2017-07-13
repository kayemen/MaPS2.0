import matplotlib.pyplot as plt

from maps.core.z_stamping import load_kymograph,\
    load_correlation_window_params,\
    load_frame,\
    extract_window,\
    compute_raw_zstage,\
    compute_maxima_minima,\
    compute_zstamp,\
    compute_ideal_zstamp\

# from maps.core.z_stamping import z_stamping_step, shift_frames_and_store
from maps.settings import read_setting_from_json, setting
from maps.helpers.misc import pickle_object
from maps.helpers.img_proccessing import corr2
from maps.helpers.tiffseriesimport import writetiff
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame

import numpy as np
import logging
import os
import glob
import cv2
import time

# logging.config.dictConfig(logger_config)
logger = logging.getLogger('MaPS')

# Path to settings json for preloading settings
settings_json = 'D:\\Scripts\\MaPS\\MaPS scripts\\maps\\current_inputs.json'

# Initialize the settings object
read_setting_from_json(settings_json)

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
frame_count = 400

# Folder containing brightfield images for phase stamping. The images must be
# named sequentially
phase_image_folder = 'D:\\Scripts\\MaPS\\Data sets\\Phase_Bidi\\'

frame_no = 21

img_path = glob.glob(
    os.path.join(phase_image_folder, '*.tif')
)

# Uses GUI to select rectangular window in specified frame.
# Click and drag with mouse tto select region. Clos window to finalize
img_seq = load_image_sequence([img_path[frame_no]])
# max_heartsize_frame(img_seq[0])
#
# params = get_rect_params()
#
# x_end = params['x_end']
# height = params['height']
# y_end = params['y_end']
# width = params['width']
#
# data = [
#     ('frame', frame_no),
#     ('x_end', x_end),
#     ('height', height),
#     ('y_end', y_end),
#     ('width', width),
# ]
#
# print 'Using reference window as-'
# print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])
#
# pickle_object(data, file_name='corr_window.csv', dumptype='csv')

# z_stamp_opt, z_stamp_cf, res, bad_zooks, minp = z_stamping_step(
#     kymo_path=kymograph_path,
#     frame_count=frame_count,
#     phase_img_path=phase_image_folder,
#     use_old=use_existing_datadump_vals,
#     datafile_name='z_stamp_opt_KM-XZ_0.pkl'
# )

kymo_data = load_kymograph(kymograph_path)
# Load window parameters
x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_correlation_window_params('corr_window.csv')

z_stage_data = compute_raw_zstage(kymo_data[:frame_count, :])

(maxp, maxv, minp, minv) = compute_maxima_minima(z_stage_data)

# compute_zookzik_stats(maxp, minp)
z_stamp, _ = compute_zstamp(z_stage_data, maxp, minp, frame_no)
z_stamp_det = compute_ideal_zstamp(z_stage_data, maxp, minp, frame_no)

# Starting correlation computation
x_end_resized = x_end * setting['resampling_factor']
height_resized = height * setting['resampling_factor']
x_start_resized = x_end_resized - height_resized
# y_start_resized = y_start * setting['resampling_factor']
y_end_resized = y_end * setting['resampling_factor']
width_resized = width * setting['resampling_factor']
y_start_resized = y_end_resized - width_resized

ref_frame = load_frame(phase_image_folder, frame_no)

ref_frame_window = extract_window(ref_frame, x_start_resized, x_end_resized, y_start_resized, y_end_resized)

cv2_methods = ['cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_SQDIFF_NORMED']

slide_limit_resized = setting['slide_limit'] * setting['resampling_factor']

optimal_shifts_x = np.asarray([])
optimal_shifts_y = np.asarray([])

for frame in np.arange(0, frame_count):
    y_end_resized_frame = y_end_resized + z_stamp_det[frame]
    y_start_resized_frame = y_end_resized_frame - width_resized

    curr_frame = load_frame(phase_image_folder, frame)

    tic = time.time()

    # OpenCV method
    curr_frame_window = curr_frame[x_start_resized - 75:x_end_resized + 75, y_start_resized_frame - slide_limit_resized: y_end_resized_frame + slide_limit_resized]
    #
    # # plt.figure(111)
    # # plt.imshow(curr_frame_window, cmap=plt.cm.gray)
    # # plt.figure(112)
    # # plt.imshow(ref_frame_window, cmap=plt.cm.gray)
    # # plt.show()
    #
    corr_coeffs_alt = cv2.matchTemplate(curr_frame_window.astype('float32'), ref_frame_window.astype('float32'), eval(cv2_methods[1]))

    print 'Time taken(opencv)-', time.time() - tic

    tic = time.time()
    # corr2 method
    corr_coeffs = np.zeros((150, 50))
    for xslide in np.arange(-75, 75):
        x_end_resized_shift = x_end_resized + xslide
        x_start_resized_shift = x_end_resized_shift - height_resized

        for yslide in np.arange(-25, 25):
            y_end_resized_shift = y_end_resized_frame + yslide
            y_start_resized_shift = y_end_resized_shift - width_resized

            curr_frame_window = extract_window(curr_frame, x_start_resized_shift, x_end_resized_shift, y_start_resized_shift, y_end_resized_shift)

            corr_coeffs[xslide + 75, yslide + 25] = corr2(ref_frame_window, curr_frame_window)

    print 'Time taken(corr2)-', time.time() - tic

    # print corr_coeffs
    print np.argmax(corr_coeffs_alt - corr_coeffs)
    print np.max(corr_coeffs_alt - corr_coeffs)

    plt.imshow(np.absolute(corr_coeffs-corr_coeffs_alt)/np.max(corr_coeffs-corr_coeffs_alt), cmap=plt.cm.gray)
    plt.show()
    # plt.savefig(
    #     os.path.join(
    #         setting['data_dump'],
    #         'convex_check\\frame%d.tif' % (frame)
    #     ),
    #     bbox_inches='tight'
    # )
    # plt.gcf().clear()

    optimal_shifts_x = np.append(optimal_shifts_x, corr_coeffs_alt.max(axis=1).argmax()-75)
    optimal_shifts_y = np.append(optimal_shifts_y, corr_coeffs_alt.max(axis=0).argmax()-25)
    # print 'Similarity-', corr2(corr_coeffs, corr_coeffs_alt)
    #
    # writetiff((corr_coeffs*65535).astype('uint16'), os.path.join(setting['data_dump'], 'convex_check_corr2'), frame)
    # writetiff((corr_coeffs_alt*65535).astype('uint16'), os.path.join(setting['data_dump'], 'convex_check_opencv'), frame)
    # writetiff((np.power(corr_coeffs_alt, 25)*65535).astype('uint16'), os.path.join(setting['data_dump'], 'convex_check_opencv_gamma'), frame)


plt.plot(optimal_shifts_x, 'b')
plt.plot(optimal_shifts_y, 'g')
plt.show()
