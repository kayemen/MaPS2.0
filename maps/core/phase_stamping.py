if __name__ == '__main__':
    import sys
    sys.path.append('D:\\Scripts\\MaPS\\MaPS scripts\\')

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats, signal
import matplotlib.pyplot as plt
from skimage.transform import resize
import pywt

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2, ssim2
from maps.helpers.misc import pickle_object,\
    unpickle_object
from maps.settings import setting, read_setting_from_json
from maps.helpers.gui_modules import cv2_methods, load_frame


import os
import logging
import time
import csv
import cv2
import platform
import collections
import threading

logger = logging.getLogger(__name__)


def compute_zook_extents(maxima_point, minima_point):
    '''
    Function to compute the extent of frames to process in each zook. Added function to clean repeated code. Every zook has some frames ignored at start and end. Given the extent of the zook (start point and end point i.e maxima_point and next minima point) retuns the start and end frame numbers.
    '''
    start_slice = minima_point + setting['ignore_startzook']
    end_slice = maxima_point - setting['ignore_endzook'] + 1

    return (start_slice, end_slice)


def load_and_crop_frames(img_path, frame_no, x_start_frame, x_end_frame, y_start_frame, y_end_frame):
    '''
    Load the tiff file of the frame, resize (upsample by resampling factor if needed) if needed and return image array.
    '''
    # Upsample frame to crop
    cropped_frame = load_frame(img_path, frame_no, upsample=True, crop=True, cropParams=(x_start_frame, x_end_frame, y_start_frame, y_end_frame), index_start_number=setting['stat_index_start_at'])

    # Downsample frame before returning
    cropped_frame_downsized = resize(
        cropped_frame,
        (
            cropped_frame.shape[0] / setting['resampling_factor'],
            cropped_frame.shape[1] / setting['resampling_factor']
        ),
        preserve_range=True
    )

    return cropped_frame_downsized


def chunked_dwt_resize(dwt_array, resize_factor, chunk_size=400):
    '''
    Function to resize a large DWT array on 32 bit systems with moving window to avoid MemoryErrors. Processes the resize in chunks. Each chunk is of length chunk_size. To avoid interpolation errors at the edges of the chunk (chunk borders) there is some overlap. The overlap region is used only for computation of interpolation, and is not stored.
    The main reasons for MemoryError in large resize is due to the size of the coord_map generated while computing resize. Not the actual size of output.
    '''
    max_chunk_size = 400

    # Size of chunks used in the computation
    chunk_size = min(chunk_size, max_chunk_size)

    # No. of frames used for overlap. The larger the overlap, the more extra computation needed.
    overlap = resize_factor - 1

    # Dimensions of array before resizing
    hght = dwt_array.shape[0]
    wdth = dwt_array.shape[1]
    dpth = dwt_array.shape[2]

    # Number of chunks to be processed
    n = int(np.ceil((float(dpth)) / (chunk_size)))

    # Generating result array
    resized_dwt = np.zeros((hght, wdth, dpth * resize_factor))

    # Computing first chunk
    tempres = resize(
        dwt_array[:, :, 0: chunk_size + overlap],
        (hght, wdth, (chunk_size + overlap) * resize_factor),
        preserve_range=True,
        order=5,
        clip=False
    )
    resized_dwt[:, :, : chunk_size * resize_factor] = tempres[:, :, :-(overlap * resize_factor)]
    del tempres

    # Computing chunks before last chunk
    for i in range(1, n - 1):
        start_frame = i * chunk_size - overlap
        end_frame = (i + 1) * chunk_size + overlap

        tempres = resize(
            dwt_array[:, :, start_frame:end_frame],
            (hght, wdth, (chunk_size + 2 * overlap) * resize_factor),
            preserve_range=True,
            order=5,
            clip=False
        )

        resized_dwt[:, :, i * chunk_size * resize_factor: (i + 1) * chunk_size * resize_factor] = tempres[:, :, overlap * resize_factor:-overlap * resize_factor]
        del tempres

    # Commputing last chunk
    tempres = resize(
        dwt_array[:, :, -(chunk_size + overlap):],
        (hght, wdth, (chunk_size + overlap) * resize_factor),
        preserve_range=True,
        order=5,
        clip=False
    )
    resized_dwt[:, :, -chunk_size * resize_factor:] = tempres[:, :, overlap * resize_factor:]
    del tempres

    return resized_dwt


def upsample_dwt_array(dwt_array):
    '''
    Upsampling the dwt array if required. Depending on the architecture of system, will compute resize in single call to `resize` if 64-bit system or will use chunked calls if 32-bit
    '''
    if platform.architecture()[0] == '64bit':
        dwt_array = resize(
            dwt_array,
            (
                dwt_array.shape[0],
                dwt_array.shape[1],
                dwt_array.shape[2] * setting['time_resampling_factor']
            ),
            preserve_range=True,
            order=5,
            clip=False
        )
    else:
        dwt_array = chunked_dwt_resize(
            dwt_array,
            resize_factor=setting['time_resampling_factor'],
            chunk_size=100
        )
    return dwt_array


def write_dwt_matrix(dwt_matrix, write_to, frame_indices, validTiff=False):
    '''
    Write the frames from DWT array to disk. Will write as 64bit float values. Not valid TIFF files
    '''

    if len(frame_indices) < dwt_matrix.shape[2]:
        pass
        # raise Exception('Insufficient frame indices')
    dwt_matrix = dwt_matrix.astype('int16')
    for index, frame_id in enumerate(frame_indices):
        writetiff(dwt_matrix[:, :, index], write_to, index)


def compute_frame_dwt(img_path, frame_id, frame, cropping_params, dwt_array):
    frame_image = load_and_crop_frames(img_path, frame, *cropping_params)
    (cA, _) = pywt.dwt2(frame_image, 'db1', mode='sym')

    dwt_array[:, :, frame_id] = cA


def compute_dwt_matrix(img_path, frame_indices, crop=False, write_to=None, pkl_file_name='dwt_array.pkl'):
    '''
    Compute the DWT matrix for a sequence of images. Optionally writes the DWT array as images to disk. Also writes as pkl file
    '''
    first_image = load_frame(img_path, frame_indices[0])
    (cA, _) = pywt.dwt2(first_image, 'db1', mode='sym')

    dwt_array = np.zeros((cA.shape[0], cA.shape[1], len(frame_indices)))
    for (frame_id, frame) in enumerate(frame_indices):
        frame_image = load_frame(img_path, frame)
        (cA, _) = pywt.dwt2(frame_image, 'db1', mode='sym')

        dwt_array[:, :, frame_id] = cA

    if write_to:
        write_dwt_matrix(dwt_array, write_to, range(dwt_array.shape[2]), validTiff=True)
    pickle_object(dwt_array, pkl_file_name, dumptype='pkl')

    return dwt_array


def crop_and_compute_dwt_matrix(img_path, z_stamps, y_stamps, discarded_zooks_list, minima_points, x_end, height, y_end, width, write_to=None, pkl_file_name='dwt_array.pkl'):
    bad_zooks = [bz[0] for bz in discarded_zooks_list]

    height_resized = height * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']
    x_end_resized = x_end * setting['resampling_factor']
    y_end_resized = y_end * setting['resampling_factor']

    frame_list = []
    cropping_param_list = []

    for zook in np.arange(setting['ignore_zooks_at_start'], len(minima_points)):
        if zook in bad_zooks:
            continue
        maxima_points = (np.array(minima_points) + setting['ZookZikPeriod'] - 1)
        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])

        for frame_no in np.arange(start_frame, end_frame):
            frame_list.append(frame_no)
            y_end_frame = y_end_resized + z_stamps[frame_no]
            y_start_frame = y_end_frame - width_resized
            x_end_frame = x_end_resized + y_stamps[frame_no]
            x_start_frame = x_end_frame - width_resized

            cropping_param_list.append(
                (x_start_frame,
                 x_end_frame,
                 y_start_frame,
                 y_end_frame
                 )
            )

    dwt_threads = []

    first_image = load_and_crop_frames(img_path, frame_list[0], *cropping_param_list[0])

    (cA, _) = pywt.dwt2(first_image, 'db1', mode='sym')

    dwt_array = np.zeros((cA.shape[0], cA.shape[1], len(frame_list)))
    for (frame_id, frame) in enumerate(frame_list):
        print 'Processing frame #%d' % frame
        dwt_thread = threading.Thread(
            target=compute_frame_dwt,
            args=(
                img_path,
                frame_id,
                frame,
                cropping_param_list[frame_id],
                dwt_array
            )
        )
        dwt_thread.start()
        dwt_threads.append(dwt_thread)

    # Waiting for threads to end
    for dwt_thread in dwt_threads:
        dwt_thread.join()

    if write_to:
        write_dwt_matrix(dwt_array, write_to, range(dwt_array.shape[2]), validTiff=True)
    pickle_object(dwt_array, pkl_file_name, dumptype='pkl')

    return dwt_array


def compute_heartbeat_length(dwt_array, method='ccoeff_norm', comparison_plot=True, im_fps=480, nperseg=3000):
    '''
    Compute the length of the heartbeat. Uses the DWT of the
    '''
    color = iter(plt.cm.rainbow(np.linspace(0, 1, int(dwt_array.shape[2]))))

    time_taken = []

    hb_array = []
    # TODO
    for ref_frame in range(dwt_array.shape[2])[:100]:
        first_image = dwt_array[:, :, ref_frame]
        tic = time.time()
        x = []
        for i in range(dwt_array.shape[2]):
            second_image = dwt_array[:, :, i]
            # Corr2 method
            if method == 'corr':
                x.append(corr2(first_image, second_image))
            # SSIM
            elif method == 'ssim':
                x.append(ssim2(first_image, second_image))
            # Abs Diff
            # TODO: Will require filtering
            elif method == 'sad':
                x.append(np.sum(np.absolute(first_image - second_image)))
            # OpenCV methods ccorr, ccoeff
            elif method in ('ccoeff_norm', 'ccorr_norm', 'mse_norm'):
                x.append(cv2.matchTemplate(second_image.astype('float32'), first_image.astype('float32'), cv2_methods[method])[0][0])
        time_taken.append(time.time() - tic)
        # print x

        # For absdiff and MSE
        # if method in ('sad', 'mse_norm'):
        x = np.asarray(x)
        x = np.max(x) - x

        f, Pxx = signal.welch(x, im_fps, nperseg=nperseg)

        t = im_fps / f
        # if comparison_plot:
        #     plt.figure(211)
        #     plt.plot(x)
        #     print t[np.argmax(Pxx) - 10: np.argmax(Pxx) + 10]
        #     comparison_plot = False
        # raw_input()
        Pxx[np.where(t > 400)] = 0
        hb_array.append(t[np.argmax(Pxx)])

    cm_hb = stats.mode(hb_array)
    print 'Common heartbeat period-', cm_hb[0]
    print 'Occurs %d times' % cm_hb[1]
    plt.show()
    print hb_array
    print 'average time taken - ', sum(time_taken) / len(time_taken)

    return cm_hb[0]


def compute_canonical_heartbeat(dwt_array, mean_hb_period, sad_matrix_pkl_name='sad_matrix.pkl'):
    '''
    Processes interpolated frame wavelets. find the best continuous sequence of heartbeats
    '''
    # Populating SAD matrix
    use_pickled_sad = False

    sad_matrix = np.zeros((dwt_array.shape[2], dwt_array.shape[2]))

    threshold = 0.02

    if use_pickled_sad:
        sad_matrix = unpickle_object(sad_matrix_pkl_name)
    else:
        time_taken = []
        for i in range(dwt_array.shape[2]):
            # TODO: Threads
            ref_dwt = dwt_array[:, :, i]
            print 'Processing frame %d' % i
            tic = time.time()
            for j in range(i, dwt_array.shape[2]):
                curr_dwt = dwt_array[:, :, j]
                sad_matrix[i, j] = sad_matrix[j, i] = np.sum(
                    np.absolute(ref_dwt - curr_dwt)
                )
            time_taken.append(time.time() - tic)
            print time_taken[-1]

        pickle_object(sad_matrix, sad_matrix_pkl_name)

    # TODO: Check other values of threshold
    threshold = 0.5 * np.max(sad_matrix)
    start_frame = 0
    num_contiguous_frames = setting['ZookZikPeriod'] - (setting['ignore_startzook'] + setting['ignore_endzook'])

    # print num_contiguous_frames * setting['canonical_zook_count']
    # raw_input()

    mean_clustered_match = np.ones(num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
    stddev_clustered_match = np.ones(num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
    mean2std_clustered_match = np.zeros(num_contiguous_frames * setting['canonical_zook_count'])

    clustered_match_array = []
    match_mat_array = []

    for zook in range(setting['canonical_zook_count'] - 2):
        for frame in range(start_frame, start_frame + num_contiguous_frames - mean_hb_period):
            matching_frames = sad_matrix[frame: frame + mean_hb_period, :]

            try:
                min_pos_in_col = np.argmin(matching_frames, axis=0)
            except:
                print start_frame
                print num_contiguous_frames
                print frame
                print matching_frames.shape

            match_mat = np.zeros(matching_frames.shape)

            for col in range(len(min_pos_in_col)):
                if matching_frames[min_pos_in_col[col], col] < threshold:
                    match_mat[min_pos_in_col[col], col] = 1
                # else:
                #     print 'Threshold mismatch:', matching_frames[min_pos_in_col[col], col]

            clustered_match = np.sum(match_mat, axis=1)

            # plt.plot(clustered_match)
            # plt.show()

            mean_clustered_match[frame] = np.mean(clustered_match)
            stddev_clustered_match[frame] = np.std(clustered_match)
            mean2std_clustered_match[frame] = mean_clustered_match[frame] / stddev_clustered_match[frame]

            # print mean2std_clustered_match[frame]

            clustered_match_array.append(clustered_match)
            match_mat_array.append(match_mat)

        start_frame += num_contiguous_frames

    print np.argmax(mean2std_clustered_match)
    print np.max(mean2std_clustered_match)

    return np.argmax(mean2std_clustered_match)


# TODO: replace with class
PhaseStamp = collections.namedtuple('PhaseStamp', ['phase', 'matchval'])


def phase_stamp_images(dwt_array, canonical_hb_dwt):
    phase_stamps = []

    for frame in range(dwt_array.shape[2]):
        curr_frame = dwt_array[:, :, frame]

        sad_array = np.zeros(canonical_hb_dwt.shape[2])

        for phase in range(canonical_hb_dwt.shape[2]):
            canon_frame = canonical_hb_dwt[:, :, phase]
            sad_array[phase] = np.sum(np.abs(curr_frame - canon_frame))

        best_phase = np.argmax(sad_array)
        phase_stamps.append(PhaseStamp(phase=best_phase, matchval=sad_array[best_phase]))

    return phase_stamps


def compile_phase_z_matrix(framelist, z_stamps, phase_stamps, hb_length):
    phase_z_matrix = np.empty((hb_length, len(framelist)), dtype=np.object)

    for i in range(hb_length):
        for j in range(len(framelist)):
            phase_z_matrix[i, j] = list()

    for index, frame in enumerate(framelist):
        phase_z_matrix[phase_stamps[index].phase, z_stamps[index]].append((frame, phase_stamps[index].matchval))


def phase_stamping_step():
    frame_list = unpickle_object('processed_frame_list')
    z_stamps = unpickle_object('z_stamp_opt')

    moving_dwt_array = compute_dwt_matrix(
        img_path=setting['cropped_bf_images'],
        frame_indices=frame_list,
        write_to=setting['bf_images_dwt'],
        pkl_file_name='moving_dwt_array'
    )

    canon_frame_count = setting['canon_frac'] / 100.0 * len(frame_list)
    canon_dwt_array = upsample_dwt_array(
        moving_dwt_array[:, :, canon_frame_count]
    )[:, :, 2:-2]

    # TODO: Z stamp and crop stationary images, not use whole
    stat_dwt_array = compute_dwt_matrix(
        img_path=setting['bf_images'],  # Assuming stationary images are taken in the same imaging session as moving images
        # img_path=setting['stat_images_cropped'],
        frame_indices=range(
            setting['stat_index_start_at'],
            setting['stat_index_start_at'] + setting['stat_frame_count']
        ),
        write_to=setting['stat_images_dwt_upsampled'],
        pkl_file_name='stat_dwt_array.pkl'
    )[:, :, 2:-2]

    canon_hb_length = compute_heartbeat_length(
        stat_dwt_array,
        comparison_plot=True,
        nperseg=50000
    )

    canon_hb_start = compute_canonical_heartbeat(
        canon_dwt_array,
        np.round(canon_hb_length).astype('int')
    )

    canonical_frame_list = frame_list[canon_hb_start: canon_hb_start + canon_hb_length]

    # TODO: Write canonical frames to disk

    phi_stamps = phase_stamp_images(
        moving_dwt_array,
        moving_dwt_array[canon_hb_start: canon_hb_start + canon_hb_length]
    )

    compile_phase_z_matrix(frame_list, z_stamps, phi_stamps, canon_hb_length)


if __name__ == '__main__':
    import pickle
    from maps.helpers.logging_config import logger_config
    logging.config.dictConfig(logger_config)

    logger = logging.getLogger('maps.core.phase_stamping')

    # Path to settings json for preloading settings
    settings_json = 'D:\\Scripts\\MaPS\\MaPS scripts\\maps\\current_inputs.json'

    # Initialize the settings object
    read_setting_from_json(settings_json)

    setting['ignore_zooks_at_start'] = 0
    # setting['ignore_startzook'] = 0
    # setting['ignore_endzook'] = 0
    setting['BF_resolution'] = 0.6296
    setting['image_prefix'] = 'FRAMEX'
    setting['bf_period_path'] = 'D:\\Scripts\\MaPS\\Data sets\\Phase_cropped_old\\'
    setting['bf_period_dwt_path'] = 'D:\\Scripts\\MaPS\\Data sets\\Upsampled_BF_DWT'
    setting['workspace'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
    setting['resampling_factor'] = 5
    setting['slide_limit'] = 5
    # setting['ZookZikPeriod'] = 728
    setting['index_start_at'] = 1
    setting['num_digits'] = 5
    setting['first_minima'] = 0

    with open(os.path.join(setting['workspace'], 'processed_frame_list.pkl')) as pkfp:
        frames = pickle.load(pkfp)

    COMPUTE_DWT = True

    if COMPUTE_DWT:
        d = compute_dwt_matrix(setting['bf_period_path'], frames[:2000],
                               #    write_to=setting['bf_period_dwt_path'],
                               upsample=True)
    else:
        d = unpickle_object('dwt_array.pkl')

    # cm_hb = compute_heartbeat_length(d[:, :, 2:-2], comparison_plot=True, nperseg=50000)
    cm_hb = 94

    # print frames
    # compute_dwt_matrix(setting['bf_crop_path'], frames[:728], write_to=setting['bf_us_dwt_path'], pkl_file_name='dwt_array_bf_us.pkl', upsample=True)
    canonical_hb_start = compute_canonical_heartbeat(d[:, :, 2:-2], np.round(cm_hb).astype('int'))
    # #
    canonical_hb_frames = frames[canonical_hb_start: canonical_hb_start + np.round(cm_hb).astype('int')]
    print canonical_hb_frames
