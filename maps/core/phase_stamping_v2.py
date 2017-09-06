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
from maps.settings import setting, read_setting_from_json, THREADED
from maps.helpers.gui_modules import cv2_methods, load_frame, extract_window
from maps.helpers.logging_config import logger_config
from maps.helpers.misc import LimitedThread
from maps import settings

import os
import logging
import time
import csv
import cv2
import platform
import collections
import threading
import shutil
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError

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
    cropped_frame = load_frame(
        img_path, frame_no, upsample=True, index_start_number=setting['stat_index_start_at'])

    cropped_frame = extract_window(
        cropped_frame, x_start_frame, x_end_frame, y_start_frame, y_end_frame)

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

    # No. of frames used for overlap. The larger the overlap, the more extra
    # computation needed.
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
    resized_dwt[:, :, : chunk_size * resize_factor] = tempres[:,
                                                              :, :-(overlap * resize_factor)]
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

        resized_dwt[:, :, i * chunk_size * resize_factor: (i + 1) * chunk_size * resize_factor] = tempres[
            :, :, overlap * resize_factor:-overlap * resize_factor]
        del tempres

    # Commputing last chunk
    tempres = resize(
        dwt_array[:, :, -(chunk_size + overlap):],
        (hght, wdth, (chunk_size + overlap) * resize_factor),
        preserve_range=True,
        order=5,
        clip=False
    )
    resized_dwt[:, :, -chunk_size * resize_factor:] = tempres[:,
                                                              :, overlap * resize_factor:]
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
    return dwt_array[:, :, setting['time_resampling_factor'] - 1: -setting['time_resampling_factor'] + 1]


def write_dwt_matrix(dwt_matrix, write_to, frame_indices, validTiff=False):
    '''
    Write the frames from DWT array to disk. Will write as 64bit float values. Not valid TIFF files
    '''

    if len(frame_indices) < dwt_matrix.shape[2]:
        pass
        # raise Exception('Insufficient frame indices')
    if validTiff:
        dwt_matrix = dwt_matrix.astype('int16')

    dwt_threads = []

    if not settings.MULTIPROCESSING:
        map(
            lambda x: writetiff(
                dwt_matrix[:, :, x[1]],
                write_to,
                x[0]
            ),
            enumerate(frame_indices)
        )
    else:
        proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
        temp_var = proc_pool.map_async(
            lambda x: writetiff(
                dwt_matrix[:, :, x[1]],
                write_to,
                x[0]
            ),
            enumerate(frame_indices),
            settings.NUM_CHUNKS
        )

        try:
            temp_var.get(timeout=100)
        except TimeoutError:
            logging.info('Frame timed out in %d seconds' %
                         (settings.TIMEOUT))

    # for index, frame_id in enumerate(frame_indices):
    #     # print('Writing DWT frame #%d' % frame_id)
    #     if THREADED:
    #         dwt_threads.append(
    #             LimitedThread(
    #                 target=writetiff,
    #                 args=(dwt_matrix[:, :, index], write_to, index)
    #             )
    #         )
    #         dwt_threads[-1].start()
    #     else:
    #         writetiff(dwt_matrix[:, :, index], write_to, index)

    # for th in dwt_threads:
    #     th.join()


def load_dwt_frame(dwt_path, dwt_array, frame_index):
    # print frame_index
    dwt_frame = importtiff(dwt_path, frame_index, index_start_number=0)
    dwt_array[:, :, frame_index] = dwt_frame


def load_dwt_matrix(load_from, num_frames=None):
    first_frame = importtiff(load_from, 0)
    num_frames = len(os.listdir(load_from)
                     ) if num_frames is None else num_frames
    dwt_array = np.zeros(
        (first_frame.shape[0], first_frame.shape[1], num_frames))

    dwt_load_threads = []

    if not settings.MULTIPROCESSING:
        map(
            lambda x: load_dwt_frame(
                load_from,
                dwt_array,
                x
            ),
            range(num_frames)
        )
    else:
        proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
        temp_var = proc_pool.map_async(
            lambda x: load_dwt_frame(
                load_from,
                dwt_array,
                x
            ),
            range(num_frames),
            settings.NUM_CHUNKS
        )

        try:
            temp_var.get(timeout=100)
        except TimeoutError:
            logging.info('Frame timed out in %d seconds' %
                         (settings.TIMEOUT))

    # for index in xrange(num_frames):
    #     if THREADED:
    #         dwt_load_threads.append(
    #             LimitedThread(
    #                 target=load_dwt_frame,
    #                 args=(
    #                     load_from,
    #                     dwt_array,
    #                     index
    #                 )
    #             )
    #         )
    #         dwt_load_threads[-1].start()
    #     else:
    #         load_dwt_frame(load_from, dwt_array, index)

    # for th in dwt_load_threads:
    #     th.join()

    return dwt_array


def compute_frame_dwt(img_path, frame_id, frame, cropping_params, dwt_array):
    frame_image = load_and_crop_frames(img_path, frame, *cropping_params)
    (cA, _) = pywt.dwt2(frame_image, 'db1', mode='sym')

    dwt_array[:, :, frame_id] = cA


def compute_dwt_matrix(img_path, frame_indices, crop=False, write_to=None, pkl_file_name=None, validTiff=False):
    '''
    Compute the DWT matrix for a sequence of images. Optionally writes the DWT array as images to disk. Also writes as pkl file
    '''
    # TODO: Change frame indices to zook objects
    first_image = load_frame(img_path, frame_indices[
                             0], prefix='FRAMEX', index_start_number=0, upsample=False, num_digits=5)
    (cA, _) = pywt.dwt2(first_image, 'db1', mode='sym')

    dwt_array = np.zeros((cA.shape[0], cA.shape[1], len(frame_indices)))
    for (frame_id, frame) in enumerate(frame_indices):
        # print('DWT of frame %d' % frame)
        # TODO: Add threading here
        frame_image = load_frame(img_path, frame, prefix='FRAMEX',
                                 index_start_number=0, upsample=False, num_digits=5)
        (cA, _) = pywt.dwt2(frame_image, 'db1', mode='sym')

        dwt_array[:, :, frame_id] = cA

    writing_threads = []

    if write_to:
        if THREADED:
            writing_threads.append(
                LimitedThread(
                    target=write_dwt_matrix,
                    args=(
                        dwt_array,  # dwt_matrix
                        write_to,  # write_to
                        range(dwt_array.shape[2]),  # frame_indices
                        False if pkl_file_name is None else validTiff  # validTiff
                    )
                )
            )
            writing_threads[-1].start()
        else:
            write_dwt_matrix(
                dwt_array,  # dwt_matrix
                write_to,  # write_to
                range(dwt_array.shape[2]),  # frame_indices
                False if pkl_file_name is None else validTiff  # validTiff
            )

    for writing_thread in writing_threads:
        writing_thread.join()

    if pkl_file_name:
        print 'Pickling DWT array'
        pickle_object(dwt_array, pkl_file_name, dumptype='pkl')

    return dwt_array


def crop_and_compute_dwt_matrix(img_path, z_stamps, y_stamps, discarded_zooks_list, minima_points, x_end, height, y_end, width, write_to=None, pkl_file_name=None, validTiff=True):
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
        maxima_points = (np.array(minima_points) +
                         setting['ZookZikPeriod'] - 1)
        start_frame, end_frame = compute_zook_extents(
            maxima_points[zook], minima_points[zook])

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

    pickle_object(frame_list, 'processed_frame_list')

    dwt_threads = []

    first_image = load_and_crop_frames(
        img_path, frame_list[0], *cropping_param_list[0])

    (cA, _) = pywt.dwt2(first_image, 'db1', mode='sym')

    dwt_array = np.zeros((cA.shape[0], cA.shape[1], len(frame_list)))
    print 'Computing DWT of frames'
    for (frame_id, frame) in enumerate(frame_list):
        if THREADED:
            dwt_thread = LimitedThread(
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
        else:
            compute_frame_dwt(
                img_path,
                frame_id,
                frame,
                cropping_param_list[frame_id],
                dwt_array
            )

    # Waiting for threads to end
    for dwt_thread in dwt_threads:
        dwt_thread.join()

    writing_threads = []

    if write_to:
        if THREADED:
            writing_threads.append(
                LimitedThread(
                    target=write_dwt_matrix,
                    args=(
                        dwt_array,  # dwt_matrix
                        write_to,  # write_to
                        range(dwt_array.shape[2]),  # frame_indices
                        False if pkl_file_name is None else validTiff  # validTiff
                    )
                )
            )
            writing_threads[-1].start()
        else:
            write_dwt_matrix(
                dwt_array,  # dwt_matrix
                write_to,  # write_to
                range(dwt_array.shape[2]),  # frame_indices
                False if pkl_file_name is None else validTiff  # validTiff
            )

    for writing_thread in writing_threads:
        writing_thread.join()

    if pkl_file_name:
        print 'Pickling DWT array-started at', time.asctime()
        pickle_object(dwt_array, pkl_file_name, dumptype='pkl')

    return dwt_array


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
    return dwt_array[:, :, setting['time_resampling_factor'] - 1: -setting['time_resampling_factor'] + 1]


def interchelate_dwt_array(dwt_array, zooks=None, write_to=None, pkl_file_name=None, validTiff=False):
    # TODO: Perform upsampling zook by zook
    # print dwt_array.shape
    if zooks is not None:
        zook_gen = zooks.get_dwt_framelist(upsample=True)
        us_dwt_array = upsample_dwt_array(dwt_array[:, :, zook_gen.next()])
        for zook_lim in zook_gen:
            print zook_lim
            us_dwt_array = np.dstack(
                (us_dwt_array,
                 upsample_dwt_array(dwt_array[:, :, zook_lim]))
            )
            print us_dwt_array.shape
    else:
        us_dwt_array = upsample_dwt_array(dwt_array)

    writing_threads = []

    if write_to:
        if THREADED:
            writing_threads.append(
                LimitedThread(
                    target=write_dwt_matrix,
                    args=(
                        us_dwt_array,  # dwt_matrix
                        write_to,  # write_to
                        range(us_dwt_array.shape[2]),  # frame_indices
                        False if pkl_file_name is None else validTiff  # validTiff
                    )
                )
            )
            writing_threads[-1].start()
        else:
            write_dwt_matrix(
                us_dwt_array,  # dwt_matrix
                write_to,  # write_to
                range(us_dwt_array.shape[2]),  # frame_indices
                False if pkl_file_name is None else validTiff  # validTiff
            )

    for writing_thread in writing_threads:
        writing_thread.join()

    if pkl_file_name:
        print 'Pickling upsampled DWT array-started at', time.asctime()
        pickle_object(us_dwt_array, pkl_file_name, dumptype='pkl')

    return us_dwt_array


def compute_peak_harmonic(first_image, second_image, method, corr_array):
    # Corr2 method
    if method == 'corr':
        corr_array.append(corr2(first_image, second_image))
    # SSIM
    elif method == 'ssim':
        corr_array.append(ssim2(first_image, second_image))
    # Abs Diff
    # TODO: Will require filtering
    elif method == 'sad':
        corr_array.append(np.sum(np.absolute(first_image - second_image)))
    # OpenCV methods ccorr, ccoeff
    elif method in ('ccoeff_norm', 'ccorr_norm', 'mse_norm'):
        corr_array.append(cv2.matchTemplate(
            second_image.astype('float32'),
            first_image.astype('float32'), cv2_methods[method])[0][0]
        )


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
        # for i in range(dwt_array.shape[2]):
        #     second_image = dwt_array[:, :, i]
        if not settings.MULTIPROCESSING:
            map(
                lambda frame: compute_peak_harmonic(
                    first_image,
                    dwt_array[:, :, frame],
                    method,
                    x
                ),
                range(dwt_array.shape[2])
            )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame: compute_peak_harmonic(
                    first_image,
                    dwt_array[:, :, frame],
                    method,
                    x
                ),
                range(dwt_array.shape[2]),
                settings.NUM_CHUNKS
            )

            try:
                temp_var.get(timeout=settings.TIMEOUT)
            except TimeoutError:
                logging.info('Zook %d timed out in %d seconds' %
                             (zook.id, settings.TIMEOUT))

            del proc_pool

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

    pickle_object(cm_hb[0], 'common_heatbeat_period.pkl')
    pickle_object(time_taken, 'common_heatbeat_calctime.pkl')

    return cm_hb[0]


def compute_sad(dwt_array, ref_dwt, i, sad_matrix):
    for j in range(i, dwt_array.shape[2]):
        curr_dwt = dwt_array[:, :, j]

        sad_matrix[i, j] = sad_matrix[j, i] = np.sum(
            np.absolute(ref_dwt - curr_dwt)
        )
    # print 'row %d done' % i


def compute_canonical_heartbeat(dwt_array, zooks, mean_hb_period, sad_matrix_pkl_name=None, write_to=None, use_pickled_sad=False):
    '''
    Processes interpolated frame wavelets. find the best continuous sequence of heartbeats
    '''
    sad_matrix = np.zeros((dwt_array.shape[2], dwt_array.shape[2]))

    threshold = 0.02

    if use_pickled_sad:
        sad_matrix = unpickle_object(sad_matrix_pkl_name)
    else:
        # time_taken = []
        tic = time.time()
        sad_threads = []

        if not settings.MULTIPROCESSING:
            map(
                lambda frame_no: compute_sad(
                    dwt_array,
                    dwt_array[:, :, frame_no],
                    frame_no,
                    sad_matrix
                ),
                range(dwt_array.shape[2])
            )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame_no: compute_sad(
                    dwt_array,
                    dwt_array[:, :, frame_no],
                    frame_no,
                    sad_matrix
                ),
                range(dwt_array.shape[2]),
                settings.NUM_CHUNKS
            )

            timeout_sad = settings.TIMEOUT * 150
            try:
                temp_var.get(timeout=timeout_sad)
            except TimeoutError:
                print temp_var._number_left, 'frames left'
                logging.info('SAD generation timed out in %d seconds' %
                             (timeout_sad))
                logging.info('%d frames did not get processed' %
                             (temp_var._number_left))

            del proc_pool

        print 'Time taken for SAD matrix generation:', time.time() - tic
        if sad_matrix_pkl_name:
            pickle_object(sad_matrix, sad_matrix_pkl_name)
            print 'sad_matrix pickled'

    # TODO: Check other values of threshold
    threshold = 0.5 * np.max(sad_matrix)
    start_frame = 0
    num_contiguous_frames = (setting[
        'ZookZikPeriod'] - (setting['ignore_startzook'] + setting['ignore_endzook'])) * setting['time_resampling_factor']

    # print num_contiguous_frames * setting['canonical_zook_count']
    # raw_input()

    mean_clustered_match = np.ones(
        num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
    stddev_clustered_match = np.ones(
        num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
    mean2std_clustered_match = np.zeros(
        num_contiguous_frames * setting['canonical_zook_count'])

    clustered_match_array = []
    match_mat_array = []

    # print mean_hb_period
    for zook in range(setting['canonical_zook_count'] - 2):
        print range(int(start_frame) + (num_contiguous_frames // 4), int(start_frame + num_contiguous_frames - mean_hb_period) - (num_contiguous_frames // 4))
        for frame in range(int(start_frame), int(start_frame + num_contiguous_frames - mean_hb_period)):
            # for frame in range(int(start_frame) + (num_contiguous_frames //
            # 4), int(start_frame + num_contiguous_frames - mean_hb_period) -
            # (num_contiguous_frames // 4)):
            matching_frames = sad_matrix[frame: frame + mean_hb_period, :]

            try:
                min_pos_in_col = np.argmin(matching_frames, axis=0)
            except:
                print start_frame
                print num_contiguous_frames
                print frame
                print matching_frames.shape
                raw_input('error')

            match_mat = np.zeros(matching_frames.shape)

            for col in range(len(min_pos_in_col)):
                if matching_frames[min_pos_in_col[col], col] < threshold:
                    match_mat[min_pos_in_col[col], col] = 1
                # else:
                # print 'Threshold mismatch:',
                # matching_frames[min_pos_in_col[col], col]

            clustered_match = np.sum(match_mat, axis=1)

            # plt.plot(clustered_match)
            # plt.show()

            mean_clustered_match[frame] = np.mean(clustered_match)
            stddev_clustered_match[frame] = np.std(clustered_match)
            mean2std_clustered_match[frame] = mean_clustered_match[
                frame] / stddev_clustered_match[frame]

            # print mean2std_clustered_match[frame]

            clustered_match_array.append(clustered_match)
            match_mat_array.append(match_mat)

        start_frame += num_contiguous_frames

    pickle_object(mean_clustered_match, 'mean_clustered_match')
    pickle_object(stddev_clustered_match, 'stddev_clustered_match')

    print 'Canonical start', np.argmax(mean2std_clustered_match)
    print 'Matchval', np.max(mean2std_clustered_match)

    canon_start = np.argmax(mean2std_clustered_match)
    writing_threads = []

    if write_to:
        print 'Writing canonical heartbeat'
        if THREADED:
            writing_threads.append(
                LimitedThread(
                    target=write_dwt_matrix,
                    args=(
                        dwt_array[
                            :, :, canon_start: canon_start + mean_hb_period],
                        write_to,
                        range(mean_hb_period),
                        False
                    )
                )
            )
            writing_threads[-1].start()
        else:
            write_dwt_matrix(
                dwt_array[:, :, canon_start: canon_start + mean_hb_period],
                write_to,
                range(mean_hb_period),
                False
            )

    for writing_thread in writing_threads:
        writing_thread.join()

    return dwt_array[:, :, canon_start: canon_start + mean_hb_period]


# TODO: replace with class
PhaseStamp = collections.namedtuple(
    'PhaseStamp', ['phase', 'matchval', 'neighbourMatchVals'])


def phase_stamp_image(curr_frame, canonical_hb_dwt):
    sad_array = np.zeros(canonical_hb_dwt.shape[2])

    for phase in range(canonical_hb_dwt.shape[2]):
        canon_frame = canonical_hb_dwt[:, :, phase]
        sad_array[phase] = np.sum(np.abs(curr_frame - canon_frame))

    best_phase = np.argmin(sad_array)
    return PhaseStamp(
        phase=best_phase,
        matchval=sad_array[best_phase],
        neighbourMatchVals=(
            sad_array[(best_phase - 1) % canonical_hb_dwt.shape[2]],
            sad_array[(best_phase + 1) % canonical_hb_dwt.shape[2]]
        )
    )


def phase_stamp_images(dwt_array, canonical_hb_dwt):
    # phase_stamps = []
    tic = time.time()
    if not settings.MULTIPROCESSING:
        phase_stamps = map(
            lambda frame: phase_stamp_image(
                dwt_array[:, :, frame],
                canonical_hb_dwt
            ),
            range(dwt_array.shape[2])
        )
    else:
        proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)

        temp_var = proc_pool.map_async(
            lambda frame: phase_stamp_image(
                dwt_array[:, :, frame],
                canonical_hb_dwt
            ),
            range(dwt_array.shape[2]),
            settings.NUM_CHUNKS
        )

        try:
            phase_stamps = temp_var.get(timeout=settings.TIMEOUT * 10)
        except TimeoutError:
            logging.info('Phase stamping Timed out in %d seconds' %
                         (settings.TIMEOUT * 10))

    print 'Completed phase stamping in', time.time() - tic

    # for frame in range(dwt_array.shape[2]):
    #     curr_frame = dwt_array[:, :, frame]

    #     sad_array = np.zeros(canonical_hb_dwt.shape[2])

    #     for phase in range(canonical_hb_dwt.shape[2]):
    #         canon_frame = canonical_hb_dwt[:, :, phase]
    #         sad_array[phase] = np.sum(np.abs(curr_frame - canon_frame))

    #     # plt.plot(sad_array)
    #     # plt.show()

    #     best_phase = np.argmin(sad_array)
    #     phase_stamps.append(
    #         PhaseStamp(
    #             phase=best_phase,
    #             matchval=sad_array[best_phase],
    #             neighbourMatchVals=(
    #                 sad_array[(best_phase - 1) % canonical_hb_dwt.shape[2]],
    #                 sad_array[(best_phase + 1) % canonical_hb_dwt.shape[2]]
    #             )
    #         )
    #     )

    pickle_object(phase_stamps, 'phase_stamps')

    return phase_stamps


def compile_phase_z_matrix(framelist, z_stamps, phase_stamps, hb_length):
    phase_z_matrix = np.empty(
        (hb_length, int(np.nanmax(z_stamps)) + 1), dtype=np.object)
    phase_z_density = np.zeros(phase_z_matrix.shape)

    for i in range(hb_length):
        for j in range(int(np.nanmax(z_stamps)) + 1):
            phase_z_matrix[i, j] = list()

    for index, frame in enumerate(framelist):
        if np.isnan(z_stamps[index]):
            continue
        # print phase_stamps[index].phase, ',', z_stamps[index]
        phase_z_matrix[phase_stamps[index].phase, int(z_stamps[index])].append(
            (frame, phase_stamps[index].matchval)
        )
        phase_z_density[phase_stamps[index].phase, int(z_stamps[frame])] += 1

    pickle_object(phase_z_matrix, 'phase_z_matrix')
    pickle_object(phase_z_density, 'phase_z_density')

    # phase_z_density = np.zeros(phase_z_matrix.shape)
    # for i in range(phase_z_matrix.shape[0]):
    #     for j in range(phase_z_matrix.shape[1]):
    #         phase_z_density[i, j] = len(phase_z_matrix[i, j])

    return phase_z_matrix, phase_z_density


def get_fluorescent_filename(frame_no):
    name_format = '%%s_%%0%dd_%%s.tif' % setting['fm_num_digits']
    return name_format % (setting['fm_image_prefix'], setting['fm_index_start_at'] + frame_no, setting.get('fm_image_suffix', ''))


def find_best_frame(phi_z_cell):
    matchvals = [obj[1] for obj in phi_z_cell]
    best_fit = phi_z_cell[matchvals.index(min(matchvals))][0]
    return best_fit


def write_phase_stamped_fluorescent_images(phi_z_matrix, read_from, write_to):
    print setting['fm_images']
    phi_range, z_range = phi_z_matrix.shape

    rewrite_lookup = {}
    missing_info = []

    for phase in range(phi_range):
        for z in range(z_range):
            curr_cell = phi_z_matrix[phase, z]
            if len(curr_cell) == 0:
                missing_info.append('Final_Z%03d_T%03d.tif' % (z, phase))
                continue

            best_fit = find_best_frame(curr_cell)

            rewrite_lookup[best_fit] = 'Final_Z%03d_T%03d.tif' % (z, phase)

    for frame_no, final_name in rewrite_lookup.iteritems():
        src = os.path.join(read_from, get_fluorescent_filename(frame_no))
        dst = os.path.join(write_to, final_name)
        shutil.copy2(src, dst)
    # a = rewrite_lookup.keys()
    # a.sort()
    # for key in a:
    #     print rewrite_lookup[key]
    pickle_object(rewrite_lookup, 'final_rewrite_lookup')
    pickle_object(missing_info, 'missing_frame')


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
        # Assuming stationary images are taken in the same imaging session as
        # moving images
        img_path=setting['bf_images'],
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

    canonical_frame_list = frame_list[
        canon_hb_start: canon_hb_start + canon_hb_length]

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
    canonical_hb_start = compute_canonical_heartbeat(
        d[:, :, 2:-2], np.round(cm_hb).astype('int'))
    # #
    canonical_hb_frames = frames[
        canonical_hb_start: canonical_hb_start + np.round(cm_hb).astype('int')]
    print canonical_hb_frames
