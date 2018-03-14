if __name__ == '__main__':
    import sys
    sys.path.append('D:\\Scripts\\MaPS\\MaPS scripts\\')

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats, signal
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.external.tifffile as tff
import pywt

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2, ssim2, corr2_masked
from maps.helpers.misc import pickle_object,\
    unpickle_object
from maps.settings import setting, read_setting_from_json, THREADED
from maps.helpers.gui_modules import cv2_methods, load_frame, extract_window, apply_mask, apply_unmask, masking_window_frame
from maps.helpers.logging_config import logger_config
from maps.helpers.misc import LimitedThread
from maps import settings

import os
import logging
import time
import csv
import cv2
import code
import platform
import collections
import threading
import traceback
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

        write_timeout = settings.TIMEOUT * 10
        try:
            temp_var.get(timeout=write_timeout)
        except TimeoutError:
            logging.info('Writing DWT frames timed out in %d seconds' %
                         (write_timeout))

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

        read_timeout = settings.TIMEOUT * 10
        try:
            temp_var.get(timeout=read_timeout)
        except TimeoutError:
            logging.info('Frame timed out in %d seconds' %
                         (read_timeout))

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
        print 'Writing DWT frames'
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
            if zook_lim.stop > dwt_array.shape[2]:
                break
            # print zook_lim
            us_dwt_array = np.dstack(
                (us_dwt_array,
                 upsample_dwt_array(dwt_array[:, :, zook_lim]))
            )
            # print us_dwt_array.shape
        pickle_object(zooks, 'zooks')
    else:
        us_dwt_array = upsample_dwt_array(dwt_array)

    writing_threads = []

    if write_to:
        print 'Writing interchelated DWT frames'
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


def compute_peak_harmonic(first_image, second_image, method, corr_array, frame):
    # Corr2 method
    if method == 'corr':
        corr_array[frame] = (corr2(first_image, second_image))
    # SSIM
    elif method == 'ssim':
        corr_array[frame] = (ssim2(first_image, second_image))
    # Abs Diff
    # TODO: Will require filtering
    elif method == 'sad':
        corr_array[frame] = (np.sum(np.absolute(first_image - second_image)))
    # OpenCV methods ccorr, ccoeff
    elif method in ('ccoeff_norm', 'ccorr_norm', 'mse_norm'):
        corr_array[frame] = (cv2.matchTemplate(
            second_image.astype('float32'),
            first_image.astype('float32'), cv2_methods[method])[0][0]
        )


def compute_heartbeat_length(dwt_array, method='ccoeff_norm', comparison_plot=True, im_fps=480):
    '''
    Compute the length of the heartbeat. Uses the DWT of the
    '''
    time_taken = []

    hb_array = []

    spectra_raw = []
    # TODO
    for ref_frame in range(dwt_array.shape[2] // 2)[:300]:
        logging.info('Processing frame:%d' % ref_frame)
        first_image = dwt_array[:, :, ref_frame]
        tic = time.time()
        x = np.zeros(dwt_array.shape[2])
        # for i in range(dwt_array.shape[2]):
        #     second_image = dwt_array[:, :, i]
        if not settings.MULTIPROCESSING:
            map(
                lambda frame: compute_peak_harmonic(
                    first_image,
                    dwt_array[:, :, frame],
                    method,
                    x,
                    frame
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
                    x,
                    frame
                ),
                range(dwt_array.shape[2]),
                settings.NUM_CHUNKS
            )

            try:
                temp_var.get(timeout=settings.TIMEOUT)
            except TimeoutError:
                logging.info('Frame %d timed out in %d seconds' %
                             (ref_frame, settings.TIMEOUT))

            del proc_pool

        time_taken.append(time.time() - tic)
        # For absdiff and MSE
        # if method in ('sad', 'mse_norm'):
        x = np.asarray(x)
        x = np.max(x) - x

        spectra_raw.append(x.copy())

        f, Pxx = signal.welch(x, im_fps, nperseg=dwt_array.shape[2])

        t = im_fps / f
        Pxx[np.where(t > 400)] = 0
        hb_array.append(t[np.argmax(Pxx)])

    cm_hb = stats.mode(hb_array)

    pickle_object(cm_hb[0], 'common_heatbeat_period.pkl')
    pickle_object(time_taken, 'common_heatbeat_calctime.pkl')
    pickle_object(spectra_raw, 'raw_spectra.pkl')

    return cm_hb[0]


def standardize_brightness(image, method='minmax'):
    '''
    Standardize brightness of image and convert to range [0,1]
    '''
    image = image.astype('float64')
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
    return image
    # if method == 'minmax':
    #     image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
    # elif method == 'musigma':
    #     image = ((image - np.mean(image)) / (6 * np.std(image)))


def compute_diff_mat(dwt_array, ref_dwt, i, diff_matrix, method='sad'):
    if method == 'sad':
        for j in range(i, dwt_array.shape[2]):
            curr_dwt = dwt_array[:, :, j]

            diff_matrix[i, j] = diff_matrix[j, i] = np.sum(
                np.absolute(ref_dwt - curr_dwt)
            )

    elif method == 'corr':
        for j in range(i, dwt_array.shape[2]):
            curr_dwt = dwt_array[:, :, j]

            # Subtracting 1 to make max min behave same as SAD
            diff_matrix[i, j] = diff_matrix[j, i] = 1 - cv2.matchTemplate(
                ref_dwt.astype('float32'),
                curr_dwt.astype('float32'),
                cv2_methods['ccoeff_norm']
            )[0][0]

    elif method == 'nsad':
        for j in range(i, dwt_array.shape[2]):
            curr_dwt = dwt_array[:, :, j]

            diff_matrix[i, j] = diff_matrix[j, i] = np.sum(
                np.absolute(ref_dwt - curr_dwt)
            ) / ref_dwt.size


def compute_canonical_heartbeat(dwt_array, zooks, mean_hb_period, diff_method='sad', canon_method='count', diff_matrix_pkl_name=None, write_to=None, use_pickled_diff=False):
    '''
    Processes interpolated frame wavelets. find the best continuous sequence of heartbeats
    diff_methods:
        sad - Sum of absolute difference
        ssad - Standardized brightness SAD
        nsad - Normalized SAD - normalized by the number of pixels in 1 dwt frame
        nssad - Normalized, standardized SAD
        corr - Correlation coefficient

    canon_methods:
        count - Count the number of minima in each row. Take mean2std
        sum - Mean of each row. Take mean2std
        combo - Combination of count and sum
    '''
    diff_matrix = np.zeros((dwt_array.shape[2], dwt_array.shape[2]))

    threshold = 0.02

    # Standardize brightness if required
    if diff_method in ('nssad', 'ssad'):
        print 'Standardizing brightness'
        diff_method = diff_method.replace('ss', 's')

        if not settings.MULTIPROCESSING or True:
            for i in range(dwt_array.shape[2]):
                dwt_array[:, :, i] = standardize_brightness(dwt_array[:, :, i])
            # dwt_array = map(
            #     lambda frame_no: standardize_brightness(
            #         dwt_array[:, :, frame_no]
            #     ),
            #     range(dwt_array.shape[2])
            # )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame_no: standardize_brightness(
                    dwt_array[:, :, frame_no]
                ),
                range(dwt_array.shape[2]),
                settings.NUM_CHUNKS
            )

            timeout_standardize = settings.TIMEOUT * 30
            try:
                temp_var.get(timeout=timeout_standardize)
            except TimeoutError:
                print temp_var._number_left, 'frames left'
                logging.info('DWT standardization timed out in %d seconds' %
                             (timeout_standardize))
                logging.info('%d frames did not get processed' %
                             (temp_var._number_left))

            del proc_pool

    if use_pickled_diff:
        diff_matrix = unpickle_object(diff_matrix_pkl_name)
    else:
        # time_taken = []
        print 'Using method "%s" for diff_matrix calculation' % diff_method
        tic = time.time()
        sad_threads = []

        if not settings.MULTIPROCESSING:
            map(
                lambda frame_no: compute_diff_mat(
                    dwt_array,
                    dwt_array[:, :, frame_no],
                    frame_no,
                    diff_matrix,
                    diff_method
                ),
                range(dwt_array.shape[2])
            )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame_no: compute_diff_mat(
                    dwt_array,
                    dwt_array[:, :, frame_no],
                    frame_no,
                    diff_matrix,
                    diff_method
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
        if diff_matrix_pkl_name:
            pickle_object(diff_matrix, diff_matrix_pkl_name)
            print 'diff_matrix pickled'

    # TODO: Check other values of threshold
    threshold = 0.5 * np.max(diff_matrix)
    start_frame = 0
    num_contiguous_frames = (setting[
        'ZookZikPeriod'] - (setting['ignore_startzook'] + setting['ignore_endzook'])) * setting['time_resampling_factor']

    # print num_contiguous_frames * setting['canonical_zook_count']
    # raw_input()

    # Using number of frames
    if canon_method in ('count', 'combo'):
        mean_clustered_match = np.ones(
            num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
        stddev_clustered_match = np.ones(
            num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
        mean2std_clustered_match = np.zeros(
            num_contiguous_frames * setting['canonical_zook_count'])
    # Using mean of row vals
    if canon_method in ('sum', 'combo'):
        mean_clustered_match_new = np.ones(
            num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
        stddev_clustered_match_new = np.ones(
            num_contiguous_frames * setting['canonical_zook_count']) * mean_hb_period
        mean2std_clustered_match_new = np.zeros(
            num_contiguous_frames * setting['canonical_zook_count'])

    clustered_match_array = []
    match_mat_array = []

    # print mean_hb_period
    # TODO: To be optimized
    for zook in range(setting['canonical_zook_count']-2):
        # for zook in range(setting['canonical_zook_count'] - 2):
        # print range(int(start_frame) + (num_contiguous_frames // 4),
        # int(start_frame + num_contiguous_frames - mean_hb_period) -
        # (num_contiguous_frames // 4))
        if setting['central_zook_canon']:
            ignore_region = (num_contiguous_frames // 4)
            frame_range = range(int(start_frame) + ignore_region, int(
                start_frame + num_contiguous_frames - mean_hb_period) - ignore_region)
        else:
            frame_range = range(int(start_frame), int(
                start_frame + num_contiguous_frames - mean_hb_period))

        for frame in frame_range:
            matching_frames = diff_matrix[frame: frame + mean_hb_period, :]

            if canon_method in ('count', 'combo'):
                min_pos_in_col = np.argmin(matching_frames, axis=0)

                match_mat = np.zeros(matching_frames.shape)

                # TODO: To be vectorized
                # TODO: Raise threshold
                for col in range(len(min_pos_in_col)):
                    if matching_frames[min_pos_in_col[col], col] < threshold:
                        match_mat[min_pos_in_col[col], col] = 1
                    # else:
                    # print 'Threshold mismatch:',
                    # matching_frames[min_pos_in_col[col], col]

                clustered_match = np.sum(match_mat, axis=1)

                mean_clustered_match[frame] = np.mean(clustered_match)
                stddev_clustered_match[frame] = np.std(clustered_match)
                mean2std_clustered_match[frame] = mean_clustered_match[
                    frame] / stddev_clustered_match[frame]

            if canon_method in ('sum', 'combo'):
                clustered_match_new = np.mean(matching_frames, axis=1)

                mean_clustered_match_new[frame] = np.mean(clustered_match_new)
                stddev_clustered_match_new = np.std(clustered_match_new)
                mean2std_clustered_match_new = mean_clustered_match_new[
                    frame] / stddev_clustered_match_new[frame]

            # clustered_match_array.append(
            #     (clustered_match, clustered_match_new))
            # match_mat_array.append(match_mat)

        start_frame += num_contiguous_frames

    # pickle_object(mean_clustered_match, 'mean_clustered_match')
    # pickle_object(stddev_clustered_match, 'stddev_clustered_match')
    # pickle_object(clustered_match_array, 'clustered_match_array')
    # pickle_object(match_mat_array, 'match_mat_array')

    if canon_method == 'count':
        canon_start = np.argmax(mean2std_clustered_match)
    elif canon_method == 'sum':
        canon_start = np.argmax(mean2std_clustered_match_new)
    else:
        canon_start = np.argmax(
            mean2std_clustered_match + mean2std_clustered_match_new)

    # if setting['central_zook_canon']:
    #     canon_start +=

    # if diff_method == 'sad':
    #     print 'Using argmax'
    #     canon_start = np.argmax(mean2std_clustered_match)
    # else:
    #     print 'Using argmin'
    #     canon_start = np.argmin(mean2std_clustered_match)

    pickle_object(canon_start, 'canon_hb_start')
    print 'Canonical start:', canon_start
    print 'True Canonical start frame:', zooks.get_true_frame_number(canon_start)
    # print 'Matchval', np.max(mean2std_clustered_match)

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

    return (dwt_array[:, :, canon_start: canon_start + mean_hb_period], diff_matrix, canon_start)


# TODO: replace with class
PhaseStamp = collections.namedtuple(
    'PhaseStamp', ['phase', 'matchval', 'neighbourMatchVals'])


def phase_stamp_image(curr_frame, canonical_hb_dwt, method='corr'):
    diff_array = np.zeros(canonical_hb_dwt.shape[2])

    if method == 'sad':
        for phase in range(canonical_hb_dwt.shape[2]):
            canon_frame = canonical_hb_dwt[:, :, phase]
            diff_array[phase] = np.sum(np.abs(curr_frame - canon_frame))

    elif method == 'corr':
        for phase in range(canonical_hb_dwt.shape[2]):
            canon_frame = canonical_hb_dwt[:, :, phase]
            diff_array[phase] = 1 - cv2.matchTemplate(
                curr_frame.astype('float32'),
                canon_frame.astype('float32'),
                cv2_methods['ccoeff_norm']
            )[0][0]

    else:
        raise NotImplementedError(
            'Method %s not implemented for phase stamping' % method)

    best_phase = np.argmin(diff_array)
    return PhaseStamp(
        phase=best_phase,
        matchval=diff_array[best_phase],
        neighbourMatchVals=(
            diff_array[(best_phase - 1) % canonical_hb_dwt.shape[2]],
            diff_array[(best_phase + 1) % canonical_hb_dwt.shape[2]]
        )
    )


def phase_stamp_images(dwt_array, canonical_hb_dwt, method='corr'):
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

    pickle_object(phase_stamps, 'phase_stamps')

    return phase_stamps


def mask_canonical_heartbeat(canonical_hb_dwt, num_masks=10, mask_pkl_file_name='mask_array', masked_canon_dwt_array='masked_canon_dwt', use_pickled_masks=False):
    if not use_pickled_masks:
        frames_per_mask = int(
            np.ceil(canonical_hb_dwt.shape[2] / float(num_masks)))

        mask_array = np.zeros(canonical_hb_dwt.shape)

        canon_shape = canonical_hb_dwt[:, :, 0].shape

        for mask in range(num_masks - 1):
            print mask
            us_canon_frames = resize(
                canonical_hb_dwt[:, :, mask *
                                 frames_per_mask: (mask + 1) * frames_per_mask],
                (
                    canon_shape[0] * 4,
                    canon_shape[1] * 4,
                    frames_per_mask
                ),
                preserve_range=True
            )

            print us_canon_frames.shape

            _, mask_mat = masking_window_frame(
                us_canon_frames.astype('float32'), crop_selection=False, mask_selection=True)

            mask_mat = resize(
                mask_mat,
                (
                    mask_mat.shape[0] * 0.25,
                    mask_mat.shape[1] * 0.25,
                ),
                preserve_range=True
            )

            mask_array[:, :, mask * frames_per_mask: (mask + 1) * frames_per_mask] = np.dstack(
                [mask_mat] * frames_per_mask
            )

        if canonical_hb_dwt.shape[2] % num_masks or True:
            mask = mask + 1
            print mask

            rem_frames = canonical_hb_dwt.shape[2] % frames_per_mask

            us_canon_frames = resize(
                canonical_hb_dwt[:, :, mask * frames_per_mask:],
                (
                    canon_shape[0] * 4,
                    canon_shape[1] * 4,
                    rem_frames if rem_frames != 0 else frames_per_mask
                ),
                preserve_range=True
            )

            print us_canon_frames.shape

            _, mask_mat = masking_window_frame(
                us_canon_frames.astype('float32'), crop_selection=False, mask_selection=True)
            mask_mat = resize(
                mask_mat,
                (
                    mask_mat.shape[0] * 0.25,
                    mask_mat.shape[1] * 0.25,
                ),
                preserve_range=True
            )
            mask_array[:, :, mask * frames_per_mask:] = np.dstack(
                [mask_mat] * (rem_frames if rem_frames !=
                              0 else frames_per_mask)
            )

        masked_canon_dwt = []

        for phase in range(mask_array.shape[2]):
            masked_canon_dwt.append(apply_mask(
                canonical_hb_dwt[:, :, phase], mask_array[:, :, phase]))

        pickle_object(mask_array, mask_pkl_file_name)
        pickle_object(masked_canon_dwt, masked_canon_dwt_array)

    else:
        mask_array = unpickle_object('mask_array')
        masked_canon_dwt = unpickle_object('masked_canon_dwt')

    try:
        write_dwt_matrix(
            np.dstack(
                map(
                    lambda i: apply_unmask(
                        masked_canon_dwt[i],
                        mask_array[:, :, i]
                    ),
                    range(len(masked_canon_dwt))
                )
            ).astype('float32'),
            setting['canon_dwt_masked'],
            range(len(masked_canon_dwt))
        )
    except:
        import traceback
        import code
        traceback.print_exc()
        code.interact(local=locals())

    return mask_array, masked_canon_dwt


def phase_stamp_image_masked(curr_frame, canonical_hb_dwt, masks_array, method='corr', temp=0):
    # print temp
    USE_CV_MASK = False

    tic = time.time()
    diff_array = np.zeros(len(canonical_hb_dwt))
    # diff_array = np.zeros(canonical_hb_dwt.shape[2])

    if method == 'sad':
        for phase in range(len(canonical_hb_dwt)):
            canon_frame = canonical_hb_dwt[phase]

            masked_frame = apply_mask(curr_frame, masks_array[:, :, phase])

            diff_array[phase] = np.sum(np.abs(masked_frame - canon_frame))

    elif method == 'corr':
        for phase in range(len(canonical_hb_dwt)):
            # for phase in range(canonical_hb_dwt.shape[2]):
            canon_frame = canonical_hb_dwt[phase]
            # canon_frame = canonical_hb_dwt[phase]
            # masked_frame = apply_mask(curr_frame, masks_array[:, :, phase])
            # code.interact(local=locals())
            try:
                diff_array[phase] = 1 - cv2.matchTemplate(
                    curr_frame.astype('float32'),
                    canon_frame.astype('float32'),
                    cv2_methods['ccorr_norm'],
                    mask=masks_array[:, :, phase].astype('float32')
                )[0][0]

                # diff_array[phase] = 1 - cv2.matchTemplate(
                #     curr_frame.astype('float32'),
                #     canon_frame.astype('float32'),
                #     cv2_methods['ccoeff_norm'],
                #     # mask=masks_array[:, :, phase].astype('float32')
                # )[0][0]

                # diff_array[phase] = 1 - corr2_masked(curr_frame.astype('float32'), canon_frame.astype('float32'), masks_array[:, :, phase].astype('float32'))

            except:
                traceback.print_exc()
                print 'error in', phase
                diff_array[phase] = 1
    # elif method == 'corr_mult_mask':
    #     for phase in range(canonical_hb_dwt.shape[2]):
    #         canon_frame = canonical_hb_dwt[phase]
    #         masked_frame = apply_mask(curr_frame, masks_array[:, :, phase])

    #         try:
    #             diff_array[phase] = 1 - cv2.matchTemplate(
    #                 masked_frame.astype('float32'),
    #                 canon_frame.astype('float32'),
    #                 cv2_methods['ccoeff_norm']
    #             )[0][0]
    #         except:
    #             traceback.print_exc()
    #             print 'error in', phase
    #             diff_array[phase] = 1

    else:
        raise NotImplementedError(
            'Method %s not implemented for phase stamping' % method)

    # plt.plot(diff_array)
    # plt.show()

    best_phase = np.argmin(diff_array)
    # print best_phase

    return PhaseStamp(
        phase=best_phase,
        matchval=diff_array[best_phase],
        neighbourMatchVals=(
            diff_array[(best_phase - 1) % len(canonical_hb_dwt)],
            diff_array[(best_phase + 1) % len(canonical_hb_dwt)]
        )
    )


def phase_stamp_images_masked(dwt_array, canonical_hb_dwt, masks_array, method='corr'):
    tic = time.time()

    if not settings.MULTIPROCESSING or True:
        # print dwt_array.shape[2]
        results = map(
            lambda frame: phase_stamp_image_masked(
                dwt_array[:, :, frame],
                canonical_hb_dwt,
                masks_array,
                method,
                frame
            ),
            range(dwt_array.shape[2])
        )
    else:
        proc_pool = ThreadPool(processes=5)

        temp_var = proc_pool.map_async(
            lambda frame: phase_stamp_image_masked(
                dwt_array[:, :, frame],
                canonical_hb_dwt,
                masks_array,
                method,
                frame
            ),
            range(dwt_array.shape[2]),
            settings.NUM_CHUNKS
        )
        print 'Spawned all processes'

        try:
            results = temp_var.get(timeout=settings.TIMEOUT * 50)
        except TimeoutError:
            logging.info('Phase stamping Timed out in %d seconds' %
                         (settings.TIMEOUT * 50))

    phase_stamps = results
    # phase_stamps = [r[0] for r in results]
    # correlation_matrix = np.dstack([r[1].transpose() for r in results])
    # correlation_matrix = correlation_matrix.reshape(np.prod(correlation_matrix.shape[:2]), -1)

    print 'Completed phase stamping in', time.time() - tic

    pickle_object(phase_stamps, 'masked_phase_stamps')
    # pickle_object(correlation_matrix, 'masked_correlation_matrixi')

    return phase_stamps
    # return phase_stamps, correlation_matrix


def get_row(pz_row, rank=1):
    # matchvals = [i[1] for i in cell]
    for cell in pz_row:
        cell = sorted(cell, key=lambda x: x[1])
    # matches = map(lambda cell: cell[[i[1] for i in cell].index(
        # min([i[1] for i in cell]))][1] if len(cell) else np.NAN, pz_row)
    matches = map(lambda cell: cell[rank - 1][1]
                  if len(cell) >= rank else np.NAN, pz_row)
    return matches


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def correct_phase_stamp_matchvals(phase_stamps):
    # Adjust phase stamps to remove degradation over time and degradation
    # over zook
    phase_stamps_matchvals = [i.matchval for i in phase_stamps]
    phase_stamps_lp_matchvals = moving_average(phase_stamps_matchvals, n=1000)
    corrected_phase_stamp_mathvals = phase_stamps_matchvals[:-999] - \
        (phase_stamps_lp_matchvals - min(phase_stamps_lp_matchvals))

    corrected_phase_stamps = []
    for ind, i in enumerate(phase_stamps[:-999]):
        # try:
        # i.matchval = corrected_phase_stamp_mathvals[ind]
        corrected_phase_stamps.append(PhaseStamp(
            phase=i.phase, neighbourMatchVals=i.neighbourMatchVals, matchval=corrected_phase_stamp_mathvals[ind]))
        # except:
    # phase_stamps = phase_stamps[:-999]

    return corrected_phase_stamps


def compute_phi_z_density(pz_mat):
    pz_density = np.zeros(pz_mat.shape)

    for row in pz_density.shape[0]:
        for col in pz_density.shape[1]:
            pz_density[i, j] = len(pz_mat[i, j])

    return pz_density


# TODO: COMMENT
def filter_phi_z_mat(pz_mat, phase_stamps, z_stamps, percentile=50, percentile_filt=False, pkl_name='negative_phase_stamps'):
    pz_mat_density = np.zeros(pz_mat.shape)

    '''
    Remove negative slope from phase stamps. Set the values to have high differences
    '''
    diff_vals = np.asarray([frame.matchval for frame in phase_stamps])
    diff_diff = diff_vals[1:] - diff_vals[:-1]
    problem_frames = [((frame + 1), phase_stamps[frame + 1]) for (frame, diffval)
                      in enumerate(diff_diff) if (0 < diffval and diffval < 30)]

    pickle_object(problem_frames, pkl_name)

    for bad_frame in problem_frames:
        for pz_cell in pz_mat[bad_frame[1].phase, :]:
            cell_frames = [i[0] for i in pz_cell]
            if bad_frame[0] in cell_frames:
                pz_cell.pop(cell_frames.index(bad_frame[0]))

    '''
    Remove bottom 50% of frames
    '''
    filt_pz_mat = np.empty(pz_mat.shape, dtype=np.object)

    if percentile_filt:
        for phi in range(pz_mat.shape[0]):
            pz_row_matchvals = reduce(
                lambda x, y: x + y, map(lambda cell: [i[1] for i in cell], pz_mat[phi, :]))
            pz_median = np.percentile(np.asarray(pz_row_matchvals), percentile)
            for ind, cell in enumerate(pz_mat[phi, :]):
                # print cell
                cell = [i for i in cell if i[1] < pz_median]
                # print cell
                # print '-'*80
                filt_pz_mat[phi, ind] = cell
                pz_mat_density[phi, ind] = len(cell)

    else:
        filt_pz_mat = pz_mat

    return filt_pz_mat, pz_mat_density


def phi_z_histogram(pz_density):
    row_filling = np.sum(pz_density, axis=1)
    plt.figure()
    plt.plot(row_filling)
    plt.show(block=False)

    mean_filling = np.mean(row_filling)

    for i in range(len(row_filling)):
        print i, '->', row_filling[i]

    print 'Threshold at 20%'
    print np.where(row_filling < 0.2 * mean_filling)


def fill_phi_z_row(pz_mat, lut_name='phiz_fill_lut.txt'):
    '''
    Fill blank cells in phi z matrix
    '''
    with open(os.path.join(setting['workspace'], lut_name)) as fp:
        raw_lines = fp.readlines()

    fill_lut = {}

    for line in raw_lines:
        t = line.split(':')
        fill_lut[int(t[0])] = [int(i) for i in t[1].split(',')]

    for from_row, to_rows in fill_lut.iteritems():
        filler_row = pz_mat[from_row, :]
        for to_row in to_rows:
            print to_row
            # filled_row = pz_mat[to_row, :]
            pz_mat[to_row, :] = filler_row
            # for i in range(pz_mat.shape[1]):
            #     pz_mat[to_row, i].extend(filler_row[i])

    # pz_density = compute_phi_z_density

    # return pz_mat, pz_density
    return pz_mat


def compile_phase_z_matrix(framelist, z_stamps, phase_stamps, hb_length, pz_mat_pkl_name='phase_z_matrix', pz_density_pkl_name='phase_z_density'):
    print 'Downsampling by', setting['phiz_downsampling_factor']
    z_stamps_shifted = z_stamps - np.nanmin(z_stamps)
    z_stamps_shifted = z_stamps_shifted / setting['phiz_downsampling_factor']

    phase_z_matrix = np.empty(
        (hb_length, int(np.nanmax(z_stamps_shifted)) + 1), dtype=np.object)
    phase_z_density = np.zeros(phase_z_matrix.shape)

    for i in range(hb_length):
        for j in range(int(np.nanmax(z_stamps_shifted)) + 1):
            phase_z_matrix[i, j] = list()

    phase_stamps = correct_phase_stamp_matchvals(phase_stamps)

    for index, frame in enumerate(framelist[:-999]):
        if np.isnan(z_stamps_shifted[frame]):
            continue
        # print phase_stamps[index].phase, ',', z_stamps_shifted[index]
        phase_z_matrix[phase_stamps[index].phase, int(z_stamps_shifted[frame])].append(
            (frame, phase_stamps[index].matchval)
        )
        phase_z_density[phase_stamps[index].phase,
                        int(z_stamps_shifted[frame])] += 1

    pickle_object(phase_z_matrix, pz_mat_pkl_name)
    pickle_object(phase_z_density, pz_density_pkl_name)

    return phase_z_matrix, phase_z_density


def get_fluorescent_filename(frame_no):
    name_format = '%%s%%0%dd_%%s.tif' % setting['fm_num_digits']
    ret_name = name_format % (setting['fm_image_prefix'], setting[
        'fm_index_start_at'] + frame_no, setting.get('fm_image_suffix', ''))
    ret_name = ret_name.replace('_.tif', '.tif')
    return ret_name


def find_best_frame(phi_z_cell):
    matchvals = [obj[1] for obj in phi_z_cell]
    best_fit = phi_z_cell[matchvals.index(min(matchvals))][0]
    return best_fit


def write_final_frame(src, dst, method='copy'):
    '''
    valid methods='copy', 'adj_brightness'
    '''
    THRESHOLD = 200
    NORM_POINT = 10000

    if method == 'copy':
        shutil.copy2(src, dst)
    elif method == 'adj_brightness':
        frame = tff.imread(src)
        frame[np.where(frame > THRESHOLD)] = NORM_POINT / \
            (np.max(frame)-THRESHOLD)*frame[np.where(frame > THRESHOLD)]
        tff.imsave(dst, frame.astype('int16'))
        # tff.imsave(dst, frame)
    elif method == 'med_adj_brightness':
        frame = tff.imread(src)
        frame[np.where(frame > THRESHOLD)] = NORM_POINT / \
            (np.median(frame)-THRESHOLD)*frame[np.where(frame > THRESHOLD)]
        tff.imsave(dst, frame.astype('int16'))


def write_phase_stamped_fluorescent_images(phi_z_matrix, read_from, write_to):
    # print setting['fm_images']
    phi_range, z_range = phi_z_matrix.shape

    # TODO: Attempting to change data structure
    # num_frames = len(os.listdir(setting['fm_images']))
    # rewrite_lookup = {i: [] for i in range(1, num_frames + 1)}

    # for phase in range(phi_range):
    #     for z in range(z_range):
    #         curr_cell = phi_z_matrix[phase, z]

    #         if len(curr_cell) == 0:
    #             missing_info.append('Final_Z%03d_T%03d.tif' % (z, phase))

    #         best_fit = find_best_frame(curr_cell)

    #         rewrite_lookup[best_fit].append(
    #             'Final_Z%03d_T%03d.tif' % (z, phase)
    #         )
    rewrite_lookup = {}
    missing_info = []

    for phase in range(phi_range):
        for z in range(z_range):
            curr_cell = phi_z_matrix[phase, z]
            if len(curr_cell) == 0:
                missing_info.append('Final_Z%03d_T%03d.tif' % (z, phase))
                continue

            best_fit = find_best_frame(curr_cell)

            rewrite_lookup['Final_Z%03d_T%03d.tif' % (z, phase)] = best_fit

    ref_fluorescent_image_path = os.path.join(
        read_from, get_fluorescent_filename(setting['fm_index_start_at']))

    dummy_image = tff.imread(ref_fluorescent_image_path)
    dummy_image = dummy_image * 0

    for final_name, frame_no in rewrite_lookup.iteritems():
        # TODO: Read and write instead of copy to allow brightness/motion
        # adjustment
        src = os.path.join(read_from, get_fluorescent_filename(frame_no))
        dst = os.path.join(write_to, final_name)
        write_final_frame(src, dst, method='adj_brightness')
        # shutil.copy2(src, dst)

    # Writing dummy images for missing frames
    for missing_frame in missing_info:
        tff.imsave(os.path.join(write_to, missing_frame), dummy_image)

    # a = rewrite_lookup.keys()
    # a.sort()
    # for key in a:
    #     print rewrite_lookup[key]
    pickle_object(rewrite_lookup, 'final_rewrite_lookup')
    pickle_object(missing_info, 'missing_frame')


def check_fluorescent_bleaching(zooks, read_from):
    fm_frame_objs = []
    plt.figure()
    mean_vals = np.zeros(
        (len(range(0, len(zooks[0].get_framelist()), 10)), len(zooks[::20])))
    max_vals = np.zeros(
        (len(range(0, len(zooks[0].get_framelist()), 10)), len(zooks[::20])))
    for i, frame in enumerate(range(0, len(zooks[0].get_framelist()), 10)):
        # mean_vals = []
        # max_vals = []

        # for zook in zooks[::20]:
        # for i in range(0,20):
        #     mean_vals = []
        #     for frame_no in zook.get_framelist()[i::20]:
        #         print frame_no, 'loaded'
        #         fm_frame_objs.append(tff.imread(os.path.join(read_from, get_fluorescent_filename(frame_no))))
        #         mean_vals.append(np.max(fm_frame_objs[-1]))
        #     plt.plot(mean_vals)
        #     print '-'*80
        # mean_vals = []
        for j, zook in enumerate(zooks[::20]):
            # for frame_no in zook.get_framelist():
            frame_no = zook.get_framelist()[frame]
            # print frame_no, 'loaded'
            fm_frame_objs.append(tff.imread(os.path.join(
                read_from, get_fluorescent_filename(frame_no))))
            mean_vals[i, j] = np.mean(fm_frame_objs[-1])
            max_vals[i, j] = np.max(fm_frame_objs[-1])
            # mean_vals.append(np.mean(fm_frame_objs[-1]))
            # max_vals.append(np.max(fm_frame_objs[-1]))
            # mean_vals.append(np.max(fm_frame_objs[-1]))
        # plt.plot(mean_vals.transpose())
        # plt.plot(max_vals)
    # plt.title('Mean vals - separate')
    # plt.figure()
    # plt.title('Mean vals - curvefit')
    # plt.plot(np.mean(mean_vals.transpose(), axis=1))
    # plt.show()

    # plot_data = []
    # for frame in fm_frame_objs:
    #     plot_data.append(np.mean(frame))


def write_brother_frames(phi_z_matrix, read_from, write_to, brother_frame_loc_file):
    # setting['brother_frames']
    for (phase, z) in brother_frame_locs:
        curr_cell = phi_z_matrix[phase, z]
        if len(curr_cell) <= 1:
            print 'Brother frames at (%d, %d) not possible. Not enough frames' % (phase, z)
        else:
            # Name the frames and get frame numbers in decreasing order of
            # matchval
            brother_frame_names = ['Brother_frames_Z%03d_T%03d_F%02d.tif' % (
                z, phase, index) for index in range(len(curr_cell))]
            # Sort cell frames by match val
            curr_cell.sort(key=lambda t: t[1].matchval)
            brother_frame_ids = [frame[0] for frame in curr_cell]

        # Write frames
        for (frame_no, final_name) in zip(brother_frame_ids, brother_frame_names):
            src = os.path.join(read_from, get_fluorescent_filename(frame_no))
            dst = os.path.join(write_to, final_name)
            shutil.copy2(src, dst)


def write_pz_density(pz_density, name):
    filename = os.path.join(setting['workspace'], name)
    tff.imsave(filename, pz_density)

    max_count = pz_density.max()

    enhanced_pz = np.dstack([np.zeros(pz_density.shape, dtype='uint8')]*3)
    # enhanced_pz[pz_density==1, 2] = 255
    for i in range(pz_density.shape[0]):
        for j in range(pz_density.shape[1]):
            if pz_density[i, j] == 1:
                enhanced_pz[i, j, :] = np.array([0, 0, 255])
            elif pz_density[i, j] > 1:
                val = (float(pz_density[i, j]) / max_count) * 128
                enhanced_pz[i, j, :] = np.array([128+val, 255-val, 0])

    filename = os.path.join(setting['workspace'], 'enhanced_'+name)
    tff.imsave(filename, enhanced_pz.astype('uint8'))


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


def plot_matrix_rows(matrix, title='', col_sep=1, start_col=None, num_cols=None, row_length=None, legend_key_string='Row'):
    if start_col is None:
        start_col = 0
    if num_cols is None:
        num_cols = matrix.shape[0] - start_col
    if row_length is None:
        row_length = matrix.shape[1]

    disp_mat = matrix[:row_length, start_col:start_col+num_cols:col_sep]

    offset_stack = np.zeros(disp_mat.shape)
    offset_preval = np.zeros(disp_mat.shape)

    offset_range = (np.max(disp_mat) - np.min(disp_mat)) * 1.5
    legend_keys = []
    for i in range(offset_stack.shape[1]):
        offset_stack[:, i] = (offset_stack.shape[1] - i - 1) * offset_range
        offset_preval[:5, i] = i * offset_range
        legend_keys.append('%s %d' %
                           (legend_key_string, start_col + i*col_sep+1))

    plt.figure()
    plt.gca().set_prop_cycle(None)
    plt.plot(disp_mat + offset_stack)
    plt.title(title+'(Stacked)')
    plt.legend(legend_keys)
    plt.gca().set_prop_cycle(None)
    plt.plot(offset_stack, linestyle='--')

    # plt.get_current_fig_manager().full_screen_toggle()
    plt.savefig(os.path.join(setting[
        'workspace'], title+'(Stacked)'+'.svg'), format='svg', dpi=600)

    # plt.figure()
    # plt.plot(disp_mat + offset_preval)
    # plt.title(title+'(Superimposed)')
    # plt.legend(legend_keys)

    # plt.savefig(os.path.join(setting[
    #                 'workspace'], title+'(Superimposed)'+'.svg'), format='svg', dpi=600)


def plot_results(pz_density=None, filtered_pz_density=None, phase_stamps=None, z_stamps=None, diff_matrix=None, diff_matrix_rows=None, canon_hb_stats=None, results='all'):
    if results in ('all', 'phiz'):
        plt.figure()
        plt.imshow(pz_density, cmap='custom_heatmap')
        plt.colorbar()
        plt.title('Phi-Z density')
        plt.figure()
        plt.imshow(filtered_pz_density, cmap='custom_heatmap')
        plt.colorbar()
        plt.title('Filtered Phi-Z density')
    if results in ('all', 'phase'):
        plt.figure()
        plt.plot([i.phase for i in phase_stamps][:1000])
        plt.title('Phase stamps')
        plt.figure()
        plt.plot([i.matchval for i in phase_stamps][:1000])
        plt.title('Phase matchvals')
    if results in ('all', 'z'):
        plt.figure()
        plt.plot(z_stamps)
        plt.title('Z stamps')
    if results in ('all', 'diff'):
        plt.figure()
        plt.imshow(diff_matrix)
        plt.colorbar()
        plt.title('Diff matrix')
        plt.figure()
        plt.plot(
            diff_matrix[
                :,
                diff_matrix_rows if diff_matrix_rows is not None else 1
            ]
        )
        plt.title('Diff matrix rows')
    if results in ('all', 'canon'):
        canon_hb_start = canon_hb_stats[0]
        canon_hb_len = canon_hb_stats[1]

        plot_matrix_rows(diff_matrix, 'Canon matrix diff corr',
                         5, canon_hb_start, canon_hb_len, canon_hb_len*20, )

        # canon_diff_mat = diff_matrix[
        #     :canon_hb_len * 20, canon_hb_start: canon_hb_start + canon_hb_len: 5]

        # offset_stack = np.zeros(canon_diff_mat.shape)
        # offset_preval = np.zeros(canon_diff_mat.shape)
        # for i in range(offset_stack.shape[1]):
        #     offset_stack[:, i] = i * 0.1
        #     offset_preval[:10, i] = i * 0.1

        # plt.figure()
        # plt.plot(canon_diff_mat + offset_stack)
        # plt.title('Canon matrix diff corr-stacked')
        # plt.figure()
        # plt.plot((canon_diff_mat + offset_preval)[:, :11], '-')
        # plt.plot((canon_diff_mat + offset_preval)[:, 11:22], '--')
        # plt.plot((canon_diff_mat + offset_preval)[:, 22:], '-.')
        # plt.title('Canon matrix diff corr-superimposed')

    plt.show()


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

    # cm_hb = compute_heartbeat_length(d[:, :, 2:-2], comparison_plot=True,
    # nperseg=50000)
    cm_hb = 94

    # print frames
    # compute_dwt_matrix(setting['bf_crop_path'], frames[:728],
    # write_to=setting['bf_us_dwt_path'], pkl_file_name='dwt_array_bf_us.pkl',
    # upsample=True)
    canonical_hb_start = compute_canonical_heartbeat(
        d[:, :, 2:-2], np.round(cm_hb).astype('int'))
    # #
    canonical_hb_frames = frames[
        canonical_hb_start: canonical_hb_start + np.round(cm_hb).astype('int')]
    print canonical_hb_frames
