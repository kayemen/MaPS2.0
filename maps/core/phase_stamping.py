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
from maps.helpers.gui_modules import cv2_methods


import os
import logging
import time
import csv
import cv2
import platform

logger = logging.getLogger(__name__)


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


def read_dwt_matrix(read_from):
    pass


def write_dwt_matrix(dwt_matrix, write_to, frame_indices, validTiff=False):
    '''
    Write the frames from DWT array to disk. Will write as 64bit float values. Not valid TIFF files
    '''

    if len(frame_indices) < dwt_matrix.shape[2]:
        pass
        # raise Exception('Insufficient frame indices')
    if validTiff:
        dwt_matrix = dwt_matrix.astype('int16')
    for index, frame_id in enumerate(frame_indices):
        writetiff(dwt_matrix[:, :, index], write_to, index)


def compute_dwt_matrix(img_path, frame_indices, write_to=None, upsample=False, pkl_file_name='dwt_array.pkl'):
    '''
    Compute the DWT matrix for a sequence of images. Optionally writes the DWT array as images to disk. Also writes as pkl file
    '''
    first_image = importtiff(img_path, frame_indices[0])
    (cA, (cH, cV, cD)) = pywt.dwt2(first_image, 'db1', mode='sym')

    dwt_array = np.zeros((cA.shape[0], cA.shape[1], len(frame_indices)))
    for (frame_id, frame) in enumerate(frame_indices):
        frame_image = importtiff(img_path, frame)
        (cA, _) = pywt.dwt2(frame_image, 'db1', mode='sym')

        dwt_array[:, :, frame_id] = cA

    if upsample:
        dwt_array = upsample_dwt_array(dwt_array)
        plt.imshow(dwt_array[:, :, 0], cmap='gray')
        plt.show()

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
        # plt.imshow(first_image)
        # plt.show()
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
            # Will require filtering
            elif method == 'sad':
                x.append(np.sum(np.absolute(first_image - second_image)))
            # OpenCV methods ccorr, ccoeff
            elif method in ('ccoeff_norm', 'ccorr_norm', 'mse_norm'):
                x.append(cv2.matchTemplate(second_image.astype('float32'), first_image.astype('float32'), cv2_methods[method])[0][0])
            # print x
            # raw_input()
        time_taken.append(time.time() - tic)
        # print x

        # For absdiff and MSE
        # if method in ('sad', 'mse_norm'):
        x = np.asarray(x)
        x = np.max(x) - x

        f, Pxx = signal.welch(x, im_fps, nperseg=nperseg)

        # print f
        t = im_fps / f
        if comparison_plot:
            plt.figure(211)
            plt.plot(x)
            print t[np.argmax(Pxx) - 10: np.argmax(Pxx) + 10]
            comparison_plot = False
        # raw_input()
        Pxx[np.where(t > 400)] = 0
        hb_array.append(t[np.argmax(Pxx)])
        plt.figure(212)
        cl = next(color)
        plt.plot(t, Pxx, c=cl)
        # plt.hold(True)

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

    # print sad_matrix.shape
    # print np.max(sad_matrix)
    # print np.mean(sad_matrix)
    # print np.median(sad_matrix)
    # plt.imshow(sad_matrix, cmap='hot')
    # plt.colorbar()
    # plt.show()
    # raw_input()

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
    setting['data_dump'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
    setting['resampling_factor'] = 5
    setting['slide_limit'] = 5
    # setting['ZookZikPeriod'] = 728
    setting['index_start_at'] = 1
    setting['num_digits'] = 5
    setting['first_minima'] = 0

    with open(os.path.join(setting['data_dump'], 'processed_frame_list.pkl')) as pkfp:
        frames = pickle.load(pkfp)

    COMPUTE_DWT = True

    if COMPUTE_DWT:
        d = compute_dwt_matrix(setting['bf_period_path'], frames[:500],
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
