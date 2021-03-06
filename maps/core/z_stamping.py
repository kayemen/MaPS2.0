if __name__ == '__main__':
    import sys
    sys.path.append('D:\\Scripts\\MaPS\\MaPS scripts\\')

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats
from skimage.transform import resize

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2
from maps.helpers.gui_modules import load_frame, extract_window
from maps.helpers.misc import pickle_object, unpickle_object
from maps.settings import setting, THREADED
from maps.helpers.misc import LimitedThread

import os
import logging
import maps
import csv
import cv2
import time
import threading
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# # Uncomment if manual overide of settings is needed
# setting['ignore_zooks_at_start'] = 1  # setting#
# setting['ignore_startzook'] = 7  # setting#
# setting['ignore_endzook'] = 3  # setting#
# setting['BF_resolution'] = 0.6296  # setting#
# setting['image_prefix'] = 'Phase_Bidi1_'  # setting#
# setting['workspace'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'  # setting#
# setting['resampling_factor'] = 5  # setting#
# setting['slide_limit'] = 5  # setting#
# setting['ZookZikPeriod'] = 192  # setting#
# setting['index_start_at'] = 1  # setting#
# setting['num_digits'] = 5  # setting#
# setting['first_minima'] = 0  # setting#


# TODO: add logging


def load_kymograph(kymo_path):
    '''
    Load kymograph tiff and return np array of values
    INPUTS:
        kymo_path: path to kymograph tiff files
    OUTPUTS:
        kymo_array: numpy array of kymgraph pixel values
    '''
    kymo_array = np.asarray(tff.imread(kymo_path))
    logger.info('Kymograph dimensions: %s' % str(kymo_array.shape))
    return kymo_array


def compute_raw_zstage(kymo_array, match_value=255):
    '''
    Computes raw ztage motion in pixels using kymograph as starting point for the method. This is used for further estimation of actual motion of the pixels.
    Find rightmost pixel matching match_value in each row and return as array
    Method 1: Assumes right side of pixel has white pixels, left of image has black pixels in the kymograph. Finds the rightmost black pixel.
    Method 2: calculates first order difference of each row of pixels. Uses the point with the largest difference.

    INPUTS:
        kymo_array: numpy array of kymgraph pixel values
        match_value: value to find the rightmost of
    OUTPUTS:
        z_stage: array of distances of rightmost pixel from edge
    '''
    # Use 'existing' to use pixel matching, 'new' for 1st order difference method
    method = 'new'

    z_stage = np.asarray([])

    z_stage = np.zeros(kymo_array.shape[0])
    for frame_no in xrange(len(z_stage)):
        frame_row = kymo_array[frame_no, :].astype(int)

        # Older approach. Finding extremum position of pixel of match_value
        if method == 'existing':
            matching_pixels = np.asarray(np.where(frame_row == 0))

        # New approach. Taking first order difference of row
        elif method == 'new':
            diff_row = frame_row[:-1] - frame_row[1:]
            # TODO: Once kymograph generation is done internally this can be changed to point not matching 0
            matching_pixels = np.asarray(np.where(np.absolute(diff_row) == match_value))
        z_stage[frame_no] = matching_pixels[0, 0]
        # z_stage[frame_no] = matching_pixels[0, -1]

    pickle_object(z_stage, 'z_stage')

    return z_stage


def compute_zooks(z_stage):
    '''
    Find the maxima and minima points in the z_stage data.
    From one minima to the next maxima is a zook. From a minima to the next maxima is a zik
    The fisrt minima is obtained from settings.
    Maxima points are the points where the z_stage values drop by a value greater than the threshold set. Minima points are the points with lowest value between 2 maxima points.

    INPUTS:
        z_stage: array of distances of rightmost pixel from edge
        maxima_threshold: Threshold difference above which value is considered maxima
    OUTPUTS:
        maxima_points: index of points at which maxima occur in z_stage
        maxima_values: actual values of maxima
        minima_points: index of points at which minima occur in z_stage
        minima_values: actual values of minima
    '''
    maxima_threshold = setting['kymo_zook_threshold']

    maxima_points = np.where((z_stage[1:] - z_stage[:-1]) < -maxima_threshold)[0].flatten()
    maxima_points = maxima_points[np.where(maxima_points > setting['first_minima'])]
    maxima_values = z_stage[maxima_points]

    # # TODO: TEST Include the last zook and test if it breaks anything
    # # Maxima in last zook. Uncomment after testing
    # np.append(maxima_values, z_stage[-1])

    minima_points = np.asarray(setting['first_minima'])
    for i in xrange(len(maxima_points) - 1):
        new_minimas = np.argmin(z_stage[maxima_points[i]: maxima_points[i + 1]])
        minima_points = np.append(minima_points, maxima_points[i] + new_minimas)

    # # TODO: TEST Include the last zook and test if it breaks anything
    # # Minima in last zook. Uncomment if needed
    # new_minimas = np.argmin(z_stage[maxima_points[-1]:])
    # minima_points = np.append(minima_points, maxima_points[-1] + new_minimas)

    minima_values = z_stage[minima_points]

    # logger.debug('Maxima points:\n%s' % maxima_points)
    # logger.debug('Maxima values:\n%s' % maxima_values)
    # logger.debug('Minima points:\n%s' % minima_points)
    # logger.debug('Minima values:\n%s' % minima_values)

    zooks = [(start, end) for start, end in zip(minima_points, maxima_points)]

    pickle_object(
        (maxima_points, maxima_values, minima_points, minima_values),
        'extremum_points'
    )

    pickle_object(zooks, 'zooks')

    return (maxima_points, maxima_values, minima_points, minima_values, zooks)


def compute_zookzik_stats(maxima_points, minima_points):
    '''
    Find the statistics of zook and zik. See list of outputs to see exact stats computed
    INPUTS:
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        zz_stats: Dictionary containing:-
            zook_lengths: Length of each zook. Distance from maxima to next minima
            zook_length_mean: Mean of all zook lengths
            zook_length_mode: Most common zook length
            zook_length_max: Maximum zook length
            zik_lengths: Length of each zik. Distance from minima to next maxima
            zik_length_mean: Mean of all zook lengths
            zik_length_mode: Most common zook length
            zik_length_max: Maximum zook length
    '''
    zz_stats = {}
    zz_stats['zook_lengths'] = maxima_points[:] - minima_points[:len(maxima_points)]
    zz_stats['zook_length_mean'] = np.round(np.mean(zz_stats['zook_lengths']))
    zz_stats['zook_length_mode'] = np.round(stats.mode(zz_stats['zook_lengths']))
    zz_stats['zook_length_max'] = np.max(zz_stats['zook_lengths'])

    zz_stats['zik_lengths'] = minima_points[1:] - maxima_points[:-1]
    zz_stats['zik_length_mean'] = np.round(np.mean(zz_stats['zik_lengths']))
    zz_stats['zik_length_mode'] = np.round(stats.mode(zz_stats['zik_lengths']))
    zz_stats['zik_length_max'] = np.max(zz_stats['zik_lengths'])

    logger.debug('\n'.join(['%s: %s' % (key, str(val)) for key, val in zz_stats.iteritems()]))

    pickle_object(zz_stats, 'zz_stats')

    return zz_stats


def compute_zook_extents(zook_end, zook_start):
    '''
    Function to compute the extent of frames to process in each zook. Added function to clean repeated code. Every zook has some frames ignored at start and end. Given the extent of the zook (start point and end point i.e zook_end and next minima point) retuns the start and end frame numbers.
    '''
    start_slice = zook_start + setting['ignore_startzook']
    end_slice = zook_end - setting['ignore_endzook'] + 1

    return (start_slice, end_slice)


def compute_zstamp(z_stage, maxima_points, minima_points, ref_frame_no=0):
    '''
    Remove the constant bias in z_stage values. Pull down/up to get rid of any constant shift terms. Shift done makes the reference frame have zero shift

    INPUTS:
        z_stage: array of distances of rightmost pixel from edge
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        z_stamp: Adjusted values of z_stage
        z_stamp_physical: z_stamp_values in physical units instead of frames
    '''
    z_stamp = np.zeros(z_stage.shape)

    for i in xrange(len(minima_points)):
        start_slice, end_slice = compute_zook_extents(maxima_points[i], minima_points[i])

        z_stamp[start_slice: end_slice] = \
            z_stage[start_slice: end_slice] - z_stage[ref_frame_no] * np.ones(end_slice - start_slice)

    z_stamp_physical = z_stamp * setting['BF_resolution']

    pickle_object(z_stamp, 'z_stamp')
    pickle_object(z_stamp_physical, 'z_stamp_physical')

    return (z_stamp, z_stamp_physical)


def compute_ideal_zstamp(z_stage, maxima_points, minima_points, ref_frame_no=0):
    '''
    Compute the ideal shift between the minima and maxima points deterministically. Use this to compensate for quantization errors in actual values. Find the most commons slope of shift/frame amongst all the zooks, and generates a determisitic z stamp with lines having that slope for every zook. Adjusts the y intercept for each zook so that reference frame always has shift of exactly 0.

    INPUTS:
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        z_stamp_det: deterministic version of z_stamp
    '''
    # TODO: Change to use minim and maxima not determisinstic
    minima_points_deterministic = np.arange(0, len(minima_points)) * setting['ZookZikPeriod']
    maxima_points_deterministic = minima_points_deterministic + setting['ZookZikPeriod'] - 1

    z_stamp_det = np.zeros(z_stage.shape)

    pixel_shift_length = stats.mode(z_stage[maxima_points] - z_stage[minima_points])[0]

    pixel_shift_start = stats.mode(z_stage[minima_points])[0]
    pixel_shift_per_frame = pixel_shift_length / setting['ZookZikPeriod']

    # Copied as is. Not used in previous code
    # zooklengthadjustment = (np.max(z_stage[maximapoints]-z_stage[minimapoints])-np.min(z_stage[maximapoints]-z_stage[minimapoints])) # always
    # #check to make sure that the zebrafish bondary doesn't move much between the
    # #last frame of one zook and first frame of next zook

    ref_frame_bias = (ref_frame_no - setting['index_start_at']) % setting['ZookZikPeriod']
    print ref_frame_bias

    for i in xrange(len(minima_points)):
        z_stamp_det[minima_points_deterministic[i]: maxima_points_deterministic[i] +
                    1] = np.round(np.arange(0 - ref_frame_bias, setting['ZookZikPeriod'] - ref_frame_bias) * pixel_shift_per_frame * setting['resampling_factor'])

    # if setting['plot_steps']:
    #     plt.plot(z_stamp_det)
    #     plt.show()

    pickle_object(z_stamp_det, 'z_stamp_det')

    return z_stamp_det


def load_correlation_window_params(csv_file):
    '''
    Load the bounding regions of window used for correlation. As a convention, (x_end, height) and (y_end, width) are only stored. x_start and y_start are computed as and when needed to ensure that height and width dont change due to quantization or floating point errors

    INPUTS:
        csv_file(str): name of csv file with params
    OUTPUTS:
        x_start(int): top bounding pixel
        x_end(int): bottom bounding pixel
        y_start(int): left bounding pixel
        y_end(int): right bounding pixel
        height(int): height of bounding region in pixels
        width(int): width of bounding region in pixels
        frame_no(int): the frame number in which this window was selected
    '''
    raw_data = {}
    with open(os.path.join(setting['workspace'], csv_file)) as csv_data:
        data = csv.reader(csv_data)
        for row in data:
            raw_data[row[0]] = row[1]

    height = int(raw_data['height'])
    x_end = int(raw_data['x_end'])
    x_start = x_end - height
    y_end = int(raw_data['y_end'])
    width = int(raw_data['width'])
    y_start = y_end - width
    frame_no = int(raw_data['frame'])

    return (x_start, x_end, y_start, y_end, height, width, frame_no)


def check_convexity(corr_points):
    '''
    Function to check convexity of correlation parameters.
    Method 1: Find point with maximum correlation. Check if it is greater than its neighbours. Mark as convex.
    Method 2: Find point with maximum correlation. Check if the prior and posterior partitions it splits the range into are sorted in ascending and descending order respectively.
    INPUTS:
        corr_points(array): array of correlation values in local neighbourhood of a given shift
    OUTPUTS:
        (bool): True if data set is convex
    '''
    method = 'new'

    # Existing convex checking code
    if method == 'existing':
        best_slide = np.argmax(corr_points)
        if best_slide > 0 and best_slide < (len(corr_points) - 1):
            best_point = corr_points[best_slide]
            prior_point = corr_points[best_slide - 1]
            post_point = corr_points[best_slide + 1]
            if best_point > prior_point and best_point > post_point:
                return True
        return False

    # Alternate convexity code
    elif method == 'new':
        best_slide = np.argmax(corr_points)

        if np.all(corr_points[1:best_slide] >= corr_points[:best_slide - 1]) and np.all(corr_points[best_slide + 2:] <= corr_points[best_slide + 1:-1]):
            return True
        if np.all(corr_points[1:best_slide] <= corr_points[:best_slide - 1]) and np.all(corr_points[best_slide + 2:] >= corr_points[best_slide + 1:-1]):
            return True
    return False


def calculate_frame_optimal_shift_yz(img_path, frame_no, ref_frame_window, x_start_resized, x_end_resized, y_start_resized_frame, y_end_resized_frame, z_stamp_optimal, y_stamp_optimal, z_stamp_det):
    '''
    Calculate optimal shift for single frame.
    This method does the z stamp calculation with wiggle of "setting['slide_limit']" pixels in low-res domain.
    It also computes the vertical shift in the frames to compensate for any rise/drop in the fish over a zook.
    This function can be threaded in the calling scope to improve performance.
    It uses the openCV matchTemplate method to find the correlation between the selected window and teh ROI. It then finds the maxima and computes shift in x and y directions.
    '''

    cv2_methods = {'ccorr': cv2.TM_CCORR_NORMED, 'ccoeff': cv2.TM_CCOEFF_NORMED, 'sqdiff': cv2.TM_SQDIFF_NORMED}

    slide_limit_resized = setting['slide_limit'] * setting['resampling_factor']
    slide_limit_x_resized = setting['y_slide_limit'] * setting['resampling_factor']

    # Load frame and scale
    frame = load_frame(img_path, frame_no)

    # Not needed as match template creates correct size
    # # Correlation matrix for frame. Reset to zero for each frame
    # corr_array = np.zeros((2 * slide_limit_x_resized + 1, 2 * slide_limit_resized + 1))

    # Compute shifted window params
    x_start_resized_shifted = x_start_resized - slide_limit_x_resized
    x_end_resized_shifted = x_end_resized + slide_limit_x_resized
    y_start_resized_shifted = y_start_resized_frame - slide_limit_resized
    y_end_resized_shifted = y_end_resized_frame + slide_limit_resized

    # Compute window of frame
    frame_window = extract_window(frame, x_start_resized_shifted, x_end_resized_shifted, y_start_resized_shifted, y_end_resized_shifted)

    # plt.imshow(frame_window)
    # plt.figure()
    # plt.imshow(ref_frame_window)
    # plt.show()

    # TODO: Try on entire frame instead of window
    corr_array = cv2.matchTemplate(frame_window.astype('float32'), ref_frame_window.astype('float32'), cv2_methods['ccorr'])

    # plt.imshow(corr_array)
    # plt.show()
    # Check convexity
    # if not (check_convexity(corr_array.max(axis=0)) and check_convexity(corr_array.max(axis=1))):
    #     print 'Convexity error in frame %d' % frame_no

    # Compute ideal shift using corr_array
    z_stamp_optimal[frame_no] = z_stamp_det[frame_no] + corr_array.max(axis=0).argmax() - slide_limit_resized
    y_stamp_optimal[frame_no] = corr_array.max(axis=1).argmax() - slide_limit_x_resized


def compute_frame_shift_values_yz(img_path, maxima_points, minima_points, z_stamp_det, x_end, height, y_end, width, ref_frame_no, offset=0):
    '''
    Compute optimal z stamp values for entire image sequence.
    '''
    THREADED = True

    z_stamp_optimal = np.zeros(z_stamp_det.shape)
    y_stamp_optimal = np.zeros(z_stamp_det.shape)

    # Time stamp values
    zook_time_stats = np.zeros(len(minima_points) - setting['ignore_zooks_at_start'])

    x_end_resized = x_end * setting['resampling_factor']
    height_resized = height * setting['resampling_factor']
    x_start_resized = x_end_resized - height_resized
    y_end_resized = y_end * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']
    y_start_resized = y_end_resized - width_resized

    # Load reference frame
    ref_frame = load_frame(img_path, ref_frame_no)

    # Compute window of reference frame
    ref_frame_window = extract_window(ref_frame, x_start_resized, x_end_resized, y_start_resized, y_end_resized)

    frame_threads = []

    # Create matrix of optimized z_stamps
    for zook in np.arange(setting['ignore_zooks_at_start'], len(minima_points)):
        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])

        start_frame += offset - 1
        end_frame += offset - 1

        logging.debug('Processing Zook#%d' % zook)
        # print 'Processing Zook#%d' % zook
        tic = time.time()
        for frame_no in np.arange(start_frame, end_frame):
            # Compute window params
            y_end_resized_frame = y_end_resized + z_stamp_det[frame_no]
            y_start_resized_frame = y_end_resized_frame - width_resized

            # TODO: Detect non convex corr arrays as bad zooks and make list
            if not THREADED:
                calculate_frame_optimal_shift_yz(
                    img_path, frame_no,
                    ref_frame_window,
                    x_start_resized,
                    x_end_resized,
                    y_start_resized_frame,
                    y_end_resized_frame,
                    z_stamp_optimal,
                    y_stamp_optimal,
                    z_stamp_det
                )
            else:
                frame_thread = threading.Thread(
                    target=calculate_frame_optimal_shift_yz,
                    args=(
                        img_path, frame_no,
                        ref_frame_window,
                        x_start_resized,
                        x_end_resized,
                        y_start_resized_frame,
                        y_end_resized_frame,
                        z_stamp_optimal,
                        y_stamp_optimal,
                        z_stamp_det
                    )
                )
                frame_thread.start()
                frame_threads.append(frame_thread)

        zook_time_stats[zook - setting['ignore_zooks_at_start']] = time.time() - tic

    for frame_thread in frame_threads:
        frame_thread.join()

    pickle_object(z_stamp_optimal, 'z_stamp_optimal')
    pickle_object(y_stamp_optimal, 'y_stamp_optimal')
    pickle_object(zook_time_stats, 'ZY_stamping_time')

    print 'Average time for YZ stamping:%f s/zook' % (sum(zook_time_stats) / len(zook_time_stats))

    return z_stamp_optimal, y_stamp_optimal


def compute_zstamp_curvefit(z_stamp_optimal, maxima_points, minima_points):
    '''
    Compute best line fit for each zook and residues of the shift from fit line.
    The best line fit is the linear regression curve for each zook. the residue is the difference between this computed value for each frame and the actual shift.
    INPUT:
        z_stamp_optimal(ARRAY):
        maxima_points:
        minima_points:
    OUTPUT:
        z_stamp_curvefit:
        residues:
    '''

    z_stamp_curvefit = np.zeros(z_stamp_optimal.shape)

    slopes = []

    for zook in xrange(setting['ignore_zooks_at_start'], len(minima_points)):
        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])

        z_stamps = z_stamp_optimal[start_frame: end_frame]
        frame_nos = np.arange(start_frame, end_frame)

        # Using linear regression to compute the best fitting line for each zook
        slope, intercept, _, _, _ = stats.linregress(frame_nos, z_stamps)

        slopes.append(slope)
        z_stamp_curvefit[start_frame: end_frame] = slope * frame_nos + intercept

    residues = z_stamp_optimal - z_stamp_curvefit

    pickle_object(z_stamp_curvefit, 'z_stamp_curvefit')
    pickle_object(residues, 'residues')
    pickle_object(slopes, 'slopes')

    return z_stamp_curvefit, residues, slopes


def detect_bad_zooks(residues, maxima_points, minima_points, slopes, threshold_mult=0.3):
    '''
    Find the frames in a zook which are shifted by unnaturally large amounts. The threshold for detecting if a shift is bad is if it deviates more than "c=27%" from the slope of the shifts in that zook
    INPUTS:
        residues(ARRAY): Array of residues between optimal z_stamp and the linear regression line of the z stamp values
        maxima_points:
        minima_points:
        slopes(ARRAY): array of slopes of the line in the cureve fit z_stamps for each zook
    OUTPUTS:
        bad_zooks(list): list of bad_zook tuples. Each bad zook tuple consists of the zook#, the shift value of deviant shifts and the location of deviant shifts
    '''
    bad_zooks = []

    print 'Running with threshold multiplier at %.2f' % (threshold_mult)
    for zook in xrange(setting['ignore_zooks_at_start'], len(minima_points)):
        threshold = threshold_mult * slopes[zook - setting['ignore_zooks_at_start']]

        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])
        zook_residue = residues[start_frame: end_frame]

        bad_shifts = zook_residue[np.where(np.absolute(zook_residue) > threshold)]
        bad_shift_loc = np.where(np.absolute(zook_residue) > threshold)[0] + start_frame

        is_bad_zook = bool(len(bad_shifts))

        if is_bad_zook:
            bad_zooks.append((zook, bad_shifts, bad_shift_loc))

    pickle_object(bad_zooks, 'bad_zooks')

    return bad_zooks


# TODO: change to use settings
def z_stamping_step_yz(kymo_path, frame_count, phase_img_path, use_old=False, datafile_name='z_stamp_optimal.pkl', datafile_name_x='x_stamp_optimal.pkl'):
    '''
    Single function to handle entire z stamping step
    '''
    kymo_data = load_kymograph(kymo_path)

    # Load window parameters
    x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_correlation_window_params('corr_window.csv')

    z_stage_data = compute_raw_zstage(kymo_data[:frame_count, :])

    # plt.plot(z_stage_data)

    (maxp, maxv, minp, minv, zooks) = compute_zooks(z_stage_data)

    # compute_zookzik_stats(maxp, minp)
    z_stamp, _ = compute_zstamp(z_stage_data, maxp, minp, ref_frame_no)
    z_stamp_det = compute_ideal_zstamp(z_stage_data, maxp, minp, ref_frame_no)

    # plt.plot(z_stamp_det)
    # plt.show()

    if not use_old:
        z_stamp_opt, y_stamp_opt = compute_frame_shift_values_yz(phase_img_path, maxp, minp, z_stamp_det, x_end, height, y_end, width, ref_frame_no, offset=setting['index_start_at'])

    else:
        z_stamp_opt = unpickle_object(datafile_name)
        y_stamp_opt = unpickle_object(datafile_name_x)

    z_stamp_cf, res, slope_list = compute_zstamp_curvefit(z_stamp_opt, maxp, minp)

    bad_zooks = detect_bad_zooks(res, maxp, minp, slope_list)

    return (z_stamp_opt, y_stamp_opt, z_stamp_cf, res, bad_zooks, minp)


def stationary_z_stamping(frame_count, phase_img_path, use_old=False, datafile_name='z_stamp_optimal_stat.pkl', datafile_name_x='x_stamp_optimal_stat.pkl'):
    x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_correlation_window_params('corr_window.csv')

    z_stamp_det = np.zeros(setting['stat_frame_count'])

    if not use_old:
        z_stamp_opt, y_stamp_opt = compute_frame_shift_values_yz(
            phase_img_path, [setting['stat_index_start_at'] + setting['stat_frame_count']],
            [setting['stat_index_start_at']], z_stamp_det, x_end, height, y_end, width, ref_frame_no, offset=setting['stat_index_start_at'])
    else:
        z_stamp_opt = unpickle_object(datafile_name)
        y_stamp_opt = unpickle_object(datafile_name_x)

    return z_stamp_opt, y_stamp_opt


def downsize_and_writeframe(img_path, frame_no, x_start_frame, x_end_frame, y_start_frame, y_end_frame):
    cropped_frame = load_frame(img_path, frame_no, upsample=True, crop=True, cropParams=(x_start_frame, x_end_frame, y_start_frame, y_end_frame))

    # cropped_frame = extract_window(frame, x_start_frame, x_end_frame + setting['resampling_factor'], y_start_frame, y_end_frame + setting['resampling_factor'])

    cropped_frame_downsized = resize(
        cropped_frame,
        (
            cropped_frame.shape[0] / setting['resampling_factor'],
            cropped_frame.shape[1] / setting['resampling_factor']
        ),
        preserve_range=True
    )

    if setting['validTiff']:
        cropped_frame_downsized = cropped_frame_downsized.astype('uint16')

    writetiff(cropped_frame_downsized, setting['cropped_bf_images'], frame_no)


def shift_frames_and_store_yz(img_path, z_stamps, y_stamps, discarded_zooks_list, minima_points, x_end, height, y_end, width, ref_frame_no, mask_path=''):
    bad_zooks = [bz[0] for bz in discarded_zooks_list]

    height_resized = height * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']
    x_end_resized = x_end * setting['resampling_factor']
    y_end_resized = y_end * setting['resampling_factor']

    processed_frame_list = []
    crop_write_time_stats = []
    writing_threads = []

    # TODO: Add timing
    for zook in np.arange(setting['ignore_zooks_at_start'], len(minima_points)):
        print 'Writing zook %d' % zook
        if zook in bad_zooks:
            continue

        tic = time.time()
        maxima_points = (np.array(minima_points) + setting['ZookZikPeriod'] - 1)
        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])
        for frame_no in np.arange(start_frame, end_frame):
            processed_frame_list.append(frame_no)
            y_end_frame = y_end_resized + z_stamps[frame_no]
            y_start_frame = y_end_frame - width_resized
            x_end_frame = x_end_resized + y_stamps[frame_no]
            x_start_frame = x_end_frame - width_resized

            writing_thread = LimitedThread(
                target=downsize_and_writeframe,
                args=(
                    img_path,
                    frame_no,
                    x_start_frame,
                    x_end_frame,
                    y_start_frame,
                    y_end_frame
                )
            )
            writing_thread.start()
            writing_threads.append(writing_thread)

        crop_write_time_stats.append(time.time() - tic)

    # Wait for all writing threads to finish
    for writing_thread in writing_threads:
        writing_thread.join()
        print len([1 for th in writing_threads if th.isAlive()]), ' threads still processing'

    pickle_object(processed_frame_list, 'processed_frame_list.pkl', dumptype='pkl')
    pickle_object(crop_write_time_stats, 'Cropping_time_stats')

    print 'Average time for Cropping and writing to disk:%f s/zook' % (sum(crop_write_time_stats) / len(crop_write_time_stats))

    return processed_frame_list


if __name__ == '__main__':
    from maps.helpers.logging_config import logger_config
    logging.config.dictConfig(logger_config)

    logger = logging.getLogger('maps.core.z_stamping')

    # Path to settings json for preloading settings
    settings_json = 'D:\\Scripts\\MaPS\\MaPS scripts\\maps\\current_inputs.json'

    # Initialize the settings object
    read_setting_from_json(settings_json)

    setting['ignore_zooks_at_start'] = 1
    setting['ignore_startzook'] = 7
    setting['ignore_endzook'] = 3
    setting['BF_resolution'] = 0.6296
    setting['image_prefix'] = 'Phase_Bidi1_'
    setting['workspace'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
    setting['resampling_factor'] = 5
    setting['slide_limit'] = 5
    setting['ZookZikPeriod'] = 192
    setting['index_start_at'] = 1
    setting['num_digits'] = 5
    setting['first_minima'] = 0

    z_stamp_opt, z_stamp_cf, res, bad_zooks = z_stamping_step(
        'D:\Scripts\MaPS\Data sets\KM-XZ_0.tif',
        9999,
        'D:\\Scripts\\MaPS\\Data sets\\Phase_Bidi\\'
    )

    # if setting['plot_steps']:
    #     plt.plot(z_stamp_opt)
    #     plt.show()

    for bad_zook in bad_zooks:
        print '=' * 80
        print 'Zook #%d' % bad_zook[0]
        print 'Fault locations:'
        print bad_zook[2]
        print 'Fault values:'
        print bad_zook[1]

    # if setting['plot_steps']:
    #     plt.plot(res)
    #     plt.show()

    # # TODO: Set to true to plot histogram and store for each bad zook
    # if is_bad_zook and plot_hist:
    #     hist, bins = np.histogram(zook_residue, bins=100)
    #     width = 0.8 * (bins[1] - bins[0])
    #     center = (bins[:-1] + bins[1:]) / 2
    #     plt.bar(center, hist, align='center', width=width)
    #     plt.savefig(
    #         os.path.join(
    #             setting['workspace'],
    #             'Hist vals\\%d_per\\zook%d.png' % (int(threshold_mult * 100), zook)
    #         ),
    #         bbox_inches='tight'
    #     )
    #     plt.gcf().clear()
