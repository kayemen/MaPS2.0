if __name__ == '__main__':
    import sys
    sys.path.append('D:\\Scripts\\MaPS\\MaPS scripts\\')

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats
from skimage.transform import resize

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2
from maps.helpers.misc import pickle_object, make_or_clear_directory
from maps.settings import setting

import os
import logging
import maps
import csv
import cv2
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# # Uncomment if manual overide of settings is needed
# setting['ignore_zooks_at_start'] = 1  # setting#
# setting['ignore_startzook'] = 7  # setting#
# setting['ignore_endzook'] = 3  # setting#
# setting['BF_resolution'] = 0.6296  # setting#
# setting['image_prefix'] = 'Phase_Bidi1_'  # setting#
# setting['data_dump'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'  # setting#
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
    logger.debug('Kymograph dimensions: %s' % str(kymo_array.shape))
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
        z_stage[frame_no] = matching_pixels[0, -1]
    return z_stage


def compute_maxima_minima(z_stage, maxima_threshold=10):
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
    maxima_points = np.where((z_stage[1:] - z_stage[:-1]) < -maxima_threshold)[0].flatten()
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

    logger.debug('Maxima points:\n%s' % maxima_points)
    logger.debug('Maxima values:\n%s' % maxima_values)
    logger.debug('Minima points:\n%s' % minima_points)
    logger.debug('Minima values:\n%s' % minima_values)

    return (maxima_points, maxima_values, minima_points, minima_values)


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

    return zz_stats


def compute_zook_extents(maxima_point, minima_point):
    '''
    Function to compute the extent of frames to process in each zook. Added function to clean repeated code. Every zook has some frames ignored at start and end. Given the extent of the zook (start point and end point i.e maxima_point and next minima point) retuns the start and end frame numbers.
    '''
    start_slice = minima_point + setting['ignore_startzook']
    end_slice = maxima_point - setting['ignore_endzook'] + 1

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
        # TODO: Can bias removal happen at end
        z_stamp[start_slice: end_slice] = \
            z_stage[start_slice: end_slice] - z_stage[ref_frame_no] * np.ones(end_slice - start_slice)
    # Bias removal. Check if it should be z_stage[0] or z_stage[setting['ignore_startzook']]
    # z_stamp -= np.ones(z_stamp.shape)*z_stage[0]

    z_stamp_physical = z_stamp * setting['BF_resolution']

    if setting['plot_steps']:
        plt.plot(z_stamp)
        plt.show()

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

    for i in xrange(len(minima_points)):
        z_stamp_det[minima_points_deterministic[i]: maxima_points_deterministic[i] +
                    1] = np.round(np.arange(0 - ref_frame_no, setting['ZookZikPeriod'] - ref_frame_no) * pixel_shift_per_frame * setting['resampling_factor'])

    if setting['plot_steps']:
        plt.plot(z_stamp_det)
        plt.show()

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
    with open(os.path.join(setting['data_dump'], csv_file)) as csv_data:
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


def extract_window(frame, x_start, x_end, y_start, y_end):
    '''
    Return the ROI from a frame within the bounding box specified by x_start, x_end, y_start, y_end
    '''
    frame_win = frame[int(x_start): int(x_end) + 1, int(y_start): int(y_end) + 1]
    return frame_win


def load_frame(img_path, frame_no, upsample=True):
    '''
    Load the tiff file of the frame, resize (upsample by resampling factor if needed) if needed and return image array.
    '''
    img = importtiff(img_path, frame_no, prefix=setting['image_prefix'], index_start_number=setting['index_start_at'], num_digits=setting['num_digits'])
    if upsample:
        img_resized = resize(
            img,
            (
                img.shape[0] * setting['resampling_factor'],
                img.shape[1] * setting['resampling_factor']
            ),
            preserve_range=True
        )

        return img_resized
    else:
        return img


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

        if np.all(corr_points[1:best_slide] >= corr_points[:best_slide - 1]) and np.all(corr_points[best_slide + 2:] <= corr_points[best_slide + 2:-1]):
            return True
    return False


def calculate_frame_optimal_shift(img_path, frame_no, ref_frame_window, x_start_resized, x_end_resized, y_start_resized_frame, y_end_resized_frame, z_stamp_optimal, z_stamp_det):
    '''
    Calculate optimal shift for single frame.
    This method does the z stamp calculation with wiggle of "setting['slide_limit']" pixels in low-res domain.
    This function can be threaded in the calling scope to improve performance.
    It first finds the correlation at large gaps, checks if they form a convex set, and then searches in the smaller space around the maxima if convex..
    '''
    slide_limit_resized = setting['slide_limit'] * setting['resampling_factor']

    # Load frame and scale
    frame = load_frame(img_path, frame_no)
    # Correlation matrix for frame. Reset to zero for each frame
    corr_array = np.zeros(2 * slide_limit_resized + 1)

    for slide_amount in np.arange(-slide_limit_resized, slide_limit_resized + 1, setting['resampling_factor']):
        pass
        # Compute shifted window params
        y_start_resized_shifted = y_start_resized_frame + slide_amount
        y_end_resized_shifted = y_end_resized_frame + slide_amount

        # Compute window of frame
        frame_window = extract_window(frame, x_start_resized, x_end_resized, y_start_resized_shifted, y_end_resized_shifted)

        # Compute correlation
        try:
            corr_array[slide_amount + slide_limit_resized] = corr2(ref_frame_window, frame_window)
        except:
            corr_array[slide_amount + slide_limit_resized] = -1
            # TODO: Raise error and stop process
            print 'Error in Frame:', frame_no

    # Compute best slide amount
    best_slide_amount = np.argmax(corr_array) - slide_limit_resized

    # Compute convexity
    is_convex = check_convexity(corr_array)

    if is_convex:
        # Compute best maxima in local region
        for slide_amount in np.arange(max(best_slide_amount - setting['resampling_factor'] + 1, -slide_limit_resized), min(best_slide_amount + setting['resampling_factor'], slide_limit_resized + 1)):
            # Compute shifted window params
            y_start_resized_shifted = y_start_resized_frame + slide_amount
            y_end_resized_shifted = y_end_resized_frame + slide_amount

            # Compute window of frame
            frame_window = extract_window(frame, x_start_resized, x_end_resized, y_start_resized_shifted, y_end_resized_shifted)

            # Compute correlation
            try:
                corr_array[slide_amount + slide_limit_resized] = corr2(ref_frame_window, frame_window)
            except:
                corr_array[slide_amount + slide_limit_resized] = -1
                # TODO: Raise error and stop process
                print 'Error in Frame:', frame_no

    else:
        # Find optima in overall region
        for slide_amount in np.arange(-slide_limit_resized, slide_limit_resized + 1):
            # Compute shifted window params
            y_start_resized_shifted = y_start_resized_frame + slide_amount
            y_end_resized_shifted = y_end_resized_frame + slide_amount

            # Compute window of frame
            frame_window = extract_window(frame, x_start_resized, x_end_resized, y_start_resized_shifted, y_end_resized_shifted)

            # Compute correlation
            try:
                corr_array[slide_amount + slide_limit_resized] = corr2(ref_frame_window, frame_window)
            except:
                corr_array[slide_amount + slide_limit_resized] = -1
                # TODO: Raise error and stop process
                print 'Error in Frame:', frame_no

    # Compute ideal shift using corr_array
    z_stamp_optimal[frame_no] = z_stamp_det[frame_no] + np.argmax(corr_array) - slide_limit_resized


def compute_frame_shift_values(img_path, maxima_points, minima_points, z_stamp_det):
    '''
    Compute optimal z stamp values for entire image sequence.
    '''
    z_stamp_optimal = np.zeros(z_stamp_det.shape)

    # Time stamp values
    zook_time_stats = np.zeros(len(minima_points) - setting['ignore_zooks_at_start'])

    # Load window parameters
    x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_correlation_window_params('corr_window.csv')

    x_start_resized = x_start * setting['resampling_factor']
    x_end_resized = x_end * setting['resampling_factor']
    y_start_resized = y_start * setting['resampling_factor']
    y_end_resized = y_end * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']

    # Shifting deterministic z stamp to make reference frame have 0 shift
    # TODO: TEST if resampling can be done here instead of before rounding step
    # z_stamp_det = (z_stamp_det - z_stamp_det[ref_frame_no]) * setting['resampling_factor']
    # z_stamp_det = (z_stamp_det - z_stamp_det[ref_frame_no])

    # Load reference frame
    ref_frame = load_frame(img_path, ref_frame_no)

    # Compute window of reference frame
    ref_frame_window = extract_window(ref_frame, x_start_resized, x_end_resized, y_start_resized, y_end_resized)

    # Create matrix of optimized z_stamps
    for zook in np.arange(setting['ignore_zooks_at_start'], len(minima_points)):
        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])

        print 'Processing Zook#%d' % zook
        tic = time.time()
        for frame_no in np.arange(start_frame, end_frame):
            # Compute window params
            y_start_resized_frame = y_start_resized + z_stamp_det_shifted[frame_no]
            y_end_resized_frame = y_end_resized + z_stamp_det_shifted[frame_no]

            # TODO: Run as spawned thread
            calculate_frame_optimal_shift(
                img_path, frame_no,
                ref_frame_window,
                x_start_resized,
                x_end_resized,
                y_start_resized_frame,
                y_end_resized_frame,
                z_stamp_optimal,
                z_stamp_det_shifted
            )
        print time.time() - tic
        zook_time_stats[zook - setting['ignore_zooks_at_start']] = time.time() - tic

    z_stamp_optimal_resized = z_stamp_optimal / setting['resampling_factor']

    return z_stamp_optimal, z_stamp_optimal_resized


def compute_zstamp_curvefit(z_stamp_optimal, maxima_points, minima_points):
    '''
    Compute best line fit for each zook and residues of the shift from fit line
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

    return z_stamp_curvefit, residues, slopes


def detect_bad_zooks(residues, maxima_points, minima_points, slopes):
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

    sdev = stats.tstd(residues)
    # threshold = sdev * 3

    c = 0.27  # threshold multiplier

    print 'Running with threshold multiplier at %.2f' % (c)
    for zook in xrange(setting['ignore_zooks_at_start'], len(minima_points)):
        threshold = c * slopes[zook - setting['ignore_zooks_at_start']]

        start_frame, end_frame = compute_zook_extents(maxima_points[zook], minima_points[zook])
        zook_residue = residues[start_frame: end_frame]

        bad_shifts = zook_residue[np.where(np.absolute(zook_residue) > threshold)]
        bad_shift_loc = np.where(np.absolute(zook_residue) > threshold)[0] + start_frame

        is_bad_zook = bool(len(bad_shifts))

        if is_bad_zook:
            bad_zooks.append((zook, bad_shifts, bad_shift_loc))

    return bad_zooks


def shift_frames_and_store(img_path, z_stamps):
    # TODO
    pass


def z_stamping_step(kymo_path, frame_count, phase_img_path, use_old=False, datafile_name='z_stamp_opt.pkl'):
    '''
    Single function to handle entire z stamping step
    '''
    kymo_data = load_kymograph(kymo_path)
    z_stage_data = compute_raw_zstage(kymo_data[:frame_count, :])

    (maxp, maxv, minp, minv) = compute_maxima_minima(z_stage_data)

    # compute_zookzik_stats(maxp, minp)
    z_stamp, _ = compute_zstamp(z_stage_data, maxp, minp)
    z_stamp_det = compute_ideal_zstamp(z_stage_data, maxp, minp)

    if not use_old:
        z_stamp_opt, z_stamp_opt_resized = compute_frame_shift_values(phase_img_path, maxp, minp, z_stamp_det)

        # TODO: Save as csv as well
        pickle_object(z_stamp_opt, datafile_name, dumptype='pkl')

    else:
        import pickle
        with open(os.path.join(setting['data_dump'], datafile_name)) as pkfp:
            z_stamp_opt = pickle.load(pkfp)

    z_stamp_cf, res, slope_list = compute_zstamp_curvefit(z_stamp_opt, maxp, minp)

    bad_zooks = detect_bad_zooks(res, maxp, minp, slope_list)

    return (z_stamp_opt, z_stamp_cf, res, bad_zooks)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from maps.helpers.logging_config import logger_config
    logging.config.dictConfig(logger_config)

    logger = logging.getLogger('maps.core.z_stamping')

    setting['ignore_zooks_at_start'] = 1
    setting['ignore_startzook'] = 7
    setting['ignore_endzook'] = 3
    setting['BF_resolution'] = 0.6296
    setting['image_prefix'] = 'Phase_Bidi1_'
    setting['data_dump'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
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

    plt.plot(z_stamp_opt)
    plt.show()
    if setting['plot_steps']:

    for bad_zook in bad_zooks:
        print '=' * 80
        print 'Zook #%d' % bad_zook[0]
        print 'Fault locations:'
        print bad_zook[2]
        print 'Fault values:'
        print bad_zook[1]

    plt.plot(res)
    plt.show()

    if setting['plot_steps']:

    # # TODO: Set to true to plot histogram and store for each bad zook
    # if is_bad_zook and plot_hist:
    #     hist, bins = np.histogram(zook_residue, bins=100)
    #     width = 0.8 * (bins[1] - bins[0])
    #     center = (bins[:-1] + bins[1:]) / 2
    #     plt.bar(center, hist, align='center', width=width)
    #     plt.savefig(
    #         os.path.join(
    #             setting['data_dump'],
    #             'Hist vals\\%d_per\\zook%d.png' % (int(c * 100), zook)
    #         ),
    #         bbox_inches='tight'
    #     )
    #     plt.gcf().clear()
