if __name__ == '__main__':
    import sys
    sys.path.append('D:\\Scripts\\MaPS\\MaPS scripts\\')

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats
import matplotlib.pyplot as plt
from skimage.transform import resize

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2

import os
import logging
import maps
logger = logging.getLogger(__name__)

# TODO: Use settings module instead of hard coded values (replace # setting#)
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
    Find rightmost pixel matching match_value in each row and return as array
    INPUTS:
        kymo_array: numpy array of kymgraph pixel values
        match_value: value to find the rightmost of
    OUTPUTS:
        z_stage: array of distances of rightmost pixel from edge
    '''
    z_stage = np.asarray([])

    z_stage = np.zeros(kymo_array.shape[0])
    for frame_no in xrange(len(z_stage)):
        frame_row = kymo_array[frame_no, :].astype(int)

        # Older approach. Finding extremum position of pixel of match_value
        # matching_pixels = np.asarray(np.where(frame_row == 0))

        # New approach. Taking first order difference of row
        diff_row = frame_row[:-1] - frame_row[1:]
        # TODO: Once kymograph generation is done internally this can be changed to point not matching 0
        matching_pixels = np.asarray(np.where(np.absolute(diff_row) == match_value))
        z_stage[frame_no] = matching_pixels[0, -1]
    return z_stage


def generate_zstage_plot(z_stage, **params):
    '''
    Generate and return the plot image for z_stage values
    '''
    pass


def compute_maxima_minima(z_stage, maxima_threshold=10):
    '''
    Find the maxima and minima points in the z_stage data.
    From one minima to the next maxima is a zook. From a minima to the next maxima is a zik
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

    first_minima = 0  # setting#
    minima_points = np.asarray(first_minima)
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

    # TODO: add logging
    # for key, val in zz_stats.iteritems():
    #     print key, ': ', val
    return zz_stats


def generate_zookzik_stat_plots(arg):
    '''
    Generate plots for zook_zik_stats
    '''
    pass


def compute_zook_extents(maxima_point, minima_point):
    ignore_startzook = 7  # setting#
    ignore_endzook = 3  # setting#

    start_slice = minima_point + ignore_startzook
    end_slice = maxima_point - ignore_endzook + 1

    # TODO: return slice object instead
    return (start_slice, end_slice)


def compute_zstamp(z_stage, maxima_points, minima_points, ref_frame_no=0):
    '''
    Remove the constant bias in z_stage values. Pull down/up to get rid of any constant shift terms
    INPUTS:
        z_stage: array of distances of rightmost pixel from edge
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        z_stamp: Adjusted values of z_stage
        z_stamp_physical: z_stamp_values in physical units instead of frames
    '''
    BF_resolution = 0.6296  # setting#
    # ignore_startzook = 7  # setting#
    # ignore_endzook = 3  # setting#

    # TODO: TEST Check if transpose is needed
    z_stamp = np.zeros(z_stage.shape)

    for i in xrange(len(minima_points)):
        # start_slice = minima_points[i] + ignore_startzook
        # end_slice = maxima_points[i] - ignore_endzook + 1
        start_slice, end_slice = compute_zook_extents(maxima_points[i], minima_points[i])
        # TODO: Can bias removal happen at end
        z_stamp[start_slice: end_slice] = \
            z_stage[start_slice: end_slice] - z_stage[0] * np.ones(end_slice - start_slice)
    # Bias removal. Check if it should be z_stage[0] or z_stage[ignore_startzook]
    # z_stamp -= np.ones(z_stamp.shape)*z_stage[0]

    z_stamp_physical = z_stamp * BF_resolution

    return (z_stamp, z_stamp_physical)


def compute_ideal_zstamp(z_stage, maxima_points, minima_points):
    '''
    Compute the ideal minima and maxima points deterministically. Use this to compensate for quantization errors in actual values
    INPUTS:
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        z_stamp_det: deterministic version of z_stamp
    '''
    ZookZikPeriod = 192  # setting#
    resampling_factor = 5  # setting#

    minima_points_deterministic = np.arange(0, len(minima_points)) * ZookZikPeriod
    maxima_points_deterministic = minima_points_deterministic + ZookZikPeriod - 1

    z_stamp_det = np.zeros(z_stamp.shape)

    pixel_shift_length = stats.mode(z_stage[maxima_points] - z_stage[minima_points])[0]

    pixel_shift_start = stats.mode(z_stage[minima_points])[0]
    pixel_shift_per_frame = pixel_shift_length / ZookZikPeriod

    # Copied as is. Not used in previous code
    # zooklengthadjustment = (np.max(z_stage[maximapoints]-z_stage[minimapoints])-np.min(z_stage[maximapoints]-z_stage[minimapoints])) # always
    # #check to make sure that the zebrafish bondary doesn't move much between the
    # #last frame of one zook and first frame of next zook

    for i in xrange(len(minima_points)):
        z_stamp_det[minima_points_deterministic[i]: maxima_points_deterministic[i] + 1] = np.round(np.arange(0, ZookZikPeriod) * pixel_shift_per_frame) * resampling_factor

    return z_stamp_det


def load_correlation_window_params(csv_file):
    x_start = 69
    x_end = 121
    y_end = 263
    width = 60
    y_start = y_end - width
    frame_no = 20
    return (x_start, x_end, y_start, y_end, width, frame_no)


def extract_window(frame, x_start, x_end, y_start, y_end):
    return frame[x_start: x_end + 1, y_start: y_end + 1]


def load_frame_and_upscale(img_path, frame_no):
    '''
    Load the tifffile of the frame, resize (upsample by resampling factor) and return cropped window)
    '''
    resampling_factor = 5  # setting#
    image_prefix = 'Phase_Bidi1_'  # setting#
    index_start_at = 1  # setting#
    num_digits = 5  # setting#

    img = importtiff(img_path, frame_no, prefix=image_prefix, index_start_number=index_start_at, num_digits=num_digits)
    img_resized = resize(img, (img.shape[0] * resampling_factor, img.shape[1] * resampling_factor), preserve_range=True)

    return img_resized


def check_convexity(corr_points):
    best_slide = np.argmax(corr_points)
    if best_slide > 0 and best_slide < (len(corr_points) - 1):
        best_point = corr_points[best_slide]
        prior_point = corr_points[best_slide - 1]
        post_point = corr_points[best_slide + 1]
        if best_point > prior_point and best_point > post_point:
            return True
    return False


def compute_optimal_z_stamp(img_path, maxima_points, minima_points, z_stamp_det):
    '''
    load ref frame
    findo window in ref frame
    '''
    ignore_zooks_at_start = 1  # setting#
    resampling_factor = 5  # setting#
    slide_limit = 5  # setting#

    slide_limit_resized = slide_limit * resampling_factor

    z_stamp_optimal = np.zeros(z_stamp_det.shape)

    # Load window parameters
    x_start, x_end, y_start, y_end, width, ref_frame_no = load_correlation_window_params(None)

    x_start_resized = x_start * resampling_factor
    x_end_resized = x_end * resampling_factor
    y_start_resized = y_start * resampling_factor
    y_end_resized = y_end * resampling_factor
    width_resized = width * resampling_factor

    print 'base'
    print y_start_resized
    print y_end_resized

    z_stamp_det_shifted = (z_stamp_det - z_stamp_det[ref_frame_no]) * resampling_factor

    plt.plot(z_stamp_det_shifted)
    plt.show()

    # Load reference frame
    ref_frame = load_frame_and_upscale(img_path, ref_frame_no)

    # Compute window of reference frame
    ref_frame_window = extract_window(ref_frame, x_start_resized, x_end_resized, y_start_resized, y_end_resized)

    # Create matrix of optimized z_stamps

    for i in np.arange(ignore_zooks_at_start, len(minima_points)):
        start_frame, end_frame = compute_zook_extents(maxima_points[i], minima_points[i])

        for frame_no in np.arange(start_frame, end_frame):
            # Compute window params
            y_start_resized_frame = y_start_resized + z_stamp_det_shifted[frame_no]
            y_end_resized_frame = y_end_resized + z_stamp_det_shifted[frame_no]

            # print '-' * 25, frame_no, '-' * 25
            # print y_start_resized_frame
            # print y_end_resized_frame

            # Load frame and scale
            frame = load_frame_and_upscale(img_path, frame_no)
            # frame_window = extract_window(frame, x_start_resized, x_end_resized, )

            # Correlation matrix for frame. Reset to zero for each frame
            corr_array = np.zeros(2 * slide_limit_resized + 1)

            for slide_amount in np.arange(-slide_limit_resized, slide_limit_resized + 1, resampling_factor):
                pass
                # Compute shifted window params
                y_start_resized_shifted = y_start_resized_frame + slide_amount
                y_end_resized_shifted = y_end_resized_frame + slide_amount

                # print y_start_resized_shifted
                # print y_end_resized_shifted

                # Compute window of frame
                frame_window = extract_window(frame, x_start_resized, x_end_resized, y_start_resized_shifted, y_end_resized_shifted)

                # Compute correlation
                try:
                    corr_array[slide_amount + slide_limit_resized] = corr2(ref_frame_window, frame_window)
                except:
                    corr_array[slide_amount + slide_limit_resized] = -1
                    # print 'Frame:', frame_no
                    # print 'Slide:', slide_amount + slide_limit
                    # print 'ref shape:', ref_frame_window.shape
                    # print 'frame shape:', frame_window.shape
                    # import traceback
                    # traceback.print_exc()

            # Compute best slide amount
            best_slide_amount = np.argmax(corr_array) - slide_limit_resized
            # print best_slide_amount
            # is_convex = False
            # Compute convexity
            is_convex = check_convexity(corr_array)

            if is_convex:
                # Compute best maxima in local region
                for slide_amount in np.arange(max(best_slide_amount - resampling_factor + 1, -slide_limit_resized), min(best_slide_amount + resampling_factor, slide_limit_resized + 1)):
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
                        # print 'Frame:', frame_no
                        # print 'Slide:', slide_amount + slide_limit
                        # print 'ref shape:', ref_frame_window.shape
                        # print 'frame shape:', frame_window.shape
                        # import traceback
                        # traceback.print_exc()

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
                        # print 'Frame:', frame_no
                        # print 'Slide:', slide_amount + slide_limit
                        # print 'ref shape:', ref_frame_window.shape
                        # print 'frame shape:', frame_window.shape
                        # import traceback
                        # traceback.print_exc()

            # Compute ideal shift using corr_array
            z_stamp_optimal[frame_no] = z_stamp_det_shifted[frame_no] + np.argmax(corr_array) - slide_limit_resized

            # print np.argmax(corr_array) - slide_limit_resized
            # plt.plot(corr_array)
            # plt.show()

    z_stamp_optimal_resized = z_stamp_optimal / resampling_factor

    plt.plot(z_stamp_optimal_resized)
    plt.show()

    return z_stamp_optimal


def check_optimal_z_stamp():
    pass


def shift_frames(img_path, z_stamps):
    pass


def z_stamping_step():
    pass

if __name__ == '__main__':
    from maps.helpers.logging_config import logger_config
    logging.config.dictConfig(logger_config)

    logger = logging.getLogger('maps.core.z_stamping')
    kymo_data = load_kymograph('D:\Scripts\MaPS\Data sets\KM-XZ_0.tif')
    z_stage_data = compute_raw_zstage(kymo_data[:1000, :])
    # plt.plot(z_stage_data)
    # plt.show()
    (maxp, maxv, minp, minv) = compute_maxima_minima(z_stage_data)
    # compute_zookzik_stats(maxp, minp)
    z_stamp, _ = compute_zstamp(z_stage_data, maxp, minp)
    z_stamp_det = compute_ideal_zstamp(z_stage_data, maxp, minp)
    # plt.plot(z_stamp)
    # plt.show()
    # plt.plot(z_stamp_det)
    # plt.show()
    #
    img_pth = 'D:\\Scripts\\MaPS\\Data sets\\Phase_Bidi\\'  # setting#
    #
    z_stamp_opt = compute_optimal_z_stamp(img_pth, maxp, minp, z_stamp_det)
    #
    # # for frame in range(len(z_stamp_opt)):
    # #     print z_stamp_opt[frame]
