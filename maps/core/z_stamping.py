import numpy as np
import skimage.external.tifffile as tff
from scipy import stats
import matplotlib.pyplot as plt

import os
import sys


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
        # Taking first order difference of row
        diff_row = frame_row[:-1] - frame_row[1:]
        # TODO: Once kymograph generation is done internally this can be changed to point not matching 0
        matching_pixels = np.asarray(np.where(np.absolute(diff_row) == match_value))
        # Older approach. Finding extremum position of pixel of match_value
        # matching_pixels = np.asarray(np.where(frame_row == 0))
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


def compute_zstamp(z_stage, maxima_points, minima_points):
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
    ignore_startzook = 7  # setting#
    ignore_endzook = 3  # setting#
    BF_resolution = 0.6296  # setting#

    # TODO: TEST Check if transpose is needed
    z_stamp = np.zeros(z_stage.shape)

    for i in xrange(len(minima_points)):
        start_slice = minima_points[i] + ignore_startzook
        end_slice = maxima_points[i] - ignore_endzook + 1
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


if __name__ == '__main__':
    kymo_data = load_kymograph('D:\Scripts\MaPS\Data sets\KM-XZ_0.tif')
    z_stage_data = compute_raw_zstage(kymo_data[:1000, :])
    (maxp, maxv, minp, minv) = compute_maxima_minima(z_stage_data)
    # compute_zookzik_stats(maxp, minp)
    z_stamp, _ = compute_zstamp(z_stage_data, maxp, minp)
    z_stamp_det = compute_ideal_zstamp(z_stage_data, maxp, minp)
