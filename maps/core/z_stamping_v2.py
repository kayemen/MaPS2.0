if __name__ == '__main__':
    import sys
    import os
    print 'Appending to path:'
    print os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats
from skimage.transform import resize

from maps import settings
from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.img_proccessing import corr2
from maps.helpers.gui_modules import load_frame, extract_window, cv2_methods, create_image_overlay
from maps.helpers.misc import pickle_object, unpickle_object, write_error_plot
from maps.settings import setting, read_setting_from_json
from maps.helpers.misc import LimitedThread

import os
import logging
import collections
import maps
import csv
import cv2
import time
import threading
import traceback
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Zook = collections.namedtuple('Zook', ('id', 'start', 'end', 'is_valid'))


class Zook:

    def __init__(self, zook_id, start, end):
        self.start = start
        self.end = end
        self.dwt_start = None
        self.dwt_end = None
        self.dwt_start_us = None
        self.dwt_end_us = None
        self.dwt_computed = False
        self.dwt_upsampled = False
        self.is_bad = False
        self.is_bad_zstage = False
        self.is_bad_convexity = False
        self.is_bad_shift = False
        self.generic_error = False
        self.override = False
        self.is_trimmed = False
        self.id = zook_id

    def __len__(self):
        return int(self.end - self.start + 1)

    def __repr__(self):
        return 'Zook {id}: {start}->{end} - {len} usable frames'.format(id=self.id, len=len(self), start=self.start, end=self.end)

    def __iter__(self):
        for frame in xrange(self.start, self.end + 1):
            yield frame

    def get_framelist(self):
        return range(self.start, self.end - 1)

    def trim(self):
        if self.is_trimmed:
            return
        self.start += setting['ignore_startzook']
        self.end -= setting['ignore_endzook']
        self.is_trimmed = True

    def untrim(self):
        if not self.is_trimmed:
            return
        self.start -= setting['ignore_startzook']
        self.end += setting['ignore_endzook']
        self.is_trimmed = False


class Zooks:

    def __init__(self):
        self.zook_list = []

    def __repr__(self):
        return 'Zookset containing {len} zooks ({blen} bad zooks)'.format(len=len(self.zook_list), blen=len(self.get_bad_zooks()))

    def __len__(self):
        return len(self.zook_list) - len(self.get_bad_zooks())

    def __iter__(self):
        self.update_badzooks()
        for zook in self.zook_list:
            if zook.is_bad:
                continue
            yield zook

    def __getitem__(self, key):
        if isinstance(key, slice):
            subzooks = Zooks()
            subzooks.zook_list = self.zook_list[key]
            return subzooks
        return self.zook_list[key]

    def trim_zooks(self):
        for zook in self:
            zook.trim()

    def untrim_zooks(self):
        for zook in self:
            zook.untrim()

    def get_bad_zooks(self):
        self.update_badzooks()
        bad_zooks = [zook for zook in self.zook_list if zook.is_bad]
        return bad_zooks

    def get_framelist(self):
        framelist = []

        self.update_badzooks()

        for zook in self.zook_list:
            if not zook.is_bad:
                framelist += zook.get_framelist()
        return framelist

    def get_dwt_framelist(self, upsample=False):
        start = 0
        for zook in self:
            zook.dwt_computed = True
            yield slice(start, start + len(zook))
            zook.dwt_start = start
            start += len(zook)
            zook.dwt_end = start - 1
            if upsample:
                zook.dwt_start_us = (zook.dwt_start * setting['time_resampling_factor']) + (
                    setting['time_resampling_factor'] - 1)
                zook.dwt_end_us = (
                    zook.dwt_end * setting['time_resampling_factor']) - (setting['time_resampling_factor'] - 1)

    def get_true_frame_number(self, dwt_frame_no):
        # corr_zook =
        return dwt_frame_no

    def update_badzooks(self):
        for zook in self.zook_list:
            zook.is_bad = (zook.is_bad_zstage or zook.is_bad_convexity or zook.is_bad_shift or zook.generic_error) and (
                not zook.override)

    def override_bad_zooks(self, zooklist):
        for zook in zooklist:
            zook.override = True

        self.update_badzooks()

    def add_zook(self, zook_start, zook_end):
        '''
        Add zook to set
        '''
        # assert isinstance(zook, Zook)
        self.zook_list.append(Zook(len(self) + 1, zook_start, zook_end))

    def get_maxima_points(self):
        return [zook.end for zook in self]

    def get_minima_points(self):
        return [zook.start for zook in self]


def compute_raw_zstage(kymo_path, match_value=255, median_filt=False, kernel_size=5):
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
    method = 'new'

    kymo_array = np.asarray(tff.imread(kymo_path))

    # Add median filtering
    if median_filt:
        kymo_array = cv2.medianBlur(kymo_array, kernel_size)

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
            # TODO: Once kymograph generation is done internally this can be
            # changed to point not matching 0
            matching_pixels = np.asarray(
                np.where(np.absolute(diff_row) == match_value))
        z_stage[frame_no] = matching_pixels[0, 0]
        # z_stage[frame_no] = matching_pixels[0, -1]

    pickle_object(z_stage, 'z_stage')

    return z_stage


def compute_zooks(z_stage):
    zookset = Zooks()

    maxima_threshold = setting.get('kymo_zook_threshold', 20)

    maxima_points = np.where(
        (z_stage[1:] - z_stage[:-1]) < -maxima_threshold)[0].flatten()
    maxima_points = maxima_points[
        np.where(maxima_points > setting.get('first_minima', 0))]
    # maxima_values = z_stage[maxima_points]

    # # TODO: TEST Include the last zook and test if it breaks anything
    # # Maxima in last zook. Uncomment after testing
    # np.append(maxima_values, z_stage[-1])

    minima_points = np.asarray([setting.get('first_minima', 0)])
    for i in xrange(len(maxima_points) - 1):
        # OLD
        # new_minima = np.argmin(z_stage[maxima_points[i]: maxima_points[i + 1]])
        # NEW
        new_minima = 1
        minima_points = np.append(minima_points, maxima_points[i] + new_minima)

    # # TODO: TEST Include the last zook and test if it breaks anything
    # # Minima in last zook. Uncomment if needed
    # new_minimas = np.argmin(z_stage[maxima_points[-1]:])
    # minima_points = np.append(minima_points, maxima_points[-1] + new_minimas)

    # minima_values = z_stage[minima_points]

    for start, end in zip(minima_points, maxima_points):
        zookset.add_zook(start, end)

    # pickle_object(
    #     (maxima_points, maxima_values, minima_points, minima_values),
    #     'extremum_points'
    # )

    pickle_object(zookset, 'zooks')

    return zookset


def compute_zookzik_stats(zooks):
    '''
    Find the statistics of zook and zik. See list of outputs to see exact stats computed
    INPUTS:
        maxima_points: index of points at which maxima occur in z_stage
        minima_points: index of points at which minima occur in z_stage
    OUTPUTS:
        zz_stats: Dictionary containing:-
            lengths: Length of each zook. Distance from maxima to next minima
            mean: Mean of all zook lengths
            mode: Most common zook length
            max: Maximum zook length
    '''
    zz_stats = {}

    # maxima_points =
    # minima_points =

    zz_stats['lengths'] = [len(zook) for zook in zooks]
    zz_stats['mean'] = np.round(np.mean(zz_stats['lengths']))
    zz_stats['mode'] = np.round(
        stats.mode(zz_stats['lengths']))
    zz_stats['max'] = np.max(zz_stats['lengths'])

    logger.debug('\n'.join(['%s: %s' % (key, str(val))
                            for key, val in zz_stats.iteritems()]))

    pickle_object(zz_stats, 'zz_stats')

    return zz_stats


def drop_bad_zooks_zstage(zooks, zz_stats, z_stage, threshold=0.9, threshold_mult=3):
    # Find bad zooks after zstage calculation to drop bad zooks
    for zook in zooks:
        if len(zook) < zz_stats['mean'] * threshold:
            zook.is_bad_zstage = True

        slope = (z_stage[zook.end] - z_stage[zook.start]) / len(zook)

        shifts = z_stage[zook.start + 1: zook.end] - \
            z_stage[zook.start: zook.end - 1]

        if bool(len(np.where(shifts > threshold_mult * int(slope + 1))[0])):
            # print zook.id, 'is a bad zook'
            # plt.plot(shifts)
            # plt.show()
            zook.is_bad_zstage = True

    zooks.update_badzooks()


def compute_zstamp(z_stage, zooks, ref_frame_no=0, z_stamp_pkl_name='z_stamp'):
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

    for zook in zooks:
        z_stamp[zook.start: zook.end + 1] = z_stage[
            zook.start: zook.end + 1] - z_stage[ref_frame_no]

    z_stamp_physical = z_stamp * setting['BF_resolution']

    pickle_object(z_stamp, z_stamp_pkl_name)
    pickle_object(z_stamp_physical, z_stamp_pkl_name + '_physical')

    return (z_stamp, z_stamp_physical)


def compute_deterministic_zstamp(z_stamp, zooks, ref_frame_no=0, z_stamp_pkl_name='z_stamp_det'):

    maxima_points = np.asarray([zook.end for zook in zooks])
    minima_points = np.asarray([zook.start for zook in zooks])

    pixel_shift_length = stats.mode(
        z_stamp[maxima_points] - z_stamp[minima_points])[0]

    pixel_shift_start = stats.mode(z_stamp[minima_points])[0]

    mode_zook_len = stats.mode([len(zook) for zook in zooks])[0]

    pixel_shift_per_frame = float(pixel_shift_length) / mode_zook_len

    z_stamp_det = np.zeros(z_stamp.shape)

    for zook in zooks:
        z_stamp_det[zook.start: zook.end + 1] = np.round(
            np.arange(-ref_frame_no, -ref_frame_no + len(zook)) *
            pixel_shift_per_frame * setting['resampling_factor']
        )

    pickle_object(z_stamp_det, z_stamp_pkl_name)

    zooks.trim_zooks()

    return z_stamp_det


def load_window_params(csv_name='corr_window.csv'):
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
    raw_data = unpickle_object(csv_name, dumptype='csv')
    raw_data = {key: int(val) for key, val in raw_data}

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

        # Checking if best slide is at the extreme edge
        if best_slide == 0 or best_slide == (len(corr_points) - 1):
            # plt.plot(corr_points)
            # plt.show()
            # print 'Edge of window'
            return False

        # Checking maxima convex
        if np.all(corr_points[1:best_slide] >= corr_points[:best_slide - 1]) and np.all(corr_points[best_slide + 2:] <= corr_points[best_slide + 1:-1]):
            return True

        # Checking minima convex (techincally concave)
        if np.all(corr_points[1:best_slide] <= corr_points[:best_slide - 1]) and np.all(corr_points[best_slide + 2:] >= corr_points[best_slide + 1:-1]):
            return True

        # TODO: Confirm if workable
        # Check if maxima is in middle half of curve
        if 0.1 * len(corr_points) <= best_slide <= 0.9 * len(corr_points):
            return True
    return False


def calculate_frame_optimal_shift_yz(img_path, prefix, frame_no, offset, ref_frame_window, x_end_resized, height, y_end_resized, width, z_stamp_optimal, y_stamp_optimal, z_stamp_det, zook):
    '''
    Calculate optimal shift for single frame.
    This method does the z stamp calculation with wiggle of "setting['slide_limit']" pixels in low-res domain.
    It also computes the vertical shift in the frames to compensate for any rise/drop in the fish over a zook.
    This function can be threaded in the calling scope to improve performance.
    It uses the openCV matchTemplate method to find the correlation between the selected window and teh ROI. It then finds the maxima and computes shift in x and y directions.
    '''
    PLOT = False
    DEBUG = False

    try:
        x_start_resized = x_end_resized - height
        y_end_resized_frame = y_end_resized + z_stamp_det[frame_no]
        y_start_resized_frame = y_end_resized_frame - width

        slide_limit_resized = setting[
            'slide_limit'] * setting['resampling_factor']
        slide_limit_x_resized = setting[
            'y_slide_limit'] * setting['resampling_factor']

        # Load frame and scale
        frame = load_frame(img_path, frame_no,
                           index_start_number=offset, prefix=prefix)

        # Compute shifted window params
        x_start_resized_shifted = x_start_resized - slide_limit_x_resized
        x_end_resized_shifted = x_end_resized + slide_limit_x_resized
        y_start_resized_shifted = y_start_resized_frame - slide_limit_resized
        y_end_resized_shifted = y_end_resized_frame + slide_limit_resized

        # Compute window of frame
        frame_window = extract_window(frame, x_start_resized_shifted,
                                      x_end_resized_shifted, y_start_resized_shifted, y_end_resized_shifted)

        if DEBUG:
            print frame_window.shape
            print ref_frame_window.shape

        if PLOT:
            plt.figure()
            plt.imshow(create_image_overlay(
                frame.astype('uint16'), overlay_type='rectangle',
                overlay_data={
                    'pt1': (int(y_start_resized_shifted), int(x_start_resized_shifted), ),
                    'pt2': (int(y_end_resized_shifted), int(x_end_resized_shifted), ),
                    'color': (0, 65535, 0),
                    'thickness': 10
                }
            ))
            plt.show(block=False)

        # TODO: Try on entire frame instead of window
        corr_array = cv2.matchTemplate(frame_window.astype(
            'float32'), ref_frame_window.astype('float32'), cv2_methods['ccorr_norm'])

        # if PLOT:
        #     plt.figure()
        #     plt.imshow(corr_array, cmap='gray')
        #     plt.colorbar()
        #     # plt.figure()
        #     # plt.plot(corr_array.max(axis=0))
        #     # plt.plot(corr_array.max(axis=1))
        #     plt.show(block=False)
        #     print 'plotted'

        # Check convexity
        if not (check_convexity(corr_array.max(axis=0)) and check_convexity(corr_array.max(axis=1))):
            print 'Convexity error in frame %d' % frame_no
            zook.is_bad_convexity = True

            if DEBUG:
                print check_convexity(corr_array.max(axis=0))
                print check_convexity(corr_array.max(axis=1))
                plt.figure()
                plt.plot(corr_array.max(axis=0))
                plt.plot(corr_array.max(axis=1))
                plt.show()
                print 'plotted'

        # Compute ideal shift using corr_array
        try:
            y_shift = corr_array.max(axis=0).argmax() - slide_limit_resized
            x_shift = corr_array.max(axis=1).argmax() - slide_limit_x_resized

            z_stamp_optimal[frame_no] = z_stamp_det[frame_no] + y_shift
            y_stamp_optimal[frame_no] = x_shift

            if setting['write_ref_window']:
                write_window = extract_window(
                    frame_window,
                    x_shift + slide_limit_x_resized,
                    x_shift + slide_limit_x_resized + height,
                    y_shift + slide_limit_resized,
                    y_shift + slide_limit_resized + width
                )
                write_window = resize(
                    write_window,
                    (
                        write_window.shape[0] / setting['resampling_factor'],
                        write_window.shape[1] / setting['resampling_factor']
                    ),
                    preserve_range=True
                ).astype('uint16')

                writetiff(write_window, setting[
                          'cropped_ref_windows'], frame_no)

                if PLOT:
                    plt.figure()
                    plt.imshow(create_image_overlay(
                        frame_window.astype('uint16'), overlay_type='rectangle',
                        overlay_data={
                            'pt1': (int(y_shift + slide_limit_resized), int(x_shift + slide_limit_x_resized), ),
                            'pt2': (int(y_shift + slide_limit_resized + width), int(x_shift + slide_limit_x_resized + height), ),
                            'color': (0, 65535, 0),
                            'thickness': 10
                        }
                    ))
                    plt.show(block=True)
                    print 'plotted'

        except:
            traceback.print_exc()
    except:
        traceback.print_exc()
        print 'error in frame ', frame_no
        print frame_window.shape
        print ref_frame_window.shape
        zook.generic_error = True
        # raw_input()
    # print frame_no, ',',


def compute_optimal_yz_stamp(img_path, zooks, z_stamp_det, offset=0, prefix=None, corr_window_csv='corr_window.csv', z_stamp_pkl='z_stamp_optimal', y_stamp_pkl='y_stamp_optimal'):
    '''
    Compute optimal z stamp and y stamp values for entire image sequence.
    '''
    DEBUG = False
    PLOT = False
    # NUM_PROCESSES = 20

    z_stamp_optimal = np.empty(z_stamp_det.shape)
    y_stamp_optimal = np.empty(z_stamp_det.shape)

    y_stamp_optimal[:] = np.NAN
    z_stamp_optimal[:] = np.NAN

    x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_window_params(
        corr_window_csv)

    print '>>>', x_start, x_end, y_start, y_end, height, width, ref_frame_no

    # Time stamp values
    zook_time_stats = []

    x_end_resized = x_end * setting['resampling_factor']
    height_resized = height * setting['resampling_factor']
    x_start_resized = x_end_resized - height_resized
    y_end_resized = y_end * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']
    y_start_resized = y_end_resized - width_resized

    # Load reference frame
    ref_frame = load_frame(img_path, ref_frame_no,
                           index_start_number=offset, prefix=prefix)

    if PLOT:
        plt.imshow(create_image_overlay(
            ref_frame.astype('uint16'), overlay_type='rectangle',
            overlay_data={
                'pt1': (int(y_start_resized), int(x_start_resized), ),
                'pt2': (int(y_end_resized), int(x_end_resized), ),
                'color': (0, 65535, 0),
                'thickness': 10
            }
        ))
        plt.show(block=True)
        print 'plotted'

    # Compute window of reference frame
    ref_frame_window = extract_window(
        ref_frame, x_start_resized, x_end_resized, y_start_resized, y_end_resized)

    pickle_object(ref_frame_window, 'ref_frame_window')

    # Create matrix of optimized z_stamps
    for zook in zooks[setting['ignore_zooks_at_start']:]:
        start_frame, end_frame = zook.start, zook.end + 1

        logging.debug('Processing Zook#%d' % zook.id)
        # print 'Processing Zook#%d' % zook
        tic = time.time()

        if not settings.MULTIPROCESSING:
            map(
                lambda frame_no: calculate_frame_optimal_shift_yz(
                    img_path, prefix, frame_no, offset,
                    ref_frame_window,
                    x_end_resized,
                    height_resized,
                    y_end_resized,
                    width_resized,
                    z_stamp_optimal,
                    y_stamp_optimal,
                    z_stamp_det,
                    zook
                ),
                zook
            )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame_no: calculate_frame_optimal_shift_yz(
                    img_path, prefix, frame_no, offset,
                    ref_frame_window,
                    x_end_resized,
                    height_resized,
                    y_end_resized,
                    width_resized,
                    z_stamp_optimal,
                    y_stamp_optimal,
                    z_stamp_det,
                    zook
                ),
                zook,
                settings.NUM_CHUNKS,
            )
            # print 'spawned'

            try:
                temp_var.get(timeout=settings.TIMEOUT)
            except TimeoutError:
                logging.info('Zook %d timed out in %d seconds' %
                             (zook.id, settings.TIMEOUT))
                logging.info('%d frames did not get processed' %
                             (temp_var._number_left))
                zook.generic_error = True

            del proc_pool

        zook_time_stats.append(time.time() - tic)

    pickle_object(z_stamp_optimal, z_stamp_pkl)
    pickle_object(y_stamp_optimal, y_stamp_pkl)
    pickle_object(zook_time_stats, 'ZY_stamping_time')

    print 'Average time for YZ stamping:%f s/zook' % (sum(zook_time_stats) / len(zook_time_stats))

    return z_stamp_optimal, y_stamp_optimal


def compute_zstamp_curvefit(z_stamp_optimal, zooks):
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

    try:
        z_stamp_curvefit = np.zeros(z_stamp_optimal.shape)

        slopes = []

        for zook in zooks[setting['ignore_zooks_at_start']:]:
            start_frame, end_frame = zook.start, zook.end

            z_stamps = z_stamp_optimal[start_frame: end_frame]
            frame_nos = np.arange(start_frame, end_frame)

            # Using linear regression to compute the best fitting line for each
            # zook
            slope, intercept, _, _, _ = stats.linregress(
                frame_nos, z_stamps)

            slopes.append(slope)
            z_stamp_curvefit[start_frame: end_frame] = slope * \
                frame_nos + intercept

        residues = z_stamp_optimal - z_stamp_curvefit
    except:
        traceback.print_exc()

    pickle_object(z_stamp_curvefit, 'z_stamp_curvefit')
    pickle_object(residues, 'residues')
    pickle_object(slopes, 'slopes')

    return z_stamp_curvefit, residues, slopes


def detect_bad_zooks(residues, zooks, slopes, threshold_mult=0.3):
    '''
    Find the frames in a zook which are shifted by unnaturally large amounts. The threshold for detecting if a shift is bad is if it deviates more than "c=27%" from the slope of the shifts in that zook
    INPUTS:
        residues(ARRAY): Array of residues between optimal z_stamp and the linear regression line of the z stamp values
        maxima_points:
        minima_points:
        slopes(ARRAY): array of slopes of the line in the cureve fit z_stamps for each zook
    OUTPUTS:
        #, the shift value of deviant shifts and the location of deviant shifts
        bad_zooks(list): list of bad_zook tuples. Each bad zook tuple consists of the zook
    '''
    bad_zooks = []

    print 'Running with threshold multiplier at %.2f' % (threshold_mult)
    # print slopes
    for zno, zook in enumerate(zooks[setting['ignore_zooks_at_start']:]):
        # min threshold is 2. if slope is higher, error tolerance is higher
        threshold = max(2, threshold_mult * slopes[zno])

        start_frame, end_frame = zook.start, zook.end
        zook_residue = residues[start_frame: end_frame - 1]

        # plt.plot(zook_residue)
        # plt.show()
        # print zook_residue
        # print threshold
        # raw_input()

        bad_shifts = zook_residue[
            np.where(np.absolute(zook_residue) > threshold)]
        bad_shift_loc = np.where(np.absolute(zook_residue) > threshold)[
            0] + start_frame

        is_bad_zook = bool(len(bad_shifts)) or np.any(np.isnan(zook_residue))

        if is_bad_zook:
            bad_zooks.append((zook, bad_shifts, bad_shift_loc))
            zook.is_bad_shift = True

    zooks.update_badzooks()

    pickle_object(bad_zooks, 'bad_zooks')

    return bad_zooks


def find_relative_brightness(img, num_rows=4):
    intensity = 0

    if setting.get('rel_intensity_method', 'median') == 'median':
        intensity = np.median(img[:num_rows, :])
        return intensity
    elif setting.get('rel_intensity_method', 'median') == 'mean':
        intensity = np.mean(img[:num_rows, :])
        return intensity


def adjust_relative_brightness(img, rel_intensity):
    curr_intensity = find_relative_brightness(img)

    img += (rel_intensity - curr_intensity)

    return img


def downsize_and_writeframe(img_path, prefix, frame_no, offset, x_end_resized, height_resized, y_end_resized, width_resized, z_stamps, y_stamps, write_to, adj_intensity, rel_intensity):
    try:
        y_end_frame = y_end_resized + z_stamps[frame_no]
        y_start_frame = y_end_frame - width_resized
        x_end_frame = x_end_resized + y_stamps[frame_no]
        x_start_frame = x_end_frame - height_resized

        cropped_frame = load_frame(
            img_path, frame_no, index_start_number=offset, prefix=prefix, upsample=True
        )

        cropped_frame = extract_window(
            cropped_frame,
            x_start_frame,
            x_end_frame + setting['resampling_factor'],
            y_start_frame,
            y_end_frame + setting['resampling_factor']
        )

        cropped_frame_downsized = resize(
            cropped_frame,
            (
                cropped_frame.shape[0] / setting['resampling_factor'],
                cropped_frame.shape[1] / setting['resampling_factor']
            ),
            preserve_range=True
        )

        if adj_intensity:
            cropped_frame_downsized = adjust_relative_brightness(
                cropped_frame_downsized,
                rel_intensity
            )

        if setting['validTiff']:
            cropped_frame_downsized = cropped_frame_downsized.astype('uint16')

        writetiff(cropped_frame_downsized, write_to, frame_no)
    except:
        traceback.print_exc()
        print 'write Exception in', frame_no
    # print frame_no


def shift_frames_and_store_yz(img_path, prefix, offset, z_stamps, y_stamps, zooks, write_to, crop_csv_file='crop_window.csv', mask_path='', adj_intensity=False):
    DEBUG = False

    # NUM_PROCESSES = 20

    x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_window_params(
        crop_csv_file)

    height_resized = height * setting['resampling_factor']
    width_resized = width * setting['resampling_factor']
    x_end_resized = x_end * setting['resampling_factor']
    y_end_resized = y_end * setting['resampling_factor']

    crop_write_time_stats = []

    rel_brightness = 0

    if adj_intensity:
        rel_frame = load_frame(
            img_path, ref_frame_no, index_start_number=offset, prefix=prefix, upsample=True
        )
        rel_frame_window = extract_window(
            rel_frame,
            x_end_resized - height_resized,
            x_end_resized + setting['resampling_factor'],
            y_end_resized - width_resized,
            y_end_resized + setting['resampling_factor'],
        )
        rel_frame_downsized = resize(
            rel_frame_window,
            (
                rel_frame_window.shape[0] / setting['resampling_factor'],
                rel_frame_window.shape[1] / setting['resampling_factor']
            ),
            preserve_range=True
        )
        rel_brightness = find_relative_brightness(rel_frame_downsized)
    # print zooks
    # raw_input()
    # TODO: Add timing
    for zook in zooks[setting['ignore_zooks_at_start']:]:
        logging.debug('Writing zook %d' % zook.id)
        # if zook in bad_zooks:
        #     continue

        tic = time.time()
        # start_frame, end_frame = zook.start, zook.end
        # for frame_no in np.arange(start_frame, end_frame):
        # for frame_no in zook:
        #     y_end_frame = y_end_resized + z_stamps[frame_no]
        #     y_start_frame = y_end_frame - width_resized
        #     x_end_frame = x_end_resized + y_stamps[frame_no]
        #     x_start_frame = x_end_frame - width_resized

        if not settings.MULTIPROCESSING:
            map(
                lambda frame_no: downsize_and_writeframe(
                    img_path, prefix,
                    frame_no, offset,
                    x_end_resized,
                    height_resized,
                    y_end_resized,
                    width_resized,
                    z_stamps,
                    y_stamps,
                    write_to,
                    adj_intensity,
                    rel_brightness
                ),
                zook
            )
        else:
            proc_pool = ThreadPool(processes=settings.NUM_PROCESSES)
            temp_var = proc_pool.map_async(
                lambda frame_no: downsize_and_writeframe(
                    img_path, prefix,
                    frame_no, offset,
                    x_end_resized,
                    height_resized,
                    y_end_resized,
                    width_resized,
                    z_stamps,
                    y_stamps,
                    write_to,
                    adj_intensity,
                    rel_brightness
                ),
                zook,
                settings.NUM_CHUNKS
            )
            try:
                temp_var.get(timeout=settings.TIMEOUT * 20)
            except TimeoutError:
                logging.info('Zook %d timed out in %d seconds' %
                             (zook.id, settings.TIMEOUT * 20))
            del proc_pool

        # if WRITE_REFERENCE_WINDOW:
        #     x_start, x_end, y_start, y_end, height, width, ref_frame_no = load_window_params(
        #         crop_csv_file)

        crop_write_time_stats.append(time.time() - tic)

    pickle_object(crop_write_time_stats, 'cropping_time_stats')

    print 'Average time for Cropping and writing to disk:%f s/zook' % (sum(crop_write_time_stats) / (len(crop_write_time_stats) + 1))


if __name__ == '__main__':
    try:
        import glob
        from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame, get_rect_params, masking_window_frame
        from maps.helpers.logging_config import logger_config

        logging.config.dictConfig(logger_config)

        read_setting_from_json('job2')

        setting['ignore_zooks_at_start'] = 0
        setting['ignore_startzook'] = 7
        setting['ignore_endzook'] = 3
        setting['BF_resolution'] = 0.6296
        setting['validTiff'] = True
        # setting['image_prefix'] = 'Phase_Bidi1_'
        # setting['workspace'] = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'
        setting['resampling_factor'] = 5
        setting['slide_limit'] = 5
        setting['y_slide_limit'] = 10
        # setting['ZookZikPeriod'] = 192
        # setting['index_start_at'] = 1
        # setting['num_digits'] = 5
        # setting['first_minima'] = 0

        MOVING_FRAMES = False

        '''Moving frames'''
        if MOVING_FRAMES:
            ref_frame_no = 21

            frame_count = 1000

            CORR_GUI = True

            if CORR_GUI:
                img_path = glob.glob(
                    os.path.join(setting['bf_images'], '*.tif')
                )
                img_seq = load_image_sequence(
                    [img_path[setting['index_start_at'] + ref_frame_no]])
                max_heartsize_frame(img_seq[0])

                params = get_rect_params()

                data = [
                    ('frame', ref_frame_no),
                    ('x_end', params['x_end']),
                    ('height', params['height']),
                    ('y_end', params['y_end']),
                    ('width', params['width']),
                ]

                print 'Using reference window as-'
                print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

                pickle_object(data, file_name='corr_window.csv',
                              dumptype='csv')

            COMPUTE_NEW = True

            if COMPUTE_NEW:
                kymo_path = glob.glob(
                    os.path.join(setting['km_path'], 'mov_*.tif')
                )[1]
                print kymo_path
                z_stage = compute_raw_zstage(kymo_path)

                zooks = compute_zooks(z_stage[:frame_count])

                zz_stats = compute_zookzik_stats(zooks)

                drop_bad_zooks_zstage(zooks, zz_stats)

                z_stamp, _ = compute_zstamp(
                    z_stage[:frame_count], zooks, ref_frame_no)

                z_stamp_det = compute_deterministic_zstamp(
                    z_stamp, zooks, ref_frame_no)

                # plt.plot(z_stamp)
                # plt.figure()
                # plt.plot(z_stamp_det)
                # plt.show()
                # sys.exit()

                z_stamp_opt, y_stamp_opt = compute_optimal_yz_stamp(
                    setting['bf_images'], zooks, z_stamp_det, offset=setting['index_start_at'], prefix=setting['image_prefix'])
            else:
                z_stamp_opt = unpickle_object('z_stamp_optimal')
                y_stamp_opt = unpickle_object('y_stamp_optimal')
                zooks = unpickle_object('zooks')
                zooks.trim_zooks()

            z_stamp_cf, res, slopes = compute_zstamp_curvefit(
                z_stamp_opt, zooks)

            # plt.plot(res)
            plt.plot(z_stamp_opt)
            # plt.plot(z_stamp_cf)
            plt.show()

            bz = detect_bad_zooks(res, zooks, slopes)

            CROP_GUI = False

            if CROP_GUI:
                img_path = glob.glob(
                    os.path.join(setting['bf_images'], '*.tif')
                )
                img_seq = load_image_sequence(
                    [img_path[setting['index_start_at'] + ref_frame_no]])
                masking_window_frame(img_seq[0])
                crop_params = get_rect_params()

                data = [
                    ('frame', ref_frame_no),
                    ('x_end', crop_params['x_end']),
                    ('height', crop_params['height']),
                    ('y_end', crop_params['y_end']),
                    ('width', crop_params['width']),
                ]

                print 'Using cropping window as-'
                print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

                pickle_object(data, file_name='crop_window.csv',
                              dumptype='csv')

            WRITE_IMAGES = True

            if WRITE_IMAGES:
                shift_frames_and_store_yz(
                    setting['bf_images'], setting[
                        'image_prefix'], setting['index_start_at'],
                    z_stamp_opt, y_stamp_opt, zooks, setting[
                        'cropped_bf_images']
                )

        STAT_FRAMES = True

        '''Stationary frames'''
        if STAT_FRAMES:
            ref_frame_no = 21

            frame_count = 1000

            CORR_GUI = False

            if CORR_GUI:
                img_path = glob.glob(
                    os.path.join(setting['bf_images'], '*.tif')
                )
                img_seq = load_image_sequence(
                    [img_path[setting['stat_index_start_at'] + ref_frame_no]])
                max_heartsize_frame(img_seq[0])

                params = get_rect_params()

                data = [
                    ('frame', ref_frame_no),
                    ('x_end', params['x_end']),
                    ('height', params['height']),
                    ('y_end', params['y_end']),
                    ('width', params['width']),
                ]

                print 'Using reference window as-'
                print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

                pickle_object(data, file_name='corr_window_stat.csv',
                              dumptype='csv')

            COMPUTE_NEW = False

            if COMPUTE_NEW:
                z_stage = np.zeros(frame_count)
                # z_stage[-1] = 100

                # kymo_path = glob.glob(
                #     os.path.join(setting['km_path'], 'stat_*.tif')
                # )[0]
                # print kymo_path
                # z_stage = compute_raw_zstage(kymo_path)

                # plt.plot(z_stage)
                # plt.show()

                # zooks = compute_zooks(z_stage)
                zooks = Zooks()
                zooks.add_zook(0, frame_count - 1)
                pickle_object(zooks, 'stat_zooks')

                # zz_stats = compute_zookzik_stats(zooks)

                # drop_bad_zooks_zstage(zooks, zz_stats)
                # print zooks
                # print [zook for zook in zooks]

                z_stamp, _ = compute_zstamp(
                    z_stage[:frame_count], zooks, ref_frame_no)

                z_stamp_det = compute_deterministic_zstamp(
                    z_stamp, zooks, ref_frame_no)

                # plt.plot(z_stamp)
                # plt.figure()
                # plt.plot(z_stamp_det)
                # plt.show()
                # sys.exit()

                z_stamp_opt, y_stamp_opt = compute_optimal_yz_stamp(
                    setting['bf_images'], zooks, z_stamp_det, offset=setting['stat_index_start_at'], prefix=setting['image_prefix'], corr_window_csv='corr_window_stat.csv', z_stamp_pkl='z_stamp_optimal_stat', y_stamp_pkl='y_stamp_optimal_stat')
            else:
                z_stamp_opt = unpickle_object('z_stamp_optimal_stat')
                y_stamp_opt = unpickle_object('y_stamp_optimal_stat')
                zooks = unpickle_object('stat_zooks')
                zooks.trim_zooks()

            z_stamp_cf, res, slopes = compute_zstamp_curvefit(
                z_stamp_opt, zooks)

            # plt.plot(res)
            # plt.plot(z_stamp_opt)
            # plt.plot(z_stamp_cf)
            # plt.show()

            # print zooks
            # bz = detect_bad_zooks(res, zooks, slopes)
            # print zooks

            CROP_GUI = False

            if CROP_GUI:
                img_path = glob.glob(
                    os.path.join(setting['bf_images'], '*.tif')
                )
                img_seq = load_image_sequence(
                    [img_path[setting['index_start_at'] + ref_frame_no]])
                masking_window_frame(img_seq[0])
                crop_params = get_rect_params()

                data = [
                    ('frame', ref_frame_no),
                    ('x_end', crop_params['x_end']),
                    ('height', crop_params['height']),
                    ('y_end', crop_params['y_end']),
                    ('width', crop_params['width']),
                ]

                print 'Using cropping window as-'
                print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

                pickle_object(data, file_name='crop_window_stat.csv',
                              dumptype='csv')

            WRITE_IMAGES = True

            if WRITE_IMAGES:
                shift_frames_and_store_yz(
                    setting['bf_images'], setting[
                        'image_prefix'], setting['stat_index_start_at'],
                    z_stamp_opt, y_stamp_opt, zooks, setting[
                        'stat_images_cropped'], 'crop_window_stat.csv'
                )
    except:
        traceback.print_exc()

    import code
    code.interact(local=locals())

    # plt.plot(z_stamp_det)
    # plt.show()
