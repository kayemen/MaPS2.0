if __name__ == '__main__':
    import sys
    import os
    print 'Appending to path:'
    print os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import skimage.external.tifffile as tff
from scipy import stats, signal
from skimage.transform import resize

from maps import settings
from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.helpers.gui_modules import load_frame, extract_window, cv2_methods, create_image_overlay, load_image_sequence, masking_window_frame
from maps.helpers.misc import pickle_object, unpickle_object, write_error_plot,\
    make_or_clear_directory
from maps.settings import setting, read_setting_from_json
from maps.helpers.img_proccessing import corr2, corr2_masked, butter_lowpass_filtfilt

import os
import re
import shutil
import logging
import collections
import maps
import csv
import glob
import cv2
import time
import threading
import traceback
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError
import matplotlib.pyplot as plt


def get_fluorescent_filename(frame_no):
    name_format = '%%s%%0%dd_%%s.tif' % setting['fm_num_digits']
    ret_name = name_format % (setting['fm_image_prefix'], setting[
        'fm_index_start_at'] + frame_no, setting.get('fm_image_suffix', ''))
    ret_name = ret_name.replace('_.tif', '.tif')
    return ret_name


def get_cropped_brightfield_filename(frame_no):
    name_format = '%%s%%0%dd_%%s.tif' % 5
    ret_name = name_format % (
        'FRAMEX', frame_no, setting.get('image_suffix', ''))
    ret_name = ret_name.replace('_.tif', '.tif')
    return ret_name


def find_best_frame(phi_z_cell):
    matchvals = [obj[1] for obj in phi_z_cell]
    best_fit = phi_z_cell[matchvals.index(min(matchvals))][0]
    return best_fit


def write_final_frame(src, dst, adj_brightness=False, method='copy', params=None):
    '''
    valid methods='copy', 'adj_brightness'
    '''
    THRESHOLD = 200
    NORM_POINT = 10000

    if method == 'copy' or adj_brightness is False:
        shutil.copy2(src, dst)
    elif method == 'adj_brightness':
        frame = tff.imread(src)
        frame[np.where(frame > THRESHOLD)] = NORM_POINT / \
            (np.max(frame) - THRESHOLD) * frame[np.where(frame > THRESHOLD)]
        tff.imsave(dst, frame.astype('int16'))
        # tff.imsave(dst, frame)
    elif method == 'med_adj_brightness':
        if params is None:
            raise Exception('No median point set')
        else:
            shift_pt = np.percentile(params, 70)

        frame = tff.imread(src)
        frame = shift_pt / (np.median(frame)) * frame
        tff.imsave(dst, frame.astype('int16'))


def write_phase_stamped_fluorescent_images(phi_z_matrix, read_from, write_to, brightness_adjust=False, brightness_adjust_params=('med_adj_brightness', None)):
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
                missing_info.append('Final_T%03d_Z%03d.tif' % (phase, z))
                continue

            best_fit = find_best_frame(curr_cell)

            rewrite_lookup['Final_T%03d_Z%03d.tif' % (phase, z)] = best_fit

    ref_fluorescent_image_path = os.path.join(
        read_from, get_fluorescent_filename(setting['fm_index_start_at']))

    dummy_image = tff.imread(ref_fluorescent_image_path)
    dummy_image = dummy_image * 0

    for final_name, frame_no in rewrite_lookup.iteritems():
        # TODO: Read and write instead of copy to allow brightness/motion
        # adjustment
        src = os.path.join(read_from, get_fluorescent_filename(frame_no))
        dst = os.path.join(write_to, final_name)
        write_final_frame(src, dst, adj_brightness=brightness_adjust, method=brightness_adjust_params[0],
                          params=brightness_adjust_params[1])
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


def check_fluorescent_bleaching(zooks, read_from, PLOT=False):

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
        if PLOT:
            plt.plot(mean_vals.transpose())
        # plt.plot(max_vals)
    avg_brightness = np.mean(mean_vals.transpose(), axis=1)

    if PLOT:
        plt.title('Mean vals - separate')
        plt.figure()
        plt.title('Mean vals - curvefit')
        plt.plot(avg_brightness)
        plt.show()

    return avg_brightness


def write_brother_frames(phi_z_matrix, read_from_fluor, read_from_phase, write_to, brother_frame_file):
    # setting['brother_frames']
    # File format -
    # Z-<start>:<end>,P-<start>:<end>
    '''
    phi, z
    '''
    fluor_bro_frames = os.path.join(write_to, 'fluor')
    bf_bro_frames = os.path.join(write_to, 'phase')

    make_or_clear_directory(fluor_bro_frames)
    make_or_clear_directory(bf_bro_frames)
    brother_frame_locs = []

    dummy_file = glob.glob(os.path.join(read_from_fluor, '*.tif'))[0]
    dummy_image_fluor = tff.imread(dummy_file)
    dummy_image_fluor = dummy_image_fluor * 0

    dummy_file = glob.glob(os.path.join(read_from_phase, '*.tif'))[0]
    dummy_image_phase = tff.imread(os.path.join(dummy_file))
    dummy_image_phase = dummy_image_phase * 0

    with open(os.path.join(setting['workspace'], brother_frame_file), 'r') as fp:
        for line in fp.readlines():
            line_vals = re.match(
                r'Z-(\d*):(\d*),P-(\d*):(\d*)', line.strip()).groups()
            print line_vals
            brother_frame_locs.append(map(lambda x: int(x), line_vals))

    for (zstart, zend, pstart, pend) in brother_frame_locs:
        num_brother_frames = 0
        for phase in range(pstart, pend):
            for z in range(zstart, zend):
                curr_cell = phi_z_matrix[phase, z]
                if len(curr_cell) > num_brother_frames:
                    num_brother_frames = len(curr_cell)

        for phase in range(pstart, pend):
            for z in range(zstart, zend):
                curr_cell = phi_z_matrix[phase, z]

                # Name the frames and get frame numbers in decreasing order of
                # matchval
                brother_frame_names = [
                    'Brother_frames_T%03d_Z%03d_F%%06d_B%02d.tif' % (
                        phase, z, index
                    ) for index in range(len(curr_cell))
                ] + [
                    'Brother_frames_T%03d_Z%03d_Fxxxxxx_B%02d.tif' % (
                        phase, z, index
                    ) for index in range(len(curr_cell), num_brother_frames)
                ]
                # Sort cell frames by match val
                curr_cell.sort(key=lambda t: t[0])
                brother_frame_ids = [frame[0] for frame in curr_cell] + \
                    [None for _ in range(len(curr_cell), num_brother_frames)]

        # Write frames
        for (frame_no, name_format) in zip(brother_frame_ids, brother_frame_names):
            if frame_no is not None:
                print frame_no
                final_name = name_format % (frame_no)

                src = os.path.join(
                    read_from_fluor, get_fluorescent_filename(frame_no))
                dst = os.path.join(fluor_bro_frames, final_name)
                shutil.copy2(src, dst)

                src = os.path.join(
                    read_from_phase, get_cropped_brightfield_filename(frame_no))
                dst = os.path.join(bf_bro_frames, final_name)
                shutil.copy2(src, dst)
            else:
                dst = os.path.join(fluor_bro_frames, name_format)
                tff.imsave(dst, dummy_image_fluor)

                dst = os.path.join(bf_bro_frames, name_format)
                tff.imsave(dst, dummy_image_phase)


def adjust_brightness(curr_frame, method='clip'):
    # curr_frame = img_seq[:, :, frame_no]

    if np.all(curr_frame == 0):
        return curr_frame

    if method == 'clip':
        curr_frame = np.clip(curr_frame, np.percentile(
            curr_frame, 5), np.percentile(curr_frame, 95))
        # curr_frame = (curr_frame.astype('float32') - np.min(curr_frame)
        #               ) * 65535 / (np.max(curr_frame) - np.min(curr_frame))
    elif method == 'gamma':
        pass

    return curr_frame


def create_fluorescent_masks(read_from, pz_matrix, from_z=None, to_z=None, mask_pkl_name='fluor_masks', ref_frame_pkl_name='fluor_ref_frames', read_pickled_masks=None):
    if from_z is None:
        from_z = 0
    if to_z is None:
        to_z = pz_matrix.shape[1]

    total_masks = to_z - from_z

    mask_array = [None for i in range(pz_matrix.shape[1])]
    ref_frames = [None for i in range(pz_matrix.shape[1])]

    mask_dir = os.path.join(setting['workspace'], 'fluor_masks')
    make_or_clear_directory(mask_dir, clear=False)

    if read_pickled_masks is not None:
        import glob
        mask_list = glob.glob(os.path.join(mask_dir, '*.tif'))
        for m in mask_list:
            mask_id = int(os.path.basename(m).split('.')[0][1:])
            if from_z <= mask_id < to_z:
                mask_array[mask_id] = tff.imread(m)

    pass_all = ''
    # for mask_no, z in enumerate(range(from_z, to_z)):
    z = from_z
    while from_z <= z < to_z:
        mask_no = z - from_z
        if mask_array[z] is not None and pass_all:
            z = z + 1
            continue

        print 'Generating mask %d(%d/%d)' % (z, mask_no + 1, total_masks)
        # Load all images which have this z stamp
        img_list = [os.path.join(read_from, 'Final_T%03d_Z%03d.tif' % (
            phi, z)) for phi in range(pz_matrix.shape[0])]

        print 'loaded %d images' % len(img_list)

        img_seq = np.dstack(load_image_sequence(img_list))

        for frame_no in range(img_seq.shape[2]):
            img_seq[:, :, frame_no] = adjust_brightness(
                img_seq[:, :, frame_no])

        ref_frames[z], mask_array[z] = masking_window_frame(
            img_seq, crop_selection=False,
            mask_selection=True,
            curr_mask=mask_array[z]
        )
        print 'ref frame', ref_frames[z]

        tff.imsave(os.path.join(mask_dir, 'Z%03d.tif' % z), mask_array[z])

        next_frame = raw_input(
            'Next frame? (Enter frame number between %d and %d. Leave blank to go to next frame. Press "y" to pass all remaining frames: ' % (from_z, to_z))

        if next_frame.lower() == 'y':
            pass_all = True
            z = z + 1
        elif next_frame == '':
            z = z + 1
        elif next_frame.isdigit():
            z = int(next_frame)

    pickle_object(mask_array, mask_pkl_name)
    pickle_object(ref_frames, ref_frame_pkl_name)

    return mask_array, ref_frames


def invert_mask_array(mask_array, inv_mask_pkl_name='fluor_inv_masks'):
    inv_mask_array = [None for m in range(len(mask_array))]
    for mask, mask_val in enumerate(mask_array):
        if mask_val is None:
            continue
        inv_mask_array[mask] = 255 - mask_array[mask]

    pickle_object(inv_mask_array, inv_mask_pkl_name)

    return inv_mask_array


def find_optimal_shift(frame, ref_frame, frame_mask, slide_limit_x, slide_limit_y, plot=False):
    PLOT = True

    if np.all(frame == 0):
        return (0, 0)
    if np.all(ref_frame == 0):
        return (0, 0)
        # return (np.nan, np.nan)

    # print frame.shape
    # print ref_frame.shape
    # print frame_mask.shape

    corr_array = cv2.matchTemplate(
        frame.astype('float32'),
        ref_frame.astype('float32'),
        cv2_methods['ccorr_norm'],
        mask=frame_mask.astype('float32')
    )

    # corr_array = np.zeros((2*slide_limit_x-1, 2*slide_limit_y-1))
    # for i in range(-slide_limit_x, slide_limit_x-1):
    #     for j in range(-slide_limit_y, slide_limit_y-1):
    #         # print(slide_limit_x+i, -slide_limit_x+i)
    #         # print(slide_limit_y+j, -slide_limit_y+j)
    #         curr_frame = frame[
    #             slide_limit_x+i:-slide_limit_x+i+1,
    #             slide_limit_y+j:-slide_limit_y+j+1
    #         ]
    #         corr_array[i+slide_limit_x, j+slide_limit_y] = corr2_masked(
    #             curr_frame.astype('float32'),
    #             ref_frame.astype('float32'),
    #             frame_mask.astype('float32')
    #         )

    # print corr_array.shape
    # print corr_array

    y_shift = corr_array.max(axis=0).argmax() - slide_limit_y - 1
    x_shift = corr_array.max(axis=1).argmax() - slide_limit_x - 1

    # print(x_shift, y_shift)
    # if plot and (abs(x_shift) > slide_limit_x or abs(y_shift) > slide_limit_y):
    if plot:
        plt.subplot(221)
        plt.imshow(frame)
        plt.subplot(222)
        plt.imshow(ref_frame)
        plt.subplot(223)
        plt.imshow(corr_array, cmap='gray')
        plt.subplot(224)
        plt.plot(corr_array.max(axis=0))
        plt.plot(corr_array.max(axis=1))
        plt.legend(['y', 'x'], loc='upper right')
        plt.show()
    # raw_input()
    return (x_shift, y_shift)


def calculate_fluorescent_image_shift(read_from, pz_matrix, mask_array, ref_frames):
    PLOT = False

    s_l_x = 15
    s_l_y = 15

    phi_end, z_end = pz_matrix.shape

    x_stamps = np.zeros(pz_matrix.shape)
    y_stamps = np.zeros(pz_matrix.shape)

    # Computing intra z shift
    for z in range(z_end):
        # for z in range(97, 98):
        ref_frame_phase = ref_frames[z]
        if ref_frame_phase is None:
            # continue
            ref_frame_phase = 0
        print 'z ', z, ' ref frame at ', ref_frame_phase

        if z != 0:
            # Find relative phase stamp of curr_z with prev_z
            rel_x_stamp = 0
            rel_y_stamp = 0

        x_stamp_fix_z = np.ones((phi_end)) * np.nan
        y_stamp_fix_z = np.ones((phi_end)) * np.nan

        mask = mask_array[z]

        ref_frame = tff.imread(
            os.path.join(read_from, 'Final_T%03d_Z%03d.tif' %
                         (ref_frame_phase, z))
        )

        # ref_frame = adjust_brightness(ref_frame)

        # masked_ref_frame = (
        #     ref_frame * (mask / np.max(mask))
        # )[
        #     s_l_x: -s_l_x + 1,
        #     s_l_y: -s_l_y + 1
        # ]

        print 'Processing Z ', z
        for phi in range(phi_end):
            curr_frame = tff.imread(
                os.path.join(read_from, 'Final_T%03d_Z%03d.tif' %
                             (phi, z))
            )

            # curr_frame = adjust_brightness(curr_frame)

            # print 'phi-', phi

            # print curr_frame.shape
            # print ref_frame.shape
            # print mask.shape
            x_stamps[phi, z], y_stamps[phi, z] = find_optimal_shift(
                curr_frame, ref_frame[
                    s_l_x: -s_l_x + 1,
                    s_l_y: -s_l_y + 1
                ],
                mask[
                    # s_l_x: -s_l_x + 1,
                    # s_l_y: -s_l_y + 1
                    2*s_l_x: -2*s_l_x,
                    2*s_l_y: -2*s_l_y
                ],
                # mask,
                s_l_x, s_l_y,
                True if (z == 97) else False
                # curr_frame, masked_ref_frame, s_l_x, s_l_y,
                # (z in [105, 106, 107]) and phi == 10
            )
            # if not ((-s_l_x < x_stamps[phi, z] < s_l_x) and (-s_l_y < y_stamps[phi, z] < s_l_y)):
            #     print phi

    # plt.imshow(x_stamps, cmap='custom_heatmap')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(y_stamps, cmap='custom_heatmap')
    # plt.colorbar()
    # plt.show()

    '''# Computing inter z shift
    for z_prev, z_curr in zip(range(z_end - 1), range(1, z_end)):
        print 'comparing Z%d-Z%d' % (z_prev, z_curr)

        if ref_frames[z_prev] is None or ref_frames[z_curr] is None:
            continue

        mask = mask_array[z_prev]
        # curr_mask = mask_array[z_curr]

        inter_x_stamps = np.ones(phi_end) * np.nan
        inter_y_stamps = np.ones(phi_end) * np.nan

        # plt.subplot(211)
        # plt.plot(x_stamps[:, z_prev])
        # plt.subplot(212)
        # plt.plot(y_stamps[:, z_prev])
        # plt.show()
        for phi in range(phi_end):
            ref_frame = tff.imread(
                os.path.join(read_from, 'Final_Z%03d_T%03d.tif' %
                             (z_prev, phi))
            )

            ref_frame = adjust_brightness(ref_frame)

            x_prev = x_stamps[phi, z_prev]
            y_prev = y_stamps[phi, z_prev]

            # masked_ref_frame = (
            #     ref_frame * (mask / np.max(mask))
            # )[
            #     s_l_x + int(x_prev): -s_l_x + int(x_prev),
            #     s_l_y + int(y_prev): -s_l_y + int(y_prev)
            # ]

            curr_frame = tff.imread(
                os.path.join(read_from, 'Final_Z%03d_T%03d.tif' %
                             (z_curr, phi))
            )

            curr_frame = adjust_brightness(curr_frame)

            # masked_curr_frame = curr_frame * (curr_mask / np.max(curr_mask))

            # plt.imshow(masked_curr_frame)
            # plt.show()
            print 'Shifts-', (x_prev, y_prev)
            # print 'Z_next-', masked_curr_frame.shape
            # print 'Z_prev-', masked_ref_frame.shape

            try:
                inter_x_stamps[phi], inter_y_stamps[phi] = find_optimal_shift(
                    curr_frame, ref_frame[
                        s_l_x + int(x_prev): -s_l_x + int(x_prev),
                        s_l_y + int(y_prev): -s_l_y + int(y_prev)
                    ], mask[
                        s_l_x + int(x_prev): -s_l_x + int(x_prev),
                        s_l_y + int(y_prev): -s_l_y + int(y_prev)
                    ], s_l_x, s_l_y, True
                    # masked_curr_frame, masked_ref_frame, s_l_x, s_l_y, True
                    # curr_frame, masked_ref_frame, s_l
                )
            except:
                import traceback
                traceback.print_exc()
                print 'Z-', z_prev
                # import code
                # code.interact(local=locals())

        try:
            hist, _, _ = np.histogram2d(
                inter_x_stamps, inter_y_stamps, bins=2 * max(s_l_x, s_l_y) + 1, normed=False)

            inter_x_stamp = hist.max(axis=0).argmax() - s_l_x
            inter_y_stamp = hist.max(axis=1).argmax() - s_l_y

            x_stamps[phi, z_curr] += inter_x_stamp
            y_stamps[phi, z_curr] += inter_y_stamp

        except:
            import traceback
            traceback.print_exc()
            import code
            code.interact(local=locals())
    '''
    # plt.imshow(x_stamps)
    # plt.figure()
    # plt.imshow(y_stamps)
    # plt.show()

    pickle_object(x_stamps, 'x_stamps_fluor')
    pickle_object(y_stamps, 'y_stamps_fluor')

    return x_stamps, y_stamps


def calculate_relative_fluorescent_image_shift(read_from, pz_matrix, mask_array, ref_frames, x_stamps, y_stamps):
    from skimage.transform import resize

    PLOT = False

    s_l_x = 15
    s_l_y = 15

    # phi_end, z_end = pz_matrix.shape
    z_start = 0
    z_end = pz_matrix.shape[1]-1
    phi = 20

    x_stamps_new = np.zeros(pz_matrix.shape)
    y_stamps_new = np.zeros(pz_matrix.shape)

    x_prev = 0
    y_prev = 0

    for z_prev, z_next in zip(range(z_start, z_end-1), range(z_start+1, z_end)):
        ref_frame = tff.imread(
            os.path.join(read_from, 'Final_T%03d_Z%03d.tif' %
                         (phi, z_prev))
        )

        # ref_frame = resize(
        #     ref_frame,
        #     (
        #         ref_frame.shape[0]*4,
        #         ref_frame.shape[1]*4,
        #     )
        # )

        curr_frame = tff.imread(
            os.path.join(read_from, 'Final_T%03d_Z%03d.tif' %
                         (phi, z_next))
        )

        # curr_frame = resize(
        #     curr_frame,
        #     (
        #         curr_frame.shape[0]*4,
        #         curr_frame.shape[1]*4,
        #     )
        # )

        mask_prev = mask_array[z_prev]

        # print(max(0, s_l_x + int(x_stamps[phi, z_prev])),
        #   min(0, -s_l_x + int(x_stamps[phi, z_prev]) - 1))
        mask_prev = mask_prev[
            max(0, s_l_x + int(x_stamps[phi, z_prev])):min(0, -s_l_x + int(x_stamps[phi, z_prev]) - 1),
            max(0, s_l_y + int(y_stamps[phi, z_prev])):min(0, -s_l_y + int(y_stamps[phi, z_prev]) - 1)
        ]
        mask_next = mask_array[z_next]
        mask_next = mask_next[
            max(0, s_l_x + int(x_stamps[phi, z_next])):min(0, -s_l_x + int(x_stamps[phi, z_next]) - 1),
            max(0, s_l_y + int(y_stamps[phi, z_next])):min(0, -s_l_y + int(y_stamps[phi, z_next]) - 1)
        ]

        # ref_frame_mask = ref_frame * (mask_prev/255)
        # print np.max(ref_frame - ref_frame_mask), np.min(ref_frame - ref_frame_mask)
        # plt.imshow(ref_frame - ref_frame_mask)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(mask_next)
        # plt.show()
        # print(np.all(mask_prev == mask_next))
        # mask = np.ones(ref_frame.shape)
        mask = (mask_prev/255)*mask_next

        # mask = resize(
        #     mask,
        #     (
        #         mask.shape[0]*4,
        #         mask.shape[1]*4
        #     )
        # )

        # plt.subplot(211)
        # plt.imshow(ref_frame*(mask/255))
        # plt.subplot(212)
        # plt.imshow(curr_frame*(mask/255))
        # plt.show()
        print '-----------------------'
        print z_next
        print ref_frame.shape
        print curr_frame.shape

        x_new, y_new = find_optimal_shift(
            curr_frame, ref_frame[
                s_l_x: -s_l_x + 1,
                s_l_y: -s_l_y + 1
            ],
            mask[
                s_l_x: -s_l_x + 1,
                s_l_y: -s_l_y + 1
            ],
            s_l_x, s_l_y,
            False
        )

        x_stamps_new[:, z_next] = x_new
        y_stamps_new[:, z_next] = y_new
        # THRESH = 0
        # if x_new < 0:
        #     x_stamps_new[:, z_next] = x_new + 1
        # else:
        # x_stamps_new[:, z_next] = x_new
        # if y_new < 0:
        #     y_stamps_new[:, z_next] = y_new + 1
        # else:
        # if abs(x_new) > THRESH:
        #     x_stamps_new[:, z_next] = x_new
        # if abs(y_new) > THRESH:
        #     y_stamps_new[:, z_next] = y_new

        print(x_stamps_new[0, z_next], y_stamps_new[0, z_next])
        print(x_prev, y_prev)

        x_stamps_new[:, z_next], y_stamps_new[:, z_next] = x_stamps_new[
            :, z_next
        ] + x_prev, y_stamps_new[:, z_next] + y_prev
        print(x_stamps_new[0, z_next], y_stamps_new[0, z_next])

        x_prev, y_prev = x_stamps_new[phi, z_next], y_stamps_new[phi, z_next]
        # raw_input()

    # plt.plot(range(z_start, z_end), x_stamps_new[0, z_start:z_end])
    # plt.show()

    # plt.plot(range(166), x_stamps_new[0, :])
    # plt.plot(range(166), y_stamps_new[0, :])

    # plt.legend(['x', 'y'])
    # plt.figure()

    # slopex, interceptx, _, _, _ = stats.linregress(
    #     range(z_start, z_end), x_stamps_new[0, z_start:z_end]
    # )

    # x_cf = slopex * np.asarray(range(z_start, z_end)) + interceptx
    plt.figure()
    plt.plot(range(z_end - z_start + 1), x_stamps_new[0, :])
    plt.plot(range(z_end - z_start + 1), y_stamps_new[0, :])

    plt.legend(['x', 'y'])
    plt.title('Prefiltering')

    fx1, Sxx = signal.welch(x_stamps_new[20, z_start:z_end])
    x_cf = butter_lowpass_filtfilt(x_stamps_new[20, z_start:z_end], 0.1, 1)

    x_stamps_new[:, z_start:z_end] = x_stamps_new[:, z_start:z_end] - x_cf
    fx2, Sxf = signal.welch(x_stamps_new[20, z_start:z_end])

    # slopey, intercepty, _, _, _ = stats.linregress(
    #     range(z_start, z_end), y_stamps_new[0, z_start:z_end]
    # )

    # y_cf = slopey * np.asarray(range(z_start, z_end)) + intercepty
    fy1, Syy = signal.welch(y_stamps_new[20, z_start:z_end])
    y_cf = butter_lowpass_filtfilt(y_stamps_new[20, z_start:z_end], 0.05, 1)

    y_stamps_new[:, z_start:z_end] = y_stamps_new[:, z_start:z_end] - y_cf
    fy2, Syf = signal.welch(y_stamps_new[20, z_start:z_end])

    # print slope, intercept
    # plt.figure()
    # plt.plot(fx1, Sxx)
    # plt.plot(fy1, Syy)
    # plt.legend(['x', 'y'])
    # plt.title('Prefilt spectrum')
    # plt.imshow(x_stamps_new)
    # plt.colorbar()
    # plt.figure()
    # plt.plot(fx2, Sxf)
    # plt.plot(fy2, Syf)
    # plt.legend(['x', 'y'])
    # plt.title('Postfilt spectrum')
    # plt.imshow(y_stamps_new)
    # plt.colorbar()

    # plt.figure()
    plt.plot(range(z_end - z_start), x_cf)
    plt.plot(range(z_end - z_start), y_cf)

    plt.legend(['x', 'y', 'x_cf', 'y_cf'])
    plt.title('LPF')

    plt.figure()
    plt.plot(range(z_end - z_start + 1), x_stamps_new[0, :])
    plt.plot(range(z_end - z_start + 1), y_stamps_new[0, :])

    plt.legend(['x', 'y'])
    plt.title('Postfiltering')
    # plt.figure()
    # plt.plot(range(1, 166), x_stamps_new[0, 1:]-x_stamps_new[0, :-1])
    # plt.plot(range(1, 166), y_stamps_new[0, 1:]-y_stamps_new[0, :-1])
    plt.show()

    return x_stamps_new, y_stamps_new


def match_copied_frame_shifts(x_stamps_fluor, y_stamps_fluor, lut_name='phiz_fill_lut.txt'):
    with open(os.path.join(setting['workspace'], lut_name)) as fp:
        raw_lines = fp.readlines()

    fill_lut = {}

    for line in raw_lines:
        t = line.split(':')
        fill_lut[int(t[0])] = [int(i) for i in t[1].split(',')]

    x_stamps_fluor_new = np.copy(x_stamps_fluor)
    y_stamps_fluor_new = np.copy(y_stamps_fluor)

    for from_row, to_rows in fill_lut.iteritems():
        x_fill_row = x_stamps_fluor[from_row, :]
        y_fill_row = y_stamps_fluor[from_row, :]

        for to_row in to_rows:
            x_stamps_fluor_new[to_row, :] = x_fill_row
            y_stamps_fluor_new[to_row, :] = y_fill_row

    return (x_stamps_fluor_new, y_stamps_fluor_new)

def adjust_fluor_brightness(frame, mask, min_val=20):
    index_arr = (mask != 0)

    curr_min = np.min(frame[index_arr])
    curr_max = np.max(frame[index_arr])

    print curr_min, frame.dtype, curr_max

    frame[index_arr] = ((
        (frame[index_arr] - curr_min) * (65535.0 - curr_min - min_val) / (curr_max - curr_min)) + min_val).astype('uint16')

    print frame.min(), frame.max()


def shift_and_crop_fluor_images(read_from, write_to, mask_write_to, pz_matrix, x_stamps, y_stamps, mask_array, inv_mask_array, adj_brightness=False):
    s_l_x = 15
    s_l_y = 15

    phi_end, z_end = pz_matrix.shape

    # x_stamps = np.zeros(pz_matrix.shape)
    # y_stamps = np.zeros(pz_matrix.shape)

    # for z in range(80, 90):
    print '%s -> %s' % (read_from, write_to)
    for z in range(z_end):
        if mask_array[z] is None:
            continue
        for phi in range(phi_end):
            framename = os.path.join(
                read_from, 'Final_T%03d_Z%03d.tif' % (phi, z))
            # print 'Writing ', framename, 'to', write_to
            print 'Writing (T=%d, z=%d)|' % (phi, z),
            # print (z, phi)

            mask = mask_array[z]
            inv_mask = inv_mask_array[z]

            frame = tff.imread(framename).astype('uint16')
            exp_shape = frame[s_l_x: -s_l_x - 1, s_l_y:-s_l_y - 1].shape
            if mask is not None:
                if adj_brightness:
                    adjust_fluor_brightness(frame, mask)
                frame_inv = frame * (inv_mask / np.max(inv_mask))
                frame = frame * (mask / np.max(mask))
            else:
                frame_inv = frame
            # print 'orig', (z, phi), new_ref_frame.shape
            # print(x_stamps[phi, z], y_stamps[phi, z])
            x_shift, y_shift = int(x_stamps[phi, z]), int(y_stamps[phi, z])

            # print(x_shift, y_shift)

            cropped_frame = frame[
                max(0, s_l_x + x_shift):min(0, -s_l_x + x_shift - 1),
                max(0, s_l_y + y_shift):min(0, -s_l_y + y_shift - 1)
            ]
            cropped_frame_inv = frame_inv[
                max(0, s_l_x + x_shift):min(0, -s_l_x + x_shift - 1),
                max(0, s_l_y + y_shift):min(0, -s_l_y + y_shift - 1)
            ]

            if cropped_frame.shape != exp_shape:
                print 'here'
                # print cropped_frame.shape
                # print exp_shape
                right_pad = abs(cropped_frame.shape[1] - exp_shape[1])
                top_pad = abs(cropped_frame.shape[0] - exp_shape[0])
                cropped_frame = cv2.copyMakeBorder(
                    cropped_frame, top_pad, 0, 0, right_pad, cv2.BORDER_CONSTANT)
                cropped_frame_inv = cv2.copyMakeBorder(
                    cropped_frame_inv, top_pad, 0, 0, right_pad, cv2.BORDER_CONSTANT)
                # raw_input()

            try:
                tff.imsave(
                    os.path.join(write_to, 'Final_T%03d_Z%03d.tif' %
                                 (phi, z)),
                    cropped_frame
                )
                tff.imsave(
                    os.path.join(mask_write_to, 'Final_T%03d_Z%03d.tif' %
                                 (phi, z)),
                    cropped_frame_inv
                )
            except:
                print 'Invalid dimensions'
                print([
                    max(0, s_l_x - x_shift), min(0, -s_l_x - x_shift - 1),
                    max(0, s_l_y - y_shift), min(0, -s_l_y - y_shift - 1)
                ])
                tff.imsave(
                    os.path.join(write_to, 'Final_T%03d_Z%03d.tif' %
                                 (phi, z)),
                    frame[
                        s_l_x: -s_l_x - 1,
                        s_l_y: -s_l_y - 1
                    ]
                )


def canon_correlation(moving_dwt_array, canon_hb_dwt, mask_array=None, method='corr2'):
    if mask_array is None:
        mask_array = np.ones(canon_hb_dwt.shape)

    mask_array = mask_array.clip(0, 1)
    # print mask_array.shape

    corr_matrix = np.zeros((moving_dwt_array.shape[2], canon_hb_dwt.shape[2]))

    for canon_frame in range(canon_hb_dwt.shape[2]):
        for frame in range(moving_dwt_array.shape[2]):
            if method == 'corr2':
                corr_matrix[frame, canon_frame] = corr2_masked(
                    moving_dwt_array[:, :, frame],
                    canon_hb_dwt[:, :, canon_frame],
                    mask_array[:, :, canon_frame]
                )

                # print corr_matrix[frame, canon_frame]

    return corr_matrix
