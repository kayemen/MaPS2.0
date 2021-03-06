from maps.core.z_stamping_v2 import compute_raw_zstage, compute_zooks,\
    compute_zookzik_stats, drop_bad_zooks_zstage, compute_zstamp,\
    compute_deterministic_zstamp, compute_optimal_yz_stamp,\
    compute_zstamp_curvefit, detect_bad_zooks, shift_frames_and_store_yz,\
    Zook, Zooks
# from maps.core.z_stamping import z_stamping_step_yz,
# shift_frames_and_store_yz
from maps.core.phase_stamping_v2 import crop_and_compute_dwt_matrix,\
    compute_dwt_matrix, load_dwt_matrix, interchelate_dwt_array,\
    compute_heartbeat_length, compute_canonical_heartbeat,\
    phase_stamp_images, mask_canonical_heartbeat, write_dwt_matrix, \
    phase_stamp_images_masked, compile_phase_z_matrix, \
    filter_phi_z_mat, write_brother_frames, fill_phi_z_row, \
    phi_z_histogram, plot_results, write_pz_density, plot_matrix_rows

from maps.core.post_processing import write_phase_stamped_fluorescent_images, \
    check_fluorescent_bleaching, calculate_fluorescent_image_shift, \
    shift_and_crop_fluor_images, create_fluorescent_masks, invert_mask_array, \
    canon_correlation, write_brother_frames, find_best_frame, \
    calculate_relative_fluorescent_image_shift, match_copied_frame_shifts


from maps import settings
from maps.settings import setting, read_setting_from_json, \
    display_current_settings, JOBS_DIR, display_step
from maps.helpers.misc import pickle_object, unpickle_object, zook_approval_function, make_or_clear_directory, write_log_data
from maps.helpers.gui_modules import load_image_sequence, max_heartsize_frame,\
    get_rect_params, masking_window_frame
from maps.helpers.logging_config import logger_config

import glob
import os
import time
import logging
import numpy as np
import code
import matplotlib.pyplot as plt
import sys
import uuid
import traceback
import json
# import argparse

SAVE_WORKSPACE = '-s' in sys.argv

logging.config.dictConfig(logger_config)

job_list = next(os.walk(settings.JOBS_DIR))[1]

job_choice = 0

try:
    job_choice = int(sys.argv[1])
except:
    print '\n'.join(
        map(
            lambda x, y: '%d-%s' % (x, y),
            range(1, len(job_list) + 1),
            job_list
        )
    )

    while job_choice not in range(1, len(job_list) + 1):
        job_choice = int(raw_input('Select job:'))
curr_job = job_list[job_choice - 1]
read_setting_from_json(curr_job, save_bkups=SAVE_WORKSPACE)

try:
    run_name = sys.argv[2]
except:
    run_name = time.strftime('%d%b%y_%H%M')

try:
    execfile(os.path.join(JOBS_DIR, curr_job, 'local_settings.py'))
except:
    print '''Local setting override python file not found.
    Please create a local_settings.py file in the job directory.
    Using default config parameters'''
    Y_STAGE_CORRECTION = True
    WRITE_IMAGES_TO_DISK = True
    USE_PICKLED_ZSTAMPS = False
    USE_PICKLED_ZSTAMPS_STAT = False
    USE_PICKLED_PHASESTAMPS = False
    USE_PICKLED_DWT = False
    USE_PICKLED_CMHB = False
    USE_PICKLED_SAD = False
    USE_PICKLED_CANON_DWT = False
    USE_PICKLED_PHI_Z_MATRIX = False
    WRITE_DWT = True
    WRITE_FINAL_IMAGES = True


with open(os.path.join(settings.JOBS_DIR, curr_job, 'job_metadata.txt'), 'a') as metafile:
    data = '''
--------------------------------------------------------------------------------
Run name: {run_name}
Y correction: {ycorr}
Difference matrix method: {diff_method}
Multiprocessing: {mp}
'''.format(
        run_name=run_name,
        ycorr=setting['y_correction'],
        mp=settings.MULTIPROCESSING,
        diff_method=DIFF_MAT_METHOD
    )
    metafile.write(data)


try:
    display_current_settings()

    write_log_data('-' * 40, '')
    write_log_data(run_name, '')

    '''
    ========================================================================
    Moving frame z stamping
    ========================================================================
    '''

    if USE_GUI_CORR_WINDOW and (not USE_PICKLED_ZSTAMPS):
        # Uses GUI to select rectangular window in specified frame.
        # Click and drag with mouse tto select region. Clos window to finalize
        display_step('Selecting correlation window - moving frames')
        img_path = glob.glob(
            os.path.join(setting['bf_images'], '*.tif')
        )
        img_seq = load_image_sequence(
            img_path[setting['index_start_at'] + setting['ignore_startzook']: setting['index_start_at'] + setting['ZookZikPeriod'] - setting['ignore_endzook']])
        ref_frame_no = max_heartsize_frame(
            np.dstack(img_seq),
            startval=setting['ignore_startzook']
        )

        params = get_rect_params()
    else:
        try:
            params = dict(unpickle_object(
                file_name='corr_window.csv', dumptype='csv'))
            params = {key: int(val) for key, val in params.iteritems()}
            ref_frame_no = params['frame']
        except:
            traceback.print_exc()
            params = {
                # Bottom edge of window
                'x_end': 108,
                # Height of window
                'height': 34,
                # Right edge of window
                'y_end': 250,
                # Width of window
                'width': 31,
            }

    # img_path = glob.glob(
    #     os.path.join(setting['bf_images'], '*.tif')
    # )
    # img_shape = load_image_sequence(
    #     [img_path[setting['index_start_at'] + ref_frame_no]])[0].shape

    data = [
        ('frame', ref_frame_no),
        ('x_end', params['x_end']),
        ('height', params['height']),
        ('y_end', params['y_end']),
        ('width', params['width']),
    ]

    print 'Using reference window as-'
    print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

    pickle_object(data, file_name='corr_window.csv', dumptype='csv')
    # pickle_object(stat_data, file_name='stat_corr_window.csv',
    # dumptype='csv')

    if USE_GUI_CROP_WINDOW and WRITE_CROPPED_IMAGES_TO_DISK:
        display_step('Selecting cropping window - moving frames')

        img_path = glob.glob(
            os.path.join(setting['bf_images'], '*.tif')
        )
        img_seq = load_image_sequence(
            [img_path[setting['index_start_at'] + ref_frame_no]])
        masking_window_frame(img_seq[0])

        crop_params = get_rect_params()
    else:
        try:
            crop_params = dict(unpickle_object(
                'crop_window.csv', dumptype='csv'))
            crop_params = {key: int(value)
                           for key, value in crop_params.iteritems()}
        except:
            crop_params = {
                # Bottom edge of window
                'x_end': 319,
                # Height of window
                'height': 194,
                # Right edge of window
                'y_end': 317,
                # Width of window
                'width': 213,
            }

    if WRITE_CROPPED_IMAGES_TO_DISK:
        data = [
            ('frame', ref_frame_no),
            ('x_end', crop_params['x_end']),
            ('height', crop_params['height']),
            ('y_end', crop_params['y_end']),
            ('width', crop_params['width']),
        ]

        print 'Using cropping window as-'
        print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

        pickle_object(data, file_name='crop_window.csv', dumptype='csv')

    if USE_GUI_CORR_WINDOW_STAT and (not USE_PICKLED_ZSTAMPS_STAT):
        # Uses GUI to select rectangular window in specified frame.
        # Click and drag with mouse tto select region. Clos window to finalize
        display_step('Selecting correlation window - stationary frames')
        img_path = glob.glob(
            os.path.join(setting['bf_images'], '*.tif')
        )
        img_seq = load_image_sequence(
            [img_path[setting['stat_index_start_at'] + ref_frame_no]])
        max_heartsize_frame(img_seq[0])

        params = get_rect_params()
    else:
        try:
            params = dict(unpickle_object(
                file_name='corr_window_stat.csv', dumptype='csv'))
            params = {key: int(val) for key, val in params.iteritems()}
        except:
            params = {
                # Bottom edge of window
                'x_end': 108,
                # Height of window
                'height': 34,
                # Right edge of window
                'y_end': 250,
                # Width of window
                'width': 31,
            }
    data = [
        ('frame', ref_frame_no),
        ('x_end', params['x_end']),
        ('height', params['height']),
        ('y_end', params['y_end']),
        ('width', params['width']),
    ]

    print 'Using reference window as-'
    print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])

    pickle_object(data, file_name='stat_corr_window.csv', dumptype='csv')

    if USE_GUI_CROP_WINDOW_STAT and WRITE_CROPPED_STAT_IMAGES_TO_DISK:
        display_step('Selecting cropping window - stationary frames')

        img_path = glob.glob(
            os.path.join(setting['bf_images'], '*.tif')
        )
        img_seq = load_image_sequence(
            [img_path[setting['stat_index_start_at'] + ref_frame_no]])
        masking_window_frame(img_seq[0])

        crop_params = get_rect_params()
    else:
        try:
            crop_params = dict(unpickle_object(
                'crop_window_stat.csv', dumptype='csv'))
            crop_params = {key: int(value)
                           for key, value in crop_params.iteritems()}
        except:
            crop_params = {
                # Bottom edge of window
                'x_end': 319,
                # Height of window
                'height': 194,
                # Right edge of window
                'y_end': 317,
                # Width of window
                'width': 213,
            }

    if WRITE_CROPPED_STAT_IMAGES_TO_DISK:
        data = [
            ('frame', ref_frame_no),
            ('x_end', crop_params['x_end']),
            ('height', crop_params['height']),
            ('y_end', crop_params['y_end']),
            ('width', crop_params['width']),
        ]

        print 'Using cropping window as-'
        print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

        pickle_object(data, file_name='crop_window_stat.csv', dumptype='csv')

    kymos = glob.glob(os.path.join(setting['km_path'], '*.tif'))

    display_step('Z Stamping step - moving frames')
    if USE_PICKLED_ZSTAMPS:
        z_stamp_opt = unpickle_object('z_stamp_optimal')
        y_stamp_opt = unpickle_object('y_stamp_optimal')
        zooks = unpickle_object('zooks')
        zooks.trim_zooks()
    else:
        print '\n'.join(
            map(
                lambda x, y: '%d-%s' % (x, y),
                range(1, len(kymos) + 1),
                kymos
            )
        )
        choice = 0
        while choice not in range(1, len(kymos) + 1):
            choice = int(raw_input('Select kymograph:'))

        kymo_path = glob.glob(
            os.path.join(setting['km_path'], '*.tif')
        )[choice - 1]

        write_log_data(kymo_path, 'Kymographs used')

        z_stage = compute_raw_zstage(
            kymo_path,
            median_filt=True,
            kernel_size=setting.get('median_kernel', 5)
        )

        # if DEBUG:
        #     plt.figure()
        #     plt.plot(z_stage)
        #     plt.show(block=PLOT_BLOCK)
        # assert setting['frame_count'] < z_stage_len

        zooks = compute_zooks(
            z_stage[:setting['frame_count']]
        )

        zz_stats = compute_zookzik_stats(
            zooks
        )

        write_log_data(json.dumps(zz_stats), 'ZookZik stats')

        drop_bad_zooks_zstage(
            zooks,
            zz_stats,
            z_stage,
            threshold_mult=3
        )

        print 'Initial bad zooks'
        for zook in zooks.get_bad_zooks():
            print 'Zook', zook.id, 'is potentially bad'
        print len(zooks.get_bad_zooks()), 'bad zooks detected'

        z_stamp, _ = compute_zstamp(
            z_stage[:setting['frame_count']],
            zooks,
            ref_frame_no
        )

        # if DEBUG:
        #     plt.figure()
        #     plt.plot(z_stamp)
        #     plt.show(block=PLOT_BLOCK)

        z_stamp_det = compute_deterministic_zstamp(
            z_stamp,
            zooks,
            ref_frame_no
        )

        if DEBUG:
            plt.figure()
            plt.plot(z_stamp[:1000])
            plt.plot(z_stamp_det[:1000])
            plt.show(block=PLOT_BLOCK)

        z_stamp_opt, y_stamp_opt = compute_optimal_yz_stamp(
            setting['bf_images'],
            zooks,
            z_stamp_det,
            # z_stamp,
            offset=setting['index_start_at'],
            prefix=setting['image_prefix']
        )

        pickle_object(zooks, 'zooks')
        if DEBUG:
            plt.plot(z_stamp_opt)
            plt.figure()
            plt.plot(y_stamp_opt)
            plt.show(block=PLOT_BLOCK)

    # if USE_GUI_CROP_WINDOW and WRITE_CROPPED_IMAGES_TO_DISK:
    #     display_step('Selecting cropping window - moving frames')

    #     img_path = glob.glob(
    #         os.path.join(setting['bf_images'], '*.tif')
    #     )
    #     img_seq = load_image_sequence(
    #         [img_path[setting['index_start_at'] + ref_frame_no]])
    #     masking_window_frame(img_seq[0])

    #     crop_params = get_rect_params()
    # else:
    #     try:
    #         crop_params = dict(unpickle_object(
    #             'crop_window.csv', dumptype='csv'))
    #         crop_params = {key: int(value)
    #                        for key, value in crop_params.iteritems()}
    #     except:
    #         crop_params = {
    #             # Bottom edge of window
    #             'x_end': 319,
    #             # Height of window
    #             'height': 194,
    #             # Right edge of window
    #             'y_end': 317,
    #             # Width of window
    #             'width': 213,
    #         }

    if WRITE_CROPPED_IMAGES_TO_DISK:
        # data = [
        #     ('frame', ref_frame_no),
        #     ('x_end', crop_params['x_end']),
        #     ('height', crop_params['height']),
        #     ('y_end', crop_params['y_end']),
        #     ('width', crop_params['width']),
        # ]

        # print 'Using cropping window as-'
        # print '\n'.join(['%s:%d' % (attr[0], attr[1]) for attr in data])

        # pickle_object(data, file_name='crop_window.csv', dumptype='csv')

        z_stamp_cf, residues, slopes = compute_zstamp_curvefit(
            z_stamp_opt,
            zooks
        )

        bad_zooks = detect_bad_zooks(
            residues,
            zooks,
            slopes,
            threshold_mult=1.0
        )

        # print 'Post shift bad zooks'
        # for zook in zooks.get_bad_zooks():
        #     print zook.id, 'marked bad'
        # raise Exception
        zook_approval_function(
            zooks,
            # z_stamp_opt
        )

        print zooks
        # raise Exception
        # print zooks
        pickle_object(zooks, 'zooks')

    if WRITE_CROPPED_IMAGES_TO_DISK:
        display_step('Writing cropped moving frames')
        make_or_clear_directory(setting['cropped_bf_images'], clear=True)
        shift_frames_and_store_yz(
            img_path=setting['bf_images'],
            prefix=setting['image_prefix'],
            offset=setting['index_start_at'],
            z_stamps=z_stamp_opt,
            # # Use y_stamp_opt to get Y stamping, otherwise give zeros
            y_stamps=y_stamp_opt if Y_STAGE_CORRECTION else np.zero(
                y_stamp_opt.shape),
            zooks=zooks,
            write_to=setting['cropped_bf_images'],
            adj_intensity=ADJUST_BRIGHTNESS
        )

    if WRITE_REFERENCE_WINDOW:
        display_step('Writing reference windows')
        make_or_clear_directory(setting['cropped_ref_windows'])
        shift_frames_and_store_yz(
            img_path=setting['bf_images'],
            prefix=setting['image_prefix'],
            offset=setting['index_start_at'],
            z_stamps=z_stamp_opt,
            # # Use y_stamp_opt to get Y stamping, otherwise give zeros
            y_stamps=y_stamp_opt,
            # np.zeros(y_stamp_opt.shape),
            zooks=zooks,
            write_to=setting['cropped_ref_windows'],
            crop_csv_file='corr_window.csv',
            adj_intensity=ADJUST_BRIGHTNESS
        )

    '''
    ========================================================================
    Stationary frame z stamping
    ========================================================================
    '''

    display_step('Z Stamping step - stationary frames')
    if USE_PICKLED_ZSTAMPS_STAT:
        y_stamp_opt_stat = unpickle_object('y_stamp_optimal_stat')
        z_stamp_opt_stat = unpickle_object('z_stamp_optimal_stat')
        zooks_stat = unpickle_object('zooks_stat')
        zooks_stat.trim_zooks()
    else:
        settings.TIMEOUT = 400
        # Fake Zstage for stationary frames. treating as single zook
        display_step('Z Stamping stationary images')
        z_stage_stat = np.zeros(setting['stat_frame_count'])
        zooks_stat = Zooks()
        zooks_stat.add_zook(0, setting['stat_frame_count'] - 1)
        zooks_stat.trim_zooks()
        pickle_object(zooks_stat, 'zooks_stat')

        # z_stamp_stat, _ = compute_zstamp(
        #     z_stage_stat[:setting['stat_frame_count']],
        #     zooks_stat,
        #     ref_frame_no,
        #     z_stamp_pkl_name='z_stamp_stat'
        # )

        # z_stamp_det_stat = compute_deterministic_zstamp(
        #     z_stamp_stat,
        #     zooks_stat,
        #     ref_frame_no,
        #     z_stamp_pkl_name='z_stamp_det_stat'
        # )

        z_stamp_opt_stat, y_stamp_opt_stat = compute_optimal_yz_stamp(
            setting['bf_images'],
            zooks_stat,
            z_stage_stat,
            offset=setting['stat_index_start_at'],
            prefix=setting['image_prefix'], corr_window_csv='stat_corr_window.csv', z_stamp_pkl='z_stamp_optimal_stat', y_stamp_pkl='y_stamp_optimal_stat'
        )

        pickle_object(zooks_stat, 'zooks_stat')

    if WRITE_CROPPED_STAT_IMAGES_TO_DISK:
        zook_approval_function(
            zooks_stat,
            # z_stamp_opt
        )

        # plt.plot(z_stamp_opt_stat)
        # plt.show()

    if WRITE_CROPPED_STAT_IMAGES_TO_DISK:
        display_step('Writing cropped stationary frames')

        make_or_clear_directory(setting['stat_images_cropped'])

        shift_frames_and_store_yz(
            setting['bf_images'],
            setting['image_prefix'],
            setting['stat_index_start_at'],
            z_stamp_opt_stat,
            y_stamp_opt_stat,
            zooks_stat,
            setting['stat_images_cropped'],
            'crop_window_stat.csv'
        )

    # raw_input('Copy stationary images')

    if not USE_PICKLED_PHASESTAMPS:
        '''
        ========================================================================
        Computing DWT
        ========================================================================
        '''

        if USE_PICKLED_DWT:
            # moving_dwt_array = unpickle_object('dwt_array')
            display_step('Loading moving DWT array')
            moving_dwt_array = load_dwt_matrix(
                setting['bf_images_dwt'],
                # num_frames=5000
            )

            display_step('Loading upsampled moving DWT array')
            canonical_dwt_array = load_dwt_matrix(
                setting['bf_images_dwt_upsampled']
            )

            display_step('Loading stationary DWT array')
            static_dwt_array = load_dwt_matrix(
                setting['stat_images_dwt_upsampled']
            )
        else:
            tic = time.time()
            # Use written images
            display_step('Computing DWT from cropped brightfield images')

            make_or_clear_directory(setting['bf_images_dwt'], clear=True)

            moving_dwt_array = compute_dwt_matrix(
                img_path=setting['cropped_bf_images'],
                frame_indices=zooks.get_framelist(),
                write_to=setting['bf_images_dwt'],
                validTiff=False
            )

            display_step('Computing DWT from cropped stationary images')
            static_dwt_array = compute_dwt_matrix(
                img_path=setting['stat_images_cropped'],
                frame_indices=zooks_stat.get_framelist(),
                # frame_indices=zooks.get_framelist()[:3773],
                write_to=None,
                validTiff=False
            )

            make_or_clear_directory(
                setting['bf_images_dwt_upsampled'], clear=True)

            display_step('Upsampling canonical frames')

            canon_start_frame = setting['canon_zook_offset'] * (setting['ZookZikPeriod'] - setting['ignore_startzook'] - setting['ignore_endzook'])
            canon_end_frame = canon_start_frame + setting['canon_frac']
                                 * moving_dwt_array.shape[2] / 100]
            canonical_dwt_array = interchelate_dwt_array(
                moving_dwt_array[:, :, canon_start_frame: canon_end_frame],
                zooks,
                write_to=setting['bf_images_dwt_upsampled'],
                validTiff=False
            )

            make_or_clear_directory(
                setting['stat_images_dwt_upsampled'], clear=True)

            display_step('Upsampling stationary frames')
            static_dwt_array = interchelate_dwt_array(
                static_dwt_array,
                write_to=setting['stat_images_dwt_upsampled'],
                validTiff=False
            )

        setting['canonical_zook_count'] = setting['canon_frac'] * \
            moving_dwt_array.shape[2] / 100 / setting['ZookZikPeriod']

        '''
        ========================================================================
        Computing Canonical Heartbeat
        ========================================================================
        '''

        if USE_PICKLED_CMHB:
            display_step('Loading hearbeat length')
            cm_hb = unpickle_object('common_heatbeat_period.pkl')
        else:
            display_step('Computing hearbeat length')
            cm_hb = compute_heartbeat_length(
                static_dwt_array, comparison_plot=False)

        write_log_data(np.round(cm_hb))

        print 'Canonical hearbeat length: ', np.round(cm_hb)

        if USE_PICKLED_CANON_DWT:
            display_step('Loading canonical DWT array')
            canon_hb_dwt = load_dwt_matrix(
                setting['canon_frames']
            )
            display_step('Loading canonical heartbeat start')
            canon_hb_start = unpickle_object('canon_hb_start')
            display_step('Loading diff matrix')
            diff_matrix = unpickle_object('diff_matrix')
        else:
            display_step('Computing canonical heartbeat')

            make_or_clear_directory(setting['canon_frames'])
            make_or_clear_directory(setting['canon_frames_valid'])

            canon_hb_dwt, diff_matrix, canon_hb_start = compute_canonical_heartbeat(
                canonical_dwt_array, zooks,
                int(np.round(cm_hb[0])),
                # diff_method='sad',
                diff_method=DIFF_MAT_METHOD,
                canon_method='count',
                diff_matrix_pkl_name='diff_matrix.pkl',
                write_to=setting['canon_frames'],
                use_pickled_diff=USE_PICKLED_SAD
            )
            write_dwt_matrix(
                canon_hb_dwt.astype('float32'),
                setting['canon_frames_valid'],
                range(int(np.round((cm_hb[0])))),
                True
            )

        '''
        ========================================================================
        Computing Phase stamps
        ========================================================================
        '''

        display_step('Computing phase stamps')
        phase_stamps = phase_stamp_images(
            moving_dwt_array, canon_hb_dwt,
            method=DIFF_MAT_METHOD
        )
        '''
        ========================================================================
        Computing Masked Phase stamps
        ========================================================================
        '''

        if not USE_PICKLED_MASK_ARRAY:
            display_step('Computing masked canonical heartbeat')

            make_or_clear_directory(setting['canon_dwt_masked'], clear=True)

            mask_array, masked_canon_dwt = mask_canonical_heartbeat(
                canon_hb_dwt, num_masks=10)
        else:
            display_step('Loading masked canonical heartbeat')
            mask_array = unpickle_object('mask_array')
            masked_canon_dwt = unpickle_object('masked_canon_dwt')

        # code.interact(local=locals())

        # display_step('Computing masked phase stamps')
        # # phase_stamps_masked, corr = phase_stamp_images_masked(
        # phase_stamps_masked = phase_stamp_images_masked(
        #     moving_dwt_array,
        #     # moving_dwt_array[:, :, :2000],
        #     [canon_hb_dwt[:, :, phase]
        #         for phase in range(canon_hb_dwt.shape[2])],
        #     # masked_canon_dwt,
        #     mask_array,
        #     method=DIFF_MAT_METHOD
        # )

        # code.interact(local=locals())

        # phase_stamps = phase_stamp_images(
        #     moving_dwt_array, canon_hb_dwt,
        #     method=DIFF_MAT_METHOD
        # )
    else:
        display_step('Loading hearbeat length')
        cm_hb = unpickle_object('common_heatbeat_period.pkl')
        display_step('Loading canonical heartbeat start')
        canon_hb_start = unpickle_object('canon_hb_start')
        display_step('Loading diff matrix')
        diff_matrix = unpickle_object('diff_matrix.pkl')
        display_step('Loading phase stamps')
        phase_stamps = unpickle_object('phase_stamps.pkl')
        # display_step('Loading masked phase stamps')
        # phase_stamps_masked = unpickle_object('masked_phase_stamps.pkl')

    write_log_data(canon_hb_start, 'Canonical start (upsampled)')
    write_log_data(
        canon_hb_start / setting['time_resampling_factor'], 'Canonical start (downsampled)')
    num_contig = (setting['ZookZikPeriod'] - (
        setting['ignore_startzook'] + setting['ignore_endzook'])) * setting['time_resampling_factor']
    abs_canon_start = (canon_hb_start +
                       num_contig * (canon_hb_start // num_contig)) / setting['time_resampling_factor'] + setting['ignore_startzook']
    write_log_data(
        abs_canon_start, 'Canonical start (absolute)')

    # canonical_heartbeat = processed_frame_list[]

    write_log_data(z_stamp_opt[abs_canon_start], 'Canon start Z stamp')
    write_log_data(phase_stamps[
                   canon_hb_start / setting['time_resampling_factor']], 'Canon start phase stamp')
    # write_log_data(phase_stamps_masked[
    #                canon_hb_start / setting['time_resampling_factor']], 'Canon start phase stamp (masked)')

    '''
    ========================================================================
    Computing Phi Z Matrix
    ========================================================================
    '''

    if USE_PICKLED_PHI_Z_MATRIX:
        # z_stamp_opt = unpickle_object('z_stamp_optimal')
        # z_stamp_ds = unpickle_object('z_stamp_downsampled')
        phi_z_mat = unpickle_object('phase_z_matrix')
        pz_density = unpickle_object('phase_z_density')
        # phi_z_mat_masked = unpickle_object('phase_z_matrix_masked')
        # pz_density_masked = unpickle_object('phase_z_density_masked')
        phase_stamps = unpickle_object('phase_stamps')
        # phase_stamps_masked = unpickle_object('masked_phase_stamps')
    else:
        # display_step('Downsampling z stamps')
        # z_stamp_ds = np.round(z_stamp_opt)
        # plt.plot(z_stamp_opt)
        # plt.show()
        # z_stamp_ds = np.round(z_stamp_opt / setting['resampling_factor'])
        # z_stamp_ds = z_stamp_ds + np.abs(np.nanmin(z_stamp_ds))
        # plt.plot(z_stamp_ds)
        # plt.show()

        # pickle_object(z_stamp_ds, 'z_stamp_downsampled')

        display_step('Computing phase z matrix')
        phi_z_mat, pz_density = compile_phase_z_matrix(
            zooks.get_framelist(), z_stamp_opt,
            phase_stamps, int(np.round(cm_hb))
        )

        # display_step('Computing masked phase z matrix')
        # phi_z_mat_masked, pz_density_masked = compile_phase_z_matrix(
        #     zooks.get_framelist(), z_stamp_opt,
        #     phase_stamps_masked, int(np.round(cm_hb)),
        #     pz_mat_pkl_name='phase_z_matrix_masked', pz_density_pkl_name='phase_z_density_masked'
        # )

    '''
    ========================================================================
    Filling and filtering phi z matrix
    ========================================================================
    '''

    if USE_PICKLED_FILTERED_PHIZ:
        filtered_phi_z = unpickle_object('filtered_phi_z')
        filtered_pz_density = unpickle_object('filtered_pz_density')
        # filtered_phi_z_masked = unpickle_object('filtered_phi_z_masked')
        # filtered_pz_density_masked = unpickle_object(
        #     'filtered_pz_density_masked')
    else:
        display_step('Filtering phi z matrix')
        filtered_phi_z, filtered_pz_density = filter_phi_z_mat(
            phi_z_mat, phase_stamps, z_stamp_opt, percentile_filt=FILTER_PERCENTILE, pkl_name='negative_phase_stamps')

        # display_step('Filtering masked phi z matrix')
        # filtered_phi_z_masked, filtered_pz_density_masked = filter_phi_z_mat(
        #     phi_z_mat_masked, phase_stamps, z_stamp_opt,
        #     'masked_negative_phase_stamps')

        # write_pz_density(pz_density, 'pz_density.tif')
        # write_pz_density(pz_density_masked,
        #                 'pz_density_masked.tif')

        write_pz_density(filtered_pz_density,
                         'filtered_pz_density_unfilled.tif')
        # write_pz_density(filtered_pz_density_masked,
        #                  'filtered_pz_density_masked_unfilled.tif')

        if FILL_PHIZ:
            display_step('Fill phase rows')
            # if not os.path.exists(os.path.join(setting['workspace'],
            # 'phiz_fill_lut.txt')) or :
            if FILLING_PROMPT:
                phi_z_histogram(pz_density)
                plt.figure()
                plt.imshow(pz_density, cmap='custom_heatmap')
                plt.show()
                raw_input('Create/update row filling LUT')
            # for phi in range(phi_z_mat.shape[0]):
            #     for z in range(phi_z_mat.shape[1]):
            #         print phi_z_mat[phi, z]
            #         print filtered_phi_z[phi, z]

            phi_z_mat = fill_phi_z_row(phi_z_mat, 'phiz_fill_lut.txt')
            filtered_phi_z = fill_phi_z_row(
                filtered_phi_z, 'phiz_fill_lut.txt')

            for phi in range(phi_z_mat.shape[0]):
                for z in range(phi_z_mat.shape[1]):
                    # print phi_z_mat[phi, z]
                    # print filtered_phi_z[phi, z]
                    pz_density[phi, z] = len(phi_z_mat[phi, z])
                    filtered_pz_density[phi, z] = len(filtered_phi_z[phi, z])

            # display_step('Fill masked phase rows')
            # # if not os.path.exists(os.path.join(setting['workspace'],
            # # 'phiz_fill_lut.txt')):
            # phi_z_histogram(pz_density_masked)
            # plt.figure()
            # plt.imshow(pz_density_masked, cmap='custom_heatmap')
            # plt.show()
            # raw_input('Create/update row filling LUT - masked')

            # phi_z_mat_masked = fill_phi_z_row(
            #     phi_z_mat_masked, 'phiz_fill_lut_masked.txt')
            # filtered_phi_z_masked = fill_phi_z_row(
            #     filtered_phi_z_masked, 'phiz_fill_lut_masked.txt')

        pickle_object(filtered_phi_z, 'filtered_phi_z')
        pickle_object(filtered_pz_density, 'filtered_pz_density')
        # pickle_object(filtered_phi_z_masked, 'filtered_phi_z_masked')
        # pickle_object(filtered_pz_density_masked, 'filtered_pz_density_masked')

        if FILLING_PROMPT:
            plt.figure()
            plt.imshow(pz_density, cmap='custom_heatmap')
            plt.title('Unmasked')
            # plt.figure()
            # plt.imshow(pz_density_masked, cmap='custom_heatmap')
            # plt.title('Masked')
            plt.figure()
            plt.imshow(filtered_pz_density, cmap='custom_heatmap')
            plt.title('Filtered unmasked')
            # plt.figure()
            # plt.imshow(filtered_pz_density_masked, cmap='custom_heatmap')
            # plt.title('Filtered masked')
            plt.show()

    write_pz_density(pz_density, 'pz_density.tif')
    # write_pz_density(pz_density_masked,
    #                  'pz_density_masked.tif')

    write_pz_density(filtered_pz_density, 'filtered_pz_density.tif')
    # write_pz_density(filtered_pz_density_masked,
    #                  'filtered_pz_density_masked.tif')

    '''
    ========================================================================
    Writing final images to disk
    ========================================================================
    '''

    def plot_result_data(res_type='masked'):
        if res_type == 'unmasked':
            plot_results(
                pz_density=pz_density,
                filtered_pz_density=filtered_pz_density,
                phase_stamps=phase_stamps,
                z_stamps=z_stamp_opt,
                diff_matrix=diff_matrix,
                diff_matrix_rows=460,
                canon_hb_stats=(canon_hb_start, int(np.round(cm_hb[0]))),
                results='all'
            )
        # else:
        #     plot_results(
        #         pz_density=pz_density_masked,
        #         filtered_pz_density=filtered_pz_density_masked,
        #         phase_stamps=phase_stamps_masked,
        #         z_stamps=z_stamp_opt,
        #         diff_matrix=diff_matrix,
        #         diff_matrix_rows=460,
        #         canon_hb_stats=(canon_hb_start, int(np.round(cm_hb[0]))),
        #         results='all'
        #     )

    if PLOT_RESULTS:
        plot_result_data()

    if USE_MASKED_PHASE_STAMPS:
        final_phi_z_matrix = filtered_phi_z_masked
        final_pz_density = filtered_pz_density_masked
    else:
        final_phi_z_matrix = filtered_phi_z
        final_pz_density = filtered_pz_density

    if WRITE_FINAL_IMAGES:

        display_step('Writing phi z stamped fluorescent images - ' +
                     ('masked' if USE_MASKED_PHASE_STAMPS else 'unmasked'))

        make_or_clear_directory(setting['final_images'], clear=True)

        brightness_curve = check_fluorescent_bleaching(
            zooks, setting['fm_images'], PLOT_FLUOR_BRIGHTNESS)

        write_phase_stamped_fluorescent_images(
            final_phi_z_matrix,
            read_from=setting['fm_images'],
            write_to=setting['final_images'],
            brightness_adjust=ADJUST_FLUOR_BRIGHTNESS,
            brightness_adjust_params=('med_adj_brightness', brightness_curve)
        )

        # display_step('Writing phi z stamped fluorescent images')
        # write_phase_stamped_fluorescent_images(
        #     filtered_phi_z, read_from=setting['fm_images'],
        #     write_to=setting['final_images']
        # )

        # brother_frame_locs = [
        #     (50, 150)
        # ]

        # write_brother_frames(
        #     phi_z_mat, read_from=setting['fm_images'],
        #     write_to=setting[
        #         'brother_frames'], brother_frame_locs=brother_frame_locs
        # )

    '''
    ========================================================================
    Creating masks for fluorescent frames
    ========================================================================
    '''

    if USE_PICKLED_FLUOR_MASK_ARRAY:
        fluor_masks = unpickle_object('fluor_masks')
        fluor_ref_frames = unpickle_object('fluor_ref_frames')
        # inv_fluor_masks = unpickle_object('fluor_inv_masks')
    else:
        fluor_masks, fluor_ref_frames = create_fluorescent_masks(
            read_from=setting['final_images'],
            pz_matrix=final_phi_z_matrix,
            # from_z=105,
            # to_z=110
            read_pickled_masks=True
        )

    inv_fluor_masks = invert_mask_array(fluor_masks)

    # input()

    '''
    ========================================================================
    XY stamp fluorescent images
    ========================================================================
    '''
    x_stamps_fluor = np.zeros(final_phi_z_matrix.shape)
    y_stamps_fluor = np.zeros(final_phi_z_matrix.shape)

    setting['final_images_cropped'] = setting[
        'final_images'] + '_cropped_masked_new'
    setting['final_images_cropped_ext'] = setting[
        'final_images'] + '_cropped_masked_ext'

    if WRITE_CROPPED_FLUOR_IMAGES:
        make_or_clear_directory(setting['final_images_cropped'], clear=True)
        make_or_clear_directory(
            setting['final_images_cropped_ext'], clear=True)

        shift_and_crop_fluor_images(
            setting['final_images'],
            setting['final_images_cropped'],
            setting['final_images_cropped_ext'],
            final_phi_z_matrix,
            x_stamps_fluor,
            y_stamps_fluor,
            fluor_masks,
            inv_fluor_masks,
            adj_brightness=False
        )

    if XY_STAMP_FLUOR_IMAGES:
        if USE_PICKLED_FLUOR_STAMPS:
            x_stamps_fluor_mc = unpickle_object('x_stamps_fluor')
            # y_stamps_fluor = unpickle_object('y_stamps_fluor')
        else:
            x_stamps_fluor_mc, _ = calculate_fluorescent_image_shift(
                setting['final_images_cropped_ext'],
                final_phi_z_matrix, inv_fluor_masks,
                fluor_ref_frames
            )

        y_stamps_bf = unpickle_object(
            'y_stamp_optimal')

        y_stamps_fluor_mc = np.zeros(x_stamps_fluor_mc.shape)

        if Y_CORRECTION_FLUOR:
            phi_end, z_end = final_phi_z_matrix.shape

            for phi in range(phi_end):
                for z in range(z_end):
                    if len(final_phi_z_matrix[phi, z]) > 0:
                        curr_frame = find_best_frame(
                            final_phi_z_matrix[phi, z])
                        y_stamps_fluor_mc[phi, z] = y_stamps_bf[curr_frame] * \
                            1.816 / setting['resampling_factor']

        # plt.figure()
        # plt.plot(y_stamps_fluor_mc[:, 20])
        # plt.show()

        setting['final_images_cropped_xy_stamped'] = setting[
            'final_images'] + '_cropped_masked_new_xy_stamped_new'
        setting['final_images_cropped_ext_xy_stamped'] = setting[
            'final_images'] + '_cropped_masked_ext_xy_stamped_new'

        if WRITE_CROPPED_FLUOR_IMAGES_MC:
            make_or_clear_directory(
                setting['final_images_cropped_xy_stamped'], clear=True)
            make_or_clear_directory(
                setting['final_images_cropped_ext_xy_stamped'], clear=True)

            shift_and_crop_fluor_images(
                setting['final_images'],
                setting['final_images_cropped_xy_stamped'],
                setting['final_images_cropped_ext_xy_stamped'],
                final_phi_z_matrix,
                x_stamps_fluor_mc,
                y_stamps_fluor_mc,
                fluor_masks,
                inv_fluor_masks,
                adj_brightness=False
            )

        if TZ_STAMP_FLUOR_IMAGES:
            # x_stamps_fluor = unpickle_object('x_stamps_fluor')
            # y_stamps_fluor = unpickle_object('y_stamps_fluor')
            # final_phi_z_matrix = unpickle_object('phase_z_matrix')
            # fluor_masks = unpickle_object('fluor_masks')
            # fluor_ref_frames = unpickle_object('fluor_ref_frames')

            # inv_fluor_masks = invert_mask_array(fluor_masks)

            # if USE_PICKLED_TZ_STAMPS:
            #     x_stamps_fluor_tz = unpickle_object('x_stamps_fluor')
            #     # y_stamps_fluor_tz = unpickle_object('y_stamps_fluor')
            # else:
            x_stamps_fluor_tz, _ = calculate_relative_fluorescent_image_shift(
                setting['final_images_cropped_ext_xy_stamped'],
                final_phi_z_matrix, inv_fluor_masks,
                fluor_ref_frames, x_stamps_fluor, y_stamps_fluor
            )

            x_stamps_fluor_final, y_stamps_fluor_final = match_copied_frame_shifts(
                    x_stamps_fluor_mc+x_stamps_fluor_tz, 
                    y_stamps_fluor_mc, 
                    'phiz_fill_lut.txt'
                )

            setting['final_images_cropped_tz_stamped'] = setting[
                'final_images'] + '_cropped_masked_new_tz_stamped_new'
            setting['final_images_cropped_ext_tz_stamped'] = setting[
                'final_images'] + '_cropped_masked_ext_tz_stamped_new'

            if WRITE_CROPPED_FLUOR_IMAGES_TZ:
                make_or_clear_directory(
                    setting['final_images_cropped_tz_stamped'], clear=True)
                make_or_clear_directory(
                    setting['final_images_cropped_ext_tz_stamped'], clear=True)

                # plt.plot(x_stamps_fluor[0, :])
                # plt.plot((x_stamps_fluor+x_stamps_fluor_new)[0, :])
                # plt.figure()
                # plt.imshow(x_stamps_fluor)
                # plt.title('X - before t correction')
                # plt.figure()
                # plt.plot(y_stamps_fluor[0, :])
                # plt.plot((y_stamps_fluor+y_stamps_fluor_new)[0, :])
                # plt.figure()
                # plt.imshow(y_stamps_fluor)
                # plt.title('Y - before t correction')
                # plt.figure()
                # plt.imshow(x_stamps_fluor+x_stamps_fluor_new)
                # plt.title('X - after t correction')
                # plt.figure()
                # plt.imshow(y_stamps_fluor+y_stamps_fluor_new)
                # plt.title('Y - after t correction')
                # plt.show()
                # code.interact(local=locals())
                shift_and_crop_fluor_images(
                    setting['final_images'],
                    setting['final_images_cropped_tz_stamped'],
                    setting['final_images_cropped_ext_tz_stamped'],
                    final_phi_z_matrix,
                    x_stamps_fluor_final,
                    y_stamps_fluor_final,
                    fluor_masks,
                    inv_fluor_masks,
                    adj_brightness=False
                )

    if WRITE_BROTHER_FRAMES:
        plt.figure()
        plt.imshow(final_pz_density, cmap='custom_heatmap')
        plt.title('Phi Z matrix')
        plt.colorbar()
        plt.show()

        raw_input('Create brother frame file')

        make_or_clear_directory(setting['brother_frames'], clear=True)

        write_brother_frames(
            final_phi_z_matrix,
            setting['fm_images'],
            setting['cropped_bf_images'],
            setting['brother_frames'],
            'brother_frames.txt'
        )

    # moving_dwt_array = load_dwt_matrix(
    #     setting['bf_images_dwt'],
    #     # num_frames=500
    # )

    # canon_hb_dwt = load_dwt_matrix(
    #     setting['canon_frames']
    # )

    # mask_array = unpickle_object('mask_array')

    # NUM_FRAMES = 500

    # if not USE_PICKLED_CANON_CORRELATION:
    #     for start in [0, moving_dwt_array.shape[2] // 2, moving_dwt_array.shape[2] - (NUM_FRAMES * 2)]:
    #         canon_corr_mat = canon_correlation(
    #             moving_dwt_array[:, :, start:start + NUM_FRAMES],
    #             canon_hb_dwt,
    #             # mask_array
    #         )

    #         canon_corr_mat_masked = canon_correlation(
    #             moving_dwt_array[:, :, start:start + NUM_FRAMES],
    #             canon_hb_dwt,
    #             mask_array
    #         )

    #         pickle_object(canon_corr_mat, 'canon_corr_%d' % start)
    #         pickle_object(canon_corr_mat_masked,
    #                       'canon_corr_masked_%d' % start)

    # if PLOT_CANON_CORRELATION:
    #     for start in [0, moving_dwt_array.shape[2] // 2, moving_dwt_array.shape[2] - (NUM_FRAMES * 2)]:
    #         canon_corr_mat = unpickle_object('canon_corr_%d' % start)
    #         canon_corr_mat_masked = unpickle_object(
    #             'canon_corr_masked_%d' % start)

    #         plt.figure()
    #         plt.imshow(1. - canon_corr_mat.T, cmap='jet')
    #         plt.title('Canonical correlation - unmasked (%d)' % start)
    #         plt.colorbar()
    #         # plt.get_current_fig_manager().full_screen_toggle()
    #         plt.savefig(os.path.join(setting[
    #                     'workspace'], 'Canonical correlation - unmasked (%d).svg' % start), format='svg', dpi=600)

    #         plot_matrix_rows(
    #             1. - canon_corr_mat,
    #             'Canonical correlation - unmasked (%d)' % start,
    #             col_sep=8,
    #             start_col=0,
    #             num_cols=canon_hb_dwt.shape[2],
    #             row_length=100
    #         )

    #         plt.figure()
    #         plt.imshow(1. - canon_corr_mat_masked.T, cmap='jet')
    #         plt.title('Canonical correlation - masked (%d)' % start)
    #         plt.colorbar()
    #         # plt.get_current_fig_manager().full_screen_toggle()
    #         plt.savefig(os.path.join(setting[
    #                     'workspace'], 'Canonical correlation - masked (%d).svg' % start), format='svg', dpi=600)

    #         plot_matrix_rows(
    #             1. - canon_corr_mat,
    #             'Canonical correlation - masked (%d)' % start,
    #             col_sep=8,
    #             start_col=0,
    #             num_cols=canon_hb_dwt.shape[2],
    #             row_length=100
    #         )

    #     plt.show()
        # plt.close('all')

    code.interact(local=locals())

except KeyboardInterrupt:
    display_step('DEBUG CONSOLE: Code execution stopped by user')
    code.interact(local=locals())
except:
    display_step('DEBUG CONSOLE: Write better code')
    import traceback
    traceback.print_exc()
    code.interact(local=locals())


with open(os.path.join(settings.JOBS_DIR, curr_job, 'job_metadata.txt'), 'a') as metafile:
    data = '''
Last run time: {runtime}
Steps performed:
{steps_performed}
        '''.format(
        runtime=time.asctime(),
        steps_performed='\n-'.join(settings.steps_performed)[:-1],
    )
    metafile.write(data)
