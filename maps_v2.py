from maps.core.z_stamping_v2 import compute_raw_zstage, compute_zooks,\
    compute_zookzik_stats, drop_bad_zooks_zstage, compute_zstamp,\
    compute_deterministic_zstamp, compute_optimal_yz_stamp,\
    compute_zstamp_curvefit, detect_bad_zooks, shift_frames_and_store_yz,\
    Zook, Zooks
# from maps.core.z_stamping import z_stamping_step_yz, shift_frames_and_store_yz
from maps.core.phase_stamping_v2 import crop_and_compute_dwt_matrix,\
    compute_dwt_matrix, load_dwt_matrix, interchelate_dwt_array,\
    compute_heartbeat_length, compute_canonical_heartbeat,\
    phase_stamp_images, compile_phase_z_matrix,\
    write_phase_stamped_fluorescent_images

from maps import settings
from maps.settings import setting, read_setting_from_json, \
    display_current_settings, JOBS_DIR, display_step
from maps.helpers.misc import pickle_object, unpickle_object, zook_approval_function
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
# import argparse

logging.config.dictConfig(logger_config)

job_list = next(os.walk('./jobs/'))[1]

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
read_setting_from_json(curr_job)

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
    ref_frame_no = 21

try:
    display_current_settings()

    '''
    ========================================================================
    Moving frame z stamping
    ========================================================================
    '''

    if USE_GUI_CORR_WINDOW:
        # Uses GUI to select rectangular window in specified frame.
        # Click and drag with mouse tto select region. Clos window to finalize
        display_step('Selecting correlation window - moving frames')
        img_path = glob.glob(
            os.path.join(setting['bf_images'], '*.tif')
        )
        img_seq = load_image_sequence(
            [img_path[setting['index_start_at'] + ref_frame_no]])
        max_heartsize_frame(img_seq[0])

        params = get_rect_params()
    else:
        try:
            params = dict(unpickle_object(
                file_name='corr_window.csv', dumptype='csv'))
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

    pickle_object(data, file_name='corr_window.csv', dumptype='csv')

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

        z_stage = compute_raw_zstage(
            kymo_path,
            median_filt=True,
            kernel_size=setting.get('median_kernel', 5)
        )

        if DEBUG:
            plt.figure()
            plt.plot(z_stage)
            plt.show(block=PLOT_BLOCK)
        # assert setting['frame_count'] < z_stage_len

        zooks = compute_zooks(
            z_stage[:setting['frame_count']]
        )

        zz_stats = compute_zookzik_stats(
            zooks
        )

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

        if DEBUG:
            plt.figure()
            plt.plot(z_stamp)
            plt.show(block=PLOT_BLOCK)

        z_stamp_det = compute_deterministic_zstamp(
            z_stamp,
            zooks,
            ref_frame_no
        )

        if DEBUG:
            plt.figure()
            plt.plot(z_stamp_det)
            plt.show(block=PLOT_BLOCK)

        z_stamp_opt, y_stamp_opt = compute_optimal_yz_stamp(
            setting['bf_images'],
            zooks,
            z_stamp_det,
            # z_stamp,
            offset=setting['index_start_at'],
            prefix=setting['image_prefix']
        )

        if DEBUG:
            plt.plot(z_stamp_opt)
            plt.figure()
            plt.plot(y_stamp_opt)
            plt.show(block=PLOT_BLOCK)

    if USE_GUI_CROP_WINDOW:
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

    # raise Exception
    # print zooks

    if WRITE_CROPPED_IMAGES_TO_DISK:
        display_step('Writing cropped moving frames')
        shift_frames_and_store_yz(
            setting['bf_images'],
            setting['image_prefix'],
            setting['index_start_at'],
            z_stamp_opt,
            # # Use y_stamp_opt to get Y stamping, otherwise give zeros
            y_stamp_opt,
            # np.zeros(y_stamp_opt.shape),
            zooks,
            setting['cropped_bf_images'],
            adj_intensity=ADJUST_BRIGHTNESS
        )

    '''
    ========================================================================
    Stationary frame z stamping
    ========================================================================
    '''

    if USE_GUI_CORR_WINDOW:
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

    pickle_object(data, file_name='corr_window_stat.csv', dumptype='csv')

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

        z_stamp_stat, _ = compute_zstamp(
            z_stage_stat[:setting['stat_frame_count']],
            zooks_stat,
            ref_frame_no,
            z_stamp_pkl_name='z_stamp_stat'
        )

        z_stamp_det_stat = compute_deterministic_zstamp(
            z_stamp_stat,
            zooks_stat,
            ref_frame_no,
            z_stamp_pkl_name='z_stamp_det_stat'
        )

        z_stamp_opt_stat, y_stamp_opt_stat = compute_optimal_yz_stamp(
            setting['bf_images'],
            zooks_stat,
            z_stamp_det_stat,
            offset=setting['stat_index_start_at'],
            prefix=setting['image_prefix'], corr_window_csv='corr_window_stat.csv', z_stamp_pkl='z_stamp_optimal_stat', y_stamp_pkl='y_stamp_optimal_stat'
        )

        print z_stamp_opt_stat

    if USE_GUI_CROP_WINDOW:
        display_step('Selecting cropping window - stationary frames')

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

    if WRITE_CROPPED_STAT_IMAGES_TO_DISK:
        display_step('Writing cropped stationary frames')
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

    raw_input('Copy stationary images')

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
                # num_frames=1000
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
            moving_dwt_array = compute_dwt_matrix(
                img_path=setting['cropped_bf_images'],
                frame_indices=zooks.get_framelist(),
                write_to=setting['bf_images_dwt'],
                validTiff=False
            )

            display_step('Computing DWT from cropped stationary images')
            static_dwt_array = compute_dwt_matrix(
                img_path=setting['stat_images_cropped'],
                # frame_indices=zooks_stat.get_framelist(),
                frame_indices=zooks.get_framelist()[:3773],
                write_to=None,
                validTiff=False
            )

            display_step('Upsampling canonical frames')
            canonical_dwt_array = interchelate_dwt_array(
                moving_dwt_array[:, :, : setting['canon_frac']
                                 * moving_dwt_array.shape[2] / 100],
                zooks,
                write_to=setting['bf_images_dwt_upsampled'],
                validTiff=False
            )

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

        print 'Canonical hearbeat length: ', np.round(cm_hb)

        if USE_PICKLED_CANON_DWT:
            display_step('Loading canonical DWT array')
            canon_hb_dwt = load_dwt_matrix(
                setting['canon_frames']
            )
        else:
            display_step('Computing canonical heartbeat')
            canon_hb_dwt = compute_canonical_heartbeat(
                canonical_dwt_array, zooks,
                int(np.round(cm_hb[0])), sad_matrix_pkl_name='sad_matrix.pkl',
                write_to=setting['canon_frames'],
                use_pickled_sad=USE_PICKLED_SAD
            )

        '''
        ========================================================================
        Computing Phase stamps
        ========================================================================
        '''

        display_step('Computing phase stamps')
        phase_stamps = phase_stamp_images(
            moving_dwt_array, canon_hb_dwt
        )
    else:
        display_step('Loading hearbeat length')
        cm_hb = unpickle_object('common_heatbeat_period.pkl')
        display_step('Loading phase stamps')
        phase_stamps = unpickle_object('phase_stamps.pkl')

    # canonical_heartbeat = processed_frame_list[]

    '''
    ========================================================================
    Computing Phi Z Matrix
    ========================================================================
    '''

    if USE_PICKLED_PHI_Z_MATRIX:
        z_stamp_ds = unpickle_object('z_stamp_downsampled')
        phi_z_mat = unpickle_object('phase_z_matrix')
        pz_density = unpickle_object('phase_z_density')
    else:
        display_step('Downsampling z stamps')
        # z_stamp_ds = np.round(z_stamp_opt)
        # plt.plot(z_stamp_opt)
        # plt.show()
        z_stamp_ds = np.round(z_stamp_opt / setting['resampling_factor'])
        z_stamp_ds = z_stamp_ds + np.abs(np.nanmin(z_stamp_ds))
        # plt.plot(z_stamp_ds)
        # plt.show()

        pickle_object(z_stamp_ds, 'z_stamp_downsampled')

        display_step('Computing phase z matrix')
        phi_z_mat, pz_density = compile_phase_z_matrix(
            zooks.get_framelist(), z_stamp_ds,
            phase_stamps, int(np.round(cm_hb))
        )

    '''
    ========================================================================
    Writing final images to disk
    ========================================================================
    '''

    if WRITE_FINAL_IMAGES:
        display_step('Writing phi z stamped fluorescent images')
        # write_phase_stamped_fluorescent_images(
        #     phi_z_mat, read_from=setting['fm_images'],
        #     write_to=setting['final_images']
        # )

    pz_zeros = np.zeros(pz_density.shape)
    pz_zeros[np.where(pz_density > 0)] = 1

    plt.figure()
    plt.imshow(pz_density, cmap='custom_heatmap')
    plt.colorbar()
    # plt.figure()
    # plt.imshow(pz_zeros, cmap='gray')
    # plt.colorbar()
    plt.show()

    code.interact(local=locals())

except:
    display_step('DEBUG CONSOLE: Write better code')
    import traceback
    traceback.print_exc()
    code.interact(local=locals())
