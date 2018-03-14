# from __future__ import print_function

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
    canon_correlation, write_brother_frames, calculate_relative_fluorescent_image_shift, get_fluorescent_filename


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

try:

    import numpy as np

    from scipy.signal import convolve2d as conv2d
    from skimage import restoration
    from skimage.transform import rescale
    from skimage.external import tifffile as tff

    from astropy.convolution.kernels import AiryDisk2DKernel, Gaussian2DKernel, Tophat2DKernel

    from maps.helpers.deconv_methods import richardson_lucy, damped_richardson_lucy

    import matplotlib.pyplot as plt

    import time
    import glob
    import os
    import shutil

    deconv_method = richardson_lucy

    def read_image(img_path):
        im = tff.imread(img_path)
        # im = im.astype('float64') + 1e-6
        # im /= im.max()
        return im

    def normalize_image(img):
        ret_img = img.astype('float64')
        ret_img = (ret_img - np.min(ret_img))/(np.max(ret_img)-np.min(ret_img))
        return ret_img

    def compute_error(recons_image, orig_image):
        mse = np.sum((recons_image - orig_image)**2)**0.5
        sad = np.sum(np.abs(recons_image - orig_image))

        print('MSE:', mse)
        print('SAD:', sad)

        return mse, sad


    method = 'rich-lucy'

    # Setting deconvolution method
    if method == 'rich-lucy':
        deconv_method = richardson_lucy
        deconv_params = {
            'iterations': 50
        }
    elif method == 'damp-rich-lucy':
        deconv_method = damped_richardson_lucy
        deconv_params = {
            'iterations': 100,
            'N': 3,
            # 'T': 0.8,
            'multiplier': 1
        }

    # Generating psf
    kernel_size = 31
    psf = {}
    psf_param = {}

    box_size = 7
    box_psf = np.zeros((kernel_size, kernel_size)) / box_size**2
    box_psf[int((kernel_size-box_size)/2):-int((kernel_size-box_size)/2),
            int((kernel_size-box_size)/2):-int((kernel_size-box_size)/2)] = 1
    psf['box'] = box_psf
    psf_param['box'] = box_size


    std_dev = kernel_size // 8
    gauss_psf = Gaussian2DKernel(std_dev, x_size=kernel_size,
                                y_size=kernel_size).array
    psf['gauss'] = gauss_psf
    psf_param['gauss'] = std_dev

    tophat_radius = 6
    tophat_psf = Tophat2DKernel(tophat_radius, x_size=kernel_size,
                                y_size=kernel_size).array
    psf['tophat'] = tophat_psf
    psf_param['tophat'] = tophat_radius

    radius = kernel_size // 4
    airy_psf = AiryDisk2DKernel(radius, x_size=kernel_size,
                                y_size=kernel_size).array
    psf['airy'] = airy_psf
    psf_param['airy'] = radius

    psf_name = 'tophat'
    interpolation_psf = psf[psf_name]
    # interpolation_psf = gauss_psf


    # plt.subplot(221)
    # plt.imshow(box_psf, cmap='jet')
    # plt.subplot(222)
    # plt.imshow(gauss_psf, cmap='jet')
    # plt.subplot(223)
    # plt.imshow(airy_psf, cmap='jet')
    # plt.subplot(235)
    # plt.imshow(interpolation_psf, cmap='gray')
    # plt.title('PSF')
    # plt.show()


    resampling_factor = 4

    # psf_path = os.path.join(
    #     image_path, set_name + '_interpolation_results_%d_%d_' % (kernel_size, resampling_factor))

    # result_path = os.path.join(psf_path, method)

    # make_or_clear_directory(result_path, clear=True)

    img_list = glob.glob(os.path.join(setting['final_images'], '*.tif'))

    img_index = 140

    # for j in range(10):
    #     mse_grad = np.zeros(10)
    #     sad_grad = np.zeros(10)

    #     for i in range(10):
    #         im1 = read_image(img_list[img_index+j])
    #         im2 = read_image(img_list[img_index+i])
    #         print im1.shape
    #         print im2.shape
    #         mse_grad[i], sad_grad[i] = compute_error(
    #             normalize_image(im1),
    #             normalize_image(im2)
    #         )
    #     mse_grad[np.where(mse_grad>25)] = np.nan

    #     plt.figure(1)
    #     plt.plot(mse_grad)
    #     plt.figure(2)
    #     plt.plot(sad_grad)
    
    # plt.figure(1)
    # plt.title('MSE')
    # plt.figure(2)
    # plt.title('SAD')
    # plt.show()
    
    print 'loading ', img_list[img_index]
    print 'loading ', img_list[img_index + 1]
    print 'loading ', img_list[img_index + 2]
    pre_image = read_image(img_list[img_index]).astype('float64')
    post_image = read_image(img_list[img_index + 2]).astype('float64')

    pre_image = normalize_image(pre_image)
    post_image = normalize_image(post_image)
    # pre_image = tff.imread(img_list[img_index]).astype('float64')
    # post_image = tff.imread(img_list[img_index + 2]).astype('float64')

    missing_image = read_image(img_list[img_index + 1]).astype('float64')
    missing_image = normalize_image(missing_image)
    # missing_image = tff.imread(img_list[img_index + 1]).astype('float64')

    us_pre_image = rescale(pre_image, resampling_factor, order=5)
    us_post_image = rescale(post_image, resampling_factor, order=5)
    # us_pre_image = pre_image.repeat(
    #     resampling_factor, axis=0).repeat(resampling_factor, axis=1)
    # us_post_image = pre_image.repeat(
    #     resampling_factor, axis=0).repeat(resampling_factor, axis=1)

    interpolated_image = (us_pre_image + us_post_image) / 2.

    ds_interpolated_image = normalize_image( rescale(
        interpolated_image, 1.0/resampling_factor, order=5))

    diff_image = ds_interpolated_image - missing_image

    mse1, sad1 = compute_error(ds_interpolated_image, missing_image)

    recons_image, final_psf = deconv_method(
        interpolated_image, interpolation_psf, **deconv_params)

    ds_recons_image = normalize_image(rescale(recons_image, 1.0/resampling_factor, order=5))
    # ds_recons_image = recons_image[
    #     :: resampling_factor, :: resampling_factor
    # ]

    # recons_missing_frame = deconv_method(missing_image, airy_psf, **deconv_params)

    diff_image2 = ds_recons_image - missing_image

    mse2, sad2 = compute_error(ds_recons_image, missing_image)

    with open('test_results.txt', 'a') as f:
        f.write('\n')
        f.write(
            'PSF:%s,\nResampling:%d,\ndeconv_method:%s,\ndeconv_params:%s\ndeconv_iters:%d' % (
                psf_name, resampling_factor, method, psf_param[psf_name], deconv_params['iterations']
            )
        )
        f.write('\n')
        f.write(
            'Average - MSE:%.2f,SAD:%.2f\nDeconv - MSE:%.2f,SAD:%.2f' % (
                mse1, sad1, mse2, sad2
                )
            )
        f.write('\n')
        f.write('-------------------------------------')

    plt.figure()
    plt.subplot(331)
    plt.title('Frame n')
    plt.imshow(us_pre_image, cmap='gray')
    plt.subplot(332)
    plt.title('Frame n+2')
    plt.imshow(us_post_image, cmap='gray')
    plt.subplot(333)
    plt.title('Original missing frame')
    plt.imshow(missing_image, cmap='gray')
    plt.subplot(334)
    plt.title('Average of frames')
    plt.imshow(interpolated_image, cmap='gray')
    plt.subplot(335)
    plt.imshow(recons_image, cmap='gray')
    plt.title('Deconvolved image')
    plt.subplot(336)
    plt.imshow(interpolation_psf, cmap='gray')
    plt.title('Interpolation PSF')
    plt.subplot(337)
    plt.imshow(diff_image, cmap='gray')
    plt.title('Difference - Mean interpolation')
    plt.subplot(338)
    plt.imshow(diff_image2, cmap='gray')
    plt.title('Difference - Deconvolution')
    plt.subplot(339)
    plt.axis('off')
    plt.text(0, 0, 'Average:\nMSE:%.2f\nSAD:%.2f\nDeconv:\nMSE:%.2f\nSAD:%.2f' % (mse1, sad1, mse2, sad2))

    plt.tight_layout(
        # pad=0.4, w_pad=0.5, h_pad=1.0
        )
    plt.show()

    # code.interact(local=locals())


except:
    import traceback
    traceback.print_exc()
    import code
    code.interact(local=locals())
