from __future__ import division

import numpy as np
import numpy.random as npr
from scipy.signal import fftconvolve, convolve

import scipy
from scipy import stats


print_T = True


def richardson_lucy(image, psf, iterations=50, clip=True):
    """Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = image.mean() * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for m in range(iterations):
        print('Iteration ', m)
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur,
                                     psf_mirror, 'same') / psf.sum()

    if clip:
        im_deconv[np.isnan(im_deconv)] = 0
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv, psf


def damped_richardson_lucy(image, psf, iterations=50, clip=True, N=3, T=None, multiplier=1):
    """Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = image.mean() * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for m in range(iterations):
        # print('Iteration ', m)
        relative_blur = dampen(convolve_method(
            im_deconv, psf, 'same'), image, N, T, multiplier)
        im_deconv *= (convolve_method(relative_blur,
                                      psf_mirror, 'same') / psf.sum())

    if clip:
        im_deconv[np.isnan(im_deconv)] = 0
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv, psf


def sample_noise(spectrum):
    samples = [spectrum[start:start + 300].std()
               for start in range(0, len(spectrum), 300)]

    return np.median(samples)


def modified_likelihood(I, D, T=None, multiplier=1):
    """ Equation 7 
    http://spider.ipac.caltech.edu/staff/fmasci/home/astro_refs/DampledLR94.pdf 
    """
    # assert np.all(I > 0), 'Negative values'
    global print_T
    T = T or multiplier * sample_noise(D)

    if print_T:
        # print('Using T=', T)
        print_T = False

    const_mult = (-2.0 / T**2)
    ratio = I / D
    ratio[np.isnan(ratio)] = 0

    log_ratio = np.log(ratio)
    log_ratio[np.isnan(log_ratio)] = 0

    likelihood = (D * log_ratio - I + D)

    u = const_mult * likelihood

    #u = (-2.0 / T**2) * (D * np.log( I / D) - lucy + D)
    u[np.isnan(u)] = 0
    # print 'Factor_pre', np.median(u[u > 0]), u[u >
    # 0].min()
    u_cap = np.where(u > 1, 1, u)
    # print 'Factor=', np.median([U > 0]), U[U > 0].min()
    return u_cap


def dampen(I, D, N=3, T=None, multiplier=1):

    first_term = modified_likelihood(I, D, T, multiplier)**(N - 1)
    first_term[np.isnan(first_term)] = 0
    # print first_term.mean()

    second_term = (N - (N - 1) * modified_likelihood(I, D, T, multiplier))
    second_term[np.isnan(second_term)] = 0
    # print second_term.mean()

    third_term = (D - I) / I
    third_term[np.isnan(third_term)] = 0
    # print third_term.mean()

    return 1 + first_term * second_term * third_term


def blind_richardson_lucy(image, psf, iterations=50, psf_iterations=50, clip=True):
    """Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)

    direct_time = np.prod(image.shape + psf.shape)
    fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = image.mean() * np.ones(image.shape)
    psf_estimate = psf.copy()

    psf_range = ((np.array(image.shape) -
                  np.array(psf_estimate.shape)) / 2).astype(int)

    for m in range(iterations):
        # print('Iter', m)
        relative_blur = image / \
            convolve_method(im_deconv, psf_estimate[::-1, ::-1], 'same')

        im_deconv *= convolve_method(relative_blur,
                                     psf_estimate, 'same') / psf_estimate.sum()

        # print('IMblur:', relative_blur.shape)
        # print('IM', im_deconv.shape)
        # print('PSF', psf_estimate.shape)
        # print('PSFblur', psf_blur.shape)

        for n in range(psf_iterations):
            # print('PSFIter', n)
            psf_blur = image / \
                convolve_method(im_deconv[::-1, ::-1], psf_estimate, 'same')
            psf_estimate *= (convolve_method(psf_blur,
                                             im_deconv, 'same') / im_deconv.sum())[psf_range[0]:-psf_range[0], psf_range[1]:-psf_range[1]]

            if clip:
                im_deconv[np.isnan(im_deconv)] = 0
                im_deconv[im_deconv > 1] = 1
                im_deconv[im_deconv < -1] = -1
                psf_estimate[np.isnan(psf_estimate)] = 0
                psf_estimate[psf_estimate > 1] = 1
                psf_estimate[psf_estimate < -1] = -1

    return im_deconv, psf_estimate
