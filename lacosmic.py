import numpy as np
import time
from PIL import Image
import os
from astropy import log
from astropy.nddata.utils import block_reduce, block_replicate
from scipy import ndimage


'''
    Source: (Modified from) lacosmic library, Astropy

    Returns
    -------
    cr_image : numpy array
               The cosmic ray mask for the input image (0-background pixel, 255-cosmic ray pixel)
    ncosmics_tot : int
                Total number of cosmic ray pixels in the image
    '''

def lacosmic(data, contrast, cr_threshold, neighbor_threshold,
             error=None, mask=None, background=None, effective_gain=None,
             readnoise=None, maxiter=4, border_mode='mirror'):

    block_size = 2.0
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

    clean_data = data.copy()
    if background is not None:
        clean_data += background
    final_crmask = np.zeros(data.shape, dtype=bool)

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('error and data must have the same shape')
    clean_error_image = error

    ncosmics, ncosmics_tot = 0, 0
    for iteration in range(maxiter):
        sampled_img = block_replicate(clean_data, block_size)
        convolved_img = ndimage.convolve(sampled_img, kernel,
                                         mode=border_mode).clip(min=0.0)
        laplacian_img = block_reduce(convolved_img, block_size)

        if clean_error_image is None:
            if effective_gain is None or readnoise is None:
                raise ValueError('effective_gain and readnoise must be '
                                 'input if error is not input')
            med5_img = ndimage.median_filter(clean_data, size=5,
                                             mode=border_mode).clip(min=1.e-5)
            error_image = (np.sqrt(effective_gain*med5_img + readnoise**2) /
                           effective_gain)
        else:
            error_image = clean_error_image

        snr_img = laplacian_img / (block_size * error_image)
        # this is used to remove extended structures (larger than ~5x5)
        snr_img -= ndimage.median_filter(snr_img, size=5, mode=border_mode)

        # used to remove compact bright objects
        med3_img = ndimage.median_filter(clean_data, size=3, mode=border_mode)
        med7_img = ndimage.median_filter(med3_img, size=7, mode=border_mode)
        finestruct_img = ((med3_img - med7_img) / error_image).clip(min=0.01)

        cr_mask1 = snr_img > cr_threshold
        # NOTE: to follow the paper exactly, this condition should be
        # "> contrast * block_size".  "lacos_im.cl" uses simply "> contrast"
        cr_mask2 = (snr_img / finestruct_img) > contrast
        cr_mask = cr_mask1 * cr_mask2
        if mask is not None:
            cr_mask = np.logical_and(cr_mask, ~mask)

        # grow cosmic rays by one pixel and check in snr_img
        selem = np.ones((3, 3))
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = cr_mask1 * neigh_mask
        # now grow one more pixel and lower the detection threshold
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = (snr_img > neighbor_threshold) * neigh_mask

        # previously unknown cosmic rays found in this iteration
        crmask_new = np.logical_and(~final_crmask, cr_mask)
        ncosmics = np.count_nonzero(crmask_new)

        final_crmask = np.logical_or(final_crmask, cr_mask)
        ncosmics_tot += ncosmics
        if (ncosmics_tot>0):
          log.info('Iteration {0}: Found {1} cosmic-ray pixels, '
                  'Total: {2}'.format(iteration + 1, ncosmics, ncosmics_tot))
    cr_image = 1*final_crmask*255
    return cr_image,ncosmics_tot