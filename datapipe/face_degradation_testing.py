#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-08-19 10:00:39

import cv2
import numpy as np
from basicsr.data import degradations as degradations

from utils import util_common

def face_degradation(im, sf, sig_x, sig_y, theta, nf, qf):
    '''
    Face degradation on testing data
    Input:
        im: numpy array, h x w x c, [0, 1], bgr
        sf: scale factor for super-resolution
        sig_x, sig_y, theta: parameters for generating gaussian kernel
        nf: noise level
        qf: quality factor for jpeg compression
    Output:
        im_lq: numpy array, h x w x c, [0, 1], bgr
    '''
    h, w = im.shape[:2]

    # blur
    kernel = degradations.bivariate_Gaussian(
            kernel_size=41,
            sig_x=sig_x,
            sig_y=sig_y,
            theta=theta,
            isotropic=False,
            )
    im_lq = cv2.filter2D(im, -1, kernel)

    # downsample
    im_lq = cv2.resize(im_lq, (int(w // sf), int(h // sf)), interpolation=cv2.INTER_LINEAR)

    # noise
    im_lq = degradations.add_gaussian_noise(im_lq, sigma=nf, clip=True, rounds=False)

    # jpeg compression
    im_lq = degradations.add_jpg_compression(im_lq, quality=qf)


    # resize to original size
    im_lq = cv2.resize(im_lq, (w, h), interpolation=cv2.INTER_LINEAR)

    # round and clip
    im_lq = np.clip((im_lq * 255.0).round(), 0, 255) / 255.

    return im_lq
