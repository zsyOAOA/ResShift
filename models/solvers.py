#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-06-09 14:59:55

import torch
import random
import numpy as np
from einops import rearrange

def batch_inpainging_from_grad(im_in, mask, gradx, grady):
    '''
    Recovering from gradient for batch data (torch tensro).
    Input:
        im_in: N x c x h x w, torch tensor, masked image
        mask: N x 1 x  h x w, torch tensor
        gradx, grady: N x c x h x w, torch tensor, image gradient
    '''
    im_out = torch.zeros_like(im_in.data)
    for ii in range(im_in.shape[0]):
        im_current, gradx_current, grady_current = [rearrange(x[ii,].cpu().numpy(), 'c h w -> h w c')
                                                    for x in [im_in, gradx, grady]]
        mask_current = mask[ii, 0,].cpu().numpy()
        out_current = inpainting_from_grad(im_current, mask_current, gradx_current, grady_current)
        im_out[ii,] = torch.from_numpy(rearrange(out_current, 'h w c -> c h w')).to(
                device=im_in.device,
                dtype=im_in.dtype
                )
    return im_out

def inpainting_from_grad(im_in, mask, gradx, grady):
    '''
    Input:
        im_in: h x w x c, masked image, numpy array
        mask: h x w, image mask, 1 represents missing value
        gradx: h x w x c, gradient along x-axis, numpy array
        grady: h x w x c, gradient along y-axis, numpy array
    Output:
        im_out: recoverd image
    '''
    h, w = im_in.shape[:2]
    counts_h = np.sum(1-mask, axis=0, keepdims=False)
    counts_w = np.sum(1-mask, axis=1, keepdims=False)
    if np.any(counts_h[1:-1,] == h):
        idx = find_first_index(counts_h[1:-1,], h) + 1
        im_out = fill_image_from_gradx(im_in, mask, gradx, idx)
    elif np.any(counts_w[1:-1,] == w):
        idx = find_first_index(counts_w[1:-1,], w) + 1
        im_out = inpainting_from_grad(im_in.T, mask.T, gradx.T, idx)
    else:
        idx = random.choices(list(range(1,w-1)), k=1, weights=counts_h[1:-1])[0]
        line = fill_line(im_in[:, idx, ], mask[:, idx,], grady[:, idx,])
        im_in[:, idx,] = line
        im_out = fill_image_from_gradx(im_in, mask, gradx, idx)
    if im_in.ndim > mask.ndim:
        mask = mask[:, :, None]
    im_out = im_in + im_out * mask
    return im_out

def fill_image_from_gradx(im_in, mask, gradx, idx):
    init = np.zeros_like(im_in)
    init[:, idx,] = im_in[:, idx,]
    right = np.cumsum(init[:, idx:-1, ] + gradx[:, idx+1:, ], axis=1)
    left = np.cumsum(
        init[:, idx:0:-1, ] - gradx[:, idx:0:-1, ],
        axis=1
        )[:, ::-1]
    center = im_in[:, idx, ][:, None]     # h x 1 x 3
    im_out = np.concatenate((left, center, right), axis=1)
    return im_out

def fill_line(xx, mm, grad):
    '''
    Fill one line from grad.
    Input:
        xx: n x c array, masked vector
        mm: (n,) array, mask, 1 represent missing value
        grad: (n,) array
    '''
    n = xx.shape[0]
    assert mm.sum() < n
    if mm.sum() == 0:
        return xx
    else:
        idx1 = find_first_index(mm, 1)
        if idx1 == 0:
            idx2 = find_first_index(mm, 0)
            subx = xx[idx2::-1,].copy()
            subgrad = grad[idx2::-1, ].copy()
            subx -= subgrad
            xx[:idx2,] = np.cumsum(subx, axis=0)[idx2-1::-1,]
            mm[idx1:idx2,] = 0
        else:
            idx2 = find_first_index(mm[idx1:,], 0) + idx1
            subx = xx[idx1-1:idx2-1,].copy()
            subgrad = grad[idx1:idx2,].copy()
            subx += subgrad
            xx[idx1:idx2,] = np.cumsum(subx, axis=0)
            mm[idx1:idx2,] = 0
        return fill_line(xx, mm, grad)

def find_first_index(mm, value):
    '''
    Input:
        mm: (n, ) array
        value: scalar
    '''
    try:
        out = next((idx for idx, val in np.ndenumerate(mm) if val == value))[0]
    except StopIteration:
        out = mm.shape[0]
    return out

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils import util_image
    from datapipe.masks.train import process_mask

    # mask_file_names = [x for x in Path('../lama/LaMa_test_images').glob('*mask*.png')]
    mask_file_names = [x for x in Path('./testdata/inpainting/val/places/').glob('*mask*.png')]
    file_names = [x.parents[0]/(x.stem.rsplit('_mask',1)[0]+'.png') for x in  mask_file_names]

    for im_path, mask_path in zip(file_names, mask_file_names):
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        mask = process_mask(util_image.imread(mask_path, chn='rgb', dtype='float32')[:, :, 0])
        grad_dict = util_image.imgrad(im)

        im_masked = im * (1 - mask[:, :, None])
        im_recover = inpainting_from_grad(im_masked, mask, grad_dict['gradx'], grad_dict['grady'])
        error_max = np.abs(im_recover -im).max()
        print('Error Max: {:.2e}'.format(error_max))
