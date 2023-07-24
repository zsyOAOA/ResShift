#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-10 11:15:36

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
import torch
import random
import argparse
import numpy as np
from omegaconf import OmegaConf

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from utils import util_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-i",
            "--indir",
            type=str,
            default="/mnt/lustre/share/zhangwenwei/data/imagenet/val",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "-o",
            "--outdir",
            type=str,
            default="./ImageNet-Test",
            help="Folder to save the checkpoints and training log",
            )
    args = parser.parse_args()

    img_list = sorted([x for x in Path(args.indir).glob('**/*.JPEG')])
    print(f'Number of images in imagenet validation dataset: {len(img_list)}')

    random.seed(10000)
    random.shuffle(img_list)

    gt_dir = Path(args.outdir) / 'gt'
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True)
    lq_dir = Path(args.outdir) / 'lq'
    if not lq_dir.exists():
        lq_dir.mkdir(parents=True)

    num_imgs = 3000
    configs = OmegaConf.load('./configs/degradation_testing_realesrgan.yaml')
    opts, opts_degradation = configs.opts, configs.degradation
    opts['dir_paths'] = [args.indir, ]
    opts['length'] = num_imgs
    dataset = RealESRGANDataset(opts, mode='testing')
    for ii in range(num_imgs):
        data_dict1 = dataset.__getitem__(ii)
        if (ii + 1) % 100 == 0:
            print(f'Processing: {ii+1}/{num_imgs}')
        prefix = 'realesrgan'
        data_dict2 = dataset.degrade_fun(
                opts_degradation,
                im_gt=data_dict1['gt'].unsqueeze(0),
                kernel1=data_dict1['kernel1'],
                kernel2=data_dict1['kernel2'],
                sinc_kernel=data_dict1['sinc_kernel'],
                )
        im_lq, im_gt = data_dict2['lq'], data_dict2['gt']
        im_lq, im_gt = util_image.tensor2img([im_lq, im_gt], rgb2bgr=True, min_max=(0,1) ) # uint8

        im_name = Path(data_dict1['gt_path']).stem
        im_path_gt = gt_dir / f'{im_name}.png'
        util_image.imwrite(im_gt, im_path_gt, chn='bgr', dtype_in='uint8')

        im_path_lq = lq_dir / f'{im_name}.png'
        util_image.imwrite(im_lq, im_path_lq, chn='bgr', dtype_in='uint8')

if __name__ == "__main__":
    main()
