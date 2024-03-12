#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:11:42

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import random
import argparse
import numpy as np

from utils import util_image
from utils import util_common

from datapipe.face_degradation_testing import face_degradation

parser = argparse.ArgumentParser()
parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default='/mnt/lustre/zsyue/disk/ResShift/Face/testingdata/Celeba-Test/',
        help="Folder to save the testing data.",
        )
parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default='/mnt/lustre/zsyue/datapipe/CelebA/images512x512',
        help="Txt file to record the image path.",
        )
parser.add_argument(
        "-n",
        "--num_val",
        type=int,
        default=2000,
        help="Number of images",
        )
parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="Random seed.",
        )
args = parser.parse_args()

# setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# checking save_dir
lq_dir = Path(args.save_dir) / "lq"
hq_dir = Path(args.save_dir) / "hq"

util_common.mkdir(lq_dir, delete=True)
util_common.mkdir(hq_dir, delete=True)

files_path = sorted([x for x in Path(args.input_dir).glob("*.png")])
assert args.num_val <= len(files_path)
random.shuffle(files_path)
files_path = files_path[:args.num_val]
print(f'Number of images in validation: {args.num_val}')

num_iters = 0
for ii in range(args.num_val):
    if (ii+1) %  100 == 0:
        print(f'Processing: {ii+1}/{args.num_val}')

    # degradation setting
    sf = random.uniform(1.0, 32)
    qf = random.uniform(30, 70)
    nf = random.uniform(1.0, 20)
    sig_x = random.uniform(4.0, 16)
    sig_y = random.uniform(4.0, 16)
    theta = random.random() * math.pi

    im_gt_path = files_path[ii]
    im_gt = util_image.imread(im_gt_path, chn='bgr', dtype='float32')

    im_lq = face_degradation(
            im_gt,
            sf=sf,
            sig_x=sig_x,
            sig_y=sig_y,
            theta=theta,
            qf=qf,
            nf=nf,
            )

    im_name = Path(im_gt_path).name
    im_save_path = lq_dir / im_name
    util_image.imwrite(im_lq, im_save_path, chn="bgr", dtype_in='float32')

    im_save_path = hq_dir / im_name
    util_image.imwrite(im_gt, im_save_path, chn="bgr", dtype_in='float32')

