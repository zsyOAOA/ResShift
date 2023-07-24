#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=512, help="Chopping forward.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
            "--task",
            type=str,
            default="realsrx4",
            choices=['realsrx4', 'bicsrx4'],
            help="Chopping forward.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):

    if args.task == 'realsrx4':
        configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')

    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / f'resshift_{args.task}_s{args.steps}.pth'
    if not ckpt_path.exists():
         load_file_from_url(
            url=f"https://github.com/zsyOAOA/ResShift/releases/download/v1.0/{ckpt_path.name}",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth",
            model_dir=vqgan_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    if args.chop_size == 512:
        chop_stride = 448
        chop_bs = 1
    elif args.chop_size == 256:
        chop_stride = 224
        chop_bs = 4
    elif args.chop_size == 128:
        chop_stride = 96
        chop_bs = 16
    else:
        raise ValueError("Chop size only accept 128, 256, or 512!")

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir()

    return configs, chop_stride, chop_bs

def main():
    args = get_parser()

    configs, chop_stride, chop_bs = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=chop_bs,
            use_fp16=True,
            seed=args.seed,
            )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False)

if __name__ == '__main__':
    main()
