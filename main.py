#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/training/ffhq256_bicubic8.yaml",
            help="Configs of yaml file",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
