#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-08-15 09:39:58

import argparse
import gradio as gr
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(task='realsrx4'):
    if task == 'realsrx4':
        configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    elif task == 'bicsrx4_opencv':
        configs = OmegaConf.load('./configs/bicubic_swinunet_bicubic256.yaml')
    elif task == 'bicsrx4_matlab':
        configs = OmegaConf.load('./configs/bicubic_swinunet_bicubic256.yaml')
        configs.diffusion.params.kappa = 2.0

    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / f'resshift_{task}_s15.pth'
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
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = 15
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = str(vqgan_path)

    return configs

def predict(in_path, task='realsrx4', seed=12345):
    configs = get_configs(task)
    resshift_sampler = ResShiftSampler(
            configs,
            chop_size=256,
            chop_stride=224,
            chop_bs=1,
            use_fp16=True,
            seed=seed,
            )

    out_dir = Path('restored_output')
    if not out_dir.exists():
        out_dir.mkdir()

    resshift_sampler.inference(in_path, out_dir, bs=1, noise_repeat=False)

    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")

    return im_sr, str(out_path)

title = "ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting"
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/zsyOAOA/ResShift' target='_blank'><b>ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting</b></a>.<br>
🔥 ResShift is an efficient diffusion model designed for image super-resolution or restoration.<br>
"""
article = r"""
If ResShift is helpful for your work, please help to ⭐ the <a href='https://github.com/zsyOAOA/ResShift' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/zsyOAOA/ResShift?affiliations=OWNER&color=green&style=social)](https://github.com/zsyOAOA/ResShift)

---
If our work is useful for your research, please consider citing:
```bibtex
@article{yue2023resshift,
  title={ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting},
  author={Yue, Zongsheng and Wang, Jianyi and Loy, Chen Change},
  journal={arXiv preprint arXiv:2307.12348},
  year={2023}
}
```

📋 **License**

This project is licensed under <a rel="license" href="https://github.com/zsyOAOA/ResShift/blob/master/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**

If you have any questions, please feel free to contact me via <b>zsyzam@gmail.com</b>.
![visitors](https://visitor-badge.laobi.icu/badge?page_id=zsyOAOA/ResShift)
"""
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Input: Low Quality Image"),
        gr.Dropdown(
            choices=["realsrx4", "bicsrx4_opencv", "bicsrx4_matlab"],
            value="realsrx4",
            label="Task",
            ),
        gr.Number(value=12345, precision=0, label="Ranom seed")
    ],
    outputs=[
        gr.Image(type="numpy", label="Output: High Quality Image"),
        gr.outputs.File(label="Download the output")
    ],
    title=title,
    description=description,
    article=article,
    examples=[
        ['./testdata/RealSet65/0030.jpg',  "realsrx4", 12345],
        ['./testdata/RealSet65/dog2.png',  "realsrx4", 12345],
        ['./testdata/RealSet65/bears.jpg', "realsrx4", 12345],
        ['./testdata/RealSet65/oldphoto6.png', "realsrx4", 12345],
        ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00000067.png', "bicsrx4_matlab", 12345],
        ['./testdata/Bicubicx4/lq_opencv/ILSVRC2012_val_00016898.png', "bicsrx4_opencv", 12345],
      ]
    )

demo.queue(concurrency_count=4)
demo.launch(share=False)

