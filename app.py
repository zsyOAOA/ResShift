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

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    }
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
         }

def get_configs(task='realsr', version='v3', scale=4):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if task == 'realsr':
        if version in ['v1', 'v2']:
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
        elif version == 'v3':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256_journal.yaml')
        else:
            raise ValueError(f"Unexpected version type: {version}")
        assert scale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[version]
        ckpt_path = ckpt_dir / f'resshift_{task}x{scale}_s{_STEP[version]}_{version}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif task == 'bicsr':
        configs = OmegaConf.load('./configs/bicx4_swinunet_lpips.yaml')
        assert scale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[task]
        ckpt_path = ckpt_dir / f'resshift_{task}x{scale}_s{_STEP[task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    # elif task == 'inpaint_imagenet':
        # configs = OmegaConf.load('./configs/inpaint_lama256_imagenet.yaml')
        # assert scale == 1, 'Please set scale equals 1 for image inpainting!'
        # ckpt_url = _LINK[task]
        # ckpt_path = ckpt_dir / f'resshift_{task}_s{_STEP[task]}.pth'
        # vqgan_url = _LINK['vqgan']
        # vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    # elif task == 'inpaint_face':
        # configs = OmegaConf.load('./configs/inpaint_lama256_face.yaml')
        # assert scale == 1, 'Please set scale equals 1 for image inpainting!'
        # ckpt_url = _LINK[task]
        # ckpt_path = ckpt_dir / f'resshift_{task}_s{_STEP[task]}.pth'
        # vqgan_url = _LINK['vqgan_face256']
        # vqgan_path = ckpt_dir / f'celeba256_vq_f4_dim3_face.pth'
    # elif task == 'faceir':
        # configs = OmegaConf.load('./configs/faceir_gfpgan512_lpips.yaml')
        # assert scale == 1, 'Please set scale equals 1 for face restoration!'
        # ckpt_url = _LINK[task]
        # ckpt_path = ckpt_dir / f'resshift_{task}_s{_STEP[task]}.pth'
        # vqgan_url = _LINK['vqgan_face512']
        # vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
    else:
        raise TypeError(f"Unexpected task type: {task}!")

    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    return configs

def predict(in_path, task='realsrx4', seed=12345, scale=4, version='v3'):
    configs = get_configs(task, version, scale)
    resshift_sampler = ResShiftSampler(
            configs,
            sf=scale,
            chop_size=256,
            chop_stride=224,
            chop_bs=1,
            use_amp=True,
            seed=seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    out_dir = Path('restored_output')
    if not out_dir.exists():
        out_dir.mkdir()

    resshift_sampler.inference(
            in_path,
            out_dir,
            mask_path=None,
            bs=1,
            noise_repeat=False
            )

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
@inproceedings{yue2023resshift,
  title={ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting},
  author={Yue, Zongsheng and Wang, Jianyi and Loy, Chen Change},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023},
  volume = {36},
  pages = {13294--13307},
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
            choices=["realsr", "bicsr"],
            value="realsr",
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
        ['./testdata/RealSet65/0030.jpg',  "realsr", 12345],
        ['./testdata/RealSet65/dog2.png',  "realsr", 12345],
        ['./testdata/RealSet65/bears.jpg', "realsr", 12345],
        ['./testdata/RealSet65/oldphoto6.png', "realsr", 12345],
        ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00000067.png', "bicsr", 12345],
        ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00016898.png', "bicsr", 12345],
      ]
    )

demo.queue(concurrency_count=4)
demo.launch(share=True)

