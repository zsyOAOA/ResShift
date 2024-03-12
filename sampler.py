#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=4,
            use_amp=True,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            padding_offset=16,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, mask=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
            mask: image mask for inpainting
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        offset = self.padding_offset
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        if self.configs.model.params.cond_lq and mask is not None:
            model_kwargs={
                    'lq':y0,
                    'mask': mask,
                    }
        elif self.configs.model.params.cond_lq:
            model_kwargs={'lq':y0,}
        else:
            model_kwargs = None

        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    def inference(self, in_path, out_path, mask_path=None, mask_back=True, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
            mask_path: image mask for inpainting
        '''
        def _process_per_image(im_lq_tensor, mask=None):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
                mask: image mask for inpainting, [-1, 1], 1 for unknown area
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                if mask is not None:
                    im_lq_tensor = torch.cat([im_lq_tensor, mask], dim=1)
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,
                        )
                for im_lq_pch, index_infos in im_spliter:
                    if mask is not None:
                        im_lq_pch, mask_pch = im_lq_pch[:, :-1], im_lq_pch[:, -1:,]
                    else:
                        mask_pch = None
                    with context():
                        im_sr_pch = self.sample_func(
                                im_lq_pch,
                                noise_repeat=noise_repeat,
                                mask=mask_pch,
                                )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                # print(im_lq_tensor.shape)
                with context():
                    im_sr_tensor = self.sample_func(
                            im_lq_tensor,
                            noise_repeat=noise_repeat,
                            mask=mask,
                            )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            if mask_back and mask is not None:
                mask = mask * 0.5 + 0.5
                im_lq_tensor = im_lq_tensor * 0.5 + 0.5
                im_sr_tensor = im_sr_tensor * mask + im_lq_tensor * (1 - mask)
            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)

        if self.num_gpus > 1:
            dist.barrier()

        if in_path.is_dir():
            if mask_path is None:
                data_config = {'type': 'base',
                               'params': {'dir_path': str(in_path),
                                          'transform_type': 'default',
                                          'transform_kwargs': {
                                              'mean': 0.5,
                                              'std': 0.5,
                                              },
                                          'need_path': True,
                                          'recursive': True,
                                          'length': None,
                                          }
                               }
            else:
                data_config = {'type': 'inpainting_val',
                               'params': {'lq_path': str(in_path),
                                          'mask_path': mask_path,
                                          'transform_type': 'default',
                                          'transform_kwargs': {
                                              'mean': 0.5,
                                              'std': 0.5,
                                              },
                                          'need_path': True,
                                          'recursive': True,
                                          'im_exts': ['png', 'jpg', 'jpeg', 'JPEG', 'bmp', 'PNG'],
                                          'length': None,
                                          }
                               }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for data in dataloader:
                micro_batchsize = math.ceil(bs / self.num_gpus)
                ind_start = self.rank * micro_batchsize
                ind_end = ind_start + micro_batchsize
                micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}

                if micro_data['lq'].shape[0] > 0:
                    results = _process_per_image(
                            micro_data['lq'].cuda(),
                            mask=micro_data['mask'].cuda() if 'mask' in micro_data else None,
                            )    # b x h x w x c, [0, 1], RGB

                    for jj in range(results.shape[0]):
                        im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                        im_name = Path(micro_data['path'][jj]).stem
                        im_path = out_path / f"{im_name}.png"
                        util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
            if self.num_gpus > 1:
                dist.barrier()
        else:
            im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
            if mask_path is not None:
                im_mask = util_image.imread(mask_path, chn='gray', dtype='float32')[:,:, None]  # h x w x 1
                im_mask_tensor = util_image.img2tensor(im_mask).cuda()              # 1 x c x h x w

            im_sr_tensor = _process_per_image(
                    (im_lq_tensor - 0.5) / 0.5,
                    mask=(im_mask_tensor - 0.5) / 0.5 if mask_path is not None else None,
                    )

            im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
            im_path = out_path / f"{in_path.stem}.png"
            util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass

