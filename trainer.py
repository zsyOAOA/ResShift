#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

from datapipe.datasets import create_dataset

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        # only should be run on rank: 0
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
        else:
            save_dir = Path(self.configs.save_dir) / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # text logging
        if self.rank == 0:
            logtxet_path = save_dir / 'training.log'
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a')
            self.logger.add(sys.stdout, format="{message}", level="INFO")

        # tensorboard logging
        if self.rank == 0:
            # log_dir = save_dir / 'tf_logs'
            # if not log_dir.exists():
                # log_dir.mkdir()
            # self.writer = SummaryWriter(str(log_dir))
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # image saving
        if self.rank == 0 and self.configs.train.save_images:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # checkpoint saving
        if self.rank == 0:
            ckpt_dir = save_dir / 'ckpts'
            if not ckpt_dir.exists():
                ckpt_dir.mkdir()
            self.ckpt_dir = ckpt_dir

        # ema checkpoint saving
        if self.rank == 0 and hasattr(self, 'ema_rate'):
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            if not ema_ckpt_dir.exists():
                ema_ckpt_dir.mkdir()
            self.ema_ckpt_dir = ema_ckpt_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0:
            # self.writer.close()
            pass

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)


        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(self.iters_start):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=False,
                        num_workers=self.configs.train.get('num_workers', 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info("Detailed network architecture:")
            self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_sheduler')
        self.lr_sheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            torch.save({'iters_start': self.current_iters,
                        'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                        'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                        'state_dict': self.model.state_dict()}, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

class TrainerDifIR(TrainerBase):
    def __init__(self, configs):
        # ema settings
        self.ema_rate = configs.train.ema_rate
        super().__init__(configs)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model, ckpt)

        # EMA
        if self.rank == 0:
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half().cuda()
            else:
                self.autoencoder = autoencoder.cuda()
        else:
            self.autoencoder = None

        # LPIPS metric
        if self.rank == 0:
            self.lpips_loss = lpips.LPIPS(net='vgg').cuda()

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

        # model information
        self.print_model_info()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                temp_sf = self.configs.degradation['sf']
            else:
                temp_sf = self.configs.degradation['sf']

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            noise = torch.randn(
                    size=micro_data['gt'].shape[:2] + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            model_kwargs={'lq':micro_data['lq'],} if self.configs.model.params.cond_lq else None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses, z_t, z0_pred = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses, z_t, z0_pred = compute_losses()
                    loss = losses["loss"].mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses, z_t, z0_pred = compute_losses()
                else:
                    with self.model.no_sync():
                        losses, z_t, z0_pred = compute_losses()
                loss = losses["loss"].mean() / num_grad_accumulate
                loss.backward()

            # make logging
            self.log_step_train(losses, tt, micro_data, z_t, z0_pred, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        if len(self.configs.train.milestones) > 0:
            base_lr = self.configs.train.lr
            linear_steps = self.configs.train.milestones[0]
            current_iters = self.current_iters if current_iters is None else current_iters
            if current_iters <= linear_steps:
                for params_group in self.optimizer.param_groups:
                    params_group['lr'] = (current_iters / linear_steps) * base_lr
            elif current_iters in self.configs.train.milestones:
                for params_group in self.optimizer.param_groups:
                    params_group['lr'] *= 0.5
        else:
            pass

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, num_timesteps //2, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.2e}/{:.2e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            )
                    # tensorboard
                    # self.writer.add_scalar(f'Loss-Step-{current_record}',
                                           # self.loss_mean['loss'][jj].item(),
                                           # self.log_step[phase])
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.log_step[phase] += 1
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                x1 = vutils.make_grid(batch['lq'], normalize=True, scale_each=True)  # c x h x w
                # self.writer.add_image("Training LQ Image", x1, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x1.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"lq_{self.log_step_img[phase]:05d}.png",
                           )
                x2 = vutils.make_grid(batch['gt'], normalize=True)
                # self.writer.add_image("Training HQ Image", x2, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x2.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"hq_{self.log_step_img[phase]:05d}.png",
                           )
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                x3 = vutils.make_grid(x_t, normalize=True, scale_each=True)
                # self.writer.add_image("Training Diffused Image", x3, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x3.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"diffused_{self.log_step_img[phase]:05d}.png",
                           )
                x0_pred = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z0_pred, tt),
                        self.autoencoder,
                        )
                x4 = vutils.make_grid(x0_pred, normalize=True, scale_each=True)
                # self.writer.add_image("Training Predicted Image", x4, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x4.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"x0_pred_{self.log_step_img[phase]:05d}.png",
                           )
                self.log_step_img[phase] += 1

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = [int(self.base_diffusion.num_timesteps * x) for x in [0.25, 0.5, 0.75, 1]]
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']
                num_iters = 0
                model_kwargs={'lq':im_lq,} if self.configs.model.params.cond_lq else None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        ):
                    sample_decode = {}
                    if (num_iters + 1) in indices or num_iters + 1 == 1:
                        for key, value in sample.items():
                            # if key in ['sample', 'pred_xstart']:
                            if key in ['sample', ]:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        self.base_diffusion._scale_input(value, tt-1),
                                        self.autoencoder,
                                        )
                        im_sr_progress = sample_decode['sample']
                        # im_xstart = sample_decode['pred_xstart']
                        if num_iters + 1 == 1:
                            # im_sr_all, im_xstart_all = im_sr_progress, im_xstart
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                            # im_xstart_all = torch.cat((im_xstart_all, im_xstart), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'].detach() * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=True,
                            )
                    mean_lpips += self.lpips_loss(sample_decode['sample'].detach(), im_gt).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    x1 = vutils.make_grid(im_sr_all.detach(), nrow=len(indices)+1, normalize=True, scale_each=True)
                    # self.writer.add_image('Validation Sample Progress', x1, self.log_step_img[phase])
                    if self.configs.train.save_images:
                        util_image.imwrite(
                               x1.cpu().permute(1,2,0).numpy(),
                               self.image_dir / phase / f"progress_{self.log_step_img[phase]:05d}.png",
                               )
                    x3 = vutils.make_grid(im_lq, normalize=True)
                    # self.writer.add_image('Validation LQ Image', x3, self.log_step_img[phase])
                    if self.configs.train.save_images:
                        util_image.imwrite(
                               x3.cpu().permute(1,2,0).numpy(),
                               self.image_dir / phase / f"lq_{self.log_step_img[phase]:05d}.png",
                               )
                    if 'gt' in data:
                        x4 = vutils.make_grid(im_gt, normalize=True)
                        # self.writer.add_image('Validation HQ Image', x4, self.log_step_img[phase])
                        if self.configs.train.save_images:
                            util_image.imwrite(
                                   x4.cpu().permute(1,2,0).numpy(),
                                   self.image_dir / phase / f"hq_{self.log_step_img[phase]:05d}.png",
                                   )
                    self.log_step_img[phase] += 1

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                # self.writer.add_scalar('Validation PSNR', mean_psnr, self.log_step[phase])
                # self.writer.add_scalar('Validation LPIPS', mean_lpips, self.log_step[phase])
                self.log_step[phase] += 1

            self.logger.info("="*100)

            if not self.configs.train.use_ema_val:
                self.model.train()

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if not 'relative_position_index' in key:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))

