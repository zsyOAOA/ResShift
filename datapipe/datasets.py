import random
import numpy as np
from pathlib import Path
from scipy.io import loadmat

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .degradation_bsrgan.bsrgan_light import degradation_bsrgan_variant, degradation_bsrgan
from .masks import MixedMaskGenerator

class LamaDistortionTransform:
    def __init__(self, kwargs):
        import albumentations as A
        from .aug import IAAAffine2, IAAPerspective2
        out_size = kwargs.get('pch_size', 256)
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=out_size),
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.Normalize(mean=kwargs.mean, std=kwargs.std, max_pixel_value=kwargs.max_value),
        ])

    def __call__(self, im):
        '''
        im: numpy array, h x w x c, [0,1]

        '''
        return self.transform(image=im)['image']

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'resize_ccrop_norm':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            # max edge resize if crop_size is int
            thv.transforms.Resize(size=kwargs.get('size', None)),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'rcrop_aug_norm':
        transform = thv.transforms.Compose([
            util_image.RandomCrop(pch_size=kwargs.get('pch_size', 256)),
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(max_value=kwargs.get('max_value')),  # (ndarray, hwc) --> (Tensor, chw)
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(),   # hwc --> chw
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'lama_distortions':
        transform = thv.transforms.Compose([
                LamaDistortionTransform(kwargs),
                util_image.ToTensor(max_value=1.0),   # hwc --> chw
            ])
    elif transform_type == 'rgb2gray':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),   # c x h x w, [0,1]
            thv.transforms.Grayscale(num_output_channels=kwargs.get('num_output_channels', 3)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    elif dataset_config['type'] == 'bsrgan':
        dataset = BSRGANLightDeg(**dataset_config['params'])
    elif dataset_config['type'] == 'bsrganimagenet':
        dataset = BSRGANLightDegImageNet(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    elif dataset_config['type'] == 'siddval':
        dataset = SIDDValData(**dataset_config['params'])
    elif dataset_config['type'] == 'inpainting':
        dataset = InpaintingDataSet(**dataset_config['params'])
    elif dataset_config['type'] == 'inpainting_val':
        dataset = InpaintingDataSetVal(**dataset_config['params'])
    elif dataset_config['type'] == 'deg_from_source':
        dataset = DegradedDataFromSource(**dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = BicubicFromSource(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            extra_dir_path=None,
            extra_transform_type=None,
            extra_transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            assert extra_transform_type is not None
            self.extra_transform = get_transforms(extra_transform_type, extra_transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='float32')

        im_target = self.transform(im_base)
        out = {'image':im_target, 'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BSRGANLightDegImageNet(Dataset):
    def __init__(self,
                 dir_paths=None,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=True,
                 degradation='bsrgan_light',
                 use_sharp=False,
                 rescale_gt=True,
                 ):
        super().__init__()
        file_paths_all = []
        if dir_paths is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_paths, im_exts, recursive))
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.mean = mean
        self.std = std
        self.rescale_gt = rescale_gt
        if rescale_gt:
            from albumentations import SmallestMaxSize
            self.smallest_rescaler = SmallestMaxSize(max_size=gt_size)

        self.gt_size = gt_size
        self.LR_size = int(gt_size / sf)

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=sf, use_sharp=use_sharp)
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_bsrgan_variant, sf=sf, use_sharp=use_sharp)
        else:
            raise ValueError(f'Except bsrgan or bsrgan_light for degradation, now is {degradation}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        h, w = im_hq.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            pad_h = max(0, self.gt_size - h)
            pad_w = max(0, self.gt_size - w)
            im_hq = cv2.copyMakeBorder(im_hq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if self.rescale_gt:
            im_hq = self.smallest_rescaler(image=im_hq)['image']

        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        im_lq, im_hq = self.degradation_process(image=im_hq)
        im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class BSRGANLightDeg(Dataset):
    def __init__(self,
                 dir_paths,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=False,
                 resize_back=False,
                 use_sharp=False,
                 ):
        super().__init__()
        file_paths_all = util_common.scan_files_from_folder(dir_paths, im_exts, recursive)
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.resize_back = resize_back

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.gt_size = gt_size
        self.mean = mean
        self.std = std
        self.use_sharp=use_sharp

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        # random crop
        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        # degradation
        im_lq, im_hq = degradation_bsrgan_variant(im_hq, self.sf, use_sharp=self.use_sharp)
        if self.resize_back:
            im_lq = cv2.resize(im_lq, dsize=(self.gt_size,)*2, interpolation=cv2.INTER_CUBIC)
            im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class SIDDValData(Dataset):
    def __init__(self, noisy_path, gt_path, mean=0.5, std=0.5):
        super().__init__()
        self.im_noisy_all = loadmat(noisy_path)['ValidationNoisyBlocksSrgb']
        self.im_gt_all = loadmat(gt_path)['ValidationGtBlocksSrgb']

        h, w, c = self.im_noisy_all.shape[2:]
        self.im_noisy_all = self.im_noisy_all.reshape([-1, h, w, c])
        self.im_gt_all = self.im_gt_all.reshape([-1, h, w, c])
        self.mean, self.std = mean, std

    def __len__(self):
        return self.im_noisy_all.shape[0]

    def __getitem__(self, index):
        im_gt = self.im_gt_all[index].astype(np.float32) / 255.
        im_noisy = self.im_noisy_all[index].astype(np.float32) / 255.

        im_gt = (im_gt - self.mean) / self.std
        im_noisy = (im_noisy - self.mean) / self.std

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return {'lq': im_noisy, 'gt': im_gt}

class InpaintingDataSet(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs,
            mask_kwargs,
            txt_file_path=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = [] if txt_file_path is None else util_common.readline_txt(txt_file_path)
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.mean = transform_kwargs.mean
        self.std = transform_kwargs.std
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        self.iter_i = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='uint8')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }

        mask = self.mask_generator(im, iter_i=self.iter_i)             # c x h x w, [0,1]
        self.iter_i += 1
        im_masked = im *  (1 - mask) - mask * (self.mean / self.std)   # mask area: -1
        out_dict['lq'] = im_masked
        out_dict['mask'] = (mask - self.mean) / self.std               # c x h x w, [-1,1]

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class InpaintingDataSetVal(Dataset):
    def __init__(
            self,
            lq_path,
            gt_path=None,
            mask_path=None,
            transform_type=None,
            transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = util_common.scan_files_from_folder(lq_path, im_exts, recursive)
        self.file_paths_all = file_paths_all

        # lq image path
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.gt_path = gt_path
        self.mask_path = mask_path

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_lq = util_image.imread(im_path, chn='rgb', dtype='float32')
        im_lq = self.transform(im_lq)
        out_dict = {'lq':im_lq}

        if self.need_path:
            out_dict['path'] = im_path

        # ground truth images
        if self.gt_path is not None:
            im_path = Path(self.gt_path) / Path(im_path).name
            im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
            im_gt = self.transform(im_gt)
            out_dict['gt'] = im_gt

        # image mask
        im_path = Path(self.mask_path) / Path(im_path).name
        im_mask = util_image.imread(im_path, chn='gray', dtype='float32')
        im_mask = self.transform(im_mask)
        out_dict['mask'] = im_mask        # -1 and 1

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class DegradedDataFromSource(Dataset):
    def __init__(
            self,
            source_path,
            source_txt_path=None,
            degrade_kwargs=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        file_paths_all = []
        if source_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(source_path, im_exts, recursive))
        if source_txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(source_txt_path))
        self.file_paths_all = file_paths_all

        if length is None:
            self.file_paths = file_paths_all
        else:
            assert len(file_paths_all) >= length
            self.file_paths = random.sample(file_paths_all, length)

        self.length = length
        self.need_path = need_path

        self.transform = get_transforms(transform_type, transform_kwargs)
        self.degrade_kwargs = degrade_kwargs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_source = util_image.imread(im_path, chn='rgb', dtype='float32')
        out = {'gt':self.gt_transform(im_source), 'lq':self.lq_transform(im_source)}

        if self.need_path:
            out['path'] = im_path

        return out

class BicubicFromSource(DegradedDataFromSource):
    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')

        if not hasattr(self, 'smallmax_resizer'):
            self.smallmax_resizer= util_image.SmallestMaxSize(
                    max_size = self.degrade_kwargs.get('gt_size', 256),
                    )
        if not hasattr(self, 'bicubic_transform'):
            self.bicubic_transform = util_image.Bicubic(
                scale=self.degrade_kwargs.get('scale', None),
                out_shape=self.degrade_kwargs.get('out_shape', None),
                activate_matlab=self.degrade_kwargs.get('activate_matlab', True),
                resize_back=self.degrade_kwargs.get('resize_back', False),
                )
        if not hasattr(self, 'random_cropper'):
            self.random_cropper = util_image.RandomCrop(
                pch_size=self.degrade_kwargs.get('pch_size', None),
                pass_crop=self.degrade_kwargs.get('pass_crop', False),
                )
        if not hasattr(self, 'paired_aug'):
            self.paired_aug = util_image.SpatialAug(
                    pass_aug = self.degrade_kwargs.get('pass_aug', False)
                    )

        im_gt = self.smallmax_resizer(im_gt)
        im_gt = self.random_cropper(im_gt)
        im_lq = self.bicubic_transform(im_gt)
        im_lq, im_gt = self.paired_aug([im_lq, im_gt])

        out = {'gt':self.transform(im_gt), 'lq':self.transform(im_lq)}

        if self.need_path:
            out['path'] = im_path

        return out
