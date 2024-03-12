# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path

from sampler import ResShiftSampler

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.configs = {
            "realsr": OmegaConf.load('./configs/realsr_swinunet_realesrgan256_journal.yaml'),
            "bicsr": configs = OmegaConf.load('./configs/bicx4_swinunet_lpips.yaml'),
        }

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: int = Input(description="Factor to scale image by.", default=4),
        chop_size: int = Input(
            choices=[512, 256], description="Chopping forward.", default=512
        ),
        task: str = Input(
            choices=["realsr", "bicsr"],
            description="Choose a task",
            default="realsr",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=12345
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        configs = self.configs[task]

        if task == 'realsr':
            ckpt_path = f"weights/resshift_realsrx4_s4_v3.pth"
            configs.model.ckpt_path = ckpt_path
        else:
            ckpt_path = f"weights/resshift_bicsrx4_s4.pth"
            configs.model.ckpt_path = ckpt_path
        configs.diffusion.params.steps = 4
        configs.diffusion.params.sf = scale
        configs.autoencoder.ckpt_path = f"weights/autoencoder_vq_f4.pth"

        chop_stride = 448 if chop_size == 512 else 224

        resshift_sampler = ResShiftSampler(
                configs,
                sf=scale,
                chop_size=chop_size,
                chop_stride=chop_stride,
                chop_bs=1,
                use_amp=True,
                seed=seed,
                padding_offset=configs.model.params.get('lq_size', 64),
                )

        out_path = "out_dir"
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        resshift_sampler.inference(
                str(image),
                out_path,
                mask_path=None,
                bs=1,
                noise_repeat=False
                )

        out = "/tmp/out.png"
        shutil.copy(os.path.join(out_path, os.listdir(out_path)[0]), out)

        return Path(out)
