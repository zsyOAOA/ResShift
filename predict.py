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
            "realsrx4": OmegaConf.load("./configs/realsr_swinunet_realesrgan256.yaml"),
            "bicsrx4_opencv": OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml"),
            "bicsrx4_matlab": OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml"),
        }

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: int = Input(description="Factor to scale image by.", default=4),
        chop_size: int = Input(
            choices=[512, 256], description="Chopping forward.", default=512
        ),
        task: str = Input(
            choices=["realsrx4", "bicsrx4_opencv", "bicsrx4_matlab"],
            description="Choose a task",
            default="realsrx4",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        configs = self.configs[task]

        configs.model.ckpt_path = f"weights/resshift_{task}_s15.pth"
        configs.diffusion.params.steps = 15
        configs.diffusion.params.sf = scale
        configs.autoencoder.ckpt_path = f"weights/autoencoder_vq_f4.pth"
        if task == "bicsrx4_matlab":
            configs.diffusion.params.kappa = 2.0

        chop_stride = 448 if chop_size == 512 else 224

        resshift_sampler = ResShiftSampler(
            configs,
            chop_size=chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=True,
            seed=seed,
        )

        out_path = "out_dir"
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        resshift_sampler.inference(str(image), out_path, bs=1, noise_repeat=False)

        out = "/tmp/out.png"
        shutil.copy(os.path.join(out_path, os.listdir(out_path)[0]), out)

        return Path(out)
