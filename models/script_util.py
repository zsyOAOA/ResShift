import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps, SpacedDiffusionDDPM

def create_gaussian_diffusion(
    *,
    normalize_input,
    schedule_name,
    sf=4,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
    )

def create_gaussian_diffusion_ddpm(
    *,
    beta_start,
    beta_end,
    sf=4,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    timestep_respacing=None,
    scale_factor=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    return SpacedDiffusionDDPM(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarTypeDDPM.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarTypeDDPM.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarTypeDDPM.LEARNED_RANGE
        ),
        scale_factor=scale_factor,
        sf=sf,
    )
