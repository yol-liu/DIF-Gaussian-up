from contextlib import contextmanager

import numpy as np
from scipy.ndimage import binary_dilation, zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@contextmanager
def evaluating(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def resolved_latent_mode(model, latent_mode=None):
    if hasattr(model, 'probability'):
        from probabilistic_config import ALIASES
        return ALIASES.get(latent_mode, latent_mode) or model.probability['evaluation']['estimator']
    return latent_mode or model.cfg.get('sampling', {}).get('eval_mode', 'prior_mean')


def predict_volume(model, data, chunk_size, latent_mode=None, num_samples=None, oracle=False,
                   return_details=False, generator=None, return_samples=False):
    kwargs = {'is_eval': True, 'eval_npoint': chunk_size}
    if getattr(model, 'use_latent', False):
        latent_mode = resolved_latent_mode(model, latent_mode)
        kwargs.update(latent_mode=latent_mode, num_samples=num_samples)
        if hasattr(model, 'probability'):
            kwargs.update(oracle=oracle, generator=generator, return_samples=return_samples)
        elif latent_mode in ('posterior_mean_oracle', 'gt_posterior_mean', 'gt_posterior_sample'):
            if not oracle:
                raise ValueError('Posterior evaluation requires explicit oracle=True')
            kwargs['gt_volume'] = data.get('gt_volume')
            kwargs['oracle'] = True
    details = model(data, **kwargs)
    output = details['points_pred']
    if output.ndim == 4:
        output = output.mean(0)
    if output.ndim != 3:
        raise ValueError(f'Expected [B,1,N] predictions, got {output.shape}')
    details['points_pred'] = output
    return details if return_details else output


def legacy_downsample(volume, scale=0.5):
    return zoom(volume, scale, order=3, prefilter=False) if scale != 1 else volume.copy()


def volume_metrics(target, prediction, spacing):
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite volume prediction')
    clipped = np.clip(prediction, 0, 1)
    target_gradient = np.stack(np.gradient(target, *spacing))
    prediction_gradient = np.stack(np.gradient(clipped, *spacing))
    magnitude = np.linalg.norm(target_gradient, axis=0)
    edge = magnitude > 0.02
    radius_mm = 2.0
    radii = np.ceil(radius_mm / np.asarray(spacing)).astype(int)
    axes = [np.arange(-radius, radius + 1) * pitch for radius, pitch in zip(radii, spacing)]
    offsets = np.meshgrid(*axes, indexing='ij')
    band = binary_dilation(edge, structure=sum(axis ** 2 for axis in offsets) <= radius_mm ** 2)
    window = min(7, min(target.shape))
    window -= 1 - window % 2
    return {
        'psnr': float(peak_signal_noise_ratio(target, clipped, data_range=1.)),
        'ssim': float(structural_similarity(target, clipped, data_range=1., win_size=window)) if window >= 3 else None,
        'unclipped_mse': float(np.mean((target - prediction) ** 2)),
        'clipping_fraction': float(np.mean((prediction < 0) | (prediction > 1))),
        'boundary_error': float(np.mean(np.abs(target[band] - clipped[band]))) if band.any() else None,
        'gradient_error': float(np.mean(np.abs(target_gradient - prediction_gradient))),
    }


def scale_comparisons(target, full_prediction, coarse_prediction, spacing):
    coarse_target = legacy_downsample(target)
    filtered_prediction = legacy_downsample(full_prediction)
    coarse_spacing = np.asarray(spacing) * (np.asarray(target.shape) - 1) / (np.asarray(coarse_target.shape) - 1)
    return {
        'S1': volume_metrics(target, full_prediction, spacing),
        'S2': volume_metrics(coarse_target, filtered_prediction, coarse_spacing),
        'S3': volume_metrics(coarse_target, coarse_prediction, coarse_spacing),
        'S4': {'unclipped_mse': float(np.mean((coarse_prediction - filtered_prediction) ** 2))},
    }
