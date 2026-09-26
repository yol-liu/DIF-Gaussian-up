import torch
import torch.nn.functional as functional


def project_attenuation(model_or_volume, geometry, rays, samples=128):
    if samples < 2:
        raise ValueError('Ray quadrature needs at least two samples')
    if geometry.get('attenuation_unit') != 'mm^-1':
        raise ValueError('Explicit attenuation_unit=mm^-1 is required; normalized CT needs calibration')
    start, end = rays['start'], rays['end']
    extent = torch.as_tensor(geometry['volume_extent'], device=start.device, dtype=start.dtype)
    if extent.ndim == 2:
        extent = extent[:, None]
    direction = end - start
    parallel = direction.abs() < 1e-12
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    first, second = -start / safe_direction, (1 - start) / safe_direction
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    low = torch.where(parallel, torch.full_like(low, -torch.inf), low)
    high = torch.where(parallel, torch.full_like(high, torch.inf), high)
    entry = low.max(-1).values.clamp(0, 1)
    exit_point = high.min(-1).values.clamp(0, 1)
    valid = (exit_point > entry) & ~(parallel & ((start < 0) | (start > 1))).any(-1)
    interval = (exit_point - entry).clamp_min(0) * valid
    fractions = (torch.arange(samples, device=start.device, dtype=start.dtype) + 0.5) / samples
    positions = start[..., None, :] + direction[..., None, :] * (
        entry[..., None] + interval[..., None] * fractions)[..., None]
    if callable(model_or_volume):
        values = model_or_volume(positions.flatten(1, 2)).reshape(*start.shape[:2], samples)
    else:
        grid = positions[..., [2, 1, 0]] * 2 - 1
        values = functional.grid_sample(model_or_volume, grid.unsqueeze(3), mode='bilinear',
                                        padding_mode='border', align_corners=True)[:, 0, :, :, 0]
    length_mm = torch.linalg.vector_norm(direction * extent, dim=-1) * interval
    return values.mean(-1) * length_mm


def calibrated_ray_loss(prediction, measurement, calibration):
    if calibration.get('status') != 'PASS' or not calibration.get('gt_projection_match'):
        raise ValueError('Ray loss is blocked until GT/operator calibration passes')
    return functional.smooth_l1_loss(prediction, measurement)
