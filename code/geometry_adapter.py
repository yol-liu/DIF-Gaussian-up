"""Scanner-frame geometry in mm; detector uv indexes the original tensor H,W.

The repository transposes H,W before grid_sample, so grid x indexes original H.
Pooled level s has stride s and first receptive-field center (s-1)/2 pixels.
All grid_sample calls use align_corners=True. Scanner Z flips normalized Z.
"""

import torch
import torch.nn.functional as functional


def xyz_to_scanner_tensor(tensor):
    signs = tensor.new_tensor([1., 1., -1.])
    return tensor * signs[:, None] * signs[None, :]


def feature_grid(detector_grid, image_shape, feature_shape, stride):
    original = detector_grid.new_tensor(image_shape)
    shape = detector_grid.new_tensor(feature_shape)
    pixel = (detector_grid + 1) * (original - 1) / 2
    coordinate = (pixel - (stride - 1) / 2) / stride
    grid = 2 * coordinate / (shape - 1).clamp_min(1) - 1
    singleton = shape == 1
    valid = torch.where(singleton, coordinate.abs() <= .5, (coordinate >= 0) & (coordinate <= shape - 1)).all(-1)
    return torch.where(singleton, torch.zeros_like(grid), grid), valid


def sample_detector_features(features, grid):
    batch, views, channels = features.shape[:3]
    query_shape = grid.shape[2:-1]
    safe_grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    sampled = functional.grid_sample(
        features.flatten(0, 1).transpose(-1, -2),
        safe_grid.reshape(batch * views, -1, 1, 2), align_corners=True,
        mode='bilinear', padding_mode='zeros')
    return sampled[..., 0].reshape(batch, views, channels, *query_shape)


def aligned_anchor_features(features, projected, image_shape, stride, fusion):
    """Use the same pixel centers and validity as geometric cross-attention."""
    uv = projected['uv']
    grid, inside = feature_grid(uv, image_shape, features.shape[-2:], stride)
    valid = projected['valid'] & inside & torch.isfinite(uv).all(-1) & (uv.abs() <= 1).all(-1)
    values = sample_detector_features(features, grid)  # B,V,C,N
    mask = valid[:, :, None]
    if fusion == 'mean':
        return values.masked_fill(~mask, 0.).sum(1) / mask.sum(1).clamp_min(1)
    if fusion == 'max':
        return values.masked_fill(~mask, torch.finfo(values.dtype).min).max(1).values.masked_fill(~mask.any(1), 0.)
    raise ValueError(f'Unsupported aligned fusion: {fusion}')


class GeometryAdapter:
    def __init__(self, data):
        dtype = torch.float64 if data['angles'].dtype == torch.float64 else torch.float32
        self.angles = data['angles'].to(dtype)
        self.batch, self.views = self.angles.shape
        self.extent = data['volume_extent'].to(dtype)
        self.detector = self._per_view(data['detector_extent'].to(dtype), vector=True)
        self.dso = self._per_view(data['DSO'].to(dtype))
        self.dsd = self._per_view(data['DSD'].to(dtype))
        self.view_mask = data.get('view_mask', torch.ones_like(self.angles, dtype=torch.bool)).bool()
        self.view_mask = self.view_mask & torch.isfinite(self.angles)
        self.angles = torch.nan_to_num(self.angles)
        for key in ('scanner_origin_offset', 'detector_offset'):
            if key in data and torch.any(data[key] != 0):
                raise ValueError(f'Uncalibrated nonzero {key}; legacy projection does not implement this offset')
        for name, value in [('volume_extent', self.extent), ('detector_extent', self.detector), ('DSO', self.dso), ('DSD', self.dsd)]:
            if not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f'{name} must contain finite positive physical lengths')
        self.position_scale = torch.linalg.vector_norm(self.extent, dim=-1).clamp_min(1e-6)

    def _per_view(self, value, vector=False):
        if vector:
            return value[:, None].expand(-1, self.views, -1) if value.ndim == 2 else value
        return value.reshape(self.batch, 1).expand(-1, self.views) if value.numel() == self.batch else value

    def physical(self, points):
        return (points.to(self.extent.dtype) - .5) * self.extent[:, None] * points.new_tensor([1., 1., -1.])

    def position(self, points):
        return self.physical(points) / self.position_scale[:, None, None]

    def project(self, points):
        physical = self.physical(points)
        finite = torch.isfinite(physical).all(-1)
        physical = torch.nan_to_num(physical, nan=0., posinf=0., neginf=0.)
        cosine, sine = self.angles.cos()[:, :, None], self.angles.sin()[:, :, None]
        rotated_x = cosine * physical[:, None, :, 0] + sine * physical[:, None, :, 1]
        rotated_y = -sine * physical[:, None, :, 0] + cosine * physical[:, None, :, 1]
        rotated_z = physical[:, None, :, 2].expand_as(rotated_x)
        depth = self.dso[..., None] - rotated_x
        valid = finite[:, None] & (depth > 1e-4) & self.view_mask[..., None]
        denominator = torch.where(valid, depth, torch.ones_like(depth))
        scale = 2 * self.dsd / self.detector[..., 0]
        scale_other = 2 * self.dsd / self.detector[..., 1]
        scales = torch.stack([scale, scale_other], -1)[:, :, None]
        uv = torch.stack([rotated_z, rotated_y], -1) / denominator[..., None] * scales
        zeros = torch.zeros_like(rotated_x)
        rotation_x = torch.stack([cosine.expand_as(zeros), sine.expand_as(zeros), zeros], -1)
        rotation_y = torch.stack([-sine.expand_as(zeros), cosine.expand_as(zeros), zeros], -1)
        rotation_z = torch.stack([zeros, zeros, torch.ones_like(zeros)], -1)
        first = rotation_z / denominator[..., None] + rotated_z[..., None] * rotation_x / denominator[..., None].square()
        second = rotation_y / denominator[..., None] + rotated_y[..., None] * rotation_x / denominator[..., None].square()
        jacobian = torch.stack([first, second], -2) * scales[..., None]
        distance = torch.sqrt(depth.square() + rotated_y.square() + rotated_z.square()) / self.dso[..., None]
        return {'uv': uv.masked_fill(~valid[..., None], 2.),
                'jacobian': jacobian.masked_fill(~valid[..., None, None], 0.),
                'valid': valid, 'distance': distance.masked_fill(~valid, 0.)}

    def ray_encoding(self, projected, uv):
        extra = uv.ndim - 3
        shape = (self.batch, self.views) + (1,) * extra
        depth = self.dsd.reshape(shape)
        detector = self.detector.reshape(self.batch, self.views, *((1,) * (uv.ndim - 3)), 2)
        detector_y, detector_z = (uv * detector / 2).unbind(-1)[::-1]
        cosine, sine = self.angles.cos().reshape(shape), self.angles.sin().reshape(shape)
        direction = torch.stack([-depth * cosine - detector_y * sine,
                                  -depth * sine + detector_y * cosine, detector_z], -1)
        direction = functional.normalize(direction, dim=-1, eps=1e-6)
        distance = projected['distance']
        while distance.ndim < uv.ndim - 1:
            distance = distance.unsqueeze(-1)
        angular = torch.stack([sine.expand_as(uv[..., 0]), cosine.expand_as(uv[..., 0])], -1)
        return torch.cat([direction, distance.expand_as(uv[..., 0]).unsqueeze(-1), uv, angular], -1)
