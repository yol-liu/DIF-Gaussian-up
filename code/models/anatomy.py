import torch
from torch import nn
import torch.nn.functional as functional

from models.utils import query_view_feats
from geometry_adapter import GeometryAdapter


def psd_tensor(parameters):
    matrix = parameters.new_zeros(*parameters.shape[:-1], 3, 3)
    row, column = torch.tril_indices(3, 3, device=parameters.device)
    matrix[..., row, column] = parameters
    diagonal = torch.arange(3, device=parameters.device)
    matrix[..., diagonal, diagonal] = functional.softplus(matrix[..., diagonal, diagonal]) + 1e-4
    return matrix @ matrix.transpose(-1, -2)


def physical_projection(points, angles, geometry):
    projected = GeometryAdapter(dict(geometry, angles=angles)).project(points)
    if not projected['valid'].all():
        raise ValueError('Support center must be in front of the source plane')
    return projected['uv'], projected['jacobian']


def structure_loss(predicted, target, confidence):
    error = (predicted - target).square().sum((-1, -2))
    return (error * confidence).sum() / confidence.sum().clamp_min(1e-8)


class AnatomySupportEncoder(nn.Module):
    def __init__(self, channels, full_channels, resolution, config):
        super().__init__()
        self.arm = config.get('arm', 'G0')
        if self.arm not in ('G0', 'Gfree', 'Ggeo', 'Gaux', 'Ganat'):
            raise ValueError(f'Unknown anatomy arm: {self.arm}')
        self.resolution = resolution
        hidden = config.get('hidden_dim', 32)
        self.length_mm = float(config.get('length_mm', 3.2))
        self.kappa = float(config.get('kappa', 4.))
        if not 0 < self.length_mm <= 20 or not 0 <= self.kappa <= 20:
            raise ValueError('Support length/kappa exceed the declared finite bounds')
        self.context = nn.Sequential(nn.Conv3d(channels, hidden, 1), nn.SiLU(),
                                     nn.Conv3d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.structure = nn.Conv1d(hidden, 6, 1)
        self.support = nn.Conv1d(hidden, 6, 1)
        self.offsets = nn.Conv1d(hidden, 8, 1)
        self.level_projections = nn.ModuleList([nn.Conv1d(full_channels, hidden, 1),
                                               nn.Conv1d(channels, hidden, 1)])
        self.residual = nn.Sequential(nn.Conv1d(3 * hidden, hidden, 1), nn.SiLU(),
                                      nn.Conv1d(hidden, channels, 1))
        for head in (self.structure, self.support, self.offsets, self.residual[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, center_features, encoder_output, data):
        batch, channels, anchors = center_features.shape
        if anchors != self.resolution ** 3:
            raise ValueError('Anatomy context requires the configured regular anchor grid')
        context = self.context(center_features.reshape(batch, channels, *([self.resolution] * 3))).flatten(2)
        tensor = psd_tensor(self.structure(context).transpose(1, 2))
        tensor = tensor / tensor.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        identity = torch.eye(3, device=context.device, dtype=context.dtype)
        if self.arm == 'Ganat':
            support = self.length_mm ** 2 * torch.linalg.inv(identity + self.kappa * tensor)
        else:
            raw = psd_tensor(self.support(context).tanh().transpose(1, 2))
            support = self.length_mm ** 2 * raw / raw.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        support = support + 1e-4 * identity
        geometry = {key: data[key] for key in ('volume_extent', 'detector_extent', 'DSO', 'DSD')}
        uv, jacobian = physical_projection(data['points_gs'], data['angles'], geometry)
        footprint = jacobian @ support[:, None] @ jacobian.transpose(-1, -2)
        detector_pixels = data['detector_resolution'][:, None, None]
        floor = torch.diag_embed((0.1 * 2 / detector_pixels).square()).expand_as(footprint)
        factor = torch.linalg.cholesky(footprint + floor)
        offsets = torch.stack([torch.zeros_like(uv), factor[..., 0], -factor[..., 0],
                               factor[..., 1], -factor[..., 1]], -2)
        if self.arm == 'Gfree':
            free = self.offsets(context).transpose(1, 2).reshape(batch, anchors, 4, 2).tanh()
            free = free[:, None].expand(-1, uv.shape[1], -1, -1, -1) * 0.05
            offsets = torch.cat([torch.zeros_like(uv).unsqueeze(-2), free], -2)
        if self.arm == 'G0':
            offsets = offsets[..., :1, :] * 0
        grid = uv.unsqueeze(-2) + offsets
        valid = (torch.isfinite(grid) & (grid.abs() <= 1)).all(-1)
        tokens = []
        for level, feature_map in enumerate([encoder_output['feats'], encoder_output['feats_ds']]):
            mapped = grid
            if level == 1:
                original_size = grid.new_tensor(encoder_output['feats'].shape[-2:])
                feature_size = grid.new_tensor(feature_map.shape[-2:])
                pixel = (grid + 1) * (original_size - 1) / 2
                mapped = 2 * ((pixel - 7.5) / 16) / (feature_size - 1).clamp_min(1) - 1
                mapped = torch.where(feature_size > 1, mapped, torch.zeros_like(mapped))
            level_valid = valid & (mapped.abs() <= 1).all(-1)
            samples = query_view_feats(feature_map, mapped.flatten(2, 3), fusion=None)
            samples = samples.reshape(batch, feature_map.shape[2], anchors, grid.shape[-2], uv.shape[1])
            weights = level_valid.permute(0, 2, 3, 1).unsqueeze(1).to(samples.dtype)
            pooled = (samples * weights).sum((-1, -2)) / weights.sum((-1, -2)).clamp_min(1)
            tokens.append(self.level_projections[level](pooled))
        refined = center_features + self.residual(torch.cat([context] + tokens, 1))
        return refined, tensor
