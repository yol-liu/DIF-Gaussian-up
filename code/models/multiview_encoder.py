"""Fixed 2D maps, shared 3D anchor tokens, overlapping local 3D attention.

Cross masks are [B,anchor,V*scales*samples], True for usable observations.
Spatial neighborhoods read the whole token grid even when queries are chunked.
No K*K attention or 3D-to-2D map update is constructed.
"""

import itertools
import math

import torch
from torch import nn
import torch.nn.functional as functional

from geometry_adapter import GeometryAdapter, feature_grid, sample_detector_features
from models.anatomy import psd_tensor, structure_loss


def masked_max(features, valid):
    masked = features.masked_fill(~valid[..., None], torch.finfo(features.dtype).min)
    return masked.max(-2).values.masked_fill(~valid.any(-1)[..., None], 0.)


class LocalCrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, query, tokens, valid, bias=None):
        batch, count, width = query.shape
        query = self.query(query).reshape(batch, count, self.heads, self.head_dim)
        keys = self.key(tokens).reshape(batch, count, tokens.shape[-2], self.heads, self.head_dim)
        values = self.value(tokens).reshape_as(keys)
        logits = torch.einsum('bnhd,bnthd->bnht', query.float(), keys.float()) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias.float()
        weights = logits.masked_fill(~valid[:, :, None], -torch.finfo(logits.dtype).max).softmax(-1)
        weights = weights * valid[:, :, None]
        dropped = functional.dropout(weights, self.dropout, self.training)
        output = torch.einsum('bnht,bnthd->bnhd', dropped.to(values.dtype), values).reshape(batch, count, width)
        output = self.output(output).masked_fill(~valid.any(-1)[..., None], 0.)
        return output, weights


class SpatialLocalAttention(nn.Module):
    def __init__(self, width, heads, resolution, window, chunk_size, dropout):
        super().__init__()
        coordinates = torch.stack(torch.meshgrid(*([torch.arange(resolution)] * 3), indexing='ij'), -1).reshape(-1, 3)
        radius = window // 2
        offsets = torch.tensor(list(itertools.product(range(-radius, radius + 1), repeat=3)))
        neighbors = coordinates[:, None] + offsets[None]
        valid = ((neighbors >= 0) & (neighbors < resolution)).all(-1)
        safe = neighbors.clamp(0, resolution - 1)
        indices = safe[..., 0] * resolution ** 2 + safe[..., 1] * resolution + safe[..., 2]
        self.register_buffer('indices', indices, persistent=False)
        self.register_buffer('valid', valid, persistent=False)
        self.norm = nn.LayerNorm(width)
        self.attention = LocalCrossAttention(width, heads, dropout)
        self.relative_position = nn.Sequential(nn.Linear(3, width // 2), nn.SiLU(), nn.Linear(width // 2, heads))
        self.chunk_size = chunk_size
        self.resolution = resolution

    def forward(self, tokens, positions):
        normalized = self.norm(tokens)
        outputs = []
        for start in range(0, tokens.shape[1], self.chunk_size):
            indices = self.indices[start:start + self.chunk_size]
            neighbors = normalized[:, indices]
            delta = (positions[:, indices] - positions[:, start:start + self.chunk_size, None]) * self.resolution
            bias = self.relative_position(delta).permute(0, 1, 3, 2)
            valid = self.valid[start:start + self.chunk_size][None].expand(tokens.shape[0], -1, -1)
            update, _ = self.attention(normalized[:, start:start + self.chunk_size], neighbors, valid, bias)
            outputs.append(update)
        return torch.cat(outputs, 1)


class GeometryBlock(nn.Module):
    def __init__(self, width, heads, resolution, config):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.observation_norm = nn.LayerNorm(width)
        self.cross_attention = LocalCrossAttention(width, heads, config['dropout'])
        self.spatial_attention = SpatialLocalAttention(width, heads, resolution, config['spatial_window'],
                                                       config['anchor_chunk_size'], config['dropout'])
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Dropout(config['dropout']), nn.Linear(width * 2, width))
        self.dropout = nn.Dropout(config['dropout'])

    def finish(self, tokens, cross, positions):
        tokens = tokens + self.dropout(cross)
        tokens = tokens + self.dropout(self.spatial_attention(tokens, positions))
        return tokens + self.dropout(self.ffn(tokens))


class ViewUpdate(nn.Module):
    """3D-to-2D half of an alternating block: coarsest-map pixels read the anchors whose
    projections fall nearest to them, then attend within their own view. The change is
    upsampled into the finer maps so the next 3D block samples updated 2D evidence."""

    def __init__(self, width, heads, scales, neighbors, radius, dropout):
        super().__init__()
        self.neighbors = neighbors
        self.radius = radius
        self.coarse = max(scales)
        self.pixel_norm = nn.LayerNorm(width)
        self.anchor_norm = nn.LayerNorm(width)
        self.relative = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.cross_attention = LocalCrossAttention(width, heads, dropout)
        self.position = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.propagate = nn.ModuleDict({str(scale): nn.Conv2d(width, width, 1) for scale in scales if scale != self.coarse})

    def forward(self, maps, tokens, projected, image_shape):
        feature_map = maps[self.coarse]
        batch, views, width, height, breadth = feature_map.shape
        # detector-grid coordinates of every coarse pixel center (inverse of feature_grid)
        step = feature_map.new_tensor([2 * self.coarse / (size - 1) for size in image_shape])
        rows = torch.arange(height, device=feature_map.device, dtype=feature_map.dtype)
        cols = torch.arange(breadth, device=feature_map.device, dtype=feature_map.dtype)
        centers = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), -1).reshape(-1, 2)
        pixel_uv = (centers * self.coarse + (self.coarse - 1) / 2) * 2 / feature_map.new_tensor([size - 1 for size in image_shape]) - 1
        uv = projected['uv'].to(feature_map.dtype)
        usable = projected['valid'] & (uv.abs() <= 1).all(-1)
        delta = (pixel_uv[None, None, :, None] - uv[:, :, None]) / step
        distance = delta.square().sum(-1).masked_fill(~usable[:, :, None], float('inf'))
        count = min(self.neighbors, distance.shape[-1])
        nearest, indices = distance.topk(count, dim=-1, largest=False)
        valid = nearest <= self.radius ** 2
        anchors = self.anchor_norm(tokens)
        flat = indices.reshape(batch, -1)
        gathered = torch.gather(anchors, 1, flat[..., None].expand(-1, -1, width)).reshape(*indices.shape, width)
        offsets = torch.gather(delta, 3, indices[..., None].expand(-1, -1, -1, -1, 2))
        depth = torch.gather(projected['distance'].to(feature_map.dtype)[:, :, None].expand(-1, -1, indices.shape[2], -1), 3, indices)
        keys = gathered + self.relative(torch.cat([offsets, depth[..., None]], -1))
        pixels = feature_map.flatten(3).transpose(-1, -2)
        update, _ = self.cross_attention(self.pixel_norm(pixels).flatten(0, 1), keys.flatten(0, 1), valid.flatten(0, 1))
        pixels = pixels.flatten(0, 1) + update
        attended = self.self_norm(pixels) + self.position(pixel_uv)
        pixels = pixels + self.self_attention(attended, attended, attended, need_weights=False)[0]
        pixels = pixels + self.ffn(pixels)
        updated = pixels.reshape(batch, views, height, breadth, width).permute(0, 1, 4, 2, 3)
        change = (updated - feature_map).flatten(0, 1)
        maps = {**maps, self.coarse: updated}
        for scale, projection in self.propagate.items():
            target = maps[int(scale)]
            upsampled = functional.interpolate(change, size=target.shape[-2:], mode='bilinear', align_corners=False)
            maps[int(scale)] = target + projection(upsampled).reshape(target.shape)
        return maps, valid.any(-1).float().mean().detach()


class MultiViewGeometryEncoder(nn.Module):
    def __init__(self, ds_channels, resolution, config):
        super().__init__()
        self.config = config
        self.resolution = resolution
        self.scales = config['feature_scales']
        self.chunk_size = config['anchor_chunk_size']
        width = config['d_model']
        channels = {4: 256, 8: 512, 16: ds_channels}
        self.feature_projections = nn.ModuleDict({str(scale): nn.Conv2d(channels[scale], width, 1) for scale in self.scales})
        self.base_projection = nn.Linear(ds_channels, width)
        self.position_encoding = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.ray_encoding = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.scale_encoding = nn.Parameter(torch.empty(len(self.scales), width))
        nn.init.normal_(self.scale_encoding, std=.02)
        self.blocks = nn.ModuleList([GeometryBlock(width, config['num_heads'], resolution, config)
                                     for _ in range(config['num_blocks'])])
        # alternating 2D/3D: after every block but the last, the 2D maps read the 3D tokens
        self.view_updates = nn.ModuleList([
            ViewUpdate(width, config['num_heads'], self.scales, config['view_update_neighbors'],
                       config['view_update_radius_pixels'], config['dropout'])
            for _ in range(config['num_blocks'] - 1)]) if config['view_update'] else None
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, ds_channels)
        # residual_base: output = masked-max base + gated update (starts at B0). Without it the
        # transformer output alone is the anchor feature, so no gate is needed.
        self.residual_base = config['residual_base']
        if self.residual_base:
            self.residual_gate = nn.Parameter(torch.tensor(float(config['residual_init'])))
        anatomy = config['anatomy']
        self.anatomy_enabled = anatomy['enabled']
        self.guided_sampling = anatomy['guided_sampling']
        if self.anatomy_enabled:
            self.structure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 6))
            nn.init.normal_(self.structure_head[-1].weight, std=.02)
            nn.init.zeros_(self.structure_head[-1].bias)
        coordinates = torch.stack(torch.meshgrid(*([torch.arange(resolution) / resolution] * 3), indexing='ij'), -1).reshape(-1, 3)
        self.register_buffer('anchor_layout', coordinates, persistent=False)

    def _offsets(self, projected, structure, image_shape, scale, progress):
        uv = projected['uv']
        step = uv.new_tensor([2 * scale / (size - 1) for size in image_shape])
        pattern = uv.new_tensor([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        fixed = (pattern * self.config['fixed_radius_pixels'] * step).expand(*uv.shape[:-1], 5, 2)
        if structure is None or not self.guided_sampling:
            return fixed, uv.new_tensor(0.)
        anatomy = self.config['anatomy']
        with torch.autocast(device_type=uv.device.type, enabled=False):
            structure = structure.float()
            identity = torch.eye(3, device=uv.device)
            support = anatomy['length_mm'] ** 2 * torch.linalg.inv(identity + anatomy['kappa'] * structure)
            support = support + anatomy['sigma_floor_mm'] ** 2 * identity
            jacobian = projected['jacobian'].float() / step.float()[None, None, None, :, None]
            covariance = jacobian @ support[:, None] @ jacobian.transpose(-1, -2)
            covariance = covariance + self.config['footprint_min_pixels'] ** 2 * torch.eye(2, device=uv.device)
            factor = torch.linalg.cholesky(covariance)
            lengths = torch.linalg.vector_norm(factor, dim=-2, keepdim=True).clamp_min(1e-8)
            factor = factor * lengths.clamp(self.config['footprint_min_pixels'], self.config['footprint_max_pixels']) / lengths
            first, second = factor[..., 0], factor[..., 1]
            guided = torch.stack([torch.zeros_like(first), first, -first, second, -second], -2) * step
        return fixed.lerp(guided.to(fixed.dtype), progress), support.diagonal(dim1=-2, dim2=-1).sum(-1).mean().detach()

    def _observations(self, maps, geometry, points, structure, image_shape, progress):
        projected = geometry.project(points)
        shared = self.shared_observations(geometry, points, structure, progress) if self.config['sampling_mode'] == 'shared3d' else None
        all_tokens, all_valid, rates, radii = [], [], [], []
        for index, scale in enumerate(self.scales):
            if shared is None:
                offsets, _ = self._offsets(projected, structure, image_shape, scale, progress)
                detector_grid = projected['uv'][..., None, :] + offsets
                sample_valid = projected['valid'][..., None]
                ray_projected = projected
            else:
                detector_grid = shared['uv']
                offsets = detector_grid - projected['uv'][..., None, :]
                sample_valid = shared['valid']
                ray_projected = shared
            grid, inside_level = feature_grid(detector_grid, image_shape, maps[scale].shape[-2:], scale)
            valid = sample_valid & inside_level
            valid = valid & torch.isfinite(detector_grid).all(-1) & (detector_grid.abs() <= 1).all(-1)
            features = sample_detector_features(maps[scale], grid).permute(0, 3, 1, 4, 2)
            ray = geometry.ray_encoding(ray_projected, detector_grid).permute(0, 2, 1, 3, 4)
            tokens = features + self.ray_encoding(ray.to(features.dtype)) + self.scale_encoding[index]
            all_tokens.append(tokens)
            all_valid.append(valid.permute(0, 2, 1, 3))
            rates.append(valid.float().mean().detach())
            pixel_step = offsets.new_tensor([2 * scale / (size - 1) for size in image_shape])
            radii.append(torch.linalg.vector_norm(offsets[..., 1:, :] / pixel_step, dim=-1).detach())
        tokens = torch.stack(all_tokens, 3).flatten(2, 4)
        valid = torch.stack(all_valid, 3).flatten(2, 4)
        return tokens, valid, torch.stack(rates), torch.stack([radius.min() for radius in radii]).min(), torch.stack([radius.max() for radius in radii]).max()

    def shared_observations(self, geometry, points, structure=None, progress=1.):
        """Five shared scanner-space points: center plus a regular tetrahedron.

        The tetrahedron has zero mean and isotropic second moment. This keeps the
        observation count identical to the five-point detector stencil. No separate
        per-view or per-scale offsets/clamps can break the 3D correspondence.
        """
        pattern = points.new_tensor([[0., 0., 0.], [1., 1., 1.], [1., -1., -1.],
                                     [-1., 1., -1.], [-1., -1., 1.]]) / math.sqrt(3.)
        offsets = (pattern * self.config['shared_radius_mm']).expand(*points.shape[:2], 5, 3)
        if structure is not None and self.guided_sampling:
            anatomy = self.config['anatomy']
            with torch.autocast(device_type=points.device.type, enabled=False):
                identity = torch.eye(3, device=points.device)
                support = anatomy['length_mm'] ** 2 * torch.linalg.inv(identity + anatomy['kappa'] * structure.float())
                support = support + anatomy['sigma_floor_mm'] ** 2 * identity
                factor = torch.linalg.cholesky(support)
                guided = torch.einsum('bnij,kj->bnki', factor, pattern.float())
            offsets = offsets.lerp(guided.to(offsets.dtype), progress)
        # GeometryAdapter.physical flips Z; offsets are in scanner mm.
        neighbors = points[:, :, None] + offsets * points.new_tensor([1., 1., -1.]) / geometry.extent[:, None, None]
        batch, count, samples = neighbors.shape[:3]
        projected = geometry.project(neighbors.flatten(1, 2))
        result = {key: value.reshape(batch, geometry.views, count, samples, *value.shape[3:])
                  for key, value in projected.items()}
        in_volume = ((neighbors >= 0) & (neighbors <= 1)).all(-1)
        result['valid'] = result['valid'] & in_volume[:, None]
        result['points'] = neighbors
        return result

    def forward(self, encoder_output, base_features, data):
        anchors = data['points_gs']
        if anchors.shape[1:] != self.anchor_layout.shape or not torch.allclose(anchors, self.anchor_layout[None].to(anchors), atol=1e-6, rtol=0):
            raise ValueError('Transformer requires XYZ anchors in np.mgrid order at index/gs_res')
        geometry = GeometryAdapter(data)
        maps = {}
        for scale in self.scales:
            feature_map = encoder_output['pyramid'][scale]
            projected = self.feature_projections[str(scale)](feature_map.flatten(0, 1))
            maps[scale] = projected.reshape(*feature_map.shape[:2], *projected.shape[1:])
        positions = geometry.position(anchors)
        tokens = self.base_projection(base_features.transpose(1, 2)) + self.position_encoding(positions.to(base_features.dtype))
        structure = None
        diagnostics = {}
        anatomy = self.config['anatomy']
        warmup = anatomy['warmup_steps']
        progress = 1. if not self.training or warmup == 0 else min(1., max(0., float(data.get('optimizer_step', 0)) / warmup))
        for block_index, block in enumerate(self.blocks):
            cross_chunks, rates, concentration, minima, maxima = [], [], [], [], []
            for start in range(0, anchors.shape[1], self.chunk_size):
                stop = start + self.chunk_size
                observations, valid, rate, minimum, maximum = self._observations(
                    maps, geometry, anchors[:, start:stop], structure[:, start:stop] if structure is not None else None,
                    encoder_output['feats'].shape[-2:], progress)
                cross, weights = block.cross_attention(block.query_norm(tokens[:, start:stop]), block.observation_norm(observations), valid)
                cross_chunks.append(cross)
                rates.append(rate)
                minima.append(minimum)
                maxima.append(maximum)
                view_mass = weights.reshape(*weights.shape[:3], geometry.views, len(self.scales), 5).sum((-1, -2))
                concentration.append(view_mass.max(-1).values.mean().detach())
            tokens = block.finish(tokens, torch.cat(cross_chunks, 1), positions)
            if self.view_updates is not None and block_index < len(self.view_updates):
                maps, covered = self.view_updates[block_index](
                    maps, tokens, geometry.project(anchors), encoder_output['feats'].shape[-2:])
                diagnostics[f'block{block_index}/view_update_pixel_coverage'] = covered
            diagnostics[f'block{block_index}/attention_max_view_mass'] = torch.stack(concentration).mean()
            for index, scale in enumerate(self.scales):
                diagnostics[f'block{block_index}/valid_scale{scale}'] = torch.stack(rates)[:, index].mean()
            diagnostics[f'block{block_index}/footprint_min_pixels'] = torch.stack(minima).min()
            diagnostics[f'block{block_index}/footprint_max_pixels'] = torch.stack(maxima).max()
            if block_index == 0 and self.anatomy_enabled:
                with torch.autocast(device_type=tokens.device.type, enabled=False):
                    raw = self.structure_head(tokens.float())
                    structure = psd_tensor(raw)
                    structure = structure / structure.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-8)
        output = self.output_projection(self.output_norm(tokens)).transpose(1, 2)
        if self.residual_base:
            delta = self.residual_gate * output
            diagnostics['residual_base_norm_ratio'] = (delta.float().norm() / base_features.float().norm().clamp_min(1e-8)).detach()
            output = base_features + delta
        diagnostics['anchor_feature_std'] = output.float().std().detach()
        diagnostics['token_variance'] = tokens.float().var(dim=1, unbiased=False).mean().detach()
        diagnostics['anatomy_sampling_fraction'] = tokens.new_tensor(progress if self.guided_sampling else 0.)
        auxiliary = {'diagnostics': diagnostics}
        if self.training and self.anatomy_enabled and data.get('auxiliary_supervision', True):
            if 'structure_target' not in data or 'structure_confidence' not in data:
                raise ValueError('Anatomy training requires scanner-frame GT structure_target/confidence')
            auxiliary['structure_loss'] = structure_loss(structure, data['structure_target'], data['structure_confidence'])
        return output, auxiliary


class GeometryPointFusion(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        width = config['d_model']
        self.feature_projection = nn.Conv2d(channels, width, 1)
        self.query_projection = nn.Linear(channels + 3, width)
        self.ray_encoding = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = LocalCrossAttention(width, config['num_heads'], config['dropout'])
        self.output_projection = nn.Linear(width, channels)
        self.gate = nn.Parameter(torch.tensor(float(config['residual_init'])))
        self.chunk_size = config['point_chunk_size']

    def prepare(self, feature_maps):
        mapped = self.feature_projection(feature_maps.flatten(0, 1))
        return mapped.reshape(*feature_maps.shape[:2], *mapped.shape[1:])

    def forward(self, maps, projected_maps, interpolated, data):
        geometry = GeometryAdapter(data)
        outputs = []
        ratios, concentrations = [], []
        for start in range(0, data['points'].shape[1], self.chunk_size):
            points = data['points'][:, start:start + self.chunk_size]
            shared = interpolated[:, :, start:start + self.chunk_size].transpose(1, 2)
            projected = geometry.project(points)
            grid = projected['uv']
            valid = projected['valid'] & (grid.abs() <= 1).all(-1)
            features = sample_detector_features(maps, grid).permute(0, 3, 1, 2)
            base = masked_max(features, valid.transpose(1, 2))
            tokens = sample_detector_features(projected_maps, grid).permute(0, 3, 1, 2)
            tokens = tokens + self.ray_encoding(geometry.ray_encoding(projected, grid).transpose(1, 2).to(tokens.dtype))
            query = self.query_projection(torch.cat([shared, geometry.position(points).to(shared.dtype)], -1))
            update, weights = self.attention(self.query_norm(query), self.token_norm(tokens), valid.transpose(1, 2))
            delta = self.gate * self.output_projection(update)
            delta = delta.masked_fill(~valid.any(1)[..., None], 0.)
            outputs.append((base + delta).transpose(1, 2))
            ratios.append((delta.float().norm() / base.float().norm().clamp_min(1e-8)).detach())
            concentrations.append(weights.max(-1).values.mean().detach())
        self.last_diagnostics = {'point/residual_base_norm_ratio': torch.stack(ratios).mean(),
                                 'point/attention_max_view_mass': torch.stack(concentrations).mean()}
        return torch.cat(outputs, -1)
