"""Joint dense-observation variational model. D reaches the sparse decoder only via z."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch import nn

from models.model import DIF_Gaussian
from models.latent_np import BoundedGaussianHead, LocalLatentGrid, FeatureFiLM, reparameterize
from models.gt_encoder import GTEncoder
from probabilistic_config import resolve_probability, configure_normalization, ALIASES, MODES


def context_inputs(data):
    # Strip reconstruction GT and privileged observations at the context boundary.
    return {key: value for key, value in data.items()
            if not key.startswith(('dense_', 'teacher_', 'gt_')) and key != 'points_gt'}


def dense_inputs(data):
    required = ('projs', 'angles', 'view_ids', 'points_proj', 'points_gs_proj')
    if any('dense_' + key not in data for key in required):
        raise ValueError('dense_np posterior requires real dense_* projections/geometry/view IDs')
    result = context_inputs(data)
    # Dense posterior never reads CT-derived auxiliary targets, including anatomy labels.
    result = {key: value for key, value in result.items() if not key.startswith('structure_')}
    result['auxiliary_supervision'] = False
    result.pop('view_mask', None)
    for key in required + ('view_mask',):
        if 'dense_' + key in data:
            result[key] = data['dense_' + key]
    return result


def observation_descriptor(data):
    """Permutation invariant unique-ID count + circular coverage/moments (7 numbers)."""
    angles = data['angles']
    masks = data.get('view_mask', torch.ones_like(angles, dtype=torch.bool)).bool()
    ids = data.get('view_ids')
    rows = []
    for b in range(angles.shape[0]):
        selected = angles[b, masks[b]]
        if ids is not None:
            selected_ids = ids[b, masks[b]]
            if selected_ids.unique().numel() != selected_ids.numel():
                raise ValueError('Repeated raw view IDs are not independent observations')
        if selected.numel() == 0:
            rows.append(angles.new_zeros(7))
            continue
        ordered = selected.remainder(2 * torch.pi).sort().values
        gaps = torch.cat([ordered[1:] - ordered[:-1], ordered[:1] + 2 * torch.pi - ordered[-1:]])
        coverage = 1 - gaps.max() / (2 * torch.pi)
        rows.append(torch.stack([angles.new_tensor(float(len(selected))).log1p(), coverage,
                                 selected.sin().mean(), selected.cos().mean(),
                                 (2 * selected).sin().mean(), (2 * selected).cos().mean(),
                                 torch.sqrt(selected.sin().mean().square() + selected.cos().mean().square())]))
    return torch.stack(rows)


@dataclass
class ContextState:
    features: Dict[str, Any]
    data: Dict[str, Any]
    pooled: torch.Tensor
    grid: Optional[torch.Tensor] = None


@dataclass
class ProbabilityState:
    context: ContextState
    features: Dict[str, Any]
    z_g: torch.Tensor
    z_grid: Optional[torch.Tensor]
    kl_info: Dict[str, torch.Tensor]


class DenseProbabilisticDIFGaussian(DIF_Gaussian):
    def init(self):
        super().init()
        self.probability = resolve_probability(self.cfg['probabilistic'])
        self.use_latent = True
        self.mode = self.probability['mode']
        latent = self.probability['latent']
        self.use_local = latent['use_local']
        self.global_dim = latent['global_dim']
        self.local_dim = latent['local_dim']
        self.grid_res = latent['grid_res']
        channels = self.image_encoder.ds_ch
        head_options = {key: latent[key] for key in ('sigma_min', 'sigma_max', 'initial_sigma')}
        self.prior_head = BoundedGaussianHead(channels + 7, self.global_dim, **head_options)
        self.posterior_head = BoundedGaussianHead(channels + 7 + (32 if self.mode == 'gt_cvae' else 0),
                                                  self.global_dim, **head_options)
        if self.mode == 'gt_cvae':
            self.gt_encoder = GTEncoder(out_ch=32)
        if self.use_local:
            # Pool/interpolation geometry reused; old residual heads are not part of this model.
            self.local_grid = LocalLatentGrid(channels, self.global_dim, self.local_dim, grid_res=self.grid_res)
            del self.local_grid.prior_net, self.local_grid.delta_net
            self.local_grid.coordinate_convention = 'cell_centers'
            self.local_prior_head = BoundedGaussianHead(channels + self.global_dim, self.local_dim, **head_options)
            self.local_posterior_head = BoundedGaussianHead(
                channels + self.global_dim + (32 if self.mode == 'gt_cvae' else 0), self.local_dim, **head_options)
        self.feature_film = FeatureFiLM(self.image_encoder.out_ch,
                                       self.global_dim + (self.local_dim if self.use_local else 0))
        nn.init.normal_(self.feature_film.net[-1].weight, std=self.probability['film_init_std'])
        configure_normalization(self, self.probability['normalization'])
        self.reference = None

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'probability') and self.probability['normalization'] == 'frozen_bn':
            configure_normalization(self, 'frozen_bn')
        if getattr(self, 'reference', None) is not None:
            # Reference and compatible decoder/backbone stay fixed during prior-only fitting.
            for child in self.children():
                child.eval()
            self.prior_head.train(mode)
            if self.use_local:
                self.local_prior_head.train(mode)
        return self

    def encode_context(self, data, prepare=True):
        clean = context_inputs(data)
        if prepare:
            clean = self.prepare_views(clean)
        features = self.encode_projs(clean)
        pooled = torch.cat([features['anchor_feats'].mean(-1), observation_descriptor(clean)], -1)
        grid = self.local_grid.pool_anchors_to_grid(features['anchor_feats'], clean['points_gs']) if self.use_local else None
        return ContextState(features, clean, pooled, grid)

    def infer_prior(self, context):
        return self.prior_head(context.pooled)

    def record_feature_gradient(self, name, feature):
        if feature.requires_grad and hasattr(self, 'probability_gradients'):
            def record(gradient):
                self.probability_gradients[name] = gradient.detach().float().norm()
            feature.register_hook(record)

    def infer_posterior(self, data, context):
        if self.mode == 'dense_np':
            dense = self.encode_context(dense_inputs(data), prepare=False)
            pooled, grid = dense.pooled, dense.grid
            self.record_feature_gradient('dense_conditioning', pooled)
        else:
            if data.get('gt_volume') is None:
                raise ValueError('gt_cvae posterior requires gt_volume')
            gt = self.gt_encoder(data['gt_volume'], context.data['points_gs'])
            pooled = torch.cat([context.pooled, gt['gt_global_feat']], -1)
            grid = None
            if self.use_local:
                gt_grid = self.local_grid.pool_anchors_to_grid(gt['gt_anchor_feat'], context.data['points_gs'])
                grid = torch.cat([context.grid, gt_grid], 1)
        return self.posterior_head(pooled), grid

    def local_stats(self, head, grid, z_g):
        b, c, r, _, _ = grid.shape
        cells = grid.flatten(2).transpose(1, 2)
        mu, lv = head(torch.cat([cells, z_g[:, None].expand(-1, cells.shape[1], -1)], -1))
        return mu.transpose(1, 2).reshape(b, -1, r, r, r), lv.transpose(1, 2).reshape(b, -1, r, r, r)

    def sample_state(self, context, prior, posterior=None, posterior_grid=None, mean=False, generator=None):
        mu, lv = posterior if posterior is not None else prior
        z_g = mu if mean else reparameterize(mu, lv, generator)
        info = dict(mu_p_g=prior[0], logvar_p_g=prior[1])
        if posterior is not None:
            info.update(mu_q_g=mu, logvar_q_g=lv)
        z_grid, conditioning = None, z_g
        if self.use_local:
            p_l = self.local_stats(self.local_prior_head, context.grid, z_g)
            q_l = None
            if posterior is not None:
                head = self.reference.local_posterior_head if self.reference is not None else self.local_posterior_head
                q_l = self.local_stats(head, posterior_grid, z_g)
                if self.reference is not None:
                    q_l = tuple(value.detach() for value in q_l)
                info.update(mu_q_l=q_l[0], logvar_q_l=q_l[1])
            info.update(mu_p_l=p_l[0], logvar_p_l=p_l[1])
            mu_l, lv_l = q_l if q_l is not None else p_l
            z_grid = mu_l if mean else reparameterize(mu_l, lv_l, generator)
            local = self.local_grid.interpolate_to_anchors(z_grid, context.data['points_gs'])
            conditioning = torch.cat([z_g[:, None].expand(-1, local.shape[1], -1), local], -1)
        features = dict(context.features)
        features['gs_params'] = dict(context.features['gs_params'])
        features['gs_params']['feats'] = self.feature_film(context.features['gs_params']['feats'], conditioning)
        if 'diagnostics' in features:
            features['diagnostics'] = dict(features['diagnostics'])
        return ProbabilityState(context, features, z_g, z_grid, info)

    def state_with_global(self, state, z_g):
        """Explicit diagnostic intervention, holding sparse features and local grid fixed."""
        conditioning = z_g
        if self.use_local:
            local = self.local_grid.interpolate_to_anchors(state.z_grid, state.context.data['points_gs'])
            conditioning = torch.cat([z_g[:, None].expand(-1, local.shape[1], -1), local], -1)
        features = dict(state.features, gs_params=dict(state.features['gs_params']))
        features['gs_params']['feats'] = self.feature_film(state.context.features['gs_params']['feats'], conditioning)
        return ProbabilityState(state.context, features, z_g, state.z_grid, state.kl_info)

    def decode_points(self, state, query_chunk):
        # Query-only overrides. C's masks/calibration/features always come from the fixed state.
        data = dict(state.context.data)
        for key in self.registered_point_keys:
            if key in query_chunk:
                data[key] = query_chunk[key]
        return self.forward_points(state.features, data)

    def decode_chunked(self, state, chunk_size):
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        data = state.context.data
        n = data['points'].shape[1]
        return torch.cat([self.decode_points(state, {key: data[key][..., left:left+chunk_size, :]
                                                     for key in self.registered_point_keys})
                          for left in range(0, n, chunk_size)], 2)

    def validate_pair(self, data, strict=True):
        dense = dense_inputs(data)
        for b in range(data['projs'].shape[0]):
            cm = data.get('view_mask', torch.ones_like(data['angles'], dtype=torch.bool))[b].bool()
            dm = dense.get('view_mask', torch.ones_like(dense['angles'], dtype=torch.bool))[b].bool()
            c, d = data['view_ids'][b, cm], dense['view_ids'][b, dm]
            if c.unique().numel() != c.numel() or d.unique().numel() != d.numel():
                raise ValueError('Repeated raw view IDs')
            if not torch.isin(c, d).all() or (strict and not len(c) < len(d)):
                raise ValueError('Main dense_np training requires C strictly contained in D')
            positions = (c[:, None] == d[None]).long().argmax(1)
            if not torch.equal(data['projs'][b, cm], dense['projs'][b, dm][positions]) or not torch.allclose(
                    data['angles'][b, cm], dense['angles'][b, dm][positions]):
                raise ValueError('C/D shared views must have identical pixels and angles')

    def forward(self, data, is_eval=False, eval_npoint=100000, gt_volume=None,
                latent_mode=None, num_samples=None, generator=None, oracle=False,
                return_samples=False, return_debug=False, optimizer_step=None):
        if gt_volume is not None:
            data = dict(data, gt_volume=gt_volume)
        if optimizer_step is not None:
            data = dict(data, optimizer_step=optimizer_step)
        training = not is_eval
        if training:
            self.probability_gradients = {}
        if training:
            mode = ('dense' if self.mode == 'dense_np' else 'gt') + '_posterior_sample'
            if self.probability['mean_code']:
                mode = 'dense_posterior_mean'
            count = num_samples if num_samples is not None else self.probability['train_samples']
            if self.mode == 'dense_np':
                self.validate_pair(data)
        else:
            mode = ALIASES.get(latent_mode, latent_mode) or self.probability['evaluation']['estimator']
            if mode not in MODES:
                raise ValueError(f'Unknown latent mode {mode}')
            if 'posterior' in mode and not oracle:
                raise ValueError('Posterior deployment request requires explicit oracle=True')
            count = num_samples if num_samples is not None else self.probability['evaluation']['num_samples']
            if mode != 'prior_mc_mean':
                count = 1
        if not isinstance(count, int) or count < 1:
            raise ValueError('num_samples must be a positive integer')
        use_posterior = 'posterior' in mode
        if use_posterior and not mode.startswith('dense' if self.mode == 'dense_np' else 'gt'):
            raise ValueError('Posterior mode does not match the trained model')
        if self.probability['mean_code'] and mode not in ('prior_latent_mean', 'dense_posterior_mean'):
            raise ValueError('Deterministic mean-code control only supports latent-mean estimators')
        context = self.encode_context(data)
        if training:
            self.record_feature_gradient('sparse_conditioning', context.pooled)
        prior = self.infer_prior(context)
        posterior, posterior_grid = None, None
        if use_posterior:
            if self.probability['objective'] == 'frozen_reference':
                if self.reference is None:
                    raise ValueError('BLOCKED_REFERENCE: attach a validated trained dense_np checkpoint')
                with torch.no_grad():
                    posterior, posterior_grid = self.reference.infer_posterior(data, context)
            else:
                posterior, posterior_grid = self.infer_posterior(data, context)
        if generator is None and is_eval:
            generator = torch.Generator(device=prior[0].device).manual_seed(self.probability['evaluation']['seed'])
        samples, infos, states = [], [], []
        mean_prediction = m2 = None
        for index in range(count):
            state = self.sample_state(context, prior, posterior, posterior_grid,
                                      mean=mode.endswith('mean') and mode != 'prior_mc_mean', generator=generator)
            prediction = self.decode_points(state, context.data) if training else self.decode_chunked(state, eval_npoint)
            if training or return_samples:
                samples.append(prediction)
            if return_debug:
                states.append(state)
            infos.append(state.kl_info)
            if mean_prediction is None:
                mean_prediction, m2 = prediction, torch.zeros_like(prediction)
            else:
                delta = prediction - mean_prediction
                mean_prediction = mean_prediction + delta / (index + 1)
                m2 = m2 + delta * (prediction - mean_prediction)
        variance = m2 / (count - 1) if count > 1 else None
        rec = self.probability['reconstruction']
        observation_var = rec['observation_sigma'] ** 2 if rec['likelihood'] == 'gaussian' else None
        result = {'points_pred': mean_prediction, 'predictive_var_latent': variance,
                  'observation_variance': observation_var,
                  'predictive_var_total': variance + observation_var if variance is not None and observation_var is not None else None,
                  'latent_mode': mode, 'oracle': use_posterior, 'num_samples': count}
        if samples:
            result['sample_predictions'] = torch.stack(samples)
        if training:
            result['kl_infos'] = infos
            if 'structure_loss' in context.features:
                result['structure_loss'] = context.features['structure_loss']
        if return_debug:
            result['states'] = states
        return result
