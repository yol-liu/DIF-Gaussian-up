'''
Probabilistic wrapper around DIF_Gaussian (C1 hierarchical CVAE).

Sampling and decoding are deliberately split so every decode chunk shares the
same anatomy sample (state). Latent only modulates Gaussian features via
zero-init FiLM; center / scale / rotation stay deterministic.
'''

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from models.model import DIF_Gaussian
from models.latent_np import (
    GlobalLatent,
    LocalLatentGrid,
    FeatureFiLM,
    pool_global_ctx,
    gaussian_kl,
)
from models.gt_encoder import GTEncoder


@dataclass
class SampleState:
    '''Fixed anatomy sample shared across all decode chunks.'''
    feats_proj: torch.Tensor
    gs_params: Dict[str, torch.Tensor]
    z_g: Optional[torch.Tensor] = None
    z_grid: Optional[torch.Tensor] = None
    # variational diagnostics (detached where appropriate)
    kl_info: Dict[str, Any] = field(default_factory=dict)


def build_model(cfg):
    '''Factory: probabilistic model only when latent.use_latent is true.'''
    latent_cfg = cfg.get('latent', {}) if hasattr(cfg, 'get') else getattr(cfg, 'latent', {})
    if isinstance(latent_cfg, dict):
        use_latent = latent_cfg.get('use_latent', False)
    else:
        use_latent = getattr(latent_cfg, 'use_latent', False)
    if use_latent:
        return ProbabilisticDIFGaussian(cfg)
    return DIF_Gaussian(cfg)


class ProbabilisticDIFGaussian(DIF_Gaussian):
    '''
    Amortized hierarchical conditional VAE over structured Gaussian primitives.

    API:
      state = model.encode_and_sample(data, gt_volume=None, sample_from=...)
      mu    = model.decode(data, state)          # any chunk, same state
      model.render(rays, state)                  # placeholder for C2
    '''

    def init(self):
        super().init()
        latent_cfg = self.cfg.get('latent', {})
        self.use_latent = bool(latent_cfg.get('use_latent', False))
        self.use_local = bool(latent_cfg.get('use_local', False))
        self.global_dim = int(latent_cfg.get('global_dim', 64))
        self.local_dim = int(latent_cfg.get('local_dim', 16))
        self.grid_res = int(latent_cfg.get('grid_res', 4))
        self.prior_sigma_floor = float(latent_cfg.get('prior_sigma_floor', 1e-2))
        self.default_sample_from = 'posterior'

        sampling_cfg = self.cfg.get('sampling', {})
        if sampling_cfg:
            self.default_sample_from = sampling_cfg.get('sample_from', 'posterior')

        if not self.use_latent:
            return

        # conditioning trunk channel = downsampled encoder channel
        ctx_dim = self.image_encoder.ds_ch
        mid_ch = self.image_encoder.out_ch
        gt_ch = int(latent_cfg.get('gt_feat_ch', 32))

        self.global_latent = GlobalLatent(
            ctx_dim=ctx_dim,
            latent_dim=self.global_dim,
            gt_dim=gt_ch,
            prior_sigma_floor=self.prior_sigma_floor,
        )
        # GT encoder is created lazily / always present for training; inference
        # simply never calls it when gt_volume is None.
        self.gt_encoder = GTEncoder(in_ch=1, out_ch=gt_ch)

        film_in = self.global_dim
        if self.use_local:
            self.local_latent = LocalLatentGrid(
                ctx_dim=ctx_dim,
                global_dim=self.global_dim,
                latent_dim=self.local_dim,
                gt_dim=gt_ch,
                grid_res=self.grid_res,
                prior_sigma_floor=self.prior_sigma_floor,
            )
            film_in = self.global_dim + self.local_dim

        self.feature_film = FeatureFiLM(
            feat_dim=mid_ch,
            latent_dim=film_in,
        )

    # ------------------------------------------------------------------ #
    # Sampling API
    # ------------------------------------------------------------------ #
    def encode_and_sample(
        self,
        data,
        gt_volume=None,
        sample_from=None,
        num_samples=1,
    ):
        '''
        Encode projections and draw latent sample(s).

        sample_from: 'prior' | 'posterior' | 'mean'
          - prior:      conditional prior (inference)
          - posterior:  residual posterior (training; requires gt_volume)
          - mean:       prior mean (= deterministic baseline under zero-init FiLM)

        num_samples > 1 returns a list of SampleState (one encode, S samples).
        '''
        sample_from = sample_from or self.default_sample_from
        assert sample_from in ('prior', 'posterior', 'mean'), \
            f'unknown sample_from: {sample_from}'

        feats_dict = self.encode_projs(data)
        if not self.use_latent:
            state = SampleState(
                feats_proj=feats_dict['feats_proj'],
                gs_params=feats_dict['gs_params'],
            )
            return [state] * num_samples if num_samples > 1 else state

        anchor_feats = feats_dict['anchor_feats']  # [B, C, K]
        gs_params = feats_dict['gs_params']
        global_ctx = pool_global_ctx(anchor_feats, mode='mean')  # [B, C]

        # GT features for posterior (training only)
        gt_global_feat = None
        gt_anchor_feat = None
        if sample_from == 'posterior':
            if gt_volume is None:
                raise ValueError('sample_from="posterior" requires gt_volume')
            gt_out = self.gt_encoder(
                gt_volume, anchor_xyz=data['points_gs']
            )
            gt_global_feat = gt_out['gt_global_feat']
            gt_anchor_feat = gt_out['gt_anchor_feat']

        mu_p_g, logvar_p_g = self.global_latent.prior(global_ctx)
        if sample_from == 'posterior':
            mu_q_g, logvar_q_g = self.global_latent.posterior(
                global_ctx, gt_global_feat
            )
        else:
            mu_q_g, logvar_q_g = mu_p_g, logvar_p_g

        # local grid context (shared across samples)
        grid_ctx = None
        mu_p_l = logvar_p_l = mu_q_l = logvar_q_l = None
        if self.use_local:
            grid_ctx = self.local_latent.pool_anchors_to_grid(
                anchor_feats, data['points_gs']
            )

        states = []
        for _ in range(num_samples):
            if sample_from == 'mean':
                z_g = mu_p_g
            elif sample_from == 'posterior':
                z_g = self.global_latent.sample(mu_q_g, logvar_q_g)
            else:
                z_g = self.global_latent.sample(mu_p_g, logvar_p_g)

            z_grid = None
            z_local_anchors = None
            if self.use_local:
                # top-down: local prior/posterior conditioned on this z_g
                mu_p_l, logvar_p_l = self.local_latent.prior(z_g, grid_ctx)
                if sample_from == 'posterior':
                    gt_grid = self.local_latent.pool_anchors_to_grid(
                        gt_anchor_feat, data['points_gs']
                    )
                    mu_q_l, logvar_q_l = self.local_latent.posterior(
                        z_g, grid_ctx, gt_grid
                    )
                else:
                    mu_q_l, logvar_q_l = mu_p_l, logvar_p_l

                if sample_from == 'mean':
                    z_grid = mu_p_l
                elif sample_from == 'posterior':
                    z_grid = self.local_latent.sample(mu_q_l, logvar_q_l)
                else:
                    z_grid = self.local_latent.sample(mu_p_l, logvar_p_l)

                z_local_anchors = self.local_latent.interpolate_to_anchors(
                    z_grid, data['points_gs']
                )  # [B, K, D_l]

            # FiLM conditioning per anchor
            if self.use_local:
                z_g_exp = z_g.unsqueeze(1).expand(-1, z_local_anchors.shape[1], -1)
                z_cond = torch.cat([z_g_exp, z_local_anchors], dim=-1)  # [B, K, D]
            else:
                z_cond = z_g  # [B, D] broadcast inside FiLM

            gs_params_mod = {
                k: v for k, v in gs_params.items() if k != 'feats'
            }
            # ONLY modulate features; geometry stays deterministic
            gs_params_mod['feats'] = self.feature_film(gs_params['feats'], z_cond)

            kl_info = {
                'mu_p_g': mu_p_g,
                'logvar_p_g': logvar_p_g,
                'mu_q_g': mu_q_g,
                'logvar_q_g': logvar_q_g,
            }
            if self.use_local:
                kl_info.update({
                    'mu_p_l': mu_p_l,
                    'logvar_p_l': logvar_p_l,
                    'mu_q_l': mu_q_l,
                    'logvar_q_l': logvar_q_l,
                })

            states.append(SampleState(
                feats_proj=feats_dict['feats_proj'],
                gs_params=gs_params_mod,
                z_g=z_g,
                z_grid=z_grid,
                kl_info=kl_info,
            ))

        return states if num_samples > 1 else states[0]

    def decode(self, data, state: SampleState):
        '''
        Decode query points given a fixed SampleState.
        Must NOT resample latents — all chunks share the same state.
        '''
        feats_dict = {
            'feats_proj': state.feats_proj,
            'gs_params': state.gs_params,
        }
        return self.forward_points(feats_dict, data)

    def render(self, rays, state: SampleState):
        # Reserved for C2 (analytic X-ray / projection loss).
        raise NotImplementedError(
            'render(rays, state) is reserved for C2; not implemented in C1'
        )

    def forward(self, data, is_eval=False, eval_npoint=100000,
                gt_volume=None, sample_from=None, num_samples=1):
        if not self.use_latent:
            return super().forward(data, is_eval=is_eval, eval_npoint=eval_npoint)

        # inference samples the conditional prior; 'mean' is for ablation
        sample_from = sample_from or (
            'prior' if is_eval else self.default_sample_from
        )
        if is_eval and sample_from == 'posterior':
            sample_from = 'prior'

        if not is_eval:
            state = self.encode_and_sample(
                data, gt_volume=gt_volume, sample_from=sample_from, num_samples=1
            )
            pred = self.decode(data, state)
            return {
                'points_pred': pred,
                'state': state,
                'kl_info': state.kl_info,
            }

        # eval: encode once, decode in chunks sharing state
        if num_samples > 1:
            states = self.encode_and_sample(
                data, gt_volume=None, sample_from=sample_from, num_samples=num_samples
            )
            preds = []
            for state in states:
                preds.append(self._decode_chunked(data, state, eval_npoint))
            return {
                'points_pred': torch.stack(preds, dim=0),  # [S, B, C, N]
                'states': states,
            }

        state = self.encode_and_sample(
            data, gt_volume=None, sample_from=sample_from, num_samples=1
        )
        return {
            'points_pred': self._decode_chunked(data, state, eval_npoint),
            'state': state,
        }

    def _decode_chunked(self, data, state, eval_npoint):
        total_npoint = data['points_proj'].shape[2]
        n_batch = int(np.ceil(total_npoint / eval_npoint))
        pred_list = []
        for i in range(n_batch):
            left = i * eval_npoint
            right = min((i + 1) * eval_npoint, total_npoint)
            tmp_data = {}
            for key in data.keys():
                if key in self.registered_point_keys:
                    tmp_data[key] = data[key][..., left:right, :]
                else:
                    tmp_data[key] = data[key]
            pred_list.append(self.decode(tmp_data, state))
        return torch.cat(pred_list, dim=2)

    def compute_kl_terms(self, kl_info):
        '''Return per-group KL tensors (with free-bits applied externally).'''
        kl_g = gaussian_kl(
            kl_info['mu_q_g'], kl_info['logvar_q_g'],
            kl_info['mu_p_g'], kl_info['logvar_p_g'],
        )  # [B, D_g]
        out = {'kl_g_per_dim': kl_g}
        if self.use_local and 'mu_q_l' in kl_info:
            kl_l = gaussian_kl(
                kl_info['mu_q_l'], kl_info['logvar_q_l'],
                kl_info['mu_p_l'], kl_info['logvar_p_l'],
            )  # [B, D_l, R, R, R]
            # flatten spatial for free-bits grouping: treat each (cell, dim)
            # then reduce; group-level free bits over D_l after spatial mean
            out['kl_l_per_dim'] = kl_l.mean(dim=(2, 3, 4))  # [B, D_l]
            out['kl_l_full'] = kl_l
        return out
