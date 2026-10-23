"""Resolved probability protocol, stored in checkpoints along with the model config."""

import torch.nn as nn
import torch.nn.functional as F


class PointChannelNorm(nn.LayerNorm):
    """No normalization across query points, so the function is invariant to chunking."""
    def forward(self, value):
        return F.layer_norm(value.transpose(1, 2), self.normalized_shape,
                            self.weight, self.bias, self.eps).transpose(1, 2)


# Anchor MLPs see every anchor of one case at once (never chunked), so per-case statistics
# across anchors reproduce BN1d's batch-1 per-channel spread without pooling cases or views.
ANCHOR_MLPS = ('gs_feats_mlp', 'gs_params_mlp')

DEFAULTS = {
    'mode': 'dense_np', 'objective': 'joint', 'normalization': 'groupnorm',
    'film_init_std': .001, 'train_samples': 1, 'mean_code': False,
    'posterior_decoder_views': 'dense', 'prior_reconstruction_weight': 0.1,
    'reference_checkpoint': None, 'reference_evidence': None,
    'latent': {'global_dim': 64, 'use_local': False, 'local_dim': 16, 'grid_res': 4,
               'sigma_min': .001, 'sigma_max': 3., 'initial_sigma': .5},
    'reconstruction': {'likelihood': 'gaussian', 'observation_sigma': .1, 'reduction': 'mean_query'},
    'kl': {'beta_final': .001, 'warmup_steps': 2000, 'free_bits': 0.,
           'free_bits_order': 'per_case', 'reduction': 'sum_latent_mean_batch'},
    'evaluation': {'estimator': 'prior_mc_mean', 'num_samples': 8, 'seed': 0},
}

ALIASES = {'prior_mean': 'prior_latent_mean', 'posterior_mean_oracle': 'gt_posterior_mean'}
MODES = {'prior_sample', 'prior_latent_mean', 'prior_mc_mean',
         'dense_posterior_sample', 'dense_posterior_mean', 'gt_posterior_sample', 'gt_posterior_mean'}


def resolve_probability(config):
    from copy import deepcopy
    result = deepcopy(DEFAULTS)
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f'Unknown probabilistic keys: {sorted(unknown)}')
    for key, value in config.items():
        if isinstance(result[key], dict):
            if set(value) - set(result[key]):
                raise ValueError(f'Unknown probabilistic.{key} keys: {set(value) - set(result[key])}')
            result[key].update(value)
        else:
            result[key] = value
    if result['mode'] not in ('deterministic', 'dense_np', 'gt_cvae'):
        raise ValueError('probabilistic.mode must be deterministic, dense_np or gt_cvae')
    if result['objective'] not in ('joint', 'frozen_reference'):
        raise ValueError('Unknown probabilistic objective')
    if result['normalization'] not in ('groupnorm', 'groupnorm_anchor_instance', 'frozen_bn'):
        raise ValueError('Use groupnorm[_anchor_instance] from scratch or frozen_bn with a trained initialization')
    if not isinstance(result['train_samples'], int) or result['train_samples'] < 1:
        raise ValueError('train_samples must be positive')
    if not 0 < result['film_init_std'] <= .1:
        raise ValueError('film_init_std must be small and nonzero')
    latent = result['latent']
    if not 0 < latent['sigma_min'] < latent['initial_sigma'] < latent['sigma_max']:
        raise ValueError('Require 0 < sigma_min < initial_sigma < sigma_max')
    if any(not isinstance(latent[key], int) or latent[key] < 1 for key in ('global_dim', 'local_dim', 'grid_res')):
        raise ValueError('Latent dimensions must be positive integers')
    rec, kl, evaluation = result['reconstruction'], result['kl'], result['evaluation']
    if rec['likelihood'] not in ('gaussian', 'mse') or rec['observation_sigma'] <= 0 or rec['reduction'] != 'mean_query':
        raise ValueError('Use Gaussian/MSE with positive observation_sigma and mean_query reduction')
    if kl['beta_final'] < 0 or kl['warmup_steps'] < 0 or kl['free_bits'] < 0:
        raise ValueError('KL weights, warmup and free bits must be nonnegative')
    if kl['free_bits_order'] not in ('per_case', 'batch_mean') or kl['reduction'] != 'sum_latent_mean_batch':
        raise ValueError('Unknown KL reduction/free-bits order')
    evaluation['estimator'] = ALIASES.get(evaluation['estimator'], evaluation['estimator'])
    if evaluation['estimator'] not in MODES or not isinstance(evaluation['num_samples'], int) or evaluation['num_samples'] < 1:
        raise ValueError('Invalid probability evaluation settings')
    if result['objective'] == 'frozen_reference' and (result['mode'] != 'dense_np' or result['mean_code']):
        raise ValueError('Frozen reference requires stochastic dense_np')
    if result['mean_code'] and (result['mode'] != 'dense_np' or latent['use_local']):
        raise ValueError('Mean-code control is global dense_np only')
    if result['posterior_decoder_views'] not in ('context', 'dense'):
        raise ValueError('posterior_decoder_views must be context or dense')
    if result['prior_reconstruction_weight'] < 0:
        raise ValueError('prior_reconstruction_weight must be nonnegative')
    return result


def configure_normalization(model, policy):
    """GN preserves per-case/view isolation; frozen BN preserves warm-start coordinates."""
    import torch.nn as nn
    if policy == 'groupnorm_anchor_instance':
        for prefix in ANCHOR_MLPS:
            if hasattr(model, prefix):
                configure_normalization(getattr(model, prefix), 'anchor_instance')
        configure_normalization(model, 'groupnorm')
    elif policy in ('groupnorm', 'anchor_instance'):
        for name, child in list(model.named_children()):
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                channels = child.num_features
                groups = min(8, max(1, channels // 2))
                while channels % groups:
                    groups -= 1
                replacement = (nn.InstanceNorm1d(channels, eps=child.eps, affine=child.affine)
                               if policy == 'anchor_instance' else
                               PointChannelNorm(channels, eps=child.eps, elementwise_affine=child.affine)
                               if isinstance(child, nn.BatchNorm1d) else
                               nn.GroupNorm(groups, channels, eps=child.eps, affine=child.affine))
                if child.affine:
                    replacement.weight.data.copy_(child.weight.data)
                    replacement.bias.data.copy_(child.bias.data)
                setattr(model, name, replacement)
            else:
                configure_normalization(child, policy)
    elif policy == 'frozen_bn':
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
