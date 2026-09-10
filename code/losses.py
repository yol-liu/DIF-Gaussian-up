'''
ELBO loss and collapse diagnostics for C1 probabilistic DIF-Gaussian.

L = λ_v · L_volume + β_g · KL_global + β_l · KL_local

Volume likelihood uses a fixed-scale Laplace or Gaussian (no learned σ).
KL uses free-bits per latent group and linear β warmup.
'''

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models.latent_np import apply_free_bits, gaussian_kl


def volume_nll(pred, target, likelihood='laplace', sigma=0.1):
    '''
    Fixed-scale observation NLL.

    pred/target: [B, 1, N] (or broadcastable)
    Laplace NLL = |x-μ|/σ + log(2σ)  (L1 is equivalent up to affine transform)
    Gaussian NLL = 0.5 (x-μ)^2/σ^2 + log(σ√(2π))
    '''
    assert sigma > 0
    diff = pred - target
    if likelihood == 'laplace':
        nll = diff.abs() / sigma + math.log(2.0 * sigma)
    elif likelihood == 'gaussian':
        nll = 0.5 * (diff / sigma).pow(2) + math.log(sigma * math.sqrt(2.0 * math.pi))
    else:
        raise ValueError(f'unsupported volume_likelihood: {likelihood}')
    return nll.mean()


def kl_warmup_scale(step, kl_warmup_steps):
    if kl_warmup_steps is None or kl_warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(kl_warmup_steps))


def elbo_loss(
    points_pred,
    points_gt,
    kl_info,
    latent_cfg,
    step=0,
    use_local=False,
):
    '''
    Returns (total_loss, metrics_dict).
    '''
    likelihood = latent_cfg.get('volume_likelihood', 'laplace')
    sigma = float(latent_cfg.get('volume_sigma', 0.1))
    lambda_v = float(latent_cfg.get('lambda_v', 1.0))
    beta_g = float(latent_cfg.get('beta_g', 1e-3))
    beta_l = float(latent_cfg.get('beta_l', 1e-2))
    free_bits_g = float(latent_cfg.get('free_bits_g', 0.5))
    free_bits_l = float(latent_cfg.get('free_bits_l', 0.1))
    kl_warmup_steps = int(latent_cfg.get('kl_warmup_steps', 20000))

    L_vol = volume_nll(points_pred, points_gt, likelihood=likelihood, sigma=sigma)

    kl_g_per_dim = gaussian_kl(
        kl_info['mu_q_g'], kl_info['logvar_q_g'],
        kl_info['mu_p_g'], kl_info['logvar_p_g'],
    )  # [B, D_g]
    KL_g = apply_free_bits(kl_g_per_dim, free_bits_g)

    KL_l = points_pred.new_zeros(())
    kl_l_per_dim = None
    if use_local and 'mu_q_l' in kl_info:
        kl_l_full = gaussian_kl(
            kl_info['mu_q_l'], kl_info['logvar_q_l'],
            kl_info['mu_p_l'], kl_info['logvar_p_l'],
        )  # [B, D_l, R, R, R]
        # group = local dims after spatial mean (one free-bits group)
        kl_l_per_dim = kl_l_full.mean(dim=(2, 3, 4))  # [B, D_l]
        KL_l = apply_free_bits(kl_l_per_dim, free_bits_l)

    warm = kl_warmup_scale(step, kl_warmup_steps)
    loss = lambda_v * L_vol + warm * beta_g * KL_g + warm * beta_l * KL_l

    metrics = {
        'loss': loss.detach(),
        'L_volume': L_vol.detach(),
        'KL_global': KL_g.detach(),
        'KL_local': KL_l.detach() if torch.is_tensor(KL_l) else torch.tensor(0.0),
        'beta_scale': warm,
        'kl_g_per_dim': kl_g_per_dim.detach(),
        'kl_l_per_dim': None if kl_l_per_dim is None else kl_l_per_dim.detach(),
        'logvar_p_g': kl_info['logvar_p_g'].detach(),
        'logvar_q_g': kl_info['logvar_q_g'].detach(),
    }
    if 'logvar_p_l' in kl_info:
        metrics['logvar_p_l'] = kl_info['logvar_p_l'].detach()
        metrics['logvar_q_l'] = kl_info['logvar_q_l'].detach()
    return loss, metrics


def active_latent_units(kl_per_dim, threshold=0.01):
    '''
    Count dimensions whose mean (over batch) KL exceeds threshold.
    kl_per_dim: [B, D]
    '''
    mean_kl = kl_per_dim.mean(dim=0)
    return int((mean_kl > threshold).sum().item()), mean_kl


def sample_diversity(volumes, metric='l1'):
    '''
    Pairwise mean distance across S samples of the same case.

    volumes: [S, B, ...]  (S >= 2)
    returns: scalar mean pairwise distance (over pairs and batch)
    '''
    assert volumes.dim() >= 2 and volumes.shape[0] >= 2
    S = volumes.shape[0]
    vols = volumes.reshape(S, volumes.shape[1], -1)  # [S, B, N]
    dists = []
    for i in range(S):
        for j in range(i + 1, S):
            diff = vols[i] - vols[j]
            if metric == 'l1':
                d = diff.abs().mean(dim=-1)
            else:
                d = diff.pow(2).mean(dim=-1).sqrt()
            dists.append(d)
    return torch.stack(dists, dim=0).mean()


@torch.no_grad()
def compute_collapse_diagnostics(
    model,
    data,
    num_samples=4,
    active_threshold=0.01,
    diversity_metric='l1',
    eval_npoint=100000,
):
    '''
    Diagnostics aimed at silent prior/posterior collapse of conditional VAEs.

    - sample_diversity: mean pairwise volume distance over S prior samples
    - KL per group / active units
    - prior/posterior variance means
    '''
    was_training = model.training
    model.eval()

    out = model(
        data,
        is_eval=True,
        eval_npoint=eval_npoint,
        sample_from='prior',
        num_samples=num_samples,
    )
    vols = out['points_pred']  # [S, B, C, N]
    diversity = sample_diversity(vols, metric=diversity_metric)

    # one posterior-free prior state for variance / (optional) KL==0 check
    state = out['states'][0] if 'states' in out else out['state']
    kl_info = state.kl_info
    diag = {
        'sample_diversity': float(diversity.item()),
        'prior_var_g_mean': float(kl_info['logvar_p_g'].exp().mean().item()),
        'post_var_g_mean': float(kl_info['logvar_q_g'].exp().mean().item()),
    }

    # KL with q=p under prior sampling path (should be ~0); still useful shape check
    kl_g = gaussian_kl(
        kl_info['mu_q_g'], kl_info['logvar_q_g'],
        kl_info['mu_p_g'], kl_info['logvar_p_g'],
    )
    n_active, _ = active_latent_units(kl_g, threshold=active_threshold)
    diag['KL_global_raw'] = float(kl_g.sum(dim=-1).mean().item())
    diag['active_units_g'] = n_active

    if 'logvar_p_l' in kl_info:
        diag['prior_var_l_mean'] = float(kl_info['logvar_p_l'].exp().mean().item())
        diag['post_var_l_mean'] = float(kl_info['logvar_q_l'].exp().mean().item())
        kl_l = gaussian_kl(
            kl_info['mu_q_l'], kl_info['logvar_q_l'],
            kl_info['mu_p_l'], kl_info['logvar_p_l'],
        ).mean(dim=(2, 3, 4))
        n_active_l, _ = active_latent_units(kl_l, threshold=active_threshold)
        diag['KL_local_raw'] = float(kl_l.sum(dim=-1).mean().item())
        diag['active_units_l'] = n_active_l

    if was_training:
        model.train()
    return diag


def metrics_to_wandb(metrics: Dict, prefix='train'):
    '''Flatten loss metrics for W&B (skips large per-dim tensors).'''
    log = {}
    for k, v in metrics.items():
        if k.endswith('_per_dim') or k.startswith('logvar_'):
            if k.startswith('logvar_') and torch.is_tensor(v):
                log[f'{prefix}/{k}_mean'] = float(v.exp().mean().item())
            continue
        if torch.is_tensor(v):
            log[f'{prefix}/{k}'] = float(v.item()) if v.numel() == 1 else float(v.mean().item())
        elif isinstance(v, (int, float)):
            log[f'{prefix}/{k}'] = v
    return log
