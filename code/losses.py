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



def volume_nll(pred, target, likelihood='laplace', sigma=0.1):
    '''
    Fixed-scale observation NLL.

    pred/target: [B, 1, N] (or broadcastable)
    Laplace NLL = |x-μ|/σ + log(2σ)  (L1 is equivalent up to affine transform)
    Gaussian NLL = 0.5 (x-μ)^2/σ^2 + log(σ√(2π))
    '''
    assert sigma > 0
    diff = pred - target
    if likelihood == 'mse':
        return diff.square().mean()
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
    from models.latent_np import apply_free_bits, gaussian_kl
    likelihood = latent_cfg.get('reconstruction_loss', 'mse')
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
        'KL_global_raw': kl_g_per_dim.sum(-1).mean().detach(),
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
    if hasattr(model, 'probability'):
        from probabilistic_diagnostics import probability_diagnostics
        return probability_diagnostics(model, data, num_samples, eval_npoint)
    from models.latent_np import apply_free_bits, gaussian_kl
    from checkpoints import rng_state, restore_rng
    saved_rng = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        if num_samples < 2:
            raise ValueError('diversity diagnostics require at least two samples')
        if data.get('gt_volume') is None:
            return {'posterior_available': False, 'USES_GT': False,
                    'KL_global_raw': None, 'active_units_g': None}
        posterior = model.encode_and_sample(
            data, gt_volume=data.get('gt_volume'), sample_from='posterior_mean'
        )
        prior = model.encode_and_sample(data, sample_from='mean')
        out = model(data, is_eval=True, eval_npoint=eval_npoint,
                    sample_from='prior', num_samples=num_samples)
        info = posterior.kl_info
        diag = {'USES_GT': True, 'sample_diversity': float(
            sample_diversity(out['sample_predictions'], metric=diversity_metric))}
        diag['prior_posterior_output_l1'] = float((
            model._decode_chunked(data, posterior, eval_npoint)
            - model._decode_chunked(data, prior, eval_npoint)).abs().mean())
        latent_cfg = model.cfg.get('latent', {})
        for group, label in [('g', 'global'), ('l', 'local')]:
            if f'mu_q_{group}' not in info:
                continue
            per_dim = gaussian_kl(info[f'mu_q_{group}'], info[f'logvar_q_{group}'],
                                  info[f'mu_p_{group}'], info[f'logvar_p_{group}'])
            if per_dim.ndim == 5:
                per_dim = per_dim.mean((2, 3, 4))
            diag[f'KL_{label}_raw'] = float(per_dim.sum(-1).mean())
            diag[f'KL_{label}_freebits_penalty'] = float(apply_free_bits(
                per_dim, latent_cfg.get(f'free_bits_{group}', 0.)))
            diag[f'kl_{group}_per_dim'] = per_dim.mean(0).tolist()
            diag[f'active_units_{group}'] = active_latent_units(per_dim, active_threshold)[0]
            for prefix, distribution in [('prior', 'p'), ('post', 'q')]:
                logvar = info[f'logvar_{distribution}_{group}']
                diag[f'{prefix}_variance_{group}_mean'] = float(logvar.exp().mean())
                diag[f'{prefix}_std_{group}_mean'] = float((0.5 * logvar).exp().mean())
        conditioning = posterior.z_g
        if model.use_local:
            local = model.local_latent.interpolate_to_anchors(posterior.z_grid, data['points_gs'])
            conditioning = torch.cat([conditioning[:, None].expand(-1, local.shape[1], -1), local], -1)
        gamma, shift = model.feature_film.net(conditioning).chunk(2, -1)
        diag['film_gamma_abs_mean'] = float(gamma.abs().mean())
        diag['film_shift_abs_mean'] = float(shift.abs().mean())
        return diag
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(saved_rng)


def metrics_to_wandb(metrics: Dict, prefix='train'):
    '''Flatten loss metrics for W&B (skips large per-dim tensors).'''
    log = {}
    for k, v in metrics.items():
        if k.endswith('_per_dim') or k.startswith('logvar_'):
            if k.startswith('logvar_') and torch.is_tensor(v):
                log[f'{prefix}/{k.replace("logvar_", "variance_")}_mean'] = float(v.exp().mean().item())
            continue
        if torch.is_tensor(v):
            log[f'{prefix}/{k}'] = float(v.item()) if v.numel() == 1 else float(v.mean().item())
        elif isinstance(v, (int, float)):
            log[f'{prefix}/{k}'] = v
    return log
