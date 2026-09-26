"""mean-query reconstruction + beta * sum-latent KL: a reweighted variational objective."""

import torch
from models.latent_np import gaussian_kl
from losses import volume_nll, kl_warmup_scale


def probability_loss(output, target, config, step):
    samples = output['sample_predictions'].float()
    rec = config['reconstruction']
    reconstruction = volume_nll(samples, target[None], rec['likelihood'], rec['observation_sigma'])
    mse = (samples - target[None]).square().mean()
    kl_cfg = config['kl']
    metrics = {'L_volume': reconstruction.detach(), 'mse_raw': mse.detach(),
               'train_samples': samples.shape[0], 'query_count': target.shape[-1]}
    penalties = []
    for group in ('g', 'l'):
        if 'mu_q_' + group not in output['kl_infos'][0]:
            continue
        per_sample = torch.stack([gaussian_kl(info['mu_q_' + group], info['logvar_q_' + group],
                                              info['mu_p_' + group], info['logvar_p_' + group]).flatten(1)
                                  for info in output['kl_infos']])  # S,B,D (all cells included)
        per_case = per_sample.mean(0)
        raw = per_case.sum(-1).mean()
        values = per_case.mean(0) if kl_cfg['free_bits_order'] == 'batch_mean' else per_case
        effective = values.clamp_min(kl_cfg['free_bits']).sum(-1).mean()
        penalties.append(effective)
        metrics.update({f'KL_{group}_raw': raw.detach(), f'KL_{group}_effective': effective.detach(),
                        f'KL_{group}_free_bits_floor': kl_cfg['free_bits'] * per_case.shape[-1],
                        f'kl_{group}_per_dim': per_case.mean(0).detach(),
                        f'KL_{group}_dimensions': per_case.shape[-1]})
        if group == 'l':
            metrics['KL_l_per_cell_normalized'] = raw.detach() / config['latent']['grid_res'] ** 3
        for source in ('p', 'q'):
            lv = output['kl_infos'][0][f'logvar_{source}_{group}'].float()
            sigma = (.5 * lv).exp()
            low, high = config['latent']['sigma_min'], config['latent']['sigma_max']
            metrics.update({f'{source}_{group}_std': sigma.mean().detach(),
                            f'{source}_{group}_variance': lv.exp().mean().detach(),
                            f'{source}_{group}_scale_near_lower': (sigma < low + .01*(high-low)).float().mean(),
                            f'{source}_{group}_scale_near_upper': (sigma > high - .01*(high-low)).float().mean()})
    beta = kl_cfg['beta_final'] * kl_warmup_scale(step, kl_cfg['warmup_steps'])
    effective_kl = sum(penalties)
    if config['mean_code']:
        # Deterministic high-information code control: both heads learn a common mean code.
        info = output['kl_infos'][0]
        effective_kl = (info['mu_q_g'] - info['mu_p_g']).square().sum(-1).mean()
        metrics['mean_code_alignment'] = effective_kl.detach()
    weighted = beta * effective_kl
    loss = weighted if config['objective'] == 'frozen_reference' else reconstruction + weighted
    metrics.update(loss=loss.detach(), warmup_beta=beta)
    metrics['weighted_mean_code_alignment' if config['mean_code'] else 'weighted_kl'] = weighted.detach()
    return loss, metrics


def probability_gradient_norms(model, loss_scale=1.):
    groups = {'prior': [], 'posterior': [], 'shared_encoder': [], 'film': [], 'decoder': []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = ('prior' if name.startswith(('prior_head.', 'local_prior_head.')) else
                 'posterior' if name.startswith(('posterior_head.', 'local_posterior_head.', 'gt_encoder.')) else
                 'film' if name.startswith('feature_film.') else
                 'decoder' if name.startswith('point_decoder.') else 'shared_encoder')
        groups[group].append(parameter.grad.detach().float().square().sum())
    result = {}
    for name, values in groups.items():
        norm = torch.stack(values).sum().sqrt() if values else torch.tensor(0.)
        result['gradient/' + name] = float(norm) if torch.isfinite(norm) else None
    result['gradient/all_finite'] = all(value is not None for value in result.values())
    for name, value in getattr(model, 'probability_gradients', {}).items():
        result['gradient/' + name] = float(value / loss_scale) if torch.isfinite(value) else None
    return result
