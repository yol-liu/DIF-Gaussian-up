"""Isolated diagnostics; they do not update BN, parameter gradients or training RNG."""

from contextlib import contextmanager

import torch

from checkpoints import rng_state, restore_rng
from evaluation_protocol import evaluating
from models.latent_np import gaussian_kl


@contextmanager
def diagnostic_context(model):
    rng = rng_state()
    try:
        with evaluating(model), torch.no_grad():
            yield
    finally:
        restore_rng(rng)


def probability_diagnostics(model, data, num_samples=4, eval_npoint=8192, seed=173):
    if num_samples < 2:
        raise ValueError('Diversity diagnostics require at least two samples')
    with diagnostic_context(model):
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        prior = model(data, is_eval=True, latent_mode='prior_mc_mean', num_samples=num_samples,
                      generator=generator, return_samples=True, return_debug=True, eval_npoint=eval_npoint)
        samples = prior['sample_predictions']
        state = prior['states'][0]
        mean_state = model.sample_state(state.context, model.infer_prior(state.context), mean=True)
        base = model.decode_chunked(mean_state, eval_npoint)
        zero = model.decode_chunked(model.state_with_global(mean_state, torch.zeros_like(mean_state.z_g)), eval_npoint)
        # Shuffling is explicitly a diagnostic intervention and needs >1 case.
        shuffled = (model.decode_chunked(model.state_with_global(mean_state, mean_state.z_g.roll(1, 0)), eval_npoint)
                    if mean_state.z_g.shape[0] > 1 else None)
        diag = {'posterior_available': False, 'oracle': None, 'sample_count': num_samples,
                'prior_sample_diversity_l1': float((samples[1:] - samples[:1]).abs().mean()),
                'latent_mean_vs_mc_l1': float((base - prior['points_pred']).abs().mean()),
                'zero_z_intervention_l1': float((zero - base).abs().mean()),
                'shuffled_z_intervention_l1': float((shuffled - base).abs().mean()) if shuffled is not None else None,
                'prior_mu_between_case_variance': float(state.kl_info['mu_p_g'].var(0, unbiased=False).mean())
                    if samples.shape[1] > 1 else None,
                'film_residual_norm': float((mean_state.features['gs_params']['feats'] -
                                             state.context.features['gs_params']['feats']).norm()),
                'KL_g_raw': None, 'kl_g_per_dim': None}
        # Sensitivity via fixed random projection, avoiding cancellation from mean output.
        with torch.enable_grad():
            z = mean_state.z_g.detach().requires_grad_(True)
            probe = model.state_with_global(mean_state, z)
            query = {key: state.context.data[key][..., :min(128, eval_npoint), :]
                     for key in model.registered_point_keys}
            prediction = model.decode_points(probe, query)
            weights = torch.randn(prediction.shape, device=prediction.device, generator=generator)
            sensitivity = torch.autograd.grad((prediction * weights).sum() / prediction.numel() ** .5, z)[0]
        diag['sensitivity_l2'] = float(sensitivity.norm())
        diag['output_change_l1'] = diag['prior_sample_diversity_l1']
        for group in ('g', 'l'):
            if f'logvar_p_{group}' in state.kl_info:
                lv = state.kl_info[f'logvar_p_{group}'].float()
                diag[f'p_{group}_variance_mean'] = float(lv.exp().mean())
                diag[f'p_{group}_std_mean'] = float((.5*lv).exp().mean())
        available = ('dense_projs' in data if model.mode == 'dense_np' else 'gt_volume' in data)
        if available:
            # Explicit samples through fixed context, including q_g-conditioned local KL.
            stats, grid = (model.reference or model).infer_posterior(data, state.context)
            q_states = [model.sample_state(state.context, model.infer_prior(state.context), stats, grid,
                                          generator=generator) for _ in range(num_samples)]
            q_predictions = torch.stack([model.decode_chunked(s, eval_npoint) for s in q_states])
            diag.update(posterior_available=True, oracle='oracle_dense' if model.mode == 'dense_np' else 'oracle_gt',
                        posterior_sample_diversity_l1=float((q_predictions[1:] - q_predictions[:1]).abs().mean()),
                        prior_posterior_output_l1=float((q_predictions.mean(0) - prior['points_pred']).abs().mean()),
                        posterior_mu_between_case_variance=float(stats[0].var(0, unbiased=False).mean()) if stats[0].shape[0] > 1 else None)
            for group in ('g', 'l'):
                if f'mu_q_{group}' not in q_states[0].kl_info:
                    continue
                kl = torch.stack([gaussian_kl(s.kl_info[f'mu_q_{group}'], s.kl_info[f'logvar_q_{group}'],
                                             s.kl_info[f'mu_p_{group}'], s.kl_info[f'logvar_p_{group}']).flatten(1)
                                  for s in q_states]).mean(0)
                diag.update({f'KL_{group}_raw': float(kl.sum(-1).mean()),
                             f'kl_{group}_per_dim': kl.mean(0).tolist(),
                             f'KL_threshold_dimensions_{group}': int((kl.mean(0) > .01).sum())})
                for source in ('p', 'q'):
                    lv = q_states[0].kl_info[f'logvar_{source}_{group}'].float()
                    sigma = (.5 * lv).exp()
                    low, high = model.probability['latent']['sigma_min'], model.probability['latent']['sigma_max']
                    diag.update({f'{source}_{group}_variance_mean': float(lv.exp().mean()),
                                 f'{source}_{group}_std_mean': float(sigma.mean()),
                                 f'{source}_{group}_scale_near_lower': float((sigma < low+.01*(high-low)).float().mean()),
                                 f'{source}_{group}_scale_near_upper': float((sigma > high-.01*(high-low)).float().mean())})
            if 'points_gt' in data and data['points_gt'].numel() == prior['points_pred'].numel():
                target = data['points_gt'].reshape_as(prior['points_pred'])
                diag['prior_mc_mse'] = float((prior['points_pred'] - target).square().mean())
                diag['posterior_mc_mse'] = float((q_predictions.mean(0) - target).square().mean())
                diag['prior_posterior_mse_gap'] = diag['prior_mc_mse'] - diag['posterior_mc_mse']
        return diag
