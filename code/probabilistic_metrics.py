"""Unclipped voxelwise Gaussian-mixture marginals; never a volume joint likelihood."""

import math
import torch


@torch.no_grad()
def mixture_metrics(sample_predictions, target, sigma, voxel_chunk=65536):
    if sigma <= 0:
        raise ValueError('Observation sigma must be positive')
    samples = sample_predictions.detach().float().reshape(sample_predictions.shape[0], -1)
    targets = target.detach().float().reshape(-1)
    totals = dict(voxelwise_marginal_nll=0., coverage_50=0., coverage_90=0.,
                  interval_width_50=0., interval_width_90=0.)
    sum_u = sum_e = sum_uu = sum_ee = sum_ue = 0.
    for start in range(0, targets.numel(), voxel_chunk):
        means, y = samples[:, start:start+voxel_chunk], targets[start:start+voxel_chunk]
        log_prob = -.5 * ((y[None] - means) / sigma).square() - math.log(sigma * math.sqrt(2*math.pi))
        totals['voxelwise_marginal_nll'] += float((-torch.logsumexp(log_prob, 0) + math.log(samples.shape[0])).sum())
        bounds = {}
        for probability in (.05, .25, .75, .95):
            low, high = means.min(0).values - 9*sigma, means.max(0).values + 9*sigma
            for _ in range(28):
                middle = (low + high) / 2
                cdf = (.5 * (1 + torch.erf((middle[None] - means) / (sigma * math.sqrt(2))))).mean(0)
                low, high = torch.where(cdf < probability, middle, low), torch.where(cdf >= probability, middle, high)
            bounds[probability] = (low+high)/2
        for level, lo, hi in ((50, .25, .75), (90, .05, .95)):
            totals[f'coverage_{level}'] += float(((y >= bounds[lo]) & (y <= bounds[hi])).sum())
            totals[f'interval_width_{level}'] += float((bounds[hi]-bounds[lo]).sum())
        if samples.shape[0] > 1:
            u = means.var(0, unbiased=True).double()
            e = (means.mean(0) - y).abs().double()
            sum_u += float(u.sum()); sum_e += float(e.sum())
            sum_uu += float(u.square().sum()); sum_ee += float(e.square().sum()); sum_ue += float((u*e).sum())
    n = targets.numel()
    result = {key: value/n for key, value in totals.items()}
    denominator = max(0., (sum_uu-sum_u**2/n)*(sum_ee-sum_e**2/n)) ** .5
    result['error_uncertainty_correlation'] = (sum_ue-sum_u*sum_e/n)/denominator if denominator > 0 else None
    result.update(mixture_samples=samples.shape[0], coverage_is_finite_mc_diagnostic=True,
                  variance_estimable=samples.shape[0] > 1)
    return result
