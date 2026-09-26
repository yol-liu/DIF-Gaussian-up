import numpy as np


def pilot_gate(baseline, candidate, minimum_cases=6):
    if baseline['train_updates'] != candidate['train_updates']:
        return {'status': 'INCONCLUSIVE', 'reason': 'Optimizer update counts differ'}
    for key in ('seed', 'loss_protocol', 'eval_scale', 'resampling_protocol', 'latent_mode'):
        if baseline[key] != candidate[key]:
            return {'status': 'INCONCLUSIVE', 'reason': f'Protocol mismatch: {key}'}
    reference = {case['name']: case for case in baseline['cases']}
    proposed = {case['name']: case for case in candidate['cases']}
    if set(reference) != set(proposed) or len(reference) < minimum_cases:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'Different or too few validation cases'}
    deltas = []
    for name, base in reference.items():
        current = proposed[name]
        if base.get('view_ids') != current.get('view_ids'):
            return {'status': 'INCONCLUSIVE', 'reason': 'Context view sets differ'}
        boundary = None
        if base.get('boundary_error') is not None and base['boundary_error'] > 0 and current.get('boundary_error') is not None:
            boundary = (base['boundary_error'] - current['boundary_error']) / base['boundary_error']
        deltas.append({'case_id': name, 'psnr': current['psnr'] - base['psnr'],
                       'ssim': current['ssim'] - base['ssim'], 'boundary_relative_improvement': boundary})
    psnr = np.asarray([case['psnr'] for case in deltas])
    ssim = np.asarray([case['ssim'] for case in deltas])
    if not np.isfinite(psnr).all() or not np.isfinite(ssim).all():
        return {'status': 'INCONCLUSIVE', 'reason': 'Non-finite paired metrics'}
    boundary = [case['boundary_relative_improvement'] for case in deltas]
    boundary_gain = all(value is not None for value in boundary) and (
        np.mean(boundary) >= .02 and np.mean(np.asarray(boundary) > 0) >= 2 / 3)
    psnr_gain = psnr.mean() >= .1 and np.mean(psnr > 0) >= 2 / 3
    protected = ssim.mean() >= -.002 and psnr.mean() >= -.1
    return {'status': 'PROMOTED_PILOT_ONLY' if protected and (psnr_gain or boundary_gain) else 'NO_PILOT_GAIN',
            'case_differences': deltas, 'mean_psnr_delta': float(psnr.mean()),
            'mean_ssim_delta': float(ssim.mean()), 'formal_claim': False}
