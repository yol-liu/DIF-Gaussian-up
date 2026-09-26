"""Generate configs/experiment_plan/{stage1,stage2}/*.yaml from the current-best baseline.

Every arm is the baseline plus an explicit override, so the diff to its comparator is exactly the
change under test. Re-running rewrites the YAML files; it never touches logs or checkpoints.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/experiment_plan'
BASE = yaml.safe_load((ROOT / 'configs/geometry_validation/R0.yaml').read_text())
MULTIVIEW = {name: yaml.safe_load((ROOT / f'configs/multiview/{name}.yaml').read_text())
             for name in ('B1', 'T1-nomax-vit', 'T1-nomax-alt')}
HIERARCHICAL = yaml.safe_load((ROOT / 'configs/geometry_validation/H0.yaml').read_text())
STRUCTURE_TARGET = {'frame': 'scanner', 'scale': 0.25, 'sigma_mm': None, 'rho_mm': None,
                    'energy_threshold': 1.0e-06, 'cache_dir': './cache/multiview_structure'}


def merge(target, override):
    for key, value in override.items():
        if value == '__delete__':
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def arm(override=(), base=None):
    config = copy.deepcopy(base if base is not None else BASE)
    for item in (override if isinstance(override, (list, tuple)) else [override]):
        merge(config, item)
    return config


def probability(mode='deterministic', **extra):
    block = {'mode': mode, 'objective': 'joint', 'normalization': 'groupnorm_anchor_instance',
             'film_init_std': 0.001, 'train_samples': 1,
             'latent': {'global_dim': 64, 'use_local': False, 'sigma_min': 0.001, 'sigma_max': 3.0,
                        'initial_sigma': 0.5},
             # MSE reconstruction as in the baseline; beta 2e-5 on MSE equals the earlier beta 1e-3 on
             # Gaussian NLL with sigma 0.1 (NLL = 50 x MSE + const).
             'reconstruction': {'likelihood': 'mse', 'observation_sigma': 0.1, 'reduction': 'mean_query'},
             'kl': {'beta_final': 2.0e-05, 'warmup_steps': 500, 'free_bits': 0.0, 'free_bits_order': 'per_case',
                    'reduction': 'sum_latent_mean_batch'},
             'evaluation': {'estimator': 'prior_mc_mean', 'num_samples': 8, 'seed': 0}}
    merge(block, extra)
    return {'model': {'latent': '__delete__', 'probabilistic': block},
            'dataset': {'paired_views': {'dense_count': 12, 'context_counts': [6]}},
            'loss': {'reconstruction_loss': 'mse'}}


PROB_CONTROL = arm(probability())
DENSE_NP = arm(probability('dense_np'))

STAGE1 = {
    'baseline-unet-transformer-meaninit': (
        'Current best (T1-nomax): UNet encoder, anchor geometry transformer with mean-initialized tokens and '
        'no max residual, point max fusion, BatchNorm, MSE.', arm()),
    'baseline-unet-transformer-meaninit-replica': (
        'Identical to the baseline; a second run measures the noise floor from GPU nondeterminism.', arm()),
    'encoder-vit-per-view': (
        'Replace the UNet 1/4-1/16 encoder with a per-view ViT trained from scratch.',
        arm({'model': {'image_encoder': MULTIVIEW['T1-nomax-vit']['model']['image_encoder']}})),
    'encoder-hierarchical-window': (
        'Replace the UNet encoder with a hierarchical window transformer (keeps the full/half-resolution '
        'conv stems and UNet decoder).', arm({'model': {'image_encoder': HIERARCHICAL['model']['image_encoder']}})),
    'anchor-fusion-maxpool': (
        'Remove the anchor transformer: anchors take the masked max over views (original DIF-Gaussian).',
        arm({'model': {'anchor_fusion': 'max', 'multiview_transformer': '__delete__'}})),
    'anchor-fusion-late-attention': (
        'Older late cross-view attention on anchors instead of the geometry transformer.',
        arm({'model': {'anchor_fusion': 'geometry_attention', 'multiview_transformer': '__delete__',
                       'view_fusion': MULTIVIEW['B1']['model']['view_fusion']}})),
    'anchor-transformer-max-residual': (
        'Geometry transformer on top of the max-fused anchor as a gated residual (tokens start from max).',
        arm({'model': {'multiview_transformer': {'residual_base': True, 'token_init': 'max'}}})),
    'anchor-transformer-no-residual-maxinit': (
        'Geometry transformer output alone, tokens start from the max-fused feature.',
        arm({'model': {'multiview_transformer': {'residual_base': False, 'token_init': 'max'}}})),
    'point-fusion-geometry-attention': (
        'Query-point view fusion by geometry attention instead of max.',
        arm({'model': {'point_fusion': 'geometry_attention'}})),
    'view-update-alternating-2d3d': (
        'Alternate 2D/3D updates: between blocks the 2D feature maps read the 3D anchor tokens.',
        arm({'model': {'multiview_transformer': {k: MULTIVIEW['T1-nomax-alt']['model']['multiview_transformer'][k]
                                                 for k in ('view_update', 'view_update_neighbors',
                                                           'view_update_radius_pixels')}}})),
    'prob-deterministic-control': (
        'Deterministic control for all probability arms: GroupNorm with per-case anchor InstanceNorm, MSE, '
        'nested C(6) subset of D(12) sampler. Differs from the baseline only by normalization and sampler.',
        PROB_CONTROL),
    'prob-gt-cvae': (
        'Conditional VAE with GT-conditioned posterior q(z|C,Y) (repaired, independent bounded heads); '
        'global z=64 via FiLM; deployment samples p(z|C).', arm(probability('gt_cvae'))),
    'prob-dense-np': (
        'Dense-view neural-process posterior q(z|D=12 views), KL(q||p(z|C=6)); global z=64 via FiLM; '
        'deployment uses p(z|C) only.', DENSE_NP),
    'prob-dense-mean-code': (
        'Deterministic mean-code control: q mean code during training, squared alignment of p to it; '
        'separates privileged-code benefit from stochastic modelling.',
        arm(probability('dense_np', mean_code=True, evaluation={'estimator': 'prior_latent_mean'}))),
}

ANATOMY = {'enabled': True, 'guided_sampling': False}


def stage2(root):
    """Geometry/loss detail arms on top of `root` (the baseline, or the structure Stage 1 promoted)."""
    def local(override=(), base=None):
        return arm(override, base=base if base is not None else root)
    # Normalization/probability details are compared with the Stage-1 probability arms, so they stay on
    # the same backbone as those arms regardless of the Stage-2 base.
    prob_control, dense_np = PROB_CONTROL, DENSE_NP
    return {
        'detail-aligned-pixel-coordinates': (
            'Anchor initialization uses the same pixel-center convention and FOV mask as cross-attention.',
            local({'model': {'multiview_transformer': {'coordinate_mode': 'aligned'}}})),
        'detail-anatomy-supervision': (
            'Auxiliary anatomy (structure tensor) supervision; sampling unchanged.',
            local({'model': {'multiview_transformer': {'anatomy': ANATOMY, 'coordinate_mode': 'legacy'}},
                 'dataset': {'structure_supervision': True, 'structure_target': STRUCTURE_TARGET}})),
        'detail-anatomy-guided-sampling': (
            'Anatomy supervision plus predicted-tensor-guided view sampling.',
            local({'model': {'multiview_transformer': {'anatomy': dict(ANATOMY, guided_sampling=True),
                                                     'coordinate_mode': 'legacy'}},
                 'dataset': {'structure_supervision': True, 'structure_target': STRUCTURE_TARGET}})),
        'detail-shared-3d-sampling': (
            'Shared 3D five-point sampling (center + tetrahedron, 8 mm) projected into every view. Requires aligned '
            'pixel coordinates, so its comparator is detail-aligned-pixel-coordinates.',
            local({'model': {'multiview_transformer': {'coordinate_mode': 'aligned', 'sampling_mode': 'shared3d',
                                                     'shared_radius_mm': 8.0}}})),
        'detail-single-scale-features': (
            'Anchor tokens read only the 1/16 feature map (component removal).',
            local({'model': {'multiview_transformer': {'feature_scales': [16]}}})),
        'detail-no-spatial-neighbours': (
            'Spatial attention window 1: no cross-anchor reads (parameters kept).',
            local({'model': {'multiview_transformer': {'spatial_window': 1}}})),
        'detail-gaussian-nll-loss': (
            'Baseline with Gaussian NLL (sigma 0.1 = 50 x MSE) instead of MSE; interacts with grad clip 1.0.',
            local({'loss': {'reconstruction_loss': 'gaussian', 'volume_sigma': 0.1}})),
        'detail-gn-standard-sampler': (
            'Probability control on the baseline 6-view sampler (no nested D): isolates the sampler effect.',
            local({'dataset': {'paired_views': '__delete__'}}, base=prob_control)),
        'detail-gn-pointchannel-anchor-norm': (
            'Probability control with the old GroupNorm policy (per-anchor LayerNorm in the anchor MLPs).',
            local({'model': {'probabilistic': {'normalization': 'groupnorm'}}}, base=prob_control)),
        'detail-dense-np-sigma-init-0.05': (
            'Dense NP with initial latent std 0.05 instead of 0.5 (less FiLM noise early).',
            local({'model': {'probabilistic': {'latent': {'initial_sigma': 0.05}}}}, base=dense_np)),
        'detail-dense-np-beta-x10': (
            'Dense NP with KL weight 2e-4 (10x).',
            local({'model': {'probabilistic': {'kl': {'beta_final': 2.0e-04}}}}, base=dense_np)),
        'detail-dense-np-train-samples-4': (
            'Dense NP with 4 latent samples per training step (averaged reconstruction).',
            local({'model': {'probabilistic': {'train_samples': 4}}}, base=dense_np)),
        'detail-dense-np-local-latent': (
            'Dense NP plus a local latent grid (4^3 cells x 16 dims) conditioned on the global sample.',
            local({'model': {'probabilistic': {'latent': {'use_local': True, 'local_dim': 16, 'grid_res': 4}}}},
                base=dense_np)),
        'detail-dense-np-multi-context-4-6-8': (
            'Dense NP trained with 4/6/8 context views (evaluated at 6).',
            local({'dataset': {'paired_views': {'context_counts': [4, 6, 8]}}}, base=dense_np)),
        'detail-dense-np-gaussian-nll': (
            'Dense NP with Gaussian NLL likelihood (sigma 0.1) and beta 1e-3 (the original repair recipe).',
            local({'model': {'probabilistic': {'reconstruction': {'likelihood': 'gaussian'}, 'kl': {'beta_final': 1.0e-03}}},
                 'loss': {'reconstruction_loss': 'gaussian', 'volume_sigma': 0.1}}, base=dense_np)),
        'detail-dense-np-frozen-reference-distill': (
            'CONDITIONAL. Prior-only distillation KL(stopgrad q_ref || p) from a trained prob-dense-np run of the '
            'same seed. Reference paths are filled per seed by the runner; blocked unless the reference shows '
            'measurable latent usage.',
            local({'model': {'probabilistic': {'objective': 'frozen_reference',
                                             'kl': {'beta_final': 1.0, 'warmup_steps': 0}}}}, base=dense_np)),
    }


def write(stage, arms):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    for name, (description, config) in arms.items():
        text = '# ' + '\n# '.join(line for line in _wrap(description)) + '\n'
        text += '# Generated by scripts/build_experiment_plan.py; edit the generator, not this file.\n'
        (folder / f'{name}.yaml').write_text(text + yaml.safe_dump(config, sort_keys=False))


def _wrap(text, width=100):
    words, line = text.split(), ''
    for word in words:
        if len(line) + len(word) + 1 > width:
            yield line
            line = word
        else:
            line = f'{line} {word}'.strip()
    if line:
        yield line


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2-base', default='baseline-unet-transformer-meaninit',
                        help='Stage-1 arm that Stage-2 details build on (a deterministic, BatchNorm arm)')
    parser.add_argument('--data-root', default='/path/to/LUNA16/C2RV-CBCT/data',
                        help='directory holding the LUNA16 projections/blocks/CT (written into every config)')
    args = parser.parse_args()
    for _, config in STAGE1.values():
        config['dataset']['root_dir'] = args.data_root
    if args.stage2_base not in STAGE1:
        parser.error(f'unknown stage-1 arm {args.stage2_base}')
    root = STAGE1[args.stage2_base][1]
    if 'probabilistic' in root['model']:
        parser.error('Stage-2 base must be a deterministic structure arm; probability arms are rebuilt on it')
    STAGE2 = stage2(root)
    write('stage1', STAGE1)
    write('stage2', STAGE2)
    (OUT / 'stage2' / 'BASE.txt').write_text(args.stage2_base + '\n')
    print(f'{len(STAGE1)} stage-1 and {len(STAGE2)} stage-2 configs (stage-2 base: {args.stage2_base}) '
          f'written to {OUT.relative_to(ROOT)}')