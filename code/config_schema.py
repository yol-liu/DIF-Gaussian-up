MODEL_KEYS = {
    'image_encoder': {'out_ch', 'type', 'vit', 'hierarchical'},
    'point_decoder': {'mlp_chs', 'legacy_head'},
    'gs': {'t', 'res', 'p_dist_scaling', 'o_scaling', 's_scaling', 'knn_k',
           'shifted_center_knn', 'normalize_weights', 'covariance_jitter'},
    'latent': {'use_latent', 'global_dim', 'use_local', 'local_dim', 'grid_res',
               'beta_g', 'beta_l', 'free_bits_g', 'free_bits_l', 'kl_warmup_steps',
               'prior_sigma_floor', 'volume_likelihood', 'volume_sigma', 'lambda_v',
               'gt_feat_ch', 'local_coordinates'},
    'sampling': {'sample_from', 'num_samples', 'eval_mode'},
    'view_fusion': {'d_model', 'num_heads', 'num_layers', 'dropout', 'geometry_hidden_dim',
                    'return_attention', 'view_dropout', 'min_train_views', 'use_max_residual', 'content_query'},
    'anatomy': {'enabled', 'arm', 'hidden_dim', 'length_mm', 'kappa', 'loss_weight'},
    'anchor_fusion': None, 'point_fusion': None, 'mask_fov': None,
    'multiview_transformer': None,
    'probabilistic': None,
}

TRANSFORMER_DEFAULTS = {
    'd_model': 128, 'num_heads': 4, 'num_blocks': 2, 'feature_scales': [4, 8, 16],
    'samples_per_scale': 5, 'spatial_window': 3, 'dropout': 0., 'anchor_chunk_size': 256,
    'point_chunk_size': 8192, 'residual_init': .001, 'fixed_radius_pixels': 1.,
    'footprint_min_pixels': .25, 'footprint_max_pixels': 2.,
    'residual_base': True, 'token_init': 'max',
    'view_update': False, 'view_update_neighbors': 16, 'view_update_radius_pixels': 2.,
    'coordinate_mode': 'legacy', 'sampling_mode': 'detector', 'shared_radius_mm': 8.,
}
ANATOMY_DEFAULTS = {'enabled': False, 'guided_sampling': False, 'length_mm': 8., 'kappa': 4.,
                    'sigma_floor_mm': .25, 'warmup_steps': 100, 'loss_weight': .01}


def resolve_transformer_config(config):
    check_keys(config, set(TRANSFORMER_DEFAULTS) | {'anatomy'}, 'multiview_transformer')
    resolved = dict(TRANSFORMER_DEFAULTS, **config)
    anatomy = config.get('anatomy', {})
    check_keys(anatomy, ANATOMY_DEFAULTS, 'multiview_transformer.anatomy')
    resolved['anatomy'] = dict(ANATOMY_DEFAULTS, **anatomy)
    for key in ('d_model', 'num_heads', 'num_blocks', 'anchor_chunk_size', 'point_chunk_size', 'spatial_window'):
        if not isinstance(resolved[key], int) or resolved[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if resolved['d_model'] < 4 or resolved['d_model'] % resolved['num_heads']:
        raise ValueError('d_model must be >=4 and divisible by num_heads')
    if resolved['spatial_window'] % 2 != 1:
        raise ValueError('spatial_window must be odd')
    scales = resolved['feature_scales']
    if not scales or len(set(scales)) != len(scales) or set(scales) - {4, 8, 16}:
        raise ValueError('feature_scales must be unique members of [4,8,16]')
    if resolved['samples_per_scale'] != 5:
        raise ValueError('Only center plus four neighbors is implemented')
    if not 0 <= resolved['dropout'] < 1 or not 0 < resolved['residual_init'] <= 1:
        raise ValueError('dropout must be in [0,1); residual_init must be small and nonzero')
    if not 0 < resolved['footprint_min_pixels'] <= resolved['footprint_max_pixels'] <= 8:
        raise ValueError('Footprint bounds must satisfy 0 < min <= max <= 8 feature pixels')
    if not 0 < resolved['fixed_radius_pixels'] <= resolved['footprint_max_pixels']:
        raise ValueError('Fixed neighborhood radius must be positive and within footprint bounds')
    if not isinstance(resolved['residual_base'], bool):
        raise ValueError('residual_base must be a boolean')
    if resolved['view_update'] and resolved['num_blocks'] < 2:
        raise ValueError('view_update alternates between blocks; it needs num_blocks >= 2')
    if not isinstance(resolved['view_update_neighbors'], int) or resolved['view_update_neighbors'] < 1 \
            or not resolved['view_update_radius_pixels'] > 0:
        raise ValueError('view_update needs a positive neighbor count and radius')
    if resolved['token_init'] not in ('max', 'mean'):
        raise ValueError('token_init must be max or mean')
    if resolved['coordinate_mode'] not in ('legacy', 'aligned'):
        raise ValueError('coordinate_mode must be legacy or aligned')
    if resolved['sampling_mode'] not in ('detector', 'shared3d'):
        raise ValueError('sampling_mode must be detector or shared3d')
    if not 0 < resolved['shared_radius_mm'] <= 50:
        raise ValueError('shared_radius_mm must be in (0,50] mm')
    if resolved['sampling_mode'] == 'shared3d' and resolved['coordinate_mode'] != 'aligned':
        raise ValueError('shared3d requires aligned coordinates')
    if resolved['token_init'] == 'mean' and resolved['residual_base']:
        raise ValueError('token_init=mean removes max from the anchor path; it requires residual_base=false')
    anatomy = resolved['anatomy']
    if anatomy['guided_sampling'] and (not anatomy['enabled'] or resolved['num_blocks'] < 2):
        raise ValueError('Guided sampling needs anatomy enabled and at least two blocks')
    if not 0 < anatomy['length_mm'] <= 50 or not 0 < anatomy['sigma_floor_mm'] <= anatomy['length_mm']:
        raise ValueError('Anatomy lengths are bounded positive mm values; floor <= length')
    if not 0 <= anatomy['kappa'] <= 20 or anatomy['warmup_steps'] < 0 or anatomy['loss_weight'] <= 0:
        raise ValueError('Invalid anatomy kappa, warmup_steps or loss_weight')
    return resolved


def check_keys(config, allowed, path):
    unknown = set(config) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {path} keys: {sorted(unknown)}')


def validate_config(config):
    check_keys(config, {'model', 'dataset', 'loss', 'transfer', 'ray_consistency'}, 'config')
    check_keys(config['model'], MODEL_KEYS, 'model')
    for section, allowed in MODEL_KEYS.items():
        if allowed is not None and section in config['model']:
            check_keys(config['model'][section], allowed, f'model.{section}')
    check_keys(config['dataset'], {'root_dir', 'gs_res', 'structure_supervision', 'teacher_views', 'subset_mode', 'structure_target', 'paired_views'}, 'dataset')
    if 'probabilistic' in config['model']:
        from probabilistic_config import resolve_probability
        probability = resolve_probability(config['model']['probabilistic'])
        config['model']['probabilistic'] = probability
        if config['model'].get('latent', {}).get('use_latent', False):
            raise ValueError('Use model.probabilistic or legacy model.latent, not both')
        if config.get('transfer', {}).get('enabled', False):
            raise ValueError('Probability objectives and feature transfer must be separate experiments')
        rec = probability['reconstruction']
        expected = {'reconstruction_loss': rec['likelihood'], 'volume_sigma': rec['observation_sigma']}
        if any(config.get('loss', {}).get(key, value) != value for key, value in expected.items()):
            raise ValueError('loss must match probabilistic.reconstruction for fair comparisons')
        config.setdefault('loss', {}).update(expected)
    if 'paired_views' in config['dataset']:
        paired = config['dataset']['paired_views']
        check_keys(paired, {'dense_count', 'context_counts'}, 'dataset.paired_views')
        counts = paired.get('context_counts', [])
        dense = paired.get('dense_count', 0)
        if not counts or any(not isinstance(n, int) or not 0 < n < dense for n in counts):
            raise ValueError('Each context count must satisfy 0 < C < D')
        if config['dataset'].get('teacher_views'):
            raise ValueError('paired_views and teacher_views are separate protocols')
    check_keys(config.get('loss', {}), {'reconstruction_loss', 'volume_sigma', 'lambda_v'}, 'loss')
    check_keys(config.get('transfer', {}), {'enabled', 'checkpoint', 'evidence', 'weight'}, 'transfer')
    transfer = config.get('transfer', {})
    if transfer.get('enabled', False) and (not transfer.get('checkpoint') or not transfer.get('evidence') or not config['dataset'].get('teacher_views')):
        raise ValueError('BLOCKED_TEACHER: alignment requires a validated dense checkpoint, provenance/quality evidence and dataset.teacher_views')
    encoder = config['model'].get('image_encoder', {})
    if encoder.get('type', 'unet') not in ('unet', 'vit', 'hierarchical'):
        raise ValueError('image_encoder.type must be unet, vit or hierarchical')
    check_keys(encoder.get('vit', {}), {'dim', 'depth', 'heads', 'patch_size'}, 'model.image_encoder.vit')
    check_keys(encoder.get('hierarchical', {}), {'dims', 'depths', 'heads', 'window'}, 'model.image_encoder.hierarchical')
    if config['dataset']['gs_res'] != config['model']['gs']['res']:
        raise ValueError('Dataset and model anchor resolutions must agree')
    if config.get('loss', {}).get('reconstruction_loss', 'mse') not in ('mse', 'laplace', 'gaussian'):
        raise ValueError('Unsupported reconstruction loss')
    if config['dataset'].get('subset_mode', 'uniform') not in ('uniform', 'nonuniform'):
        raise ValueError('Unsupported paired-view subset mode')
    anatomy = config['model'].get('anatomy', {})
    if anatomy.get('enabled') and anatomy.get('arm') in ('Gaux', 'Ganat') and not config['dataset'].get('structure_supervision'):
        raise ValueError('Gaux/Ganat training requires structure_supervision=true')
    model = config['model']
    if model.get('anchor_fusion', 'max') not in ('max', 'geometry_attention', 'geometry_transformer'):
        raise ValueError('Unknown anchor_fusion mode')
    if model.get('point_fusion', 'max') not in ('max', 'geometry_attention'):
        raise ValueError('Unknown point_fusion mode')
    enabled = model.get('anchor_fusion') == 'geometry_transformer'
    if model.get('point_fusion') == 'geometry_attention' and not enabled:
        raise ValueError('New point fusion requires the shared geometry_transformer encoder')
    if 'multiview_transformer' in model and not enabled:
        raise ValueError('multiview_transformer configuration provided while the mode is disabled')
    if enabled:
        model['multiview_transformer'] = resolve_transformer_config(model.get('multiview_transformer', {}))
        if model.get('latent', {}).get('use_latent', False) or model.get('anatomy', {}).get('enabled', False):
            raise ValueError('New Transformer arms require latent and the older anatomy encoder disabled')
        if not model.get('mask_fov', True):
            raise ValueError('Transformer requires mask_fov=true for both feature branches')
        anatomy = model['multiview_transformer']['anatomy']
        if anatomy['enabled'] and (not config['dataset'].get('structure_supervision') or
                                    config['dataset'].get('structure_target', {}).get('frame') != 'scanner'):
            raise ValueError('Transformer anatomy requires structure_supervision and scanner-frame targets')
    if 'structure_target' in config['dataset']:
        target = config['dataset']['structure_target']
        check_keys(target, {'frame', 'scale', 'sigma_mm', 'rho_mm', 'energy_threshold', 'cache_dir'}, 'dataset.structure_target')
        if target.get('frame', 'voxel_xyz') not in ('scanner', 'voxel_xyz') or not 0 < target.get('scale', .25) <= 1:
            raise ValueError('Invalid structure target frame or scale')
        if target.get('energy_threshold', 1e-6) < 0:
            raise ValueError('Structure energy threshold must be nonnegative')
        for key in ('sigma_mm', 'rho_mm'):
            if target.get(key) is not None and target[key] <= 0:
                raise ValueError(f'structure_target.{key} must be a positive physical length')
    ray = config.get('ray_consistency', {})
    check_keys(ray, {'enabled'}, 'ray_consistency')
    if ray.get('enabled', False):
        raise ValueError('BLOCKED_CALIBRATION: normalized uint8 CT/projections lack validated TIGRE-to-INR scale/operator matching; ray training is disabled')
