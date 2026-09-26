import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

from config_schema import resolve_transformer_config, validate_config
from datasets.geometry import Geometry
from geometry_adapter import GeometryAdapter, feature_grid, sample_detector_features
from models.model import DIF_Gaussian
from models.multiview_encoder import SpatialLocalAttention
from models.unet import UNet
from structure_targets import physical_structure_target
from training_diagnostics import add_structure_loss, transformer_gradient_norms
from transfer import feature_distillation, teacher_inputs, validate_transfer_provenance
from utils import load_config


def small_config(arm='T4'):
    config = load_config(Path(__file__).resolve().parents[2] / 'configs' / 'multiview' / f'{arm}.yaml')
    config.model.image_encoder.out_ch = 16
    config.model.point_decoder.mlp_chs = [32, 16, 1]
    config.model.gs.res = config.dataset.gs_res = 3
    if arm.startswith('T'):
        transformer = config.model.multiview_transformer
        transformer.d_model = 16
        transformer.anchor_chunk_size = 7
        transformer.point_chunk_size = 11
        transformer.anatomy.warmup_steps = 10
    if config.model.image_encoder.get('type', 'unet') == 'vit':
        config.model.image_encoder.vit = EasyDict(dim=32, depth=3, heads=2, patch_size=16)
    return config


def synthetic_data(batch=2, views=3, queries=23, resolution=3, image_size=32, seed=19):
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.stack(torch.meshgrid(*([torch.arange(resolution) / resolution] * 3), indexing='ij'), -1).reshape(1, -1, 3)
    data = {
        'projs': torch.rand(batch, views, 1, image_size, image_size, generator=generator),
        'angles': torch.linspace(.2, 2., views)[None].expand(batch, -1).clone(),
        'points': torch.rand(batch, queries, 3, generator=generator),
        'points_gs': anchors.expand(batch, -1, -1).clone(),
        'points_gt': torch.rand(batch, 1, queries, generator=generator),
        'volume_extent': torch.tensor([[32., 48., 64.]]).expand(batch, -1).clone(),
        'detector_extent': torch.tensor([[128., 144.]]).expand(batch, -1).clone(),
        'detector_resolution': torch.tensor([[image_size, image_size]]).expand(batch, -1).clone(),
        'DSO': torch.full((batch,), 100.), 'DSD': torch.full((batch,), 150.),
        'view_ids': torch.arange(views)[None].expand(batch, -1).clone(),
        'structure_target': torch.diag(torch.tensor([1., 0., 0.])).expand(batch, resolution ** 3, -1, -1).clone(),
        'structure_confidence': torch.ones(batch, resolution ** 3),
    }
    geometry = GeometryAdapter(data)
    data['points_gs_proj'] = geometry.project(data['points_gs'])['uv']
    data['points_proj'] = geometry.project(data['points'])['uv']
    return data


def without_gt(data):
    return {key: value for key, value in data.items() if key not in ('points_gt', 'structure_target', 'structure_confidence', 'gt_volume')}


def test_resolved_arms_and_unknown_or_blocked_configuration():
    for arm in ('B0', 'B1', 'T1', 'T1-nores', 'T1-nomax', 'T1-nomax-vit', 'T1-nomax-alt', 'T1-nomax-vit-alt', 'T2', 'T3', 'T4'):
        config = small_config(arm)
        validate_config(config)
        assert not config.model.latent.use_latent
    with pytest.raises(ValueError, match='Unknown'):
        resolve_transformer_config({'ignored_option': 1})
    with pytest.raises(ValueError, match='requires residual_base=false'):
        resolve_transformer_config({'token_init': 'mean'})
    with pytest.raises(ValueError, match='at least two blocks'):
        resolve_transformer_config({'num_blocks': 1, 'anatomy': {'enabled': True, 'guided_sampling': True}})
    config = small_config()
    config.ray_consistency.enabled = True
    with pytest.raises(ValueError, match='BLOCKED_CALIBRATION'):
        validate_config(config)
    config.ray_consistency.enabled = False
    config.transfer.enabled = True
    with pytest.raises(ValueError, match='BLOCKED_TEACHER'):
        validate_config(config)


def test_each_ablation_arm_reaches_optimizer_and_gt_free_eval():
    for arm in ('B0', 'B1', 'T1', 'T1-nores', 'T1-nomax', 'T1-nomax-vit', 'T1-nomax-alt', 'T1-nomax-vit-alt', 'T2', 'T3', 'T4'):
        torch.manual_seed(31)
        config = small_config(arm)
        model = DIF_Gaussian(config.model).train()
        data = synthetic_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        prediction = model(data, optimizer_step=10)
        reconstruction = (prediction['points_pred'] - data['points_gt']).square().mean()
        loss, _ = add_structure_loss(prediction, reconstruction, {}, config.model)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        model.eval()
        with torch.no_grad():
            result = model(without_gt(data), is_eval=True, eval_npoint=11)['points_pred']
        assert torch.isfinite(result).all()


def test_unet_multiscale_keys_preserve_original_outputs():
    torch.manual_seed(1)
    model = UNet(1, 8).eval()
    projections = torch.rand(1, 2, 1, 32, 48)
    with torch.no_grad():
        original = model(projections)
        extended = model(projections, return_pyramid=True)
    assert set(original) == {'feats', 'feats_ds'}
    for key in original:
        torch.testing.assert_close(original[key], extended[key], atol=0, rtol=0)
    for scale, channels in ((4, 256), (8, 512), (16, 1024)):
        assert extended['pyramid'][scale].shape == (1, 2, channels, 32 // scale, 48 // scale)


def test_geometry_matches_legacy_and_independent_physical_rays():
    data = synthetic_data(batch=1, views=2)
    data['angles'] = torch.tensor([[0., torch.pi / 2]], dtype=torch.float64)
    points = torch.tensor([[[.6, .7, .4], [.2, .8, .65]]], dtype=torch.float64)
    geometry = GeometryAdapter(data)
    projected = geometry.project(points)
    legacy = Geometry(dict(nVoxel=[32, 24, 16], dVoxel=[1., 2., 4.], nDetector=[64, 48],
                           dDetector=[2., 3.], DSO=100., DSD=150.))
    for index, angle in enumerate(data['angles'][0]):
        np.testing.assert_allclose(projected['uv'][0, index].numpy(), legacy.project(points[0].numpy(), float(angle)), atol=1e-12)
    expected = np.array([6.4 / (100 - 3.2) * 150 / 64, 9.6 / (100 - 3.2) * 150 / 72])
    np.testing.assert_allclose(projected['uv'][0, 0, 0], expected, atol=1e-12)
    assert projected['uv'][0, 1, 0, 1] < 0
    encoded = geometry.ray_encoding(projected, projected['uv'])
    for view, angle in enumerate(data['angles'][0]):
        source = torch.tensor([100 * angle.cos(), 100 * angle.sin(), 0.], dtype=torch.float64)
        direction = torch.nn.functional.normalize(geometry.physical(points)[0] - source, dim=-1)
        torch.testing.assert_close(encoded[0, view, :, :3], direction, atol=1e-12, rtol=1e-12)


def test_jacobian_is_per_mm_with_anisotropic_volume():
    data = synthetic_data(batch=1, views=2)
    data['angles'] = data['angles'].double()
    geometry = GeometryAdapter(data)
    points = torch.tensor([[[.3, .4, .65], [.75, .6, .2]]], dtype=torch.float64)
    jacobian = geometry.project(points)['jacobian']
    for axis, sign in enumerate((1., 1., -1.)):
        delta = torch.zeros_like(points)
        delta[..., axis] = sign * .001 / data['volume_extent'][0, axis]
        finite_difference = (geometry.project(points + delta)['uv'] - geometry.project(points - delta)['uv']) / .002
        torch.testing.assert_close(finite_difference, jacobian[..., axis], atol=1e-10, rtol=1e-6)


def test_feature_grid_axis_order_and_pooling_centers():
    image_shape = (32, 48)
    pixel = torch.tensor([13.5, 21.5])
    uv = (pixel * 2 / (torch.tensor(image_shape) - 1) - 1).reshape(1, 1, 1, 2)
    for scale in (1, 4, 8, 16):
        rows = torch.arange(image_shape[0] // scale) * scale + (scale - 1) / 2
        columns = torch.arange(image_shape[1] // scale) * scale + (scale - 1) / 2
        feature = (rows[:, None] + 10 * columns[None])[None, None, None]
        grid, valid = feature_grid(uv, image_shape, feature.shape[-2:], scale)
        assert valid.all()
        torch.testing.assert_close(sample_detector_features(feature, grid).flatten(), (pixel[0] + 10 * pixel[1]).reshape(1), atol=3e-5, rtol=1e-6)


def test_eval_view_permutation_variable_counts_and_masks():
    torch.manual_seed(4)
    model = DIF_Gaussian(small_config().model).eval()
    data = without_gt(synthetic_data())
    data['view_mask'] = torch.tensor([[True, False, True], [False, False, False]])
    changed = {key: value.clone() for key, value in data.items()}
    changed['projs'][~data['view_mask']] = float('nan')
    with torch.no_grad():
        reference = model(data, is_eval=True)['points_pred']
        masked = model(changed, is_eval=True)['points_pred']
        permutation = torch.tensor([2, 0, 1])
        for key in ('projs', 'angles', 'view_mask', 'view_ids', 'points_proj', 'points_gs_proj'):
            changed[key] = changed[key][:, permutation]
        permuted = model(changed, is_eval=True)['points_pred']
        for count in (1, 5):
            output = model(without_gt(synthetic_data(views=count)), is_eval=True)['points_pred']
            assert torch.isfinite(output).all()
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(reference, masked, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference, permuted, atol=1e-5, rtol=1e-4)


def test_training_mask_excludes_pixels_from_batchnorm_statistics():
    torch.manual_seed(5)
    first = DIF_Gaussian(small_config('T2').model).train()
    second = copy.deepcopy(first)
    data = synthetic_data()
    data['view_mask'] = torch.tensor([[True, False, True], [False, True, False]])
    changed = {key: value.clone() for key, value in data.items()}
    changed['projs'][~data['view_mask']] = 1e8
    reference = first(data)['points_pred']
    actual = second(changed)['points_pred']
    torch.testing.assert_close(reference, actual, atol=0, rtol=0)
    for (name, buffer), (_, other) in zip(first.named_buffers(), second.named_buffers()):
        if 'running_' in name:
            torch.testing.assert_close(buffer, other, atol=0, rtol=0)
    data['view_mask'][0] = False
    with pytest.raises(ValueError, match='at least one valid view'):
        first(data)


def test_invalid_depth_outside_fov_and_empty_eval_batch_are_finite():
    model = DIF_Gaussian(small_config('T2').model).eval()
    data = without_gt(synthetic_data())
    data['DSO'][:] = 2.
    data['points'][:, :3] = torch.tensor([20., 20., 20.])
    with torch.no_grad():
        assert torch.isfinite(model(data, is_eval=True)['points_pred']).all()
        data['view_mask'] = torch.zeros(2, 3, dtype=torch.bool)
        assert torch.isfinite(model(data, is_eval=True)['points_pred']).all()
    data['scanner_origin_offset'] = torch.ones(2, 3)
    with pytest.raises(ValueError, match='Uncalibrated nonzero'):
        model(data)


def test_reconstruction_reaches_attention_and_guided_structure_on_first_step():
    torch.manual_seed(6)
    model = DIF_Gaussian(small_config().model).train()
    data = synthetic_data()
    prediction = model(data, optimizer_step=10)
    reconstruction = (prediction['points_pred'] - data['points_gt']).square().mean()
    reconstruction.backward()
    norms = transformer_gradient_norms(model)
    for group in ('cross_attention', 'spatial_attention', 'point_fusion', 'structure_head'):
        assert np.isfinite(norms['gradient/' + group]) and norms['gradient/' + group] > 0, norms


def test_anchor_path_without_max_base_or_gate():
    from models.utils import query_view_feats
    for arm, fusion in (('T1-nores', 'max'), ('T1-nomax', 'mean')):
        torch.manual_seed(6)
        model = DIF_Gaussian(small_config(arm).model).train()
        assert not hasattr(model.multiview_encoder, 'residual_gate')
        data = synthetic_data()
        base = {}
        model.multiview_encoder.register_forward_hook(lambda module, inputs, output: base.update(input=inputs[1], output=output[0]))
        prediction = model(data, optimizer_step=10)
        encoder_output = model.image_encoder(data['projs'], return_pyramid=True)
        expected = query_view_feats(encoder_output['feats_ds'], GeometryAdapter(data).project(data['points_gs'])['uv'],
                                    fusion=fusion, mask_fov=True)
        assert torch.allclose(base['input'], expected, atol=1e-5)
        assert not torch.allclose(base['output'], base['input'])
        (prediction['points_pred'] - data['points_gt']).square().mean().backward()
        output_gradient = model.multiview_encoder.output_projection.weight.grad
        assert output_gradient is not None and output_gradient.abs().sum() > 0


def test_vit_encoder_matches_unet_outputs_and_masks():
    from models.vit_encoder import ViTEncoder
    torch.manual_seed(3)
    images = torch.rand(2, 3, 1, 64, 48)
    unet, vit = UNet(1, 8).eval(), ViTEncoder(1, 8, dim=32, depth=3, heads=2).eval()
    with torch.no_grad():
        reference, result = unet(images, return_pyramid=True), vit(images, return_pyramid=True)
        assert result['feats'].shape == reference['feats'].shape
        assert result['feats_ds'].shape == reference['feats_ds'].shape
        for scale in (4, 8, 16):
            assert result['pyramid'][scale].shape == reference['pyramid'][scale].shape
        mask = torch.tensor([[True, False, True], [True, True, False]])
        masked = vit(images, view_mask=mask, return_pyramid=True)
        changed = images.clone()
        changed[~mask] = torch.rand_like(changed[~mask])
        again = vit(changed, view_mask=mask, return_pyramid=True)
    assert torch.equal(masked['feats'][~mask], torch.zeros_like(masked['feats'][~mask]))
    assert torch.allclose(masked['feats'], again['feats'])


def test_view_update_pixel_centers_invert_feature_grid():
    from models.multiview_encoder import ViewUpdate
    image_shape, stride = (64, 48), 16
    height, breadth = image_shape[0] // stride, image_shape[1] // stride
    centers = torch.stack(torch.meshgrid(torch.arange(float(height)), torch.arange(float(breadth)), indexing='ij'), -1).reshape(-1, 2)
    pixel_uv = (centers * stride + (stride - 1) / 2) * 2 / torch.tensor([size - 1. for size in image_shape]) - 1
    grid, _ = feature_grid(pixel_uv[None, None], image_shape, (height, breadth), stride)
    expected = 2 * centers / torch.tensor([height - 1., breadth - 1.]) - 1
    assert torch.allclose(grid[0, 0], expected, atol=1e-6)
    assert isinstance(ViewUpdate(8, 2, [4, 8, 16], 4, 2., 0.), torch.nn.Module)


def test_alternating_update_changes_next_block_and_trains():
    torch.manual_seed(6)
    model = DIF_Gaussian(small_config('T1-nomax-alt').model).train()
    update = model.multiview_encoder.view_updates[0]
    data = synthetic_data(image_size=64)
    seen = {}
    update.register_forward_hook(lambda module, inputs, output: seen.update(before=inputs[0], after=output[0]))
    prediction = model(data, optimizer_step=10)
    for scale in (4, 8, 16):
        assert not torch.allclose(seen['before'][scale], seen['after'][scale]), scale
    assert prediction['diagnostics']['block0/view_update_pixel_coverage'] > 0
    (prediction['points_pred'] - data['points_gt']).square().mean().backward()
    for name in ('cross_attention.value.weight', 'self_attention.in_proj_weight', 'propagate.4.weight', 'relative.0.weight'):
        gradient = dict(update.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum() > 0, name


def test_masked_mean_view_fusion_ignores_invalid_views():
    from models.utils import query_view_feats
    features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float).reshape(1, 2, 3, 4, 4)
    coordinates = torch.tensor([[[[0., 0.]], [[3., 3.]]]])
    both = query_view_feats(features, coordinates.clone().fill_(0.), fusion='mean', mask_fov=True)
    single = query_view_feats(features, coordinates, fusion='mean', mask_fov=True)
    only_first = query_view_feats(features[:, :1], coordinates[:, :1], fusion='max', mask_fov=True)
    assert torch.allclose(single, only_first)
    assert not torch.allclose(both, single)
    empty = query_view_feats(features, torch.full_like(coordinates, 3.), fusion='mean', mask_fov=True)
    assert torch.equal(empty, torch.zeros_like(empty))


def test_supervision_only_head_gradient_and_real_loss_integration():
    torch.manual_seed(7)
    config = small_config('T3')
    model = DIF_Gaussian(config.model).train()
    data = synthetic_data()
    prediction = model(data)
    reconstruction = (prediction['points_pred'] - data['points_gt']).square().mean()
    total, metrics = add_structure_loss(prediction, reconstruction, {}, config.model)
    torch.testing.assert_close(total, reconstruction + .01 * prediction['structure_loss'])
    total.backward()
    assert transformer_gradient_norms(model)['gradient/structure_head'] > 0
    assert float(metrics['L_anatomy']) > 0
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(without_gt(data), is_eval=True)['points_pred']).all()


def test_chunk_invariance_and_encode_project_maps_once():
    torch.manual_seed(8)
    model = DIF_Gaussian(small_config().model).eval()
    data = without_gt(synthetic_data())
    counters = {'encoder': 0, 'point_projection': 0}
    def count_encoder(*unused):
        counters['encoder'] += 1
    def count_projection(*unused):
        counters['point_projection'] += 1
    hooks = [model.image_encoder.register_forward_hook(count_encoder),
             model.point_view_fusion.feature_projection.register_forward_hook(count_projection)]
    with torch.no_grad():
        first = model(data, is_eval=True, eval_npoint=5)['points_pred']
    assert counters == {'encoder': 1, 'point_projection': 1}
    for hook in hooks:
        hook.remove()
    model.multiview_encoder.chunk_size = 19
    model.point_view_fusion.chunk_size = 3
    for block in model.multiview_encoder.blocks:
        block.spatial_attention.chunk_size = 17
    with torch.no_grad():
        second = model(data, is_eval=True, eval_npoint=23)['points_pred']
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)


def test_spatial_neighbors_cross_chunks_without_global_or_wraparound_attention():
    torch.manual_seed(9)
    spatial = SpatialLocalAttention(8, 2, 3, 3, 1, 0.)
    positions = torch.stack(torch.meshgrid(*([torch.arange(3) / 3] * 3), indexing='ij'), -1).reshape(1, 27, 3)
    tokens = torch.randn(1, 27, 8, requires_grad=True)
    output = spatial(tokens, positions)
    output[0, 0].square().sum().backward()
    assert tokens.grad[0, 1].abs().sum() > 0
    assert tokens.grad[0, 26].abs().sum() == 0
    assert int(spatial.valid[0].sum()) == 8
    tokens = tokens.detach().requires_grad_()
    update = tokens + spatial(tokens, positions)
    next_update = spatial(update, positions)
    next_update[0, 0].square().sum().backward()
    assert tokens.grad[0, 26].abs().sum() > 0


def test_anatomy_frame_psd_energy_and_anisotropic_plane():
    axes = np.indices((33, 35, 37)).astype(np.float32)
    spacing = np.array([1., 2., 3.])
    field = axes[0] + 2 * axes[1] * 2 + axes[2] * 3
    volume = (field > field[16, 17, 18]).astype(np.float32)
    tensor, confidence = physical_structure_target(volume, spacing, np.array([[.5, .5, .5]]), frame='scanner', energy_threshold=1e-6)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor[0])
    expected = np.array([1., 2., -1.]) / np.sqrt(6)
    assert abs(eigenvectors[:, -1] @ expected) > .98
    assert eigenvalues.min() > -1e-6 and confidence[0] > 0 and confidence[0] <= 1
    tensor, confidence = physical_structure_target(np.ones_like(volume), spacing, np.array([[.5, .5, .5]]), frame='scanner')
    assert np.isfinite(tensor).all() and confidence[0] == 0


def test_guided_sampling_changes_next_block_with_warmup_and_gradient():
    torch.manual_seed(11)
    config = small_config()
    encoder = DIF_Gaussian(config.model).multiview_encoder
    data = synthetic_data()
    geometry = GeometryAdapter(data)
    projection = geometry.project(data['points_gs'])
    tensor = torch.diag(torch.tensor([.8, .1, .1])).expand(2, 27, -1, -1).clone().requires_grad_()
    fixed, _ = encoder._offsets(projection, tensor, (32, 32), 4, 0.)
    guided, _ = encoder._offsets(projection, tensor, (32, 32), 4, 1.)
    assert not torch.allclose(fixed, guided)
    guided.square().sum().backward()
    assert tensor.grad.abs().sum() > 0
    radius = torch.linalg.vector_norm(guided[..., 1:, :] / (8 / 31), dim=-1)
    assert radius.min() >= .25 - 1e-6 and radius.max() <= 2 + 1e-6


def test_shared_3d_interpolation_changes_point_fusion_query():
    torch.manual_seed(12)
    model = DIF_Gaussian(small_config('T2').model).eval()
    data = synthetic_data()
    with torch.no_grad():
        encoded = model.encode_projs(data)
        features = torch.rand(2, 16, 23)
        first = model.point_view_fusion(encoded['feats_proj'], encoded['point_feature_maps'], features, data)
        second = model.point_view_fusion(encoded['feats_proj'], encoded['point_feature_maps'], features * 4, data)
    assert (first - second).abs().max() > 0


def test_cpu_bfloat16_autocast_is_finite():
    model = DIF_Gaussian(small_config().model).train()
    data = synthetic_data()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(data, optimizer_step=10)
        loss = output['points_pred'].float().square().mean() + .01 * output['structure_loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_teacher_interface_is_mask_safe_and_requires_provenance():
    data = synthetic_data()
    data['view_mask'] = torch.ones(2, 3, dtype=torch.bool)
    for key in ('projs', 'angles', 'view_ids', 'points_proj', 'points_gs_proj'):
        data['teacher_' + key] = data[key].clone()
    inputs = teacher_inputs(data)
    assert 'view_mask' not in inputs and 'structure_target' not in inputs and 'points_gt' not in inputs
    with pytest.raises(ValueError, match='BLOCKED_TEACHER_SEMANTICS'):
        validate_transfer_provenance({}, ['train_case'], 'hash', 'hash')
    student = torch.randn(1, 8, 4, requires_grad=True)
    teacher = torch.randn(1, 8, 4, requires_grad=True)
    reliable = torch.tensor([[1., 0., 1., 0.]])
    feature_distillation(student, teacher, reliable).backward()
    assert teacher.grad is None
    assert student.grad[:, :, 1::2].abs().sum() == 0
