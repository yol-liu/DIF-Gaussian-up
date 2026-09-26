import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from checkpoints import rng_state, restore_rng, warm_start, resume_training
from evaluation_protocol import evaluating, scale_comparisons
from models.point_decoder import PointDecoder
from models.utils import query_view_feats
from models.latent_np import gaussian_kl, apply_free_bits, LocalLatentGrid
from models.probabilistic_model import ProbabilisticDIFGaussian
from models.anatomy import physical_projection, AnatomySupportEncoder, structure_loss
from datasets.geometry import Geometry
from losses import compute_collapse_diagnostics
from ray_consistency import project_attenuation, calibrated_ray_loss
from structure_targets import physical_structure_target
from transfer import feature_distillation, validate_teacher
from test_latent import make_cfg, make_data
from datasets.base import CBCT_dataset
from types import SimpleNamespace
from research_gates import pilot_gate


def test_linear_head_and_strict_legacy_roundtrip():
    linear = PointDecoder([4, 8, 1])
    legacy = PointDecoder([4, 8, 1], legacy_head=True)
    assert len(linear.mlps[-1]) == 1 and len(legacy.mlps[-1]) == 3
    replica = PointDecoder([4, 8, 1], legacy_head=True)
    replica.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval()
    replica.eval()
    inputs = torch.randn(2, 4, 13)
    torch.testing.assert_close(legacy(inputs), replica(inputs), atol=0, rtol=0)


def test_fov_negative_features_and_all_invalid():
    features = -torch.ones(1, 2, 1, 4, 4)
    coordinates = torch.tensor([[[[0., 0.], [2., 2.]], [[2., 2.], [float('nan'), 0.]]]])
    result = query_view_feats(features, coordinates, mask_fov=True)
    torch.testing.assert_close(result, torch.tensor([[[-1., 0.]]]))


def test_kl_matches_distribution_formula_and_freebits():
    mean_q, mean_p = torch.randn(2, 8), torch.randn(2, 8)
    log_q, log_p = torch.randn(2, 8), torch.randn(2, 8)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean_q, (log_q / 2).exp()),
        torch.distributions.Normal(mean_p, (log_p / 2).exp()))
    torch.testing.assert_close(gaussian_kl(mean_q, log_q, mean_p, log_p), expected)
    zero = gaussian_kl(mean_q, log_q, mean_q, log_q)
    assert zero.abs().max() < 1e-6
    assert float(apply_free_bits(torch.zeros(1, 64), 0.5)) == 32.


def test_diagnostics_posterior_and_exception_restore():
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True))
    with torch.no_grad():
        model.global_latent.delta_net[-1].bias[:8].fill_(0.5)
    data = make_data(with_gt_volume=True)
    model.train()
    diagnostic = compute_collapse_diagnostics(model, data, num_samples=2, eval_npoint=7)
    assert diagnostic['KL_global_raw'] > 0 and model.training
    assert len(diagnostic['kl_g_per_dim']) == 8
    data.pop('gt_volume')
    unavailable = compute_collapse_diagnostics(model, data)
    assert unavailable['posterior_available'] is False
    assert unavailable['KL_global_raw'] is None
    try:
        compute_collapse_diagnostics(model, data, num_samples=1)
    except ValueError:
        pass
    else:
        raise AssertionError('Diversity diagnostic must require multiple samples')
    assert model.training


def test_seed_mean_mc_and_chunk_protocol():
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True, use_local=True)).eval()
    with torch.no_grad():
        model.feature_film.net[-1].weight.normal_(0, 0.03)
    data = make_data(N=31)
    with torch.no_grad():
        torch.manual_seed(91)
        sampled = model(data, is_eval=True, sample_from='prior', num_samples=3, eval_npoint=31)['sample_predictions']
        torch.manual_seed(91)
        mean = model(data, is_eval=True, latent_mode='prior_mc_mean', num_samples=3, eval_npoint=7)['points_pred']
        torch.testing.assert_close(mean, sampled.mean(0), atol=1e-5, rtol=1e-4)
        first = model(data, is_eval=True, latent_mode='prior_mean')['points_pred']
        second = model(data, is_eval=True, latent_mode='prior_mean', eval_npoint=3)['points_pred']
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)


def test_eval_context_restores_mixed_modes():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.train()
    model[0].eval()
    try:
        with evaluating(model):
            raise RuntimeError('injected')
    except RuntimeError:
        pass
    assert model.training and not model[0].training and model[1].training


def test_resume_optimizer_scheduler_rng_and_warm_start_guard():
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), global_step=12, epoch=3,
                                    rng_by_rank=[rng_state()], config={}))
    expected = torch.rand(4)
    start, step = resume_training(model, optimizer, scheduler, checkpoint)
    torch.testing.assert_close(torch.rand(4), expected)
    assert (start, step) == (4, 12) and optimizer.param_groups[0]['lr'] == 0.05
    try:
        warm_start(model, {'weight': checkpoint['model']['weight']})
    except ValueError:
        pass
    else:
        raise AssertionError('Undeclared missing backbone must fail')


def test_local_cell_centers_interpolate_exact_cells():
    latent = LocalLatentGrid(ctx_dim=2, global_dim=2, latent_dim=1, grid_res=2)
    centers = torch.stack(torch.meshgrid(*([torch.tensor([0.25, 0.75])] * 3), indexing='ij'), -1).reshape(1, 8, 3)
    grid = torch.arange(8.).reshape(1, 1, 2, 2, 2)
    torch.testing.assert_close(latent.interpolate_to_anchors(grid, centers).flatten(), torch.arange(8.))


def geometry_fixture(dtype=torch.float64):
    config = dict(nVoxel=[16, 20, 24], dVoxel=[1., 2., 3.], nDetector=[64, 80],
                  dDetector=[2., 3.], DSO=800., DSD=1200.)
    geometry = {'volume_extent': torch.tensor([[16., 40., 72.]], dtype=dtype),
                'detector_extent': torch.tensor([[128., 240.]], dtype=dtype),
                'DSO': torch.tensor([800.], dtype=dtype), 'DSD': torch.tensor([1200.], dtype=dtype)}
    return config, geometry


def test_projection_and_physical_jacobian():
    config, geometry = geometry_fixture()
    points = torch.tensor([[[.3, .6, .7], [.5, .5, .5]]], dtype=torch.float64)
    angles = torch.tensor([[.2, 1.1]], dtype=torch.float64)
    projected, jacobian = physical_projection(points, angles, geometry)
    for view, angle in enumerate(angles[0]):
        np.testing.assert_allclose(projected[0, view].numpy(), Geometry(config).project(points[0].numpy(), float(angle)), atol=1e-10)
    for axis in range(3):
        delta = torch.zeros_like(points)
        delta[..., axis] = 1e-3 / geometry['volume_extent'][0, axis] * (-1 if axis == 2 else 1)
        plus = physical_projection(points + delta, angles, geometry)[0]
        minus = physical_projection(points - delta, angles, geometry)[0]
        torch.testing.assert_close((plus - minus) / .002, jacobian[..., axis], atol=1e-9, rtol=1e-5)


def test_footprint_covariance_linearization():
    _, geometry = geometry_fixture()
    points = torch.tensor([[[.4, .55, .6]]], dtype=torch.float64)
    angles = torch.tensor([[.7]], dtype=torch.float64)
    _, jacobian = physical_projection(points, angles, geometry)
    generator = torch.Generator().manual_seed(23)
    offsets = torch.randn(1, 2000, 3, generator=generator, dtype=torch.float64)
    errors = []
    for radius in (8., .08):
        perturbation = offsets * radius
        exact_points = points + perturbation / geometry['volume_extent'][:, None] * torch.tensor([1., 1., -1.])
        projected = physical_projection(exact_points, angles, geometry)[0][0, 0]
        linear = perturbation[0] @ jacobian[0, 0, 0].T
        exact_cov = torch.cov(projected.T)
        linear_cov = torch.cov(linear.T)
        errors.append(float((exact_cov - linear_cov).norm() / linear_cov.norm()))
    assert errors[1] < errors[0] and errors[1] < 1e-3


def test_structure_targets_plane_and_uniform():
    grid = np.indices((17, 19, 21)).astype(float)
    volume = (grid[0] + grid[1] * 2 > 26).astype(np.float32)
    target, confidence = physical_structure_target(volume, [1., 2., 3.], np.array([[.5, .5, .5]]))
    assert np.linalg.eigvalsh(target).min() >= -1e-6 and confidence[0] > 0
    _, uniform_confidence = physical_structure_target(np.ones_like(volume), [1., 2., 3.], np.array([[.5, .5, .5]]))
    assert uniform_confidence[0] == 0


def test_anatomy_arms_residual_and_gradients_without_gt():
    _, geometry = geometry_fixture(torch.float32)
    data = dict(geometry, points_gs=torch.stack(torch.meshgrid(*([torch.arange(2) / 2] * 3), indexing='ij'), -1).reshape(1, 8, 3),
                angles=torch.tensor([[0., .5]]), detector_resolution=torch.tensor([[64., 80.]]))
    features = torch.randn(1, 8, 8)
    pyramid = {'feats': torch.randn(1, 2, 4, 32, 32), 'feats_ds': torch.randn(1, 2, 8, 2, 2)}
    for arm in ('G0', 'Gfree', 'Ggeo', 'Gaux', 'Ganat'):
        encoder = AnatomySupportEncoder(8, 4, 2, dict(arm=arm, hidden_dim=4))
        output, tensor = encoder(features, pyramid, data)
        torch.testing.assert_close(output, features, atol=0, rtol=0)
        assert torch.linalg.eigvalsh(tensor).min() >= 0
        output.square().mean().backward()
        assert encoder.residual[-1].weight.grad.abs().sum() > 0
        with torch.no_grad():
            encoder.residual[-1].weight.normal_(0, .1)
        encoder.zero_grad()
        output, tensor = encoder(features, pyramid, data)
        output.square().mean().backward()
        if arm == 'Ganat':
            assert encoder.structure.weight.grad.abs().sum() > 0


def test_ray_constant_box_sphere_and_gradient():
    volume = torch.ones(1, 1, 17, 17, 17, requires_grad=True)
    rays = {'start': torch.tensor([[[-1., .5, .5], [-1., 2., .5]]]),
            'end': torch.tensor([[[2., .5, .5], [2., 2., .5]]])}
    geometry = dict(volume_extent=[10., 20., 30.], attenuation_unit='mm^-1')
    result = project_attenuation(volume, geometry, rays, samples=64)
    torch.testing.assert_close(result, torch.tensor([[10., 0.]]))
    result.sum().backward()
    assert volume.grad.abs().sum() > 0
    sphere = lambda points: ((points - .5).square().sum(-1) < .25 ** 2).float()
    result = project_attenuation(sphere, geometry, rays, samples=512)
    torch.testing.assert_close(result, torch.tensor([[5., 0.]]), atol=.04, rtol=0)
    try:
        calibrated_ray_loss(result, result, {})
    except ValueError:
        pass
    else:
        raise AssertionError('Uncalibrated ray loss must be blocked')


def test_distillation_stop_gradient_and_teacher_gate():
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8, requires_grad=True)
    feature_distillation(student, teacher).backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None
    try:
        validate_teacher({})
    except ValueError:
        pass
    else:
        raise AssertionError('Missing teacher evidence must block transfer')


def test_scale_protocol_does_not_force_point_samples_equal_filtering():
    generator = np.random.default_rng(0)
    target = generator.random((16, 16, 16), dtype=np.float32)
    prediction = target * .8
    coarse = prediction[::2, ::2, ::2]
    result = scale_comparisons(target, prediction, coarse, [1., 2., 3.])
    assert result['S4']['unclipped_mse'] > 0
    assert set(result) == {'S1', 'S2', 'S3', 'S4'}


def test_eval_projection_cache_tracks_case_angles():
    dataset = CBCT_dataset.__new__(CBCT_dataset)
    config, _ = geometry_fixture()
    dataset.cfg = {}
    dataset.geo = Geometry(config)
    dataset.name_list = ['first', 'second']
    dataset.is_train = False
    dataset.load_gt_volume = False
    dataset.dst_name = 'synthetic'
    dataset.points = np.array([[.25, .65, .45]], dtype=np.float32)
    dataset.points_proj = dataset.points_proj_key = None
    dataset.sample_projections = lambda name, **kwargs: (
        np.zeros((1, 1, 4, 4)), np.array([0. if name == 'first' else 1.]), np.array([0]))
    dataset.load_ct = lambda name, **kwargs: (np.zeros((3, 3, 3)), np.ones(3), np.zeros(3), np.eye(3).flatten())
    first, second = dataset[0], dataset[1]
    assert not np.allclose(first['points_proj'], second['points_proj'])
    np.testing.assert_allclose(second['points_proj'], dataset.project_points(dataset.points, np.array([1.])))


def test_view_dropout_subsets_all_branches_before_encoder():
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True, anchor_fusion='geometry_attention')).train()
    model.view_dropout = .9
    model.min_train_views = 1
    data = make_data(B=2, M=4)
    data['view_ids'] = torch.arange(4).expand(2, -1)
    torch.manual_seed(0)
    selected = model.prepare_views(data)
    indices = selected['view_ids'][0]
    assert len(indices) < 4
    for key in ('projs', 'angles', 'points_proj', 'points_gs_proj'):
        torch.testing.assert_close(selected[key], data[key][:, indices])


def test_pilot_gate_requires_fair_updates_and_paired_cases():
    baseline = dict(train_updates=100, seed=0, loss_protocol='mse', eval_scale=1.,
                    resampling_protocol='legacy', latent_mode='prior_mean',
                    cases=[dict(name=str(index), psnr=25., ssim=.8, boundary_error=.1, view_ids=[0, 4]) for index in range(6)])
    candidate = copy.deepcopy(baseline)
    for case in candidate['cases']:
        case['psnr'] += .2
    assert pilot_gate(baseline, candidate)['status'] == 'PROMOTED_PILOT_ONLY'
    candidate['train_updates'] = 99
    assert pilot_gate(baseline, candidate)['status'] == 'INCONCLUSIVE'
    candidate['train_updates'] = 100
    candidate['cases'][0]['view_ids'] = [1, 5]
    assert pilot_gate(baseline, candidate)['status'] == 'INCONCLUSIVE'
