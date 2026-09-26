import copy
from pathlib import Path

import pytest
import torch

from geometry_adapter import GeometryAdapter, aligned_anchor_features, feature_grid, sample_detector_features
from models.hierarchical_encoder import HierarchicalEncoder, WindowBlock
from models.model import DIF_Gaussian
from models.unet import UNet
from test_multiview_transformer import small_config, synthetic_data, without_gt
from training_diagnostics import add_structure_loss
from utils import load_config


def test_aligned_initialization_samples_actual_pixel_centers_and_masks():
    data = synthetic_data(batch=1, views=2)
    image_shape = (64, 80)
    rows = torch.arange(4) * 16 + 7.5
    cols = torch.arange(5) * 16 + 7.5
    features = (rows[:, None] + 10 * cols)[None, None, None].expand(1, 2, 1, -1, -1)
    pixel = torch.tensor([23.5, 39.5])
    uv = (pixel * 2 / (torch.tensor(image_shape) - 1) - 1).reshape(1, 1, 1, 2).expand(1, 2, 1, 2)
    projected = {'uv': uv, 'valid': torch.tensor([[[True], [False]]])}
    for fusion in ('mean', 'max'):
        result = aligned_anchor_features(features, projected, image_shape, 16, fusion)
        torch.testing.assert_close(result.flatten(), torch.tensor([418.5]))
        projected_empty = dict(projected, valid=torch.zeros_like(projected['valid']))
        assert aligned_anchor_features(features, projected_empty, image_shape, 16, fusion).eq(0).all()
    # Pixels inside the detector but outside coarse pixel centers must be masked.
    projected['uv'] = uv.clone().fill_(-1)
    assert aligned_anchor_features(features, projected, image_shape, 16, 'mean').eq(0).all()


def test_shared_samples_lie_on_the_same_physical_points_for_all_views():
    cfg = small_config('T1-nomax')
    cfg.model.multiview_transformer.update(coordinate_mode='aligned', sampling_mode='shared3d', shared_radius_mm=3.)
    encoder = DIF_Gaussian(cfg.model).multiview_encoder
    data = synthetic_data(batch=1, views=3)
    geometry = GeometryAdapter(data)
    points = torch.tensor([[[.5, .5, .5], [.6, .4, .45]]])
    shared = encoder.shared_observations(geometry, points)
    physical = geometry.physical(shared['points'].flatten(1, 2)).reshape(1, 2, 5, 3)
    center = geometry.physical(points)
    torch.testing.assert_close(physical[:, :, 0], center)
    torch.testing.assert_close((physical[:, :, 1:] - center[:, :, None]).norm(dim=-1), torch.full((1, 2, 4), 3.), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(physical[:, :, 1:].mean(2), center, atol=1e-5, rtol=1e-5)
    # Independently intersect each encoded detector ray with the sampled point.
    rays = geometry.ray_encoding(shared, shared['uv'])[..., :3]
    for view, angle in enumerate(data['angles'][0]):
        source = torch.stack([100 * angle.cos(), 100 * angle.sin(), torch.tensor(0.)])
        expected = torch.nn.functional.normalize(physical - source, dim=-1)
        torch.testing.assert_close(rays[:, view], expected, atol=1e-6, rtol=1e-5)
    for scale in (4, 8, 16):
        grid, _ = feature_grid(shared['uv'], (64, 64), (64 // scale, 64 // scale), scale)
        detector_pixels = ((grid + 1) * (64 // scale - 1) / 2) * scale + (scale - 1) / 2
        expected = (shared['uv'] + 1) * 63 / 2
        torch.testing.assert_close(detector_pixels, expected, atol=4e-6, rtol=1e-5)


def test_shifted_windows_do_not_wrap_opposite_boundaries():
    torch.manual_seed(7)
    model = WindowBlock(8, 2, window=4, shifted=True).eval()
    x = torch.randn(1, 8, 8, 8)
    changed = x.clone()
    changed[:, :, -1, -1] += 100
    with torch.no_grad():
        a, b = model(x), model(changed)
    torch.testing.assert_close(a[:, :, 0, 0], b[:, :, 0, 0], atol=0, rtol=0)
    assert not torch.allclose(a[:, :, -1, -1], b[:, :, -1, -1])


def test_hierarchy_shared_decoder_initialization_and_view_masks():
    torch.manual_seed(11)
    reference = UNet(1, 8)
    model = HierarchicalEncoder(1, 8, dims=(16, 32, 64), depths=(2, 2, 2), heads=(2, 4, 8), window=4).eval()
    source, target = reference.state_dict(), model.state_dict()
    shared = {k: source[k] for k in target if k in source and target[k].shape == source[k].shape}
    for prefix in ('inc.', 'down.0.', 'up.', 'outc.'):
        assert all(k in shared for k in target if k.startswith(prefix))
    model.load_state_dict(shared, strict=False)
    x = torch.rand(1, 3, 1, 32, 48)
    mask = torch.tensor([[True, False, True]])
    with torch.no_grad():
        result = model(x, mask, True)
        x[:, 1] = float('nan')
        changed = model(x, mask, True)
        empty = model(x, torch.zeros_like(mask), True)
    torch.testing.assert_close(result['feats'], changed['feats'], atol=0, rtol=0)
    assert empty['feats'].eq(0).all()
    for scale, channels in ((4, 256), (8, 512), (16, 1024)):
        assert result['pyramid'][scale].shape == (1, 3, channels, 32 // scale, 48 // scale)


@pytest.mark.parametrize('arm', ['R0', 'C0', 'A0', 'A1', 'S0', 'H0'])
def test_new_arms_backward_gt_free_eval_and_view_permutation(arm):
    config = load_config(Path(__file__).resolve().parents[2] / 'configs/geometry_validation' / f'{arm}.yaml')
    config.model.gs.res = config.dataset.gs_res = 3
    config.model.image_encoder.out_ch = 16
    config.model.point_decoder.mlp_chs = [32, 16, 1]
    config.model.multiview_transformer.update(d_model=16, anchor_chunk_size=13, point_chunk_size=11)
    if arm == 'H0':
        config.model.image_encoder.hierarchical.update(dims=[16, 32, 64], heads=[2, 4, 8], window=4)
    torch.manual_seed(13)
    model = DIF_Gaussian(config.model).train()
    data = synthetic_data(batch=1, views=3)
    result = model(data, optimizer_step=100)
    mse = (result['points_pred'] - data['points_gt']).square().mean()
    loss, _ = add_structure_loss(result, mse, {}, config.model)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.multiview_encoder.blocks[0].cross_attention.key.weight.grad.abs().sum() > 0
    if arm == 'H0':
        assert model.image_encoder.stages[0][0].qkv.weight.grad.abs().sum() > 0
    if arm == 'A1':
        # Isolate sampling-path gradient from the explicit GT auxiliary loss.
        model.zero_grad()
        fresh = model(data, optimizer_step=100)
        fresh['points_pred'].square().mean().backward()
        assert model.multiview_encoder.structure_head[-1].weight.grad.abs().sum() > 0
    model.eval()
    clean = without_gt(data)
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(clean)
    for key in ('projs', 'angles', 'view_ids', 'points_proj', 'points_gs_proj'):
        permuted[key] = clean[key][:, permutation]
    with torch.no_grad():
        a = model(clean, is_eval=True, eval_npoint=11)['points_pred']
        b = model(permuted, is_eval=True, eval_npoint=7)['points_pred']
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
