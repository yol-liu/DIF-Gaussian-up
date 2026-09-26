'''P0 guarantees for the fixed 6-view structural experiment plan.

Covers the three RNG streams (model init / case order / block-and-point indices),
the shared initialization contract between arms, and the arm configuration matrix.
'''
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets.base import CBCT_dataset
from datasets.geometry import Geometry
from models.probabilistic_model import build_model
from utils import load_config

ARMS = ['B0', 'B1', 'T1', 'T2', 'T3', 'T4']
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'multiview'))


def arm_config(arm):
    return load_config(os.path.join(CONFIG_DIR, f'{arm}.yaml'))


def training_dataset(sample_seed, blocks=8, points=64, npoint=16):
    dataset = CBCT_dataset.__new__(CBCT_dataset)
    dataset.cfg = {}
    dataset.name_list = ['case_a', 'case_b']
    dataset.is_train = True
    dataset.load_gt_volume = False
    dataset.dst_name = 'synthetic'
    dataset.npoint = npoint
    dataset.num_views = 6
    dataset.random_views = False
    dataset.view_offset = 0
    dataset.sample_seed = sample_seed
    dataset.epoch = 0
    dataset.points_proj = dataset.points_proj_key = None
    dataset.geo = Geometry(dict(nVoxel=[16, 20, 24], dVoxel=[1., 2., 3.], nDetector=[64, 80],
                                dDetector=[2., 3.], DSO=800., DSD=1200.))
    generator = np.random.default_rng(3)
    dataset.blocks = generator.random((blocks, points, 3)).astype(np.float32)
    dataset.load_block = lambda name, b_idx: np.full(points, b_idx, dtype=np.uint8)
    dataset.project_points = lambda pts, angles: np.zeros((len(angles), len(pts), 2), dtype=np.float32)
    dataset.sample_projections = lambda name, **kwargs: (
        np.zeros((6, 1, 4, 4), dtype=np.float32), np.arange(6, dtype=np.float32),
        np.array([0, 8, 16, 25, 33, 41]))
    return dataset


def test_block_and_point_indices_depend_only_on_seed_epoch_and_case():
    first, second = training_dataset(0), training_dataset(0)
    torch.manual_seed(1234)  # a different arm would have consumed a different amount
    np.random.seed(4321)
    torch.randn(97)
    np.random.rand(53)
    for index in (0, 1):
        left, right = first[index], second[index]
        assert left['block_idx'] == right['block_idx']
        np.testing.assert_array_equal(left['points'], right['points'])
        np.testing.assert_array_equal(left['points_gt'], right['points_gt'])
        np.testing.assert_array_equal(left['view_ids'], [0, 8, 16, 25, 33, 41])
    first.set_epoch(1)
    changed = [first[index]['block_idx'] != second[index]['block_idx'] or
               not np.array_equal(first[index]['points'], second[index]['points']) for index in (0, 1)]
    assert any(changed), 'a new epoch must draw new blocks/points'
    assert training_dataset(1)[0]['block_idx'] != training_dataset(0)[0]['block_idx'] or not np.array_equal(
        training_dataset(1)[0]['points'], training_dataset(0)[0]['points'])


def test_legacy_datasets_without_sample_seed_keep_global_rng_behaviour():
    dataset = training_dataset(None)
    np.random.seed(7)
    reference = dataset[0]['block_idx']
    np.random.seed(7)
    assert dataset[0]['block_idx'] == reference


def test_case_order_is_independent_of_model_initialization():
    orders = []
    for arm in ARMS:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(0 * 1_000_003 + 10_007)
        build_model(arm_config(arm).model)  # consumes a different amount of the global stream
        orders.append(torch.randperm(738, generator=loader_generator)[:32].tolist())
    assert all(order == orders[0] for order in orders)


def test_shared_initialization_makes_common_parameters_identical():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from train import load_common_init
    torch.manual_seed(0)
    reference = build_model(arm_config('T4').model).state_dict()
    for arm in ARMS:
        torch.manual_seed(arm.__hash__() % 1000)  # deliberately different initial weights
        model = build_model(arm_config(arm).model)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as handle:
            torch.save({'model': reference}, handle.name)
            report = load_common_init(model, handle.name)
        os.unlink(handle.name)
        loaded = model.state_dict()
        assert report['shared_count'] == len(set(reference) & set(loaded))
        for name in set(reference) & set(loaded):
            assert torch.equal(loaded[name], reference[name]), name
        for name in report['arm_specific']:
            assert name not in reference
        assert set(report['unused_in_this_arm']) == set(reference) - set(loaded)


def test_shared_initialization_rejects_shape_mismatch():
    from train import load_common_init
    import tempfile
    torch.manual_seed(0)
    model = build_model(arm_config('T1').model)
    broken = {name: value for name, value in model.state_dict().items()}
    key = next(iter(broken))
    broken[key] = torch.zeros(broken[key].shape + (2,))
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as handle:
        torch.save({'model': broken}, handle.name)
        try:
            load_common_init(model, handle.name)
            raise AssertionError('shape mismatch must be rejected')
        except ValueError as error:
            assert 'Shared initialization shape mismatch' in str(error)
        finally:
            os.unlink(handle.name)


def test_arm_matrix_changes_one_axis_at_a_time():
    configs = {arm: arm_config(arm) for arm in ARMS}
    for arm, config in configs.items():
        assert config.model.latent['use_latent'] is False
        assert config.ray_consistency['enabled'] is False
        assert config.transfer['enabled'] is False
        assert config.loss['reconstruction_loss'] == 'mse'
        assert config.model.mask_fov is True
        assert dict(config.model.gs) == dict(configs['B0'].model.gs)
        assert dict(config.model.point_decoder) == dict(configs['B0'].model.point_decoder)
    assert [configs[arm].model.anchor_fusion for arm in ARMS] == [
        'max', 'geometry_attention'] + ['geometry_transformer'] * 4
    assert [configs[arm].model.point_fusion for arm in ARMS] == ['max'] * 3 + ['geometry_attention'] * 3
    transformer = {arm: dict(configs[arm].model.multiview_transformer) for arm in ARMS[2:]}
    for arm in ARMS[3:]:
        assert {key: value for key, value in transformer[arm].items() if key != 'anatomy'} == \
               {key: value for key, value in transformer['T1'].items() if key != 'anatomy'}
    anatomy = {arm: dict(transformer[arm]['anatomy']) for arm in ARMS[2:]}
    assert [anatomy[arm]['enabled'] for arm in ARMS[2:]] == [False, False, True, True]
    assert [anatomy[arm]['guided_sampling'] for arm in ARMS[2:]] == [False, False, False, True]
    for arm in ARMS[2:]:
        assert anatomy[arm]['loss_weight'] == .01
        assert {key: value for key, value in anatomy[arm].items() if key not in ('enabled', 'guided_sampling')} == \
               {key: value for key, value in anatomy['T1'].items() if key not in ('enabled', 'guided_sampling')}
    for arm in ('T3', 'T4'):
        assert configs[arm].dataset['structure_supervision'] is True
        assert configs[arm].dataset['structure_target']['frame'] == 'scanner'
        assert configs[arm].dataset['structure_target']['scale'] == .25


def _grouping(arm, prefixes=('multiview_encoder', 'point_view_fusion', 'anchor_view_fusion')):
    torch.manual_seed(0)
    model = build_model(arm_config(arm).model)
    decayed, undecayed = [], []
    for name, _ in model.named_parameters():
        target = undecayed if any(name.startswith(prefix + '.') or f'.{prefix}.' in name
                                  for prefix in prefixes) else decayed
        target.append(name)
    return decayed, undecayed


def test_weight_decay_is_disabled_exactly_on_the_non_backbone_branches():
    backbone, _ = _grouping('B0')
    assert _grouping('B0')[1] == [], 'the max-fusion baseline has no gated branch'
    for arm in ARMS:
        decayed, undecayed = _grouping(arm)
        assert set(decayed) == set(backbone), f'{arm} changes the decayed backbone set'
        for name in undecayed:
            assert name.split('.')[0] in ('multiview_encoder', 'point_view_fusion', 'anchor_view_fusion')
        if arm != 'B0':
            assert undecayed, f'{arm} must exempt its own fusion branch'
    assert any('residual_gate' in name for name in _grouping('T1')[1])
    assert any(name.startswith('point_view_fusion.gate') for name in _grouping('T2')[1])
    assert any(name.startswith('anchor_view_fusion.') for name in _grouping('B1')[1])


def test_weight_decay_groups_reach_the_optimizer_as_configured():
    torch.manual_seed(0)
    model = build_model(arm_config('T2').model)
    decayed, undecayed = _grouping('T2')
    by_name = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [{'params': [by_name[name] for name in decayed], 'weight_decay': 1e-2},
         {'params': [by_name[name] for name in undecayed], 'weight_decay': 0.}],
        lr=2e-4, weight_decay=1e-2)
    assert [group['weight_decay'] for group in optimizer.param_groups] == [1e-2, 0.]
    assert sum(len(group['params']) for group in optimizer.param_groups) == len(list(model.parameters()))
    gate = by_name['point_view_fusion.gate']
    before = float(gate)
    gate.grad = torch.zeros_like(gate)
    optimizer.step()
    assert float(gate) == before, 'a zero-gradient gate must not be shrunk by weight decay'
