import torch

try:
    from easydict import EasyDict
except ImportError:
    # minimal stand-in so the tests run in environments without easydict
    class EasyDict(dict):
        def __init__(self, d=None):
            super().__init__()
            for k, v in (d or {}).items():
                self[k] = EasyDict(v) if isinstance(v, dict) else v

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

from models.model import DIF_Gaussian


def make_cfg(anchor_fusion=None):
    cfg = {
        'image_encoder': {'out_ch': 16},
        'point_decoder': {'mlp_chs': [32, 16, 1]},
        'gs': {
            't': 512,
            'res': 3,
            'p_dist_scaling': 1.0,
            'o_scaling': 1.0,
            's_scaling': 0.3,
        },
    }
    if anchor_fusion == 'geometry_attention':
        cfg['anchor_fusion'] = 'geometry_attention'
        cfg['point_fusion'] = 'max'
        cfg['view_fusion'] = {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.0,
            'geometry_hidden_dim': 16,
            'return_attention': False,
            'view_dropout': 0.0,
            'min_train_views': 2,
            'use_max_residual': True,
            'content_query': True,
        }
        # keep legacy GS in the model unit test for the stable recipe
        cfg['gs'].update({
            'knn_k': 3,
            'shifted_center_knn': False,
            'normalize_weights': False,
            'covariance_jitter': 0.0,
        })
    return EasyDict(cfg)


def make_data(B=2, M=4, N=20, K=27, HW=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        'projs': torch.rand(B, M, 1, HW, HW, generator=g),
        'angles': torch.rand(B, M, generator=g) * 2 * torch.pi,
        'points': torch.rand(B, N, 3, generator=g),
        'points_proj': torch.rand(B, M, N, 2, generator=g) * 2 - 1,
        'points_gs': torch.rand(B, K, 3, generator=g),
        'points_gs_proj': torch.rand(B, M, K, 2, generator=g) * 2 - 1,
        'points_gt': torch.rand(B, 1, N, generator=g),
    }


def test_legacy_max_fusion_still_runs():
    torch.manual_seed(0)
    model = DIF_Gaussian(make_cfg())
    model.eval()
    data = make_data()
    with torch.no_grad():
        pred = model(data)
    assert pred['points_pred'].shape == (2, 1, 20)
    assert torch.isfinite(pred['points_pred']).all()
    # legacy model has no attention module
    assert not hasattr(model, 'anchor_view_fusion')


def test_geometry_attention_forward_backward_finite_grads():
    torch.manual_seed(0)
    model = DIF_Gaussian(make_cfg('geometry_attention'))
    model.train()
    data = make_data()

    pred = model(data)
    assert pred['points_pred'].shape == (2, 1, 20)

    loss = torch.nn.functional.mse_loss(pred['points_pred'], data['points_gt'])
    loss.backward()

    n_attn_params = 0
    for name, p in model.anchor_view_fusion.named_parameters():
        assert p.grad is not None, f'no gradient for attention parameter {name}'
        assert torch.isfinite(p.grad).all(), f'non-finite gradient for {name}'
        n_attn_params += 1
    assert n_attn_params > 0


def test_geometry_attention_variable_view_counts():
    torch.manual_seed(0)
    model = DIF_Gaussian(make_cfg('geometry_attention'))
    model.eval()
    with torch.no_grad():
        for n_views in (3, 6):
            pred = model(make_data(M=n_views))
            assert pred['points_pred'].shape == (2, 1, 20)
            assert torch.isfinite(pred['points_pred']).all()


def test_view_dropout_respects_min_train_views():
    torch.manual_seed(0)
    model = DIF_Gaussian(make_cfg('geometry_attention'))
    model.view_dropout = 0.95  # aggressive dropout to exercise the floor
    for _ in range(20):
        mask = model.sample_view_mask(4, 5, torch.device('cpu'))
        assert mask.shape == (4, 5)
        assert (mask.sum(dim=1) >= 2).all()


def test_view_dropout_disabled_in_eval():
    torch.manual_seed(0)
    model = DIF_Gaussian(make_cfg('geometry_attention'))
    model.eval()
    data = make_data()
    with torch.no_grad():
        out1 = model(data)['points_pred']
        out2 = model(data)['points_pred']
    # deterministic in eval mode: no dropout of any kind
    assert torch.allclose(out1, out2)
