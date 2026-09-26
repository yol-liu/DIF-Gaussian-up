'''
Tests for C1 hierarchical latent / probabilistic DIF-Gaussian.

Covers:
  1. zero-init FiLM equivalence to deterministic baseline
  2. chunk-consistent decode under a shared SampleState
  3. sampling shapes + view permutation invariance of z_g
  4. KL non-negative & residual param (Δ=0 ⇒ q≡p, KL=0)
  5. gradcheck on small latent modules
  6. smoke train (~100 steps) with KL warmup + non-zero diversity
'''

import copy

import torch
import torch.nn as nn

try:
    from easydict import EasyDict
except ImportError:
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

        def get(self, key, default=None):
            return dict.get(self, key, default)


from models.model import DIF_Gaussian
from models.probabilistic_model import ProbabilisticDIFGaussian, build_model
from models.latent_np import (
    GlobalLatent,
    LocalLatentGrid,
    FeatureFiLM,
    gaussian_kl,
    reparameterize,
    apply_sigma_floor,
    pool_global_ctx,
)
from losses import elbo_loss, sample_diversity, kl_warmup_scale, volume_nll


def make_cfg(use_latent=False, use_local=False, anchor_fusion='max'):
    cfg = {
        'image_encoder': {'out_ch': 16},
        'point_decoder': {'mlp_chs': [32, 16, 1]},
        'gs': {
            't': 512,
            'res': 3,
            'p_dist_scaling': 1.0,
            'o_scaling': 1.0,
            's_scaling': 0.3,
            'knn_k': 3,
            'covariance_jitter': 1e-6,
        },
        'latent': {
            'use_latent': use_latent,
            'global_dim': 8,
            'use_local': use_local,
            'local_dim': 4,
            'grid_res': 2,
            'beta_g': 1e-3,
            'beta_l': 1e-2,
            'free_bits_g': 0.5,
            'free_bits_l': 0.1,
            'kl_warmup_steps': 50,
            'prior_sigma_floor': 1e-2,
            'volume_likelihood': 'laplace',
            'volume_sigma': 0.1,
            'lambda_v': 1.0,
            'gt_feat_ch': 8,
        },
        'sampling': {
            'sample_from': 'posterior',
            'num_samples': 1,
        },
    }
    if anchor_fusion == 'geometry_attention':
        cfg['anchor_fusion'] = 'geometry_attention'
        cfg['point_fusion'] = 'max'
        cfg['view_fusion'] = {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 1,
            'dropout': 0.0,
            'geometry_hidden_dim': 16,
            'return_attention': False,
            'view_dropout': 0.0,
            'min_train_views': 1,
        }
    return EasyDict(cfg)


def make_data(B=2, M=4, N=20, K=8, HW=16, seed=0, with_gt_volume=False, vol=8):
    g = torch.Generator().manual_seed(seed)
    data = {
        'projs': torch.rand(B, M, 1, HW, HW, generator=g),
        'angles': torch.rand(B, M, generator=g) * 2 * torch.pi,
        'points': torch.rand(B, N, 3, generator=g),
        'points_proj': torch.rand(B, M, N, 2, generator=g) * 2 - 1,
        'points_gs': torch.rand(B, K, 3, generator=g),
        'points_gs_proj': torch.rand(B, M, K, 2, generator=g) * 2 - 1,
        'points_gt': torch.rand(B, 1, N, generator=g),
    }
    if with_gt_volume:
        data['gt_volume'] = torch.rand(B, 1, vol, vol, vol, generator=g)
    return data


def _sync_shared_weights(src, dst):
    '''Copy overlapping parameters from src → dst (for equivalence tests).'''
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
    dst_sd.update(shared)
    dst.load_state_dict(dst_sd)


# --------------------------------------------------------------------------- #
# 1. Zero-init equivalence
# --------------------------------------------------------------------------- #
def test_zero_init_equivalence_global():
    torch.manual_seed(0)
    cfg_det = make_cfg(use_latent=False)
    cfg_lat = make_cfg(use_latent=True, use_local=False)

    det = DIF_Gaussian(cfg_det)
    lat = ProbabilisticDIFGaussian(cfg_lat)
    _sync_shared_weights(det, lat)

    det.eval()
    lat.eval()
    data = make_data()

    with torch.no_grad():
        out_det = det(data)['points_pred']
        # any z — FiLM last layer is zero, so output must match
        state = lat.encode_and_sample(data, sample_from='prior')
        out_lat = lat.decode(data, state)

    err = (out_det - out_lat).abs().max().item()
    print(f'  [zero-init global] max |Δ| = {err:.3e}')
    assert err < 1e-5, f'zero-init equivalence failed: {err}'


def test_zero_init_equivalence_with_local():
    torch.manual_seed(1)
    cfg_det = make_cfg(use_latent=False)
    cfg_lat = make_cfg(use_latent=True, use_local=True)

    det = DIF_Gaussian(cfg_det)
    lat = ProbabilisticDIFGaussian(cfg_lat)
    _sync_shared_weights(det, lat)

    det.eval()
    lat.eval()
    data = make_data(K=8)

    with torch.no_grad():
        out_det = det(data)['points_pred']
        state = lat.encode_and_sample(data, sample_from='prior')
        out_lat = lat.decode(data, state)

    err = (out_det - out_lat).abs().max().item()
    print(f'  [zero-init local] max |Δ| = {err:.3e}')
    assert err < 1e-5, f'zero-init+local equivalence failed: {err}'


def test_zero_init_after_loading_deterministic_ckpt():
    torch.manual_seed(2)
    cfg_det = make_cfg(use_latent=False)
    det = DIF_Gaussian(cfg_det)
    ckpt = copy.deepcopy(det.state_dict())

    cfg_lat = make_cfg(use_latent=True, use_local=True)
    lat = ProbabilisticDIFGaussian(cfg_lat)
    missing, unexpected = lat.load_state_dict(ckpt, strict=False)
    assert len(unexpected) == 0
    assert any('feature_film' in m or 'global_latent' in m for m in missing)

    det.eval()
    lat.eval()
    data = make_data()
    with torch.no_grad():
        out_det = det(data)['points_pred']
        state = lat.encode_and_sample(data, sample_from='mean')
        out_lat = lat.decode(data, state)
    err = (out_det - out_lat).abs().max().item()
    print(f'  [ckpt warm-start] max |Δ| = {err:.3e}')
    assert err < 1e-5


def test_build_model_backward_compat():
    # use_latent=false → plain DIF_Gaussian
    m = build_model(make_cfg(use_latent=False))
    assert type(m).__name__ == 'DIF_Gaussian'
    m = build_model(make_cfg(use_latent=True))
    assert isinstance(m, ProbabilisticDIFGaussian)


# --------------------------------------------------------------------------- #
# 2. Chunk consistency
# --------------------------------------------------------------------------- #
def test_chunk_consistency():
    torch.manual_seed(3)
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True, use_local=True))
    model.eval()
    data = make_data(N=40, K=8)

    with torch.no_grad():
        state = model.encode_and_sample(data, sample_from='prior')
        full = model.decode(data, state)

        # split query points into two chunks
        mid = 20
        d1 = {k: (v[..., :mid, :] if k in model.registered_point_keys else v)
              for k, v in data.items()}
        # registered keys are points / points_proj with trailing point dim
        d1['points'] = data['points'][:, :mid]
        d1['points_proj'] = data['points_proj'][:, :, :mid]
        d1['points_gt'] = data['points_gt'][:, :, :mid]

        d2 = dict(data)
        d2['points'] = data['points'][:, mid:]
        d2['points_proj'] = data['points_proj'][:, :, mid:]
        d2['points_gt'] = data['points_gt'][:, :, mid:]

        p1 = model.decode(d1, state)
        p2 = model.decode(d2, state)
        cat = torch.cat([p1, p2], dim=2)

    err = (full - cat).abs().max().item()
    print(f'  [chunk consistency] max |Δ| = {err:.3e}')
    assert err < 1e-5, f'chunk consistency failed: {err}'


# --------------------------------------------------------------------------- #
# 3. Sampling shapes + view permutation invariance
# --------------------------------------------------------------------------- #
def test_sampling_shapes_and_perm_invariance():
    torch.manual_seed(4)
    model = ProbabilisticDIFGaussian(
        make_cfg(use_latent=True, use_local=True, anchor_fusion='geometry_attention')
    )
    model.eval()
    data = make_data(B=2, M=4, K=8, with_gt_volume=True)
    gt = data['gt_volume']

    with torch.no_grad():
        state_p = model.encode_and_sample(data, sample_from='prior')
        assert state_p.z_g.shape == (2, 8)
        assert state_p.z_grid.shape == (2, 4, 2, 2, 2)

        state_q = model.encode_and_sample(
            data, gt_volume=gt, sample_from='posterior'
        )
        assert state_q.z_g.shape == (2, 8)

        # permute views → z_g distribution (prior mean) unchanged
        perm = torch.tensor([2, 0, 3, 1])
        data_perm = dict(data)
        data_perm['projs'] = data['projs'][:, perm]
        data_perm['angles'] = data['angles'][:, perm]
        data_perm['points_proj'] = data['points_proj'][:, perm]
        data_perm['points_gs_proj'] = data['points_gs_proj'][:, perm]

        s0 = model.encode_and_sample(data, sample_from='mean')
        s1 = model.encode_and_sample(data_perm, sample_from='mean')
        err = (s0.z_g - s1.z_g).abs().max().item()
        print(f'  [view-perm z_g] max |Δ| = {err:.3e}')
        assert err < 1e-4, f'view permutation changed z_g mean: {err}'


# --------------------------------------------------------------------------- #
# 4. KL non-negative & residual parameterization
# --------------------------------------------------------------------------- #
def test_kl_nonneg_and_residual_param():
    torch.manual_seed(5)
    gl = GlobalLatent(ctx_dim=16, latent_dim=8, gt_dim=16, prior_sigma_floor=1e-2)
    ctx = torch.randn(4, 16)
    gt = torch.randn(4, 16)

    mu_p, lv_p = gl.prior(ctx)
    mu_q, lv_q = gl.posterior(ctx, gt)
    # delta net is zero-init → q ≡ p
    assert torch.allclose(mu_q, mu_p, atol=1e-6)
    assert torch.allclose(lv_q, lv_p, atol=1e-6)

    kl = gaussian_kl(mu_q, lv_q, mu_p, lv_p)
    assert (kl >= -1e-6).all()
    assert kl.abs().max() < 1e-5

    # after a non-zero delta, KL >= 0
    with torch.no_grad():
        gl.delta_net[-1].weight.add_(0.1)
    mu_q2, lv_q2 = gl.posterior(ctx, gt)
    kl2 = gaussian_kl(mu_q2, lv_q2, mu_p, lv_p)
    assert (kl2 >= -1e-5).all()
    assert kl2.sum() > 0


def test_sigma_floor():
    raw = torch.tensor([-100.0, 0.0, 100.0])
    lv = apply_sigma_floor(raw, 1e-2)
    sigma = torch.exp(0.5 * lv)
    assert (sigma >= 1e-2 - 1e-8).all()


# --------------------------------------------------------------------------- #
# 5. gradcheck
# --------------------------------------------------------------------------- #
def test_gradcheck_film_and_reparam():
    torch.manual_seed(6)

    class _FilmWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.film = FeatureFiLM(feat_dim=3, latent_dim=2, hidden_dim=4)
            # break zero-init so grads are nontrivial
            nn.init.xavier_uniform_(self.film.net[-1].weight)
            nn.init.uniform_(self.film.net[-1].bias, -0.1, 0.1)

        def forward(self, feats, z):
            return self.film(feats, z).sum()

    wrap = _FilmWrap().double()
    feats = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(wrap, (feats, z), eps=1e-6, atol=1e-4)

    def _reparam_sum(mu, logvar):
        # fixed eps via manual formula for determinism in gradcheck
        std = torch.exp(0.5 * logvar)
        eps = torch.ones_like(std) * 0.5
        return (mu + eps * std).sum()

    mu = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    lv = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(_reparam_sum, (mu, lv), eps=1e-6, atol=1e-4)


def test_gradcheck_local_interpolate():
    torch.manual_seed(7)
    grid = LocalLatentGrid(
        ctx_dim=4, global_dim=2, latent_dim=3, grid_res=2, prior_sigma_floor=1e-2
    ).double()

    def _interp(z_grid, xyz):
        return grid.interpolate_to_anchors(z_grid, xyz).sum()

    z_grid = torch.randn(1, 3, 2, 2, 2, dtype=torch.double, requires_grad=True)
    xyz = torch.rand(1, 5, 3, dtype=torch.double, requires_grad=True)
    # keep xyz interior so bilinear is smooth
    xyz = xyz * 0.8 + 0.1
    xyz = xyz.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(_interp, (z_grid, xyz), eps=1e-5, atol=1e-3)


# --------------------------------------------------------------------------- #
# 6. smoke train
# --------------------------------------------------------------------------- #
def test_smoke_train_global():
    torch.manual_seed(8)
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True, use_local=False))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = make_cfg(use_latent=True)['latent']

    losses = []
    vol_losses = []
    diversities = []
    beta_scales = []
    # fixed batch to make volume-loss descent observable
    fixed = make_data(B=2, N=16, K=8, with_gt_volume=True, seed=42)
    for step in range(100):
        data = fixed if step % 2 == 0 else make_data(
            B=2, N=16, K=8, with_gt_volume=True, seed=100 + step
        )
        pred = model(data, gt_volume=data['gt_volume'], sample_from='posterior')
        loss, metrics = elbo_loss(
            pred['points_pred'], data['points_gt'], pred['kl_info'],
            cfg, step=step, use_local=False,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        vol_losses.append(float(metrics['L_volume'].item()))
        beta_scales.append(float(metrics['beta_scale']))

        if step in (0, 25, 50, 99):
            warm = kl_warmup_scale(step, cfg['kl_warmup_steps'])
            assert abs(warm - metrics['beta_scale']) < 1e-6
            model.eval()
            with torch.no_grad():
                out = model(
                    fixed, is_eval=True, sample_from='prior',
                    num_samples=4, eval_npoint=16,
                )
                div = sample_diversity(out['sample_predictions'])
                diversities.append(float(div.item()))
            model.train()

    assert max(losses) < 1e3, f'loss exploded: {max(losses)}'
    # volume term should trend down on the recurring fixed batch
    early = sum(vol_losses[:10]) / 10
    late = sum(vol_losses[-10:]) / 10
    assert late < early, f'L_volume did not drop: early={early:.4f} late={late:.4f}'
    # KL warmup: beta_scale rises from 0 toward 1 over kl_warmup_steps=50
    assert beta_scales[0] == 0.0
    assert beta_scales[25] == 0.5
    assert beta_scales[99] == 1.0
    # diversity leaves the near-zero zero-init regime and stays finite
    assert diversities[0] < 1e-2
    assert diversities[-1] > diversities[0]
    assert all(d == d and d < 1e3 for d in diversities), diversities
    print(f'  [smoke] L_vol {early:.4f} → {late:.4f}, '
          f'diversity {diversities[0]:.4e} → {diversities[-1]:.4e}')


def test_elbo_helpers():
    pred = torch.zeros(2, 1, 10)
    gt = torch.ones(2, 1, 10)
    nll = volume_nll(pred, gt, 'laplace', 0.1)
    assert nll > 0
    assert abs(kl_warmup_scale(0, 100)) < 1e-9
    assert abs(kl_warmup_scale(50, 100) - 0.5) < 1e-9
    assert abs(kl_warmup_scale(200, 100) - 1.0) < 1e-9


def test_render_placeholder():
    model = ProbabilisticDIFGaussian(make_cfg(use_latent=True))
    data = make_data()
    state = model.encode_and_sample(data, sample_from='mean')
    try:
        model.render(None, state)
        assert False, 'render should raise'
    except NotImplementedError:
        pass
