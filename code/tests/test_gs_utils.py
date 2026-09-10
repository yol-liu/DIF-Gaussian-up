import math

import torch

from models.gs_utils import build_covariance, query_gs


def make_gs_params(B, K, C, seed=0, jitter=0.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(B, K, 3, generator=g) * 0.2 + 0.1
    r = torch.randn(B, K, 4, generator=g)
    cov = build_covariance(s, r, jitter=jitter)
    return {
        'feats': torch.randn(B, C, K, generator=g),
        'offsets': torch.randn(B, K, 3, generator=g) * 0.05,
        'det': torch.linalg.det(cov),
        'inv': torch.linalg.inv(cov),
        'log_det': torch.linalg.slogdet(cov)[1],
    }


def reference_query_gs(points, gs_points, gs_params, knn_k, shifted_center_knn, normalize_weights, eps=1e-8):
    # pure-PyTorch brute-force reference for interpolation correctness
    B, N = points.shape[:2]
    C = gs_params['feats'].shape[1]
    centers = gs_points + gs_params['offsets']
    knn_ref = centers if shifted_center_knn else gs_points

    out = torch.zeros(B, C, N, dtype=points.dtype)
    for b in range(B):
        for n in range(N):
            p = points[b, n]
            d = ((knn_ref[b] - p) ** 2).sum(-1)
            idx = torch.argsort(d)[:knn_k]

            log_ws = []
            for i in idx:
                diff = p - centers[b, i]
                quad = diff @ gs_params['inv'][b, i] @ diff
                log_w = -0.5 * quad - 0.5 * gs_params['log_det'][b, i] - 1.5 * math.log(2 * math.pi)
                log_ws.append(log_w)
            log_ws = torch.stack(log_ws)

            if normalize_weights:
                w = torch.exp(log_ws - log_ws.max())
                w = w / (w.sum() + eps)
            else:
                w = torch.exp(log_ws)

            out[b, :, n] = (gs_params['feats'][b, :, idx] * w.unsqueeze(0)).sum(-1)
    return out


def test_legacy_query_gs_matches_reference():
    B, K, N, C = 2, 20, 7, 5
    torch.manual_seed(0)
    points = torch.rand(B, N, 3)
    gs_points = torch.rand(B, K, 3)
    gs_params = make_gs_params(B, K, C)

    out = query_gs(points, gs_points, gs_params)  # legacy defaults
    ref = reference_query_gs(points, gs_points, gs_params,
                             knn_k=3, shifted_center_knn=False, normalize_weights=False)
    assert out.shape == (B, C, N)
    assert torch.allclose(out, ref, atol=1e-5)


def test_normalized_query_gs_matches_reference():
    B, K, N, C = 2, 20, 7, 5
    torch.manual_seed(1)
    points = torch.rand(B, N, 3)
    gs_points = torch.rand(B, K, 3)
    gs_params = make_gs_params(B, K, C, seed=1, jitter=1e-6)

    out = query_gs(points, gs_points, gs_params,
                   knn_k=8, shifted_center_knn=True, normalize_weights=True)
    ref = reference_query_gs(points, gs_points, gs_params,
                             knn_k=8, shifted_center_knn=True, normalize_weights=True)
    assert out.shape == (B, C, N)
    assert torch.allclose(out, ref, atol=1e-5)


def test_shifted_center_knn_uses_shifted_centers():
    # one query point at origin; gaussian A anchored nearby but shifted far
    # away, gaussian B anchored farther but shifted next to the query point.
    # with k=1, legacy KNN picks A, shifted-center KNN picks B.
    B, K, C = 1, 2, 1
    points = torch.zeros(B, 1, 3)
    gs_points = torch.tensor([[[0.1, 0.0, 0.0],    # A: close anchor
                               [0.5, 0.0, 0.0]]])  # B: far anchor
    offsets = torch.tensor([[[5.0, 0.0, 0.0],      # A shifted far away
                             [-0.5, 0.0, 0.0]]])   # B shifted onto the query
    eye = torch.eye(3).expand(B, K, 3, 3).contiguous()
    gs_params = {
        'feats': torch.tensor([[[1.0, 2.0]]]),  # [B, C, K], distinguishable features
        'offsets': offsets,
        'det': torch.ones(B, K),
        'inv': eye,
        'log_det': torch.zeros(B, K),
    }

    out_legacy = query_gs(points, gs_points, gs_params, knn_k=1,
                          shifted_center_knn=False, normalize_weights=True)
    out_shifted = query_gs(points, gs_points, gs_params, knn_k=1,
                           shifted_center_knn=True, normalize_weights=True)

    # with normalized weights and k=1, output equals the selected feature
    assert torch.allclose(out_legacy, torch.tensor(1.0))
    assert torch.allclose(out_shifted, torch.tensor(2.0))


def test_normalized_weights_sum_to_one():
    # constant features per gaussian: normalized interpolation must return
    # exactly that constant, proving the weights sum to one.
    B, K, N = 2, 30, 11
    torch.manual_seed(2)
    points = torch.rand(B, N, 3)
    gs_points = torch.rand(B, K, 3)
    gs_params = make_gs_params(B, K, C=4, seed=2, jitter=1e-6)
    const = 3.14
    gs_params['feats'] = torch.full_like(gs_params['feats'], const)

    out = query_gs(points, gs_points, gs_params,
                   knn_k=8, shifted_center_knn=True, normalize_weights=True)
    assert torch.allclose(out, torch.full_like(out, const), atol=1e-4)


def test_near_zero_quaternion_no_nan():
    B, K = 2, 5
    s = torch.rand(B, K, 3) * 0.2 + 0.05
    r = torch.zeros(B, K, 4)
    r[0, 0] = 1e-20  # near-zero raw quaternion
    cov = build_covariance(s, r, jitter=1e-6)
    assert torch.isfinite(cov).all()
    assert torch.isfinite(torch.linalg.inv(cov)).all()
    assert torch.isfinite(torch.linalg.slogdet(cov)[1]).all()


def test_query_gs_cpu_device_and_dtype():
    # no hard-coded cuda allocations: everything must run on CPU, float64 included
    B, K, N, C = 1, 10, 4, 3
    s = torch.rand(B, K, 3, dtype=torch.float64) * 0.2 + 0.1
    r = torch.randn(B, K, 4, dtype=torch.float64)
    cov = build_covariance(s, r, jitter=1e-8)
    assert cov.dtype == torch.float64 and cov.device.type == 'cpu'

    gs_params = {
        'feats': torch.randn(B, C, K, dtype=torch.float64),
        'offsets': torch.randn(B, K, 3, dtype=torch.float64) * 0.05,
        'det': torch.linalg.det(cov),
        'inv': torch.linalg.inv(cov),
        'log_det': torch.linalg.slogdet(cov)[1],
    }
    out = query_gs(torch.rand(B, N, 3, dtype=torch.float64), torch.rand(B, K, 3, dtype=torch.float64),
                   gs_params, knn_k=4, shifted_center_knn=True, normalize_weights=True)
    assert out.shape == (B, C, N)
    assert torch.isfinite(out).all()


def test_knn_k_must_not_exceed_num_gaussians():
    B, K, N, C = 1, 4, 3, 2
    gs_params = make_gs_params(B, K, C)
    try:
        query_gs(torch.rand(B, N, 3), torch.rand(B, K, 3), gs_params, knn_k=K + 1)
    except ValueError:
        return
    raise AssertionError('query_gs must reject knn_k > number of gaussians')
