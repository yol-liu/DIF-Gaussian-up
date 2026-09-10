import math

import torch

# PyTorch3D provides a fast CUDA KNN. It is optional: when unavailable (e.g.,
# CPU-only test environments), we fall back to an equivalent pure-PyTorch
# implementation based on torch.cdist + topk.
try:
    from pytorch3d.ops.knn import knn_points as _p3d_knn_points
    from pytorch3d.ops.knn import knn_gather as _p3d_knn_gather
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False


def knn_search(query, ref, k):
    # query: [B, N, 3]
    # ref: [B, K, 3]
    # returns indices: [B, N, k] (indices into ref, sorted by distance)
    if _HAS_PYTORCH3D:
        _, idx, _ = _p3d_knn_points(query, ref, K=k)
        return idx
    dists = torch.cdist(query, ref)  # [B, N, K]
    _, idx = torch.topk(dists, k, dim=-1, largest=False)
    return idx


def knn_gather(x, idx):
    # x: [B, K, C]
    # idx: [B, N, k]
    # returns: [B, N, k, C]
    if _HAS_PYTORCH3D:
        return _p3d_knn_gather(x, idx)
    B, N, k = idx.shape
    C = x.shape[-1]
    idx_flat = idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, C)
    return x.gather(1, idx_flat).reshape(B, N, k, C)


def build_rotation(r, eps=1e-8):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py#L78
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    # r: [B, 4] (unnormalized quaternion, wxyz)
    # R: [B, 3, 3]
    assert r.dim() == 2 and r.shape[-1] == 4, \
        f'build_rotation expects quaternions of shape [B, 4], got {tuple(r.shape)}'

    norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(eps)
    q = r / norm

    w, x, y, z = q.unbind(-1)

    # https://www.zhihu.com/tardis/zm/art/78987582?source_id=1003 (see Eqn. 21)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_scaling_rotation(s, r, eps=1e-8):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py#L101
    # s: [B, 3]
    # r: [B, 4]
    # L: [B, 3, 3]
    assert s.dim() == 2 and s.shape[-1] == 3, \
        f'build_scaling_rotation expects scales of shape [B, 3], got {tuple(s.shape)}'

    R = build_rotation(r, eps=eps)
    L = R @ torch.diag_embed(s)
    return L


def build_covariance(s, r, jitter=0.0, eps=1e-8):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py#L28
    # s: [B, N, 3]
    # r: [B, N, 4]
    # jitter: optional isotropic diagonal jitter added before det/inv for
    #         numerical stability (0.0 keeps legacy behavior)
    # Cov: [B, N, 3, 3]
    assert s.dim() == 3 and s.shape[-1] == 3, \
        f'build_covariance expects scales of shape [B, N, 3], got {tuple(s.shape)}'
    assert r.dim() == 3 and r.shape[-1] == 4, \
        f'build_covariance expects quaternions of shape [B, N, 4], got {tuple(r.shape)}'

    b, n = s.shape[:2]
    s = s.reshape(b * n, 3)
    r = r.reshape(b * n, 4)

    L = build_scaling_rotation(s, r, eps=eps)
    Cov = L @ L.transpose(1, 2)

    Cov = Cov.reshape(b, n, 3, 3)
    if jitter > 0:
        Cov = Cov + jitter * torch.eye(3, device=Cov.device, dtype=Cov.dtype)
    return Cov


def query_gs(
    points,
    gs_points,
    gs_params,
    knn_k=3,
    shifted_center_knn=False,
    normalize_weights=False,
    eps=1e-8,
):
    # points: [B, N, 3]
    # gs_points: [B, K, 3]
    # gs_params:
    #     - det: [B, K]
    #     - inv: [B, K, 3, 3]
    #     - offsets: [B, K, 3]
    #     - feats: [B, C, K]
    #     - log_det: [B, K] (optional, used when normalize_weights=True)
    #
    # knn_k: number of neighboring Gaussians (legacy: 3)
    # shifted_center_knn:
    #     - False (legacy): KNN on the initial anchors `gs_points`
    #     - True (corrected): KNN on the actual centers `gs_points + offsets`,
    #       consistent with the centers used for weight evaluation
    # normalize_weights:
    #     - False (legacy): unnormalized Gaussian density summation
    #     - True: stable log-space softmax-normalized weights
    #
    # returns: [B, C, N]

    assert points.dim() == 3 and points.shape[-1] == 3, \
        f'query_gs expects points of shape [B, N, 3], got {tuple(points.shape)}'
    assert gs_points.dim() == 3 and gs_points.shape[-1] == 3, \
        f'query_gs expects gs_points of shape [B, K, 3], got {tuple(gs_points.shape)}'
    assert points.shape[0] == gs_points.shape[0], \
        f'batch mismatch: points {tuple(points.shape)} vs gs_points {tuple(gs_points.shape)}'

    B, K = gs_points.shape[:2]
    N = points.shape[1]
    k = knn_k
    assert k >= 1, f'knn_k must be >= 1, got {k}'
    if k > K:
        raise ValueError(f'knn_k ({k}) must be <= number of gaussians ({K})')

    # 1. det and inv of Cov
    det = gs_params['det']
    inv = gs_params['inv']
    offsets = gs_params['offsets']
    feats = gs_params['feats']
    assert det.shape == (B, K), \
        f'gs_params["det"] must be [B, K]=({B}, {K}), got {tuple(det.shape)}'
    assert inv.shape == (B, K, 3, 3), \
        f'gs_params["inv"] must be [B, K, 3, 3]=({B}, {K}, 3, 3), got {tuple(inv.shape)}'
    assert offsets.shape == (B, K, 3), \
        f'gs_params["offsets"] must be [B, K, 3]=({B}, {K}, 3), got {tuple(offsets.shape)}'
    assert feats.dim() == 3 and feats.shape[0] == B and feats.shape[2] == K, \
        f'gs_params["feats"] must be [B, C, K]=({B}, C, {K}), got {tuple(feats.shape)}'

    centers = gs_points + offsets  # actual Gaussian centers, [B, K, 3]

    # 2. query neighbors
    # neb_idx: [B, N, k]
    # points_ext: [B, N, k, 3]
    # gs': [B, N, k, ...], (1) xyz; (2) det, inv; (3) feats.
    knn_ref = centers if shifted_center_knn else gs_points
    neb_idx = knn_search(points, knn_ref, k)  # [B, N, k]
    points_ext = points.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, 3]
    neb_gs_xyz = knn_gather(centers, neb_idx)  # [B, N, k, 3]
    neb_gs_feats = knn_gather(
        feats.transpose(1, 2),  # [B, K, C]
        neb_idx
    ).permute(0, 3, 1, 2)  # [B, N, k, C] => [B, C, N, k]
    neb_det = knn_gather(
        det.unsqueeze(-1),
        neb_idx
    ).squeeze(-1)  # [B, N, k]
    neb_inv = knn_gather(
        inv.reshape(B, K, 9),
        neb_idx
    ).reshape(B, N, k, 3, 3)  # [B, N, k, 3, 3]

    # 3. calculate weights
    # let * be [B, N, k]
    # log w = -1/2 (p-mu)' inv (p-mu) - 1/2 log|Cov| - 3/2 log(2*pi)
    diff = (points_ext - neb_gs_xyz).unsqueeze(-1)  # [*, 3, 1]
    quad = (
        diff.transpose(3, 4) @ neb_inv @ diff  # [*, 1, 3] @ [*, 3, 3] @ [*, 3, 1] => [*, 1, 1]
    ).squeeze(4).squeeze(3)  # [*]

    if normalize_weights:
        log_det = gs_params.get('log_det', None)
        if log_det is not None:
            assert log_det.shape == (B, K), \
                f'gs_params["log_det"] must be [B, K]=({B}, {K}), got {tuple(log_det.shape)}'
            neb_log_det = knn_gather(log_det.unsqueeze(-1), neb_idx).squeeze(-1)  # [B, N, k]
        else:
            neb_log_det = torch.log(neb_det.clamp_min(eps))

        log_w = -0.5 * quad - 0.5 * neb_log_det - 1.5 * math.log(2 * math.pi)  # [B, N, k]
        log_w = log_w - log_w.max(dim=-1, keepdim=True).values
        w = torch.exp(log_w)
        weights = w / (w.sum(dim=-1, keepdim=True) + eps)  # [B, N, k], sums to ~1
    else:
        # legacy: unnormalized density
        # w = (2*pi)^(-3/2) * det^(-1/2) * exp(-1/2 * quad)
        weights = math.pow(2 * math.pi, -3 / 2) * \
            torch.pow(neb_det, -1 / 2) * \
            torch.exp(-1 / 2 * quad)

    # 4. sum weighted features
    # weights * gs_feats' =(sum)> [B, C, N]
    weights = weights.unsqueeze(1)  # [B, 1, N, k]
    sum_feats = (weights * neb_gs_feats).sum(dim=-1)  # [B, C, N]

    return sum_feats
