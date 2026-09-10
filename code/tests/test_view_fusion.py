import torch

from models.utils import query_view_feats
from models.view_fusion import GeometryAwareViewFusion


B, M, C, H, W, N = 2, 4, 8, 6, 6, 10
C_OUT = 16


def make_inputs(seed=0, n_views=M, in_fov=True):
    g = torch.Generator().manual_seed(seed)
    view_feats = torch.randn(B, n_views, C, H, W, generator=g)
    scale = 0.9 if in_fov else 1.0
    points_proj = (torch.rand(B, n_views, N, 2, generator=g) * 2 - 1) * scale
    points_xyz = torch.rand(B, N, 3, generator=g) * 2 - 1
    angles = torch.rand(B, n_views, generator=g) * 2 * torch.pi
    return view_feats, points_proj, points_xyz, angles


def make_fusion(return_attention=False, use_max_residual=False, content_query=False):
    torch.manual_seed(42)
    out_ch = C if use_max_residual else C_OUT
    fusion = GeometryAwareViewFusion(
        in_ch=C, out_ch=out_ch, d_model=32, num_heads=4, num_layers=2,
        dropout=0.1, geometry_hidden_dim=16, return_attention=return_attention,
        use_max_residual=use_max_residual, content_query=content_query,
    )
    fusion.eval()
    return fusion


def test_query_view_feats_max_shape():
    view_feats, points_proj, _, _ = make_inputs()
    out = query_view_feats(view_feats, points_proj, fusion='max')
    assert out.shape == (B, C, N)
    out_stack = query_view_feats(view_feats, points_proj, fusion=None)
    assert out_stack.shape == (B, C, N, M)


def test_fusion_output_shape():
    fusion = make_fusion()
    view_feats, points_proj, points_xyz, angles = make_inputs()
    out, attn = fusion(view_feats, points_proj, points_xyz, angles)
    assert out.shape == (B, C_OUT, N)
    assert attn is None
    assert torch.isfinite(out).all()


def test_max_residual_matches_max_at_init():
    # zero-init residual ⇒ fused == masked max over in-FOV views
    fusion = make_fusion(use_max_residual=True, content_query=True)
    view_feats, points_proj, points_xyz, angles = make_inputs()
    out, _ = fusion(view_feats, points_proj, points_xyz, angles)

    sampled = query_view_feats(view_feats, points_proj, fusion=None)  # [B,C,N,M]
    uv = points_proj.permute(0, 2, 1, 3)
    in_fov = (uv.abs() <= 1.0).all(dim=-1)  # [B,N,M]
    sampled_bnmc = sampled.permute(0, 2, 3, 1)
    neg_inf = torch.finfo(sampled_bnmc.dtype).min
    masked = sampled_bnmc.masked_fill(~in_fov.unsqueeze(-1), neg_inf)
    ref = masked.max(dim=2).values.transpose(1, 2)  # [B,C,N]
    assert torch.allclose(out, ref, atol=1e-5)


def test_view_permutation_invariance():
    fusion = make_fusion()
    view_feats, points_proj, points_xyz, angles = make_inputs()
    out1, _ = fusion(view_feats, points_proj, points_xyz, angles)

    perm = torch.tensor([2, 0, 3, 1])
    out2, _ = fusion(view_feats[:, perm], points_proj[:, perm], points_xyz, angles[:, perm])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_masked_views_get_zero_attention():
    fusion = make_fusion(return_attention=True)
    view_feats, points_proj, points_xyz, angles = make_inputs()
    view_mask = torch.tensor([[True, True, False, True],
                              [True, False, False, True]])
    _, attn = fusion(view_feats, points_proj, points_xyz, angles, view_mask=view_mask)
    assert attn.shape == (B, N, M)
    invalid = ~view_mask.unsqueeze(1).expand(-1, N, -1)
    assert (attn[invalid].abs() < 1e-7).all()


def test_attention_weights_sum_to_one_over_valid_views():
    fusion = make_fusion(return_attention=True)
    view_feats, points_proj, points_xyz, angles = make_inputs()
    view_mask = torch.tensor([[True, True, False, True],
                              [True, True, True, True]])
    _, attn = fusion(view_feats, points_proj, points_xyz, angles, view_mask=view_mask)
    sums = attn.sum(dim=-1)  # [B, N]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_output_ignores_masked_view_content():
    # changing the features of a masked view must not change the output
    fusion = make_fusion()
    view_feats, points_proj, points_xyz, angles = make_inputs()
    view_mask = torch.ones(B, M, dtype=torch.bool)
    view_mask[:, 1] = False

    out1, _ = fusion(view_feats, points_proj, points_xyz, angles, view_mask=view_mask)
    view_feats2 = view_feats.clone()
    view_feats2[:, 1] += 100.0
    out2, _ = fusion(view_feats2, points_proj, points_xyz, angles, view_mask=view_mask)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_all_invalid_views_no_nan():
    fusion = make_fusion(return_attention=True)
    view_feats, points_proj, points_xyz, angles = make_inputs()
    view_mask = torch.zeros(B, M, dtype=torch.bool)  # every view invalid
    out, attn = fusion(view_feats, points_proj, points_xyz, angles, view_mask=view_mask)
    assert torch.isfinite(out).all()
    assert (out == 0).all()
    assert torch.isfinite(attn).all()
    assert (attn == 0).all()


def test_out_of_fov_points_treated_as_invalid():
    fusion = make_fusion(return_attention=True)
    view_feats, points_proj, points_xyz, angles = make_inputs()
    # push all projections of view 0 outside the detector
    points_proj = points_proj.clone()
    points_proj[:, 0] = 2.0
    _, attn = fusion(view_feats, points_proj, points_xyz, angles)
    assert (attn[:, :, 0].abs() < 1e-7).all()
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_variable_view_counts():
    fusion = make_fusion()
    for n_views in (3, 6):
        view_feats, points_proj, points_xyz, angles = make_inputs(n_views=n_views)
        out, _ = fusion(view_feats, points_proj, points_xyz, angles)
        assert out.shape == (B, C_OUT, N)
        assert torch.isfinite(out).all()
