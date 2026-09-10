import torch
import torch.nn as nn

from models.utils import query_view_feats


class _CrossAttentionBlock(nn.Module):
    # one pre-token cross-attention block: query attends to per-view tokens,
    # followed by a feed-forward refinement of the query.
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, tokens, key_padding_mask):
        # query: [B*N, 1, D]
        # tokens: [B*N, M, D]
        # key_padding_mask: [B*N, M], True = ignore
        attn_out, attn_weights = self.attn(
            query, tokens, tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )  # [B*N, 1, D], [B*N, 1, M]
        query = self.norm1(query + self.dropout(attn_out))
        query = self.norm2(query + self.ffn(query))
        return query, attn_weights


class GeometryAwareViewFusion(nn.Module):
    '''
    Anchor-centric geometry-aware multi-view fusion.

    For each 3D point (Gaussian anchor), builds one token per view from the
    sampled 2D feature and its viewing geometry (projected detector coordinate,
    sin/cos of the gantry angle, and an in-FOV indicator), then fuses the
    tokens with cross-attention where the query is derived from the 3D anchor
    coordinate. No per-view-index embedding is used, so the module is
    permutation invariant over views and supports arbitrary numbers of views.
    '''

    def __init__(
        self,
        in_ch,
        out_ch,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        geometry_hidden_dim=32,
        return_attention=False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, \
            f'd_model ({d_model}) must be divisible by num_heads ({num_heads})'

        self.return_attention = return_attention

        self.feat_proj = nn.Linear(in_ch, d_model)
        # geometry features per token: (u, v, sin(angle), cos(angle), in_fov)
        self.geom_mlp = nn.Sequential(
            nn.Linear(5, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, d_model),
        )
        # query embedding from the 3D anchor coordinate
        self.query_mlp = nn.Sequential(
            nn.Linear(3, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, d_model),
        )

        self.layers = nn.ModuleList([
            _CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_ch)

    def forward(self, view_feats, points_proj, points_xyz, angles, view_mask=None):
        # view_feats: [B, M, C, H, W]
        # points_proj: [B, M, N, 2], normalized detector coordinates in [-1, 1]
        # points_xyz: [B, N, 3], normalized volume coordinates
        # angles: [B, M]
        # view_mask: optional [B, M] bool, True = valid view
        #
        # returns:
        #   fused: [B, out_ch, N]
        #   attn: [B, N, M] if return_attention else None
        assert view_feats.dim() == 5, \
            f'view_feats must be [B, M, C, H, W], got {tuple(view_feats.shape)}'
        B, M = view_feats.shape[:2]
        assert points_proj.dim() == 4 and points_proj.shape[:2] == (B, M) and points_proj.shape[-1] == 2, \
            f'points_proj must be [B, M, N, 2]=({B}, {M}, N, 2), got {tuple(points_proj.shape)}'
        N = points_proj.shape[2]
        assert points_xyz.shape == (B, N, 3), \
            f'points_xyz must be [B, N, 3]=({B}, {N}, 3), got {tuple(points_xyz.shape)}'
        assert angles.shape == (B, M), \
            f'angles must be [B, M]=({B}, {M}), got {tuple(angles.shape)}'
        if view_mask is not None:
            assert view_mask.shape == (B, M), \
                f'view_mask must be [B, M]=({B}, {M}), got {tuple(view_mask.shape)}'

        # 1. sample pixel-aligned features per view: [B, C, N, M] => [B, N, M, C]
        sampled = query_view_feats(view_feats, points_proj, fusion=None)
        sampled = sampled.permute(0, 2, 3, 1)  # [B, N, M, C]

        # 2. per-token geometry features
        uv = points_proj.permute(0, 2, 1, 3)  # [B, N, M, 2]
        in_fov = (uv.abs() <= 1.0).all(dim=-1)  # [B, N, M] bool
        sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, M, 2]
        sin_cos = sin_cos.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, 2]
        geom = torch.cat([
            uv,
            sin_cos,
            in_fov.unsqueeze(-1).to(uv.dtype),
        ], dim=-1)  # [B, N, M, 5]

        tokens = self.feat_proj(sampled) + self.geom_mlp(geom)  # [B, N, M, D]

        # 3. validity: a token is valid only if its view is valid AND the
        # point projects inside the detector FOV (grid_sample results outside
        # the FOV are extrapolated garbage and must not be treated as valid).
        valid = in_fov
        if view_mask is not None:
            valid = valid & view_mask.bool().unsqueeze(1)  # [B, N, M]

        d_model = tokens.shape[-1]
        tokens = tokens.reshape(B * N, M, d_model)
        key_padding_mask = ~valid.reshape(B * N, M)  # True = ignore

        # rows where every view is invalid would make softmax produce NaNs;
        # temporarily unmask them and zero their outputs afterwards.
        all_invalid = key_padding_mask.all(dim=-1)  # [B*N]
        safe_mask = key_padding_mask & ~all_invalid.unsqueeze(-1)

        query = self.query_mlp(points_xyz).reshape(B * N, 1, d_model)  # [B*N, 1, D]

        attn_weights = None
        for layer in self.layers:
            query, attn_weights = layer(query, tokens, safe_mask)

        fused = self.out_proj(query)  # [B*N, 1, out_ch]
        fused = fused * (~all_invalid).to(fused.dtype).reshape(B * N, 1, 1)
        fused = fused.reshape(B, N, -1).transpose(1, 2)  # [B, out_ch, N]

        attn = None
        if self.return_attention:
            attn = attn_weights.reshape(B * N, M)
            attn = attn * (~all_invalid).to(attn.dtype).unsqueeze(-1)
            attn = attn.reshape(B, N, M)

        return fused, attn
