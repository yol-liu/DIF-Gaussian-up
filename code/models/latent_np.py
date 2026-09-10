'''
Hierarchical amortized latents for DIF-Gaussian (C1).

GlobalLatent / LocalLatentGrid provide conditional prior & residual posterior.
FeatureFiLM applies zero-init modulation so latent is a perturbation of the
deterministic backbone (γ=1, β=0 at initialization).
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    # mu, logvar: [..., D]
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    '''
    KL(q || p) for diagonal Gaussians, per-dimension.
    returns: same shape as mu_q (elementwise KL contribution)
    '''
    return 0.5 * (
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
        - 1.0
    )


def apply_free_bits(kl_per_dim, free_bits):
    '''
    kl_per_dim: [B, D] (or [B, G, D] for local grid)
    free_bits: scalar lower bound per dimension (nats)
    returns scalar mean over batch of sum_D max(kl_d, free_bits)
    '''
    if free_bits is None or free_bits <= 0:
        return kl_per_dim.sum(dim=-1).mean()
    return torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()


def apply_sigma_floor(raw_logvar, sigma_floor):
    '''
    Map unconstrained raw to log-variance with sigma >= sigma_floor.
    sigma = softplus(raw) + sigma_floor; logvar = 2 log(sigma)
    '''
    sigma = F.softplus(raw_logvar) + sigma_floor
    return 2.0 * torch.log(sigma.clamp_min(1e-12))


class GlobalLatent(nn.Module):
    '''
    Case-level global latent z_g.

    Prior and posterior share the same conditioning trunk context (global_ctx).
    Posterior uses residual parameterization so Δ=0 ⇒ q ≡ p.
    '''

    def __init__(
        self,
        ctx_dim,
        latent_dim,
        gt_dim=None,
        hidden_dim=None,
        prior_sigma_floor=1e-2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_sigma_floor = prior_sigma_floor
        gt_dim = ctx_dim if gt_dim is None else gt_dim
        hidden_dim = hidden_dim or max(128, ctx_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        # residual Δ heads; zero-init so q starts at p
        self.delta_net = nn.Sequential(
            nn.Linear(ctx_dim + gt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def prior(self, global_ctx):
        # global_ctx: [B, C]
        h = self.prior_net(global_ctx)
        mu, raw_logvar = h.chunk(2, dim=-1)
        logvar = apply_sigma_floor(raw_logvar, self.prior_sigma_floor)
        return mu, logvar

    def posterior(self, global_ctx, gt_global_feat):
        # residual: mu_q = mu_p + Δmu, logvar_q = logvar_p + Δlogvar
        mu_p, logvar_p = self.prior(global_ctx)
        delta = self.delta_net(torch.cat([global_ctx, gt_global_feat], dim=-1))
        dmu, dlogvar = delta.chunk(2, dim=-1)
        return mu_p + dmu, logvar_p + dlogvar

    def sample(self, mu, logvar):
        return reparameterize(mu, logvar)


class LocalLatentGrid(nn.Module):
    '''
    Coarse 3D grid of local latents with top-down conditioning on z_g.
    Latents live on grid_res^3 cells and are trilinearly interpolated to anchors
    (no per-anchor independent sampling).
    '''

    def __init__(
        self,
        ctx_dim,
        global_dim,
        latent_dim,
        gt_dim=None,
        grid_res=4,
        hidden_dim=None,
        prior_sigma_floor=1e-2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_res = grid_res
        self.prior_sigma_floor = prior_sigma_floor
        gt_dim = ctx_dim if gt_dim is None else gt_dim
        hidden_dim = hidden_dim or max(128, ctx_dim + global_dim)

        in_prior = ctx_dim + global_dim
        self.prior_net = nn.Sequential(
            nn.Linear(in_prior, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(in_prior + gt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def pool_anchors_to_grid(self, anchor_feats, anchor_xyz):
        '''
        Mean-pool anchor features into coarse grid cells by hard binning.

        anchor_feats: [B, C, K]
        anchor_xyz:   [B, K, 3] in [0, 1]
        returns:      [B, C, R, R, R]
        '''
        B, C, K = anchor_feats.shape
        R = self.grid_res
        device = anchor_feats.device
        dtype = anchor_feats.dtype

        idx = (anchor_xyz * R).long().clamp(0, R - 1)  # [B, K, 3]
        flat = idx[..., 0] * (R * R) + idx[..., 1] * R + idx[..., 2]  # [B, K]

        feats_t = anchor_feats.transpose(1, 2)  # [B, K, C]
        out = torch.zeros(B, R * R * R, C, device=device, dtype=dtype)
        counts = torch.zeros(B, R * R * R, 1, device=device, dtype=dtype)

        flat_exp = flat.unsqueeze(-1).expand(-1, -1, C)
        out.scatter_add_(1, flat_exp, feats_t)
        ones = torch.ones(B, K, 1, device=device, dtype=dtype)
        counts.scatter_add_(1, flat.unsqueeze(-1), ones)
        out = out / counts.clamp_min(1.0)
        return out.transpose(1, 2).reshape(B, C, R, R, R)

    def prior(self, z_g, grid_ctx):
        '''
        z_g:      [B, D_g]
        grid_ctx: [B, C, R, R, R]
        returns mu, logvar: [B, D_l, R, R, R]
        '''
        B, C, R, _, _ = grid_ctx.shape
        assert R == self.grid_res
        ctx = grid_ctx.reshape(B, C, -1).transpose(1, 2)  # [B, G, C]
        zg = z_g.unsqueeze(1).expand(-1, ctx.shape[1], -1)
        h = self.prior_net(torch.cat([ctx, zg], dim=-1))  # [B, G, 2D]
        mu, raw = h.chunk(2, dim=-1)
        logvar = apply_sigma_floor(raw, self.prior_sigma_floor)
        G = R * R * R
        mu = mu.transpose(1, 2).reshape(B, -1, R, R, R)
        logvar = logvar.transpose(1, 2).reshape(B, -1, R, R, R)
        return mu, logvar

    def posterior(self, z_g, grid_ctx, gt_grid_feat):
        '''
        gt_grid_feat: [B, C_gt, R, R, R]
        residual over prior.
        '''
        mu_p, logvar_p = self.prior(z_g, grid_ctx)
        B, C, R, _, _ = grid_ctx.shape
        ctx = grid_ctx.reshape(B, C, -1).transpose(1, 2)
        gt = gt_grid_feat.reshape(B, gt_grid_feat.shape[1], -1).transpose(1, 2)
        zg = z_g.unsqueeze(1).expand(-1, ctx.shape[1], -1)
        delta = self.delta_net(torch.cat([ctx, zg, gt], dim=-1))
        dmu, dlogvar = delta.chunk(2, dim=-1)
        dmu = dmu.transpose(1, 2).reshape_as(mu_p)
        dlogvar = dlogvar.transpose(1, 2).reshape_as(logvar_p)
        return mu_p + dmu, logvar_p + dlogvar

    def sample(self, mu, logvar):
        return reparameterize(mu, logvar)

    def interpolate_to_anchors(self, z_grid, anchor_xyz):
        '''
        Trilinear interpolate grid latents to anchors.

        z_grid:     [B, D, R, R, R]
        anchor_xyz: [B, K, 3] in [0, 1]
        returns:    [B, K, D]
        '''
        # grid_sample expects align_corners coords in [-1, 1], and input
        # layout [B, C, D, H, W] with grid [B, D_out, H_out, W_out, 3] as (x,y,z)
        # corresponding to (W, H, D). Our grid is indexed (x,y,z) = (dim2,3,4).
        coords = anchor_xyz * 2.0 - 1.0  # [B, K, 3]
        grid = coords.unsqueeze(2).unsqueeze(2)  # [B, K, 1, 1, 3]
        # F.grid_sample uses (x, y, z) → (W, H, D); our tensor is [B,D,Rx,Ry,Rz]
        # so W=Rz, H=Ry, D=Rx → permute to [B, D, Rx, Ry, Rz] is already
        # (depth=x, height=y, width=z). Swap xyz → zyx for grid_sample.
        grid_zyx = torch.stack(
            [grid[..., 2], grid[..., 1], grid[..., 0]], dim=-1
        )
        sampled = F.grid_sample(
            z_grid, grid_zyx, mode='bilinear', align_corners=True, padding_mode='border'
        )  # [B, D, K, 1, 1]
        return sampled[:, :, :, 0, 0].transpose(1, 2)  # [B, K, D]


class FeatureFiLM(nn.Module):
    '''
    FiLM modulation of Gaussian features: f' = (1 + γ(z)) ⊙ f + β(z).
    Last linear layer is zero-initialized so γ=0, β=0 ⇒ f'=f at init.
    '''

    def __init__(self, feat_dim, latent_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(latent_dim, feat_dim)
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, z):
        # feats: [B, C, K]
        # z: [B, D] (broadcast over K) or [B, K, D]
        if z.dim() == 2:
            gb = self.net(z)  # [B, 2C]
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        else:
            gb = self.net(z)  # [B, K, 2C]
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = gamma.transpose(1, 2)
            beta = beta.transpose(1, 2)
        return (1.0 + gamma) * feats + beta


class AmplitudeModulation(nn.Module):
    '''
    Optional amplitude residual: a = softplus(a_det + g(z)).
    Last layer zero-init so g≡0 at init. Only used if the backbone exposes amplitude.
    '''

    def __init__(self, latent_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, amp_raw, z):
        # amp_raw: [B, K] or [B, 1, K]; z: [B, D] or [B, K, D]
        if z.dim() == 2:
            g = self.net(z).unsqueeze(-1)  # [B, 1, 1]
            if amp_raw.dim() == 2:
                g = g.squeeze(-1)
        else:
            g = self.net(z).squeeze(-1)  # [B, K]
            if amp_raw.dim() == 3:
                g = g.unsqueeze(1)
        return F.softplus(amp_raw + g)


def pool_global_ctx(anchor_feats, mode='mean'):
    '''
    Permutation-invariant pool over anchors → global context.
    anchor_feats: [B, C, K] → [B, C]
    '''
    if mode == 'mean':
        return anchor_feats.mean(dim=-1)
    if mode == 'max':
        return anchor_feats.max(dim=-1).values
    raise ValueError(f'unsupported pool mode: {mode}')
