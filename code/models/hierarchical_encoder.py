"""Native stride 4/8/16 window Transformer with the UNet's shared detail decoder.

Two shallow convolutional stages retain full/half-resolution detail. Average-pool
patch merging preserves the feature_grid center convention. Shifted partitions use
padding (not cyclic wrapping), so opposite image boundaries never attend together.
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.unet import UNet, DoubleConv, DownConv, UpConv


class WindowBlock(nn.Module):
    def __init__(self, dim, heads, window=8, shifted=False):
        super().__init__()
        self.window, self.heads = window, heads
        self.shift = window // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :] + window - 1
        self.register_buffer('relative_index', relative[0] * (2 * window - 1) + relative[1], persistent=False)

    def forward(self, x):
        b, c, h, w = x.shape
        window, shift = self.window, self.shift
        x = x.permute(0, 2, 3, 1)
        norm = self.norm1(x)
        bottom, right = (-(h + shift)) % window, (-(w + shift)) % window
        padded = F.pad(norm, (0, 0, shift, right, shift, bottom))
        ph, pw = padded.shape[1:3]
        valid = F.pad(torch.ones(h, w, dtype=torch.bool, device=x.device), (shift, right, shift, bottom))
        valid = valid.reshape(ph // window, window, pw // window, window).permute(0, 2, 1, 3).reshape(-1, window * window)
        valid = valid.repeat(b, 1)
        tokens = padded.reshape(b, ph // window, window, pw // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)
        q, k, v = self.qkv(tokens).reshape(-1, window * window, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)[None]
        mask = bias.expand(tokens.shape[0], -1, -1, -1).masked_fill(~valid[:, None, None, :], float('-inf'))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape_as(tokens)
        attended = self.proj(attended).reshape(b, ph // window, pw // window, window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(b, ph, pw, c)
        x = x + attended[:, shift:shift + h, shift:shift + w]
        return (x + self.ffn(self.norm2(x))).permute(0, 3, 1, 2)


class HierarchicalEncoder(UNet):
    def __init__(self, in_ch, out_ch, dims=(96, 192, 384), depths=(2, 2, 2), heads=(3, 6, 12), window=8):
        nn.Module.__init__(self)
        if len(dims) != 3 or len(depths) != 3 or len(heads) != 3 or window < 2:
            raise ValueError('hierarchical encoder needs three stages and window >=2')
        if any(d <= 0 or n <= 0 or h <= 0 or d % h for d, n, h in zip(dims, depths, heads)):
            raise ValueError('stage dimensions/depths/heads must be positive and dimensions divisible by heads')
        self._out_ch, self._ds_ch = out_ch, 1024
        # Exact names/shapes load the canonical UNet initialization for shared parts.
        self.inc = DoubleConv(in_ch, 64)
        self.down = nn.ModuleList([DownConv(64, 128)])
        self.merges = nn.ModuleList([nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(inp, dim, 1))
                                    for inp, dim in zip((128, *dims[:-1]), dims)])
        self.stages = nn.ModuleList([nn.Sequential(*[WindowBlock(dim, head, window, bool(i % 2)) for i in range(depth)])
                                    for dim, depth, head in zip(dims, depths, heads)])
        self.outputs = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, channels, 1), nn.BatchNorm2d(channels), nn.ReLU())
                                     for dim, channels in zip(dims, (256, 512, 1024))])
        self.up = nn.ModuleList([UpConv(1024, 512), UpConv(512, 256), UpConv(256, 128), UpConv(128, 64)])
        self.outc = nn.Sequential(nn.Conv2d(64, out_ch, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x, view_mask=None, return_pyramid=False):
        b, views = x.shape[:2]
        x = x.flatten(0, 1)
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError('hierarchical encoder requires image dimensions divisible by 16')
        indices = None
        if view_mask is not None:
            if view_mask.shape != (b, views):
                raise ValueError('view_mask must have shape [B,V]')
            if self.training and not view_mask.bool().any(-1).all():
                raise ValueError('Training cases must have at least one valid view')
            indices = view_mask.bool().flatten().nonzero().flatten()
            if indices.numel() == 0:
                result = {'feats': x.new_zeros(b, views, self.out_ch, *x.shape[-2:]),
                          'feats_ds': x.new_zeros(b, views, self.ds_ch, x.shape[-2] // 16, x.shape[-1] // 16)}
                if return_pyramid:
                    result['pyramid'] = {s: x.new_zeros(b, views, c, x.shape[-2] // s, x.shape[-1] // s)
                                         for s, c in ((4, 256), (8, 512), (16, 1024))}
                return result
            x = x.index_select(0, indices)
        full = self.inc(x)
        half = self.down[0](full)
        features, y = [], half
        for merge, stage, output in zip(self.merges, self.stages, self.outputs):
            y = stage(merge(y))
            features.append(output(y))
        y = features[-1]
        for up, skip in zip(self.up, (features[1], features[0], half, full)):
            y = up(y, skip)
        y = self.outc(y)

        def restore(value):
            if indices is not None:
                value = value.new_zeros(b * views, *value.shape[1:]).index_copy(0, indices, value)
            return value.reshape(b, views, *value.shape[1:])

        result = {'feats': restore(y), 'feats_ds': restore(features[-1])}
        if return_pyramid:
            result['pyramid'] = {s: restore(f) for s, f in zip((4, 8, 16), features)}
        return result
