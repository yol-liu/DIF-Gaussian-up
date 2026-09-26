"""Per-view ViT replacing the UNet's 1/4-1/16 encoder, same outputs as models.unet.UNet.

Patch tokens come straight from the raw projection (no conv backbone underneath). Three
depths are reassembled DPT-style into the 1/4, 1/8, 1/16 pyramid (256/512/1024 channels).
The UNet-shaped decoder upsamples back to full resolution; its only skips below 1/4 are a
two-layer full-res and half-res conv stem, kept so the point branch still gets detail.
"""


import torch
from torch import nn

from models.unet import DoubleConv, DownConv, UpConv


def sincos_position(height, width, dim, device, dtype):
    quarter = dim // 4
    frequency = 1. / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequency
    cols = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequency
    rows = torch.cat([rows.sin(), rows.cos()], -1)[:, None].expand(-1, width, -1)
    cols = torch.cat([cols.sin(), cols.cos()], -1)[None].expand(height, -1, -1)
    return torch.cat([rows, cols], -1).reshape(height * width, dim).to(dtype)


class ViTEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, dim=384, depth=8, heads=6, patch_size=16):
        super().__init__()
        if patch_size != 16 or dim % 4 or dim % heads or depth < 3:
            raise ValueError('ViT encoder needs patch_size=16, depth>=3 and dim divisible by 4 and heads')
        self._out_ch = out_ch
        self._ds_ch = 1024
        self.dim = dim
        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, patch_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.taps = (depth // 3, 2 * depth // 3, depth)
        self.tap_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.taps])
        self.reassemble = nn.ModuleDict({
            '4': nn.Sequential(nn.ConvTranspose2d(dim, 256, 4, 4), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            '8': nn.Sequential(nn.ConvTranspose2d(dim, 512, 2, 2), nn.BatchNorm2d(512), nn.ReLU(inplace=True)),
            '16': nn.Sequential(nn.Conv2d(dim, self._ds_ch, 1), nn.BatchNorm2d(self._ds_ch), nn.ReLU(inplace=True)),
        })
        self.stem_full = DoubleConv(in_ch, 64)
        self.stem_half = DownConv(64, 128)
        self.decoder = nn.ModuleList([UpConv(1024, 512), UpConv(512, 256), UpConv(256, 128), UpConv(128, 64)])
        self.head = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    @property
    def out_ch(self):
        return self._out_ch

    @property
    def ds_ch(self):
        return self._ds_ch

    def encode(self, x):
        tokens = self.patch_embed(x)
        height, width = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + sincos_position(height, width, self.dim, x.device, tokens.dtype)
        taps = []
        for index, block in enumerate(self.blocks, 1):
            tokens = block(tokens)
            if index in self.taps:
                norm = self.tap_norms[self.taps.index(index)]
                taps.append(norm(tokens).transpose(1, 2).reshape(x.shape[0], self.dim, height, width))
        return {scale: self.reassemble[str(scale)](tap) for scale, tap in zip((4, 8, 16), taps)}

    def forward(self, x, view_mask=None, return_pyramid=False):
        b, m = x.shape[:2]
        x = x.reshape(b * m, *x.shape[2:])
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError('ViT encoder needs projection sizes divisible by 16')
        indices = None
        if view_mask is not None:
            if view_mask.shape != (b, m):
                raise ValueError('view_mask must have shape [B,V]')
            valid = view_mask.bool()
            if self.training and not valid.any(-1).all():
                raise ValueError('Training cases must have at least one valid view')
            indices = valid.flatten().nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                result = {'feats': x.new_zeros(b, m, self.out_ch, *x.shape[-2:]),
                          'feats_ds': x.new_zeros(b, m, self.ds_ch, x.shape[-2] // 16, x.shape[-1] // 16)}
                if return_pyramid:
                    result['pyramid'] = {stride: x.new_zeros(b, m, channels, x.shape[-2] // stride, x.shape[-1] // stride)
                                         for stride, channels in ((4, 256), (8, 512), (16, 1024))}
                return result
            x = x.index_select(0, indices)

        pyramid = self.encode(x)
        full = self.stem_full(x)
        half = self.stem_half(full)
        y = pyramid[16]
        for block, skip in zip(self.decoder, (pyramid[8], pyramid[4], half, full)):
            y = block(y, skip)
        y = self.head(y)

        def restore_views(features):
            if indices is not None:
                features = features.new_zeros(b * m, *features.shape[1:]).index_copy(0, indices, features)
            return features.reshape(b, m, *features.shape[1:])

        result = {'feats': restore_views(y), 'feats_ds': restore_views(pyramid[16])}
        if return_pyramid:
            result['pyramid'] = {scale: restore_views(value) for scale, value in pyramid.items()}
        return result
