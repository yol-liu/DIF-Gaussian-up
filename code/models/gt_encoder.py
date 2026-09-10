'''
Lightweight 3D CNN that encodes a GT CT volume for the variational posterior.

Only used during training when gt_volume is provided. Inference never builds
or calls this module when gt_volume is None.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import index_3d


class GTEncoder(nn.Module):
    '''
    Maps a GT CT volume to a feature volume, then samples at anchors and
    pools a global feature for the hierarchical posterior.
    '''

    def __init__(self, in_ch=1, out_ch=32, mid_ch=16):
        super().__init__()
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, gt_volume, anchor_xyz=None):
        '''
        gt_volume:  [B, 1, D, H, W] (or [B, D, H, W] → unsqueezed)
        anchor_xyz: optional [B, K, 3] in [0, 1]

        returns dict:
          feat_volume:    [B, C, D', H', W']
          gt_global_feat: [B, C]
          gt_anchor_feat: [B, C, K] if anchor_xyz given else None
        '''
        if gt_volume.dim() == 4:
            gt_volume = gt_volume.unsqueeze(1)
        assert gt_volume.dim() == 5, \
            f'gt_volume must be [B,1,D,H,W], got {tuple(gt_volume.shape)}'

        feat_volume = self.net(gt_volume)
        gt_global_feat = feat_volume.mean(dim=(2, 3, 4))

        gt_anchor_feat = None
        if anchor_xyz is not None:
            # index_3d expects uv in [-1, 1]
            uv = anchor_xyz * 2.0 - 1.0
            gt_anchor_feat = index_3d(feat_volume, uv)  # [B, C, K]

        return {
            'feat_volume': feat_volume,
            'gt_global_feat': gt_global_feat,
            'gt_anchor_feat': gt_anchor_feat,
        }
