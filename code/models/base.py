import torch
import torch.nn as nn
import numpy as np

from models.utils import query_view_feats
from models.unet import UNet
from models.vit_encoder import ViTEncoder
from models.hierarchical_encoder import HierarchicalEncoder
from models.point_decoder import PointDecoder



class Recon_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init()

    def init(self):
        self.init_encoder()
        self.init_decoder(self.image_encoder.out_ch)
        self.registered_point_keys = ['points_proj']

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.get('probabilistic', {}).get('normalization') == 'frozen_bn':
            from probabilistic_config import configure_normalization
            configure_normalization(self, 'frozen_bn')
        return self

    def init_encoder(self):
        cfg = self.cfg.image_encoder
        encoder_type = cfg.get('type', 'unet')
        if encoder_type == 'unet':
            self.image_encoder = UNet(
                in_ch=1,
                out_ch=cfg.out_ch
            )
        elif encoder_type == 'vit':
            vit = cfg.get('vit', {})
            self.image_encoder = ViTEncoder(1, cfg.out_ch, dim=vit.get('dim', 384), depth=vit.get('depth', 8),
                                            heads=vit.get('heads', 6), patch_size=vit.get('patch_size', 16))
        elif encoder_type == 'hierarchical':
            self.image_encoder = HierarchicalEncoder(1, cfg.out_ch, **cfg.get('hierarchical', {}))
        else:
            raise ValueError(f'unsupported image_encoder.type: {encoder_type}')
        
    def init_decoder(self, mid_ch):
        cfg = self.cfg.point_decoder
        self.point_decoder = PointDecoder(
            channels=[mid_ch] + cfg.mlp_chs,
            residual=True, 
            use_bn=True,
            legacy_head=cfg.get('legacy_head', False),
        )

    def encode_projs(self, data):
        encoder_output = self.image_encoder(data['projs'])
        return {
            'feats_proj': encoder_output['feats']
        }
    
    def forward_points(self, feats_dict, data):
        # 1. multi-view pixel-aligned features + max-pooling
        p_feats = query_view_feats(
            view_feats=feats_dict['feats_proj'], 
            points_proj=data['points_proj'],
            fusion='max'
        )

        # 2. point-wise prediction
        p_pred = self.point_decoder(p_feats)
        return p_pred

    def forward(self, data, is_eval=False, eval_npoint=100000, return_features=False, optimizer_step=None):
        if optimizer_step is not None:
            data = dict(data, optimizer_step=optimizer_step)
        data = self.prepare_views(data) if hasattr(self, 'prepare_views') else data
        feats_dict = self.encode_projs(data) # these features are shared for any sampled 3D points

        # point-wise forward
        if not is_eval:
            result = {
                'points_pred': self.forward_points(feats_dict, data)
            }
            if 'patch_points' in data:
                # auxiliary regular K^3 patch for the SSIM loss: same shared
                # Gaussian set / encoder features, only the cheap query+decode runs again
                patch_data = dict(data, points=data['patch_points'], points_proj=data['patch_points_proj'])
                result['patch_pred'] = self.forward_points(feats_dict, patch_data)
            if 'structure_loss' in feats_dict:
                result['structure_loss'] = feats_dict['structure_loss']
            if return_features:
                result['anchor_feats'] = feats_dict['anchor_feats']
            if 'diagnostics' in feats_dict:
                result['diagnostics'] = feats_dict['diagnostics']
            return result
        else:
            if eval_npoint < 1:
                raise ValueError('eval_npoint must be positive')
            total_npoint = data['points_proj'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                
                tmp_data = {}
                for key in data.keys():
                    if key in self.registered_point_keys:
                        tmp_data[key] = data[key][..., left:right, :]
                    else: 
                        tmp_data[key] = data[key]
                
                points_pred = self.forward_points(feats_dict, tmp_data) # B, C, N
                pred_list.append(points_pred)

            result = {'points_pred': torch.cat(pred_list, dim=2)}
            if 'diagnostics' in feats_dict:
                result['diagnostics'] = feats_dict['diagnostics']
            return result
