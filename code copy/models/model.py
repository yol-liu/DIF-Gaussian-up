import torch
import torch.nn as nn
import numpy as np

from models.base import Recon_base
from models.utils import query_view_feats, MLP_1d
from models.gs_utils import query_gs, build_covariance
from models.view_fusion import GeometryAwareViewFusion



class DIF_Gaussian(Recon_base):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self):
        self.init_encoder()

        # fusion configuration (missing keys default to the legacy max-pooling model)
        self.anchor_fusion = self.cfg.get('anchor_fusion', 'max')
        self.point_fusion = self.cfg.get('point_fusion', 'max')
        assert self.anchor_fusion in ('max', 'geometry_attention'), \
            f'unsupported anchor_fusion: {self.anchor_fusion}'
        assert self.point_fusion == 'max', \
            f'unsupported point_fusion: {self.point_fusion} (only "max" is supported in this milestone)'

        # gaussians-related modules
        mid_ch = self.image_encoder.out_ch
        ds_ch = self.image_encoder.ds_ch
        self.gs_feats_mlp = MLP_1d([ds_ch, ds_ch // 4, mid_ch], use_bn=True, last_bn=True, last_act=False)
        self.gs_params_mlp = MLP_1d([ds_ch, ds_ch // 4, 3 + 4 + 3], use_bn=True, last_bn=False, last_act=False) # 3d: offsets, 4d: rotation, 3d: scaling
        self.gs_act = nn.LeakyReLU(inplace=True)

        # gaussian interpolation options (defaults reproduce the legacy behavior)
        gs_cfg = self.cfg.gs
        self.gs_knn_k = gs_cfg.get('knn_k', 3)
        self.gs_shifted_center_knn = gs_cfg.get('shifted_center_knn', False)
        self.gs_normalize_weights = gs_cfg.get('normalize_weights', False)
        self.gs_covariance_jitter = gs_cfg.get('covariance_jitter', 0.0)

        # geometry-aware view attention for Gaussian anchors
        self.view_dropout = 0.0
        self.min_train_views = 1
        if self.anchor_fusion == 'geometry_attention':
            vf_cfg = self.cfg.get('view_fusion', {})
            self.anchor_view_fusion = GeometryAwareViewFusion(
                in_ch=ds_ch,
                out_ch=ds_ch,
                d_model=vf_cfg.get('d_model', 128),
                num_heads=vf_cfg.get('num_heads', 4),
                num_layers=vf_cfg.get('num_layers', 2),
                dropout=vf_cfg.get('dropout', 0.1),
                geometry_hidden_dim=vf_cfg.get('geometry_hidden_dim', 32),
                return_attention=vf_cfg.get('return_attention', False),
            )
            self.view_dropout = vf_cfg.get('view_dropout', 0.0)
            self.min_train_views = vf_cfg.get('min_train_views', 2)

        self.init_decoder(mid_ch * 2)
        self.registered_point_keys = ['points', 'points_proj']

    def sample_view_mask(self, batch_size, n_views, device):
        # training-time view dropout: independently sample a valid subset of
        # views for each batch item, preserving at least `min_train_views`.
        keep = torch.rand(batch_size, n_views, device=device) >= self.view_dropout
        min_views = min(self.min_train_views, n_views)
        for b in range(batch_size):
            n_missing = min_views - int(keep[b].sum())
            if n_missing > 0:
                dropped = torch.nonzero(~keep[b], as_tuple=False).flatten()
                revive = dropped[torch.randperm(len(dropped), device=device)[:n_missing]]
                keep[b, revive] = True
        return keep

    def encode_projs(self, data):
        encoder_output = self.image_encoder(data['projs'])

        if self.anchor_fusion == 'geometry_attention':
            view_mask = None
            if self.training and self.view_dropout > 0:
                B, M = data['projs'].shape[:2]
                view_mask = self.sample_view_mask(B, M, data['projs'].device)
            p_feats, _ = self.anchor_view_fusion(
                view_feats=encoder_output['feats_ds'],
                points_proj=data['points_gs_proj'],
                points_xyz=data['points_gs'] * 2 - 1, # [0, 1] => [-1, 1]
                angles=data['angles'],
                view_mask=view_mask,
            )
        else:
            p_feats = query_view_feats(
                view_feats=encoder_output['feats_ds'],
                points_proj=data['points_gs_proj'],
                fusion='max'
            )

        # features
        gs_feats = self.gs_feats_mlp(p_feats)
        
        # hyper-parameters
        gs_cfg = self.cfg.gs

        t = 1. / gs_cfg.t
        gs_res = gs_cfg.res
        p_dist_scaling = gs_cfg.p_dist_scaling
        o_scaling = gs_cfg.o_scaling
        s_scaling = gs_cfg.s_scaling

        p_dist = 1 / gs_res * p_dist_scaling
        
        # other gs parameters
        gs_params = self.gs_params_mlp(p_feats)
        # gs_params[:, :6, :] = gs_params[:, :6, :] / t
        gs_params = gs_params / t

        # offsets
        max_dist = p_dist * 0.5 * o_scaling # 0.5: half (two neighboring points share the distance)
        offsets = gs_params[:, :3, :].transpose(1, 2) # [B, K, 3]
        offsets = (torch.sigmoid(offsets) - 0.5) * 2 * max_dist

        # s
        s_mean = np.sqrt(p_dist)
        s = gs_params[:, 3:6, :].transpose(1, 2) # [B, K, 3]
        s = torch.sigmoid(s) * 2 * s_scaling * s_mean + (1 - s_scaling) * s_mean # (0, 1) => ((1-k)d, (1+k)d)

        # r
        r = gs_params[:, 6:, :].transpose(1, 2) # [B, K, 4]

        # convariance matrix, det/inv
        Cov = build_covariance(s, r, jitter=self.gs_covariance_jitter) # [B, K, 3, 3]
        det = torch.linalg.det(Cov)  # [B, K]
        inv = torch.linalg.inv(Cov)  # [B, K, 3, 3]
        _, log_det = torch.linalg.slogdet(Cov) # [B, K], stable log|Cov|

        return {
            'feats_proj': encoder_output['feats'],
            'gs_params': {
                'feats': gs_feats,   # [B, C, K]
                'offsets': offsets,  # [B, K, 3]
                'det': det,          # [B, K]
                'inv': inv,          # [B, K, 3, 3]
                'log_det': log_det,  # [B, K]
            }
        }

    def forward_points(self, feats_dict, data):
        # 1. multi-view pixel-aligned features + max-pooling
        # (dense query points keep max fusion in this milestone; full
        # cross-attention over ~256^3 points would be prohibitively expensive)
        p_feats = query_view_feats(
            view_feats=feats_dict['feats_proj'], 
            points_proj=data['points_proj'],
            fusion='max'
        )

        # 2. gaussian-based interp
        interp_feats = query_gs(
            data['points'], 
            data['points_gs'], 
            feats_dict['gs_params'],
            knn_k=self.gs_knn_k,
            shifted_center_knn=self.gs_shifted_center_knn,
            normalize_weights=self.gs_normalize_weights,
        )
        interp_feats = self.gs_act(interp_feats)
        p_feats = torch.cat([p_feats, interp_feats], dim=1)

        # 3. point-wise prediction
        p_pred = self.point_decoder(p_feats)
        return p_pred
