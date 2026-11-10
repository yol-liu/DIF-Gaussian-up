import torch
import torch.nn as nn
import numpy as np

from models.base import Recon_base
from models.utils import query_view_feats, MLP_1d
from models.gs_utils import query_gs, build_covariance
from models.view_fusion import GeometryAwareViewFusion
from models.anatomy import AnatomySupportEncoder, structure_loss
from models.multiview_encoder import MultiViewGeometryEncoder, GeometryPointFusion
from config_schema import resolve_transformer_config
from geometry_adapter import GeometryAdapter, aligned_anchor_features



class DIF_Gaussian(Recon_base):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self):
        self.init_encoder()

        # fusion configuration (missing keys default to the legacy max-pooling model)
        self.anchor_fusion = self.cfg.get('anchor_fusion', 'max')
        self.point_fusion = self.cfg.get('point_fusion', 'max')
        self.mask_fov = self.cfg.get('mask_fov', True)
        assert self.anchor_fusion in ('max', 'geometry_attention', 'geometry_transformer'), \
            f'unsupported anchor_fusion: {self.anchor_fusion}'
        if self.point_fusion not in ('max', 'geometry_attention'):
            raise ValueError(f'unsupported point_fusion: {self.point_fusion}')
        if self.point_fusion == 'geometry_attention' and self.anchor_fusion != 'geometry_transformer':
            raise ValueError('Point geometry_attention requires geometry_transformer anchors')

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
                dropout=vf_cfg.get('dropout', 0.0),
                geometry_hidden_dim=vf_cfg.get('geometry_hidden_dim', 32),
                return_attention=vf_cfg.get('return_attention', False),
                # defaults favor stable warm-start from max pooling
                use_max_residual=vf_cfg.get('use_max_residual', True),
                content_query=vf_cfg.get('content_query', True),
            )
            self.view_dropout = vf_cfg.get('view_dropout', 0.0)
            self.min_train_views = vf_cfg.get('min_train_views', 2)

        self.init_decoder(mid_ch * 2)
        anatomy_cfg = self.cfg.get('anatomy', {})
        if anatomy_cfg.get('enabled', False):
            if self.cfg.get('latent', {}).get('use_latent', False):
                raise ValueError('Anatomy arms must first be evaluated without latent')
            self.anatomy_encoder = AnatomySupportEncoder(ds_ch, mid_ch, gs_cfg.res, anatomy_cfg)
        if self.anchor_fusion == 'geometry_transformer':
            if self.cfg.get('latent', {}).get('use_latent', False) or anatomy_cfg.get('enabled', False):
                raise ValueError('Transformer supports model.probabilistic; legacy latent/older anatomy are disabled')
            self.transformer_config = resolve_transformer_config(self.cfg.get('multiview_transformer', {}))
            self.multiview_encoder = MultiViewGeometryEncoder(ds_ch, gs_cfg.res, self.transformer_config)
            if self.point_fusion == 'geometry_attention':
                self.point_view_fusion = GeometryPointFusion(mid_ch, self.transformer_config)
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

    def prepare_views(self, data):
        if 'view_mask' in data:
            valid = data['view_mask'].bool()
            if valid.shape != data['projs'].shape[:2]:
                raise ValueError('view_mask must match projection batch and view dimensions')
            if self.training and not valid.any(-1).all():
                raise ValueError('Training sampler must provide at least one valid view per case')
            if self.training and self.view_dropout > 0:
                keep = valid & (torch.rand_like(valid.float()) >= self.view_dropout)
                for batch_index in range(valid.shape[0]):
                    minimum = min(self.min_train_views, int(valid[batch_index].sum()))
                    candidates = (valid[batch_index] & ~keep[batch_index]).nonzero().flatten()
                    needed = max(0, minimum - int(keep[batch_index].sum()))
                    keep[batch_index, candidates[torch.randperm(len(candidates), device=valid.device)[:needed]]] = True
                valid = keep
            return dict(data, view_mask=valid)
        if not self.training or self.view_dropout <= 0:
            return data
        count = data['projs'].shape[1]
        keep = self.sample_view_mask(1, count, data['projs'].device)[0]
        return {
            key: value[:, keep] if key in (
                'projs', 'angles', 'points_proj', 'points_gs_proj', 'view_ids',
                'patch_points_proj'
            ) else value
            for key, value in data.items()
        }

    def encode_projs(self, data):
        encoder_kwargs = {}
        if 'view_mask' in data:
            encoder_kwargs['view_mask'] = data['view_mask']
        if self.anchor_fusion == 'geometry_transformer':
            encoder_kwargs['return_pyramid'] = True
        encoder_output = self.image_encoder(data['projs'], **encoder_kwargs)

        if self.anchor_fusion == 'geometry_attention':
            view_mask = data.get('view_mask')
            p_feats, _ = self.anchor_view_fusion(
                view_feats=encoder_output['feats_ds'],
                points_proj=data['points_gs_proj'],
                points_xyz=data['points_gs'] * 2 - 1, # [0, 1] => [-1, 1]
                angles=data['angles'],
                view_mask=view_mask,
            )
        else:
            anchor_coordinates = data['points_gs_proj']
            if self.anchor_fusion == 'geometry_transformer':
                anchor_coordinates = GeometryAdapter(data).project(data['points_gs'])['uv']
            token_init = self.transformer_config['token_init'] if self.anchor_fusion == 'geometry_transformer' else 'max'
            if self.anchor_fusion == 'geometry_transformer' and self.transformer_config['coordinate_mode'] == 'aligned':
                p_feats = aligned_anchor_features(
                    encoder_output['feats_ds'], GeometryAdapter(data).project(data['points_gs']),
                    data['projs'].shape[-2:], 16, token_init)
            else:
                p_feats = query_view_feats(
                    view_feats=encoder_output['feats_ds'],
                    points_proj=anchor_coordinates,
                    fusion=token_init,
                    mask_fov=self.mask_fov,
                    view_mask=data.get('view_mask'),
                )

        auxiliary = {}
        if self.anchor_fusion == 'geometry_transformer':
            p_feats, auxiliary = self.multiview_encoder(encoder_output, p_feats, data)
            if self.point_fusion == 'geometry_attention':
                auxiliary['point_feature_maps'] = self.point_view_fusion.prepare(encoder_output['feats'])
        if hasattr(self, 'anatomy_encoder'):
            p_feats, tensor = self.anatomy_encoder(p_feats, encoder_output, data)
            if self.training and data.get('auxiliary_supervision', True) and self.anatomy_encoder.arm in ('Gaux', 'Ganat'):
                auxiliary['structure_loss'] = structure_loss(
                    tensor, data['structure_target'], data['structure_confidence'])

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
        with torch.autocast(device_type=s.device.type, enabled=False):
            stable_scales = s.float() if s.dtype in (torch.float16, torch.bfloat16) else s
            stable_rotation = r.to(stable_scales.dtype)
            Cov = build_covariance(stable_scales, stable_rotation, jitter=self.gs_covariance_jitter)
            det = torch.linalg.det(Cov)
            inv = torch.linalg.inv(Cov)
            _, log_det = torch.linalg.slogdet(Cov)

        return {
            **auxiliary,
            'feats_proj': encoder_output['feats'],
            # fused per-anchor features before GS MLPs (shared conditioning trunk
            # for any optional latent prior/posterior; ignored by deterministic path)
            'anchor_feats': p_feats,  # [B, C_ds, K]
            'gs_params': {
                'feats': gs_feats,   # [B, C, K]
                'offsets': offsets,  # [B, K, 3]
                'det': det,          # [B, K]
                'inv': inv,          # [B, K, 3, 3]
                'log_det': log_det,  # [B, K]
            }
        }

    def forward_points(self, feats_dict, data, z=None):
        # 1. multi-view pixel-aligned features + max-pooling
        # (dense query points keep max fusion in this milestone; full
        # cross-attention over ~256^3 points would be prohibitively expensive)
        if self.point_fusion == 'max':
            point_coordinates = data['points_proj']
            if self.anchor_fusion == 'geometry_transformer':
                point_coordinates = GeometryAdapter(data).project(data['points'])['uv']
            p_feats = query_view_feats(
                view_feats=feats_dict['feats_proj'], points_proj=point_coordinates,
                fusion='max', mask_fov=self.mask_fov, view_mask=data.get('view_mask'))

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
        if self.point_fusion == 'geometry_attention':
            p_feats = self.point_view_fusion(feats_dict['feats_proj'], feats_dict['point_feature_maps'], interp_feats, data)
            feats_dict['diagnostics'].update(self.point_view_fusion.last_diagnostics)
        if z is not None and getattr(self, 'z_point_proj', None) is not None:
            # Latent pathway: additive zero-init injection so z reaches the main
            # per-point evidence path, not only the FiLM-modulated anchor features.
            p_feats = p_feats + self.z_point_proj(z).unsqueeze(-1)
        p_feats = torch.cat([p_feats, interp_feats], dim=1)

        # 3. point-wise prediction
        p_pred = self.point_decoder(p_feats)
        return p_pred
