import os
import json
import yaml
import scipy
import pickle
import numpy as np
import warnings
import hashlib
from torch.utils.data import Dataset

from utils import sitk_load
from datasets.geometry import Geometry

# NumPy 2 compatibility: the LUNA16 projections were preprocessed with numpy>=2,
# whose pickles reference numpy._core.*; numpy<2 cannot unpickle them directly.
try:
    if not hasattr(np, '_core'):
        import sys as _sys
        np._core = np.core
        _sys.modules.setdefault('numpy._core', np.core)
        _sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        _sys.modules.setdefault('numpy._core.overrides', np.core.overrides)
except Exception:
    pass



class CBCT_dataset(Dataset):
    sample_seed = None  # class defaults keep partially constructed fixtures usable
    epoch = 0
    load_dense = False
    deployment_only = False

    def __init__(
            self,
            dst_name,
            cfg,
            split='train',
            num_views=10,
            npoint=5000,
            out_res_scale=1.0,
            random_views=False,
            view_offset=0,
            load_gt_volume=False,
            gt_volume_scale=1.0,
            sample_seed=None,
            load_dense=False,
            deployment_only=False,
        ):
        super().__init__()
        if not (0. < out_res_scale and out_res_scale <= 1.):
            raise ValueError
        if split == 'train' and out_res_scale != 1.:
            warnings.warn('out_res_scale does not resample cached training blocks')
        
        self.cfg = cfg
        self.dst_name = dst_name
        self.data_root = os.path.join(self.cfg.root_dir, dst_name)
        # optional full CT for C1 posterior (GT encoder); inference never needs it
        self.load_gt_volume = load_gt_volume
        self.gt_volume_scale = gt_volume_scale
        self.load_dense = load_dense
        self.deployment_only = deployment_only
        if deployment_only and (load_dense or load_gt_volume or split == 'train'):
            raise ValueError('Deployment data cannot load dense/GT or use train split')
        
        # load dataset info
        with open(os.path.join(self.data_root, 'meta_info.json'), 'r') as f:
            self.info = json.load(f)
            splits = split.split('+')
            name_list = []
            for s in splits:
                name_list += self.info[s]
            name_list = sorted(name_list)
            print('CBCT_dataset, name: {}, split: {}, len: {}.'.format(dst_name, split, len(name_list)))

        # load dataset config
        with open(os.path.join(self.data_root, self.info['dataset_config']), 'r') as f:
            dst_cfg = yaml.safe_load(f)
            out_res = np.array(dst_cfg['dataset']['resolution'])
            out_res = np.round(out_res * out_res_scale).astype(int) # to align the output shape with 'scipy.ndimage.zoom'
            if np.any(out_res < 2):
                raise ValueError('endpoint query grids need at least two samples per axis')
            self.geo = Geometry(dst_cfg['projector'])

        # prepare points
        if split == 'train':
            # load blocks' coordinates [train only]
            self.blocks = np.load(os.path.join(self.data_root, self.info['blocks_coords']))
            # SSIM patches need a C-order regular side^3 grid; verify once on block 0
            if int(cfg.get('ssim_patch', 0) or 0) >= 2:
                probe = self.blocks[0]
                side = int(round(len(probe) ** (1. / 3)))
                if side ** 3 != len(probe):
                    raise ValueError(f'SSIM patch requires cubic blocks, got {len(probe)} points')
                grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T.astype(np.float32)
                lo, hi = probe.min(0), probe.max(0)
                if not np.allclose(probe, grid * (hi - lo) / (side - 1) + lo, atol=1e-5):
                    raise ValueError('SSIM patch requires block coords to be a C-order regular grid')
        else:
            # prepare sampling points
            points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]]
            points = points.astype(np.float32)
            points = points.reshape(3, -1)
            points = points.transpose(1, 0) # N, 3
            self.points = points / (out_res - 1)
        
        # other parameters
        self.out_res_scale = out_res_scale
        self.name_list = name_list
        self.npoint = npoint
        self.is_train = (split == 'train')
        self.num_views = num_views
        self.random_views = random_views
        self.view_offset = view_offset
        # Sampling RNG kept separate from model-initialization and shuffling RNG:
        # block/point indices become a pure function of (sample_seed, epoch, case index),
        # so every arm consumes exactly the same training inputs.
        self.sample_seed = sample_seed
        self.epoch = 0

        # for acceleration when testing
        self.points_proj = None
        self.points_proj_key = None

    def __len__(self):
        return len(self.name_list)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def item_rng(self, index):
        if self.sample_seed is None:
            return None
        return np.random.default_rng([self.sample_seed, self.epoch, index])
    
    def sample_projections(self, name, n_view=None, view_ids=None, return_ids=False, rng=None):
        # -- load projections
        with open(os.path.join(self.data_root, self.info['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        context_request = n_view is None
        if n_view is None:
            n_view = self.num_views

        # -- sample projections
        if not 1 <= n_view <= len(projs):
            raise ValueError('num_views must lie within the available projection count')
        if view_ids is None:
            paired = self.cfg.get('paired_views')
            nested_eval = context_request and not self.is_train and paired and paired['dense_count'] <= len(projs)
            selection_count = paired['dense_count'] if nested_eval else n_view
            views = np.linspace(0, len(projs), selection_count, endpoint=False).astype(int)
            span = len(projs) - views[-1]
            if not self.random_views:
                offset = self.view_offset
            else:
                offset = int(rng.integers(span)) if rng is not None else int(np.random.randint(span))
            views += offset
            if nested_eval:
                views = views[np.linspace(0, len(views), n_view, endpoint=False).astype(int)]
        else:
            views = np.asarray(view_ids, dtype=np.int64)
        if views.ndim != 1 or len(np.unique(views)) != len(views) or np.any(views < 0) or np.any(views >= len(projs)):
            raise ValueError('view IDs must be unique valid projection indices')

        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]

        # -- de-normalization
        projs = projs * projs_max / 0.2

        return (projs, angles, views) if return_ids else (projs, angles)
    
    def load_ct(self, name, return_direction=False):
        image, spacing, origin, direction = sitk_load(
            os.path.join(self.data_root, self.info['image'].format(name)),
            uint8=True, return_direction=True,
        ) # float32
        if self.out_res_scale < 1.:
            original_shape = np.asarray(image.shape)
            image = scipy.ndimage.zoom(image, self.out_res_scale, order=3, prefilter=False)
            spacing = spacing * (original_shape - 1) / (np.asarray(image.shape) - 1)
        return (image, spacing, origin, direction) if return_direction else (image, spacing, origin)
    
    def load_block(self, name, b_idx):
        path = os.path.join(self.data_root, self.info['blocks_vals'].format(name, b_idx))
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values, rng=None):
        draw = rng if rng is not None else np.random
        choice = draw.choice(len(points), size=self.npoint, replace=False)
        points = points[choice]
        values = values[choice]
        return points, values

    def project_points(self, points, angles):
        points_proj = []
        for a in angles:
            p = self.geo.project(points, a)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0) # [M, N, 2]
        return points_proj

    def __getitem__(self, index):
        name = self.name_list[index]
        rng = self.item_rng(index)

        # -- load projections
        pair = self.cfg.get('paired_views') if (self.is_train or self.load_dense) else None
        dense_count = self.cfg.get('teacher_views') if self.is_train else None
        if pair:
            # One disk read / one scan normalization. C is indexed directly from D.
            dense_projs, dense_angles, dense_ids = self.sample_projections(
                name, n_view=pair['dense_count'], return_ids=True, rng=rng)
            counts = pair['context_counts'] if self.is_train else [self.num_views]
            draw = rng if rng is not None else np.random
            count = int(draw.choice(counts))
            if count >= len(dense_ids):
                raise ValueError('Dense training/oracle requires strictly more real views than context')
            if self.is_train and self.cfg.get('subset_mode', 'uniform') == 'nonuniform':
                subset = np.sort(draw.choice(len(dense_ids), count, replace=False))
            else:
                subset = np.linspace(0, len(dense_ids), count, endpoint=False).astype(int)
            projs, angles, view_ids = dense_projs[subset], dense_angles[subset], dense_ids[subset]
            # Fixed batch shape for multiple C counts; invalid padding never enters encoder/fusion.
            capacity = max(counts)
            context_mask = np.arange(capacity) < count
            projs = np.pad(projs, ((0, capacity-count), (0, 0), (0, 0), (0, 0)))
            angles = np.pad(angles, (0, capacity-count))
            view_ids = np.pad(view_ids, (0, capacity-count), constant_values=-1)
        elif dense_count is not None:
            if dense_count < self.num_views:
                raise ValueError('Teacher views must contain the sparse context')
            teacher_projs, teacher_angles, teacher_ids = self.sample_projections(name, n_view=dense_count, return_ids=True, rng=rng)
            if self.cfg.get('subset_mode', 'uniform') == 'nonuniform':
                draw = rng if rng is not None else np.random
                subset = np.sort(draw.choice(dense_count, self.num_views, replace=False))
            else:
                subset = np.linspace(0, dense_count, self.num_views, endpoint=False).astype(int)
            projs, angles, view_ids = teacher_projs[subset], teacher_angles[subset], teacher_ids[subset]
        else:
            projs, angles, view_ids = self.sample_projections(name, return_ids=True, rng=rng)

        # -- load sampling points
        patch_size = 0
        if not self.is_train:
            points = self.points
            if not self.deployment_only:
                points_gt, spacing, origin, direction = self.load_ct(name, return_direction=True)
            else:
                points_gt = None
        else:
            b_idx = int(rng.integers(len(self.blocks))) if rng is not None else np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values, rng=rng)
            points_gt = points_gt.astype(np.float32) / 255.
            points_gt = points_gt[None, :]

            # -- SSIM patch: contiguous K^3 sub-cube of the block grid.
            # SSIM needs local spatial structure; the 1024 scattered
            # training points carry none. Block coords are a C-order
            # regular side^3 grid (validated in __init__), so a patch is
            # a contiguous axis-aligned sub-cube of that grid.
            patch_size = int(self.cfg.get('ssim_patch', 0) or 0)
            if patch_size >= 2:
                side = int(round(len(block_coords) ** (1. / 3)))
                if side ** 3 != len(block_coords):
                    raise ValueError(f'SSIM patch sampling requires cubic blocks, got {len(block_coords)} points')
                k = min(patch_size, side)
                lo = (rng.integers(0, side - k + 1, size=3) if rng is not None
                      else np.random.randint(0, side - k + 1, size=3))
                axes = [np.arange(lo[a], lo[a] + k) for a in range(3)]
                flat = (axes[0][:, None, None] * side * side
                        + axes[1][None, :, None] * side
                        + axes[2][None, None, :]).reshape(-1) # C-order (i, j, k)
                patch_points = block_coords[flat].astype(np.float32)
                patch_gt = block_values[flat].astype(np.float32) / 255.
                patch_gt = patch_gt[None, :]

        # -- project points
        query_digest = None
        if not self.is_train:
            digest = hashlib.sha256()
            for start in range(0, len(points), 65536):
                digest.update(np.ascontiguousarray(points[start:start + 65536]).view(np.uint8))
            query_digest = (points.shape, str(points.dtype), digest.digest())
        cache_key = (np.asarray(angles).tobytes(), repr(vars(self.geo)), query_digest)
        if self.is_train or self.points_proj is None or cache_key != self.points_proj_key:
            points_proj = self.project_points(points, angles) # given the same geo cfg
            self.points_proj = points_proj
            self.points_proj_key = cache_key
        else:
            points_proj = self.points_proj
        patch_points_proj = None
        if self.is_train and patch_size >= 2:
            patch_points_proj = self.project_points(patch_points, angles) # [M, K^3, 2]

        # -- collect data
        ret_dict = {
            # M: the number of views
            # N: the number of sampled points
            'dst_name': self.dst_name,
            'name': name,
            'angles': angles,           # [M,]
            'view_ids': view_ids,
            'projs': projs,             # [M, 1, W, H], projections
            'points': points,           # [N, 3], center xyz of volumes ~[0, 1]
            'points_gt': points_gt,     # [1, N] (or [W', H', D'] only when is_train is False)
            'points_proj': points_proj, # [M, N, 2]
        }
        if self.is_train:
            ret_dict['block_idx'] = int(b_idx)
            if patch_size >= 2:
                ret_dict['patch_points'] = patch_points         # [K^3, 3]
                ret_dict['patch_points_proj'] = patch_points_proj
                ret_dict['patch_gt'] = patch_gt                 # [1, K^3]
        if pair:
            ret_dict.update(view_mask=context_mask, dense_projs=dense_projs, dense_angles=dense_angles,
                            dense_view_ids=dense_ids, dense_view_mask=np.ones(len(dense_ids), dtype=bool),
                            dense_points_proj=self.project_points(points, dense_angles))
        if dense_count is not None:
            ret_dict.update(teacher_projs=teacher_projs, teacher_angles=teacher_angles,
                            teacher_view_ids=teacher_ids, teacher_points_proj=self.project_points(points, teacher_angles))
        if self.deployment_only:
            ret_dict.pop('points_gt')
        elif not self.is_train:
            ret_dict['spacing'] = spacing # [3,]
            ret_dict['origin'] = origin   # [3,]
            ret_dict['direction'] = direction
        # downsampled GT volume for amortized posterior (C1); kept off by default
        if self.load_gt_volume:
            gt_vol, _, _ = sitk_load(
                os.path.join(self.data_root, self.info['image'].format(name)),
                uint8=True,
            )
            if self.gt_volume_scale < 1.0:
                gt_vol = scipy.ndimage.zoom(
                    gt_vol, self.gt_volume_scale, order=1, prefilter=False
                )
            ret_dict['gt_volume'] = gt_vol[None, ...]  # [1, D, H, W]
        return ret_dict
