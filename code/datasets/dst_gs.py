import numpy as np
from copy import deepcopy

from datasets.base import CBCT_dataset
from utils import sitk_load
from structure_targets import physical_structure_target
import os
import hashlib
import json
import tempfile
from scipy.ndimage import zoom



class CBCT_dataset_gs(CBCT_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gs_res = self.cfg.gs_res
        points_gs = np.mgrid[:gs_res, :gs_res, :gs_res] / gs_res
        self.points_gs = points_gs.reshape(3, -1).transpose(1, 0) # ~[0, 1]
        self.structure_cache = {}

    def load_structure(self, name):
        if name in self.structure_cache:
            return self.structure_cache[name]
        settings = self.cfg.get('structure_target', {})
        path = os.path.join(self.data_root, self.info['image'].format(name))
        frame = settings.get('frame', 'voxel_xyz')
        stat = os.stat(path)
        signature = {'version': 2, 'path': os.path.abspath(path), 'size': stat.st_size,
                     'mtime_ns': stat.st_mtime_ns, 'settings': dict(settings),
                     'volume_extent': (np.asarray(self.geo.v_res) * self.geo.v_spacing).tolist()}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode() + self.points_gs.tobytes()).hexdigest()
        cache_dir = settings.get('cache_dir')
        cache_path = os.path.join(cache_dir, digest + '.npz') if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=False) as saved:
                result = (saved['tensor'], saved['confidence'], float(saved['sigma_mm']), float(saved['rho_mm']))
        else:
            image, spacing, _ = sitk_load(path, uint8=True)
            original_shape = np.asarray(image.shape)
            if frame == 'scanner':
                spacing = np.asarray(self.geo.v_res) * self.geo.v_spacing / (original_shape - 1)
            target_scale = settings.get('scale', .25)
            if target_scale != 1:
                image = zoom(image, target_scale, order=1, prefilter=False)
                spacing = spacing * (original_shape - 1) / (np.asarray(image.shape) - 1)
            sigma_mm = settings.get('sigma_mm') or float(np.max(spacing))
            rho_mm = settings.get('rho_mm') or 2 * sigma_mm
            tensor, confidence = physical_structure_target(image, spacing, self.points_gs, frame=frame,
                                                           sigma_mm=sigma_mm, rho_mm=rho_mm,
                                                           energy_threshold=settings.get('energy_threshold', 0.))
            result = tensor, confidence, sigma_mm, rho_mm
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as handle:
                    temporary = handle.name
                    np.savez_compressed(handle, tensor=tensor, confidence=confidence, sigma_mm=sigma_mm, rho_mm=rho_mm)
                os.replace(temporary, cache_path)
        self.structure_cache = {name: result}
        return result

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        # projections of GS points (initial center xyz)
        points_gs = deepcopy(self.points_gs)
        points_gs_proj = self.project_points(points_gs, data_dict['angles'])

        data_dict.update({
            'points_gs': points_gs,          # [K, 3]
            'points_gs_proj': points_gs_proj # [M, K, 2]
        })
        if 'teacher_angles' in data_dict:
            data_dict['teacher_points_gs_proj'] = self.project_points(points_gs, data_dict['teacher_angles'])
        if 'dense_angles' in data_dict:
            data_dict['dense_points_gs_proj'] = self.project_points(points_gs, data_dict['dense_angles'])
        data_dict.update(self.geo.adapter_fields())
        if self.is_train and self.cfg.get('structure_supervision', False):
            name = data_dict['name']
            tensor, confidence, sigma_mm, rho_mm = self.load_structure(name)
            data_dict['structure_target'] = tensor
            data_dict['structure_confidence'] = confidence
            data_dict['structure_sigma_mm'] = np.float32(sigma_mm)
            data_dict['structure_rho_mm'] = np.float32(rho_mm)
        return data_dict
