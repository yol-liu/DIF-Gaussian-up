import hashlib
import json

import torch
import torch.nn.functional as functional


def feature_distillation(student, teacher, reliability=None):
    if student.shape != teacher.shape:
        raise ValueError('Feature transfer requires matching anchor/channel semantics')
    error = (functional.normalize(student, dim=1, eps=1e-6) -
             functional.normalize(teacher.detach(), dim=1, eps=1e-6)).square().mean(1)
    if reliability is None:
        return error.mean()
    if reliability.shape != error.shape or (reliability < 0).any() or not torch.isfinite(reliability).all():
        raise ValueError('Reliability must be finite nonnegative weights on matching anchors')
    return (error * reliability.detach()).sum() / reliability.sum().clamp_min(1e-8)


def validate_teacher(evidence):
    required = ('checkpoint_sha256', 'case_ids', 'teacher_psnr', 'sparse_psnr',
                'teacher_ssim', 'sparse_ssim', 'eval_scale', 'split')
    if any(key not in evidence for key in required):
        raise ValueError('Teacher validation evidence is incomplete')
    if evidence['eval_scale'] != 1. or evidence['split'] != 'eval':
        raise ValueError('Teacher must be validated at full resolution on eval cases')
    count = len(evidence['case_ids'])
    if count == 0 or len(set(evidence['case_ids'])) != count:
        raise ValueError('Teacher validation needs distinct cases')
    metrics = [torch.as_tensor(evidence[key]) for key in required[2:6]]
    if any(value.shape != (count,) or not torch.isfinite(value).all() for value in metrics):
        raise ValueError('Teacher metrics must be finite per-case arrays')
    teacher_psnr, sparse_psnr, teacher_ssim, sparse_ssim = metrics
    if teacher_psnr.mean() <= sparse_psnr.mean() or teacher_ssim.mean() < sparse_ssim.mean() - 0.002:
        raise ValueError('TEACHER_NOT_BETTER')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def feature_cache_key(case_id, view_ids, geometry, checkpoint_hash, augmentation):
    payload = dict(case_id=case_id, view_ids=view_ids, geometry=geometry,
                   checkpoint_hash=checkpoint_hash, augmentation=augmentation)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def teacher_inputs(data):
    result = {key: value for key, value in data.items()
              if not key.startswith('teacher_') and key not in ('gt_volume', 'points_gt', 'structure_target', 'structure_confidence')}
    for key in ('projs', 'angles', 'view_ids', 'points_proj', 'points_gs_proj'):
        result[key] = data['teacher_' + key]
    result.pop('view_mask', None)
    if 'teacher_view_mask' in data:
        result['view_mask'] = data['teacher_view_mask']
    for key in ('DSO', 'DSD', 'detector_extent', 'detector_resolution', 'detector_offset'):
        if 'teacher_' + key in data:
            result[key] = data['teacher_' + key]
    return result


def validate_transfer_provenance(evidence, train_cases, checkpoint_hash, initialization_hash):
    if evidence.get('feature_semantics') != 'shared_checkpoint_initialization':
        raise ValueError('BLOCKED_TEACHER_SEMANTICS: feature KD requires shared checkpoint initialization')
    if checkpoint_hash != initialization_hash:
        raise ValueError('Student must warm-start from the validated teacher checkpoint before feature KD')
    teacher_cases = set(evidence.get('teacher_train_cases', []))
    if not teacher_cases or not teacher_cases.issubset(set(train_cases)):
        raise ValueError('Teacher training cases must belong exclusively to the current training split')
    if evidence.get('dense_view_count', 0) <= evidence.get('sparse_view_count', 0):
        raise ValueError('Teacher must use strictly more distinct real views')
    if evidence.get('teacher_feature_variance', 0.) <= 1e-8:
        raise ValueError('Teacher feature variance must be measured and noncollapsed')
