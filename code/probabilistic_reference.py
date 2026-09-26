"""Prior-only distillation in the latent coordinates of an explicitly validated reference."""

import copy
import json
import math

import torch
from easydict import EasyDict

from transfer import file_hash
from models.probabilistic_model import build_model
from probabilistic_config import resolve_probability


def attach_reference(student, checkpoint_path, evidence_path):
    if not checkpoint_path or not evidence_path:
        raise ValueError('BLOCKED_REFERENCE: trained dense_np checkpoint and latent-usage evidence required')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    with open(evidence_path) as handle:
        evidence = json.load(handle)
    if checkpoint.get('global_step', 0) <= 0 or 'config' not in checkpoint:
        raise ValueError('BLOCKED_REFERENCE: reference must contain trained-step/config provenance')
    if evidence.get('checkpoint_sha256') != file_hash(checkpoint_path):
        raise ValueError('BLOCKED_REFERENCE: evidence does not match checkpoint')
    usage = evidence.get('latent_usage', {})
    if any(not isinstance(usage.get(key), (int, float)) or not math.isfinite(usage[key]) or usage[key] <= 1e-8
           for key in ('output_change_l1', 'sensitivity_l2')) or not evidence.get('case_ids'):
        raise ValueError('BLOCKED_REFERENCE: missing measured decoder latent-usage evidence')
    reference_config = copy.deepcopy(checkpoint['config']['model'])
    probability = resolve_probability(reference_config.get('probabilistic', {}))
    if probability['mode'] != 'dense_np' or probability['objective'] != 'joint' or probability['mean_code']:
        raise ValueError('BLOCKED_REFERENCE: expected a stochastic joint dense_np reference')
    expected, actual = copy.deepcopy(reference_config), copy.deepcopy(dict(student.cfg))
    expected['probabilistic'] = probability
    actual['probabilistic'] = resolve_probability(actual['probabilistic'])
    for configuration in (expected, actual):
        for key in ('objective', 'reference_checkpoint', 'reference_evidence', 'train_samples', 'evaluation', 'kl'):
            configuration['probabilistic'].pop(key, None)
    if expected != actual:
        raise ValueError('BLOCKED_REFERENCE: student/reference backbone, decoder and latent coordinates must match')
    reference = build_model(EasyDict(reference_config))
    reference.load_state_dict(checkpoint['model'], strict=True)
    reference.to(next(student.parameters()).device).eval().requires_grad_(False)
    # Identical initialization is mandatory, not just matching tensor dimensions.
    student.load_state_dict(checkpoint['model'], strict=True)
    student.requires_grad_(False)
    student.prior_head.requires_grad_(True)
    if student.use_local:
        student.local_prior_head.requires_grad_(True)
    # Reference is external immutable state, excluded from student checkpoints/optimizer/DDP.
    object.__setattr__(student, 'reference', reference)
    student.train(student.training)
    return {'checkpoint_sha256': evidence['checkpoint_sha256'], 'latent_usage': usage,
            'stage': 'prior_heads_only', 'teacher_gradients': False}
