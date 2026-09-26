import random

import numpy as np
import torch


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def weights_from_checkpoint(checkpoint):
    return checkpoint['model'] if 'model' in checkpoint else checkpoint


def warm_start(model, checkpoint, allowed_missing=(), reset_head=False):
    weights = dict(weights_from_checkpoint(checkpoint))
    excluded = []
    allowed = list(allowed_missing)
    if reset_head:
        prefix = f'point_decoder.mlps.{len(model.point_decoder.mlps) - 1}.'
        excluded = [name for name in weights if name.startswith(prefix)]
        weights = {name: value for name, value in weights.items() if name not in excluded}
        allowed.append(prefix)
    target = model.state_dict()
    missing = sorted(set(target) - set(weights))
    unexpected = sorted(set(weights) - set(target))
    forbidden = [name for name in missing if not any(name.startswith(prefix) for prefix in allowed)]
    mismatched = [name for name in weights if name in target and weights[name].shape != target[name].shape]
    if forbidden or unexpected or mismatched:
        raise ValueError(f'Unsafe warm_start: missing={forbidden}, unexpected={unexpected}, shape={mismatched}')
    model.load_state_dict(weights, strict=False)
    return {'loaded': sorted(weights), 'missing': missing, 'excluded': excluded, 'unexpected': unexpected}


def resume_training(model, optimizer, scheduler, checkpoint, rank=0):
    required = {'model', 'optimizer', 'scheduler', 'global_step', 'epoch', 'rng_by_rank', 'config'}
    if not required.issubset(checkpoint):
        raise ValueError('Weights-only checkpoint cannot resume exactly; use --warm_start')
    if rank >= len(checkpoint['rng_by_rank']):
        raise ValueError('Resume requires the original distributed world size')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng_by_rank'][rank])
    return checkpoint['epoch'] + 1, checkpoint['global_step']
