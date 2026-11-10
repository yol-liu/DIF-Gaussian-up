import os
import shutil
import argparse
import json
import random
import time
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dst_gs import CBCT_dataset_gs
from models.factory import build_model
from utils import convert_cuda, load_config
from evaluate import eval_one_epoch
from losses import elbo_loss, compute_collapse_diagnostics, metrics_to_wandb, ssim3d_loss, tv3d_loss, volume_nll
from checkpoints import rng_state, warm_start, resume_training
from checkpoints import weights_from_checkpoint
from transfer import feature_distillation, teacher_inputs, validate_teacher, file_hash
from transfer import validate_transfer_provenance
from training_diagnostics import add_structure_loss, transformer_gradient_norms
from geometry_adapter import GeometryAdapter


try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def points_digest(points):
    import hashlib
    array = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:16]


def load_common_init(model, path):
    '''Copy shared same-name same-shape weights; report what each arm initializes alone.'''
    import hashlib
    weights = weights_from_checkpoint(torch.load(path, map_location='cpu', weights_only=False))
    target = model.state_dict()
    shared, mismatched = {}, []
    for name, value in weights.items():
        if name in target:
            if tuple(value.shape) != tuple(target[name].shape):
                mismatched.append(name)
            else:
                shared[name] = value
    if mismatched:
        raise ValueError(f'Shared initialization shape mismatch: {mismatched}')
    model.load_state_dict(shared, strict=False)
    digest = hashlib.sha256()
    for name in sorted(shared):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(shared[name].float().numpy()).tobytes())
    return {'common_init': os.path.abspath(path), 'common_init_sha256': file_hash(path),
            'shared_parameters': sorted(shared),
            'shared_count': len(shared), 'shared_digest': digest.hexdigest(),
            'arm_specific': sorted(set(target) - set(shared)),
            'unused_in_this_arm': sorted(set(weights) - set(target))}


def _cfg_to_dict(obj):
    '''Recursively convert EasyDict / nested containers for wandb.config.'''
    if isinstance(obj, dict):
        return {k: _cfg_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_dict(v) for v in obj]
    return obj


def _wandb_enabled(args):
    return bool(getattr(args, 'wandb', False)) and _HAS_WANDB and args.local_rank == 0


def resume_path_pending(args):
    return args.resume is not None or bool(args.resume_path)


def _wandb_log(args, payload, step=None):
    if not _wandb_enabled(args):
        return
    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default='configs/default.yaml')
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--eval_scale', type=float, default=1.0)
    parser.add_argument('--val_case_count', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--max_seconds', type=float)
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--random_views', action='store_true', default=False)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--resume_path')
    parser.add_argument('--warm_start')
    parser.add_argument('--allow_missing', nargs='*', default=[])
    parser.add_argument('--reset_head', action='store_true')
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help='enable Weights & Biases logging (default: on)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='disable Weights & Biases logging')
    parser.set_defaults(wandb=True)
    parser.add_argument('--wandb_project', type=str, default='dif-gaussian')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--diag_every', type=int, default=200,
                        help='log latent collapse diagnostics every N optimizer steps')
    parser.add_argument('--val_steps', type=int, nargs='*', default=[],
                        help='fixed optimizer-update counts at which to validate and checkpoint')
    parser.add_argument('--manifest_steps', type=int, default=100,
                        help='record cases/views/block and point indices for this many initial steps')
    parser.add_argument('--common_init',
                        help='checkpoint whose shared parameters seed every arm identically')
    parser.add_argument('--no_decay_modules', nargs='*',
                        default=['multiview_encoder', 'point_view_fusion', 'anchor_view_fusion'],
                        help='module prefixes outside the shared backbone that train without weight decay')

    args = parser.parse_args()
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp training currently requires CUDA; CPU/BF16 is covered only by synthetic tests')
    if sum(bool(value) for value in (args.resume is not None, args.resume_path, args.warm_start)) > 1:
        parser.error('resume and warm_start are mutually exclusive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Three independent streams: model initialization keeps the global torch RNG,
    # case order draws from loader_generator, and block/point indices come from the
    # dataset's own (seed, epoch, index) generator. Arms therefore see identical data.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed * 1_000_003 + 10_007)

    if args.dist:
        args.local_rank = int(os.environ["LOCAL_RANK"]) # Make it compatible with different versions of DDP
        torch.distributed.init_process_group(backend='nccl' if args.device == 'cuda' else 'gloo')
        if args.device == 'cuda':
            torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank() if args.dist else 0
    args.local_rank = rank

    cfg = load_config(args.cfg_path)
    latent_cfg = cfg.model.get('latent', {}) if hasattr(cfg.model, 'get') else {}
    use_latent = bool(latent_cfg.get('use_latent', False)) if hasattr(latent_cfg, 'get') else False
    probability_cfg = cfg.model.get('probabilistic')
    use_probability = probability_cfg is not None and probability_cfg['mode'] != 'deterministic'
    if use_probability:
        from probabilistic_loss import probability_loss, probability_gradient_norms
    use_latent = use_latent or use_probability
    if use_probability and probability_cfg['mode'] == 'dense_np':
        paired = cfg.dataset.get('paired_views')
        if not paired:
            raise ValueError('dense_np training requires dataset.paired_views with real nested views')
        if args.num_views not in paired['context_counts']:
            raise ValueError('Validation view count must be covered by training context_counts')
    if probability_cfg and probability_cfg['normalization'] == 'frozen_bn' and not (
            args.warm_start or resume_path_pending(args) or probability_cfg['objective'] == 'frozen_reference'):
        raise ValueError('Frozen BN needs an explicit trained warm-start/resume/reference; use groupnorm from scratch')
    loss_cfg = dict(latent_cfg)
    loss_cfg.update(cfg.get('loss', {'reconstruction_loss': 'mse'}))

    if args.local_rank == 0:
        print(args)
        print(cfg)

        # save config
        save_dir = f'./logs/{args.name}'
        os.makedirs(save_dir, exist_ok=True)
        if not (args.resume is not None or args.resume_path) and any(
                name.endswith('.pth') for name in os.listdir(save_dir)):
            raise FileExistsError('Choose a fresh --name; existing checkpoints will not be overwritten')
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            shutil.copyfile(
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
        if os.path.abspath(args.cfg_path) != os.path.abspath(os.path.join(save_dir, 'config.yaml')):
            shutil.copyfile(args.cfg_path, os.path.join(save_dir, 'config.yaml'))

        if args.wandb:
            if not _HAS_WANDB:
                print('[wandb] package not installed; logging disabled. '
                      'Install with: pip install wandb')
            else:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.name,
                    config={
                        **_cfg_to_dict(cfg),
                        'args': vars(args),
                    },
                )

    # -- initialize training dataset/loader
    train_dst = CBCT_dataset_gs(
        dst_name=args.dst_name,
        cfg=cfg.dataset,
        split='train', 
        num_views=args.num_views, 
        npoint=args.num_points,
        out_res_scale=args.out_res_scale,
        random_views=args.random_views,
        load_gt_volume=use_latent and (not use_probability or probability_cfg['mode'] == 'gt_cvae'),
        sample_seed=args.seed,
    )
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='test',
            num_views=args.num_views,
            out_res_scale=args.eval_scale,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )
    if args.val_case_count is not None:
        eval_loader.dataset.name_list = eval_loader.dataset.name_list[:args.val_case_count]

    # -- initialize model
    model = build_model(cfg.model)
    shared_initialization = None
    if args.common_init:
        if args.warm_start or resume_path_pending(args):
            parser.error('--common_init applies to fresh runs only')
        shared_initialization = load_common_init(model, args.common_init)
        if rank == 0:
            print(json.dumps({key: value for key, value in shared_initialization.items()
                              if key != 'shared_parameters'}, indent=2))
            with open(os.path.join(save_dir, 'common_init.json'), 'w') as handle:
                json.dump(shared_initialization, handle, indent=2)
    if args.warm_start:
        loaded = warm_start(model, torch.load(args.warm_start, map_location='cpu', weights_only=False),
                            args.allow_missing, args.reset_head)
        if rank == 0:
            print(json.dumps(loaded, indent=2))
            with open(os.path.join(save_dir, 'warm_start.json'), 'w') as handle:
                json.dump(loaded, handle, indent=2)
    
    model = model.to(args.device)
    reference_provenance = None
    if use_probability and probability_cfg['objective'] == 'frozen_reference':
        from probabilistic_reference import attach_reference
        if args.warm_start or args.common_init:
            raise ValueError('Frozen-reference initialization comes from the validated reference checkpoint')
        reference_provenance = attach_reference(model, probability_cfg['reference_checkpoint'],
                                                probability_cfg['reference_evidence'])
    if args.dist:
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=use_latent or cfg.model.get('anatomy', {}).get('enabled', False),
            device_ids=[int(os.environ['LOCAL_RANK'])] if args.device == 'cuda' else None
        )
    
    # -- initialize optimizer, lr scheduler, and loss function
    # AdamW with one learning rate for every parameter. Under SGD the fusion branches'
    # step shrank with their near-zero gate, global clipping and LayerNorm chain (about
    # 1000x below the backbone); Adam's per-parameter normalization removes that scale
    # dependence. Fusion branches still skip weight decay so the gate/weight product is
    # not decayed to zero before the branch is recruited. Prefixes cover both the new
    # transformer path and the older B1 fusion.
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = undecayed if any(name.startswith(prefix + '.') or f'.{prefix}.' in name
                                  for prefix in args.no_decay_modules) else decayed
        target.append((name, parameter))
    parameter_groups = [{'params': [parameter for _, parameter in decayed], 'weight_decay': args.weight_decay},
                        {'params': [parameter for _, parameter in undecayed], 'weight_decay': 0.}]
    weight_decay_grouping = {'decayed': len(decayed), 'undecayed': len(undecayed),
                             'undecayed_parameters': [name for name, _ in undecayed],
                             'no_decay_modules': list(args.no_decay_modules)}
    if rank == 0:
        print(f'[optimizer] weight decay on {len(decayed)} tensors, off on {len(undecayed)}')
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(args.lr_decay, 1 / args.epoch)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch, global_step = 0, 0
    raw_model = model.module if args.dist else model
    resume_path = args.resume_path or (
        f'./logs/{args.name}/ep_{args.resume}.pth' if args.resume is not None else None)
    resume_checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
    transfer_cfg = cfg.get('transfer', {})
    teacher = None
    initialization_hash = None
    if transfer_cfg.get('enabled', False):
        if use_latent:
            raise ValueError('Deterministic transfer must precede probabilistic alignment')
        with open(transfer_cfg['evidence']) as handle:
            evidence = json.load(handle)
        validate_teacher(evidence)
        if set(evidence['case_ids']) != set(eval_loader.dataset.name_list):
            raise ValueError('Teacher and student must use the same fixed validation cases')
        if file_hash(transfer_cfg['checkpoint']) != evidence['checkpoint_sha256']:
            raise ValueError('Teacher evidence does not match the supplied checkpoint')
        initialization_hash = file_hash(args.warm_start) if args.warm_start else (resume_checkpoint or {}).get('transfer_initialization_hash')
        validate_transfer_provenance(evidence, train_dst.name_list, evidence['checkpoint_sha256'], initialization_hash)
        if evidence['sparse_view_count'] != args.num_views or evidence['dense_view_count'] != cfg.dataset.get('teacher_views'):
            raise ValueError('Teacher evidence must match the actual paired view counts')
        teacher = build_model(cfg.model).to(args.device)
        teacher.load_state_dict(weights_from_checkpoint(torch.load(transfer_cfg['checkpoint'], map_location='cpu', weights_only=False)), strict=True)
        teacher.eval().requires_grad_(False)
    if resume_path:
        checkpoint = resume_checkpoint
        if 'optimizer' not in checkpoint:
            raise ValueError('Weights-only checkpoint cannot resume exactly; use --warm_start')
        if checkpoint.get('config') != _cfg_to_dict(cfg):
            raise ValueError('Exact resume requires the saved model/loss/data configuration')
        if len(checkpoint.get('rng_by_rank', [])) != (torch.distributed.get_world_size() if args.dist else 1):
            raise ValueError('Exact resume requires the saved world size')
        start_epoch, global_step = resume_training(raw_model, optimizer, lr_scheduler, checkpoint, rank)
        if args.amp:
            if 'scaler' not in checkpoint:
                raise ValueError('AMP resume requires the saved GradScaler state')
            scaler.load_state_dict(checkpoint['scaler'])
    started = time.monotonic()
    stop = False
    initial_step = global_step
    view_records = []
    validation_schedule = sorted({step for step in args.val_steps if step > 0})
    validation_records = []
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    # -- training starts
    for epoch in range(start_epoch, args.epoch + 1):
        if args.dist:
            train_loader.sampler.set_epoch(epoch)
        train_dst.set_epoch(epoch)

        loss_list = []
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            stop = ((args.max_steps is not None and global_step >= args.max_steps) or
                    (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds))
            if args.dist:
                stop_tensor = torch.tensor(int(stop), device=args.device)
                torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
                stop = bool(stop_tensor.item())
            if stop:
                break
            item = convert_cuda(item, args.device)
            if rank == 0 and (args.manifest_steps < 0 or global_step < args.manifest_steps):
                view_records.append({'step': global_step, 'epoch': epoch, 'cases': item['name'],
                                     'context_view_ids': item['view_ids'].cpu().tolist(),
                                     'context_mask': item['view_mask'].cpu().tolist() if 'view_mask' in item else None,
                                     'dense_view_ids': item['dense_view_ids'].cpu().tolist() if 'dense_view_ids' in item else None,
                                     'block_idx': item['block_idx'].cpu().tolist() if 'block_idx' in item else None,
                                     'points_digest': points_digest(item['points']),
                                     'points_gt_digest': points_digest(item['points_gt']),
                                     'teacher_view_ids': item['teacher_view_ids'].cpu().tolist() if 'teacher_view_ids' in item else None,
                                     'heldout_view_ids': None})

            raw_model = model.module if isinstance(
                model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
            ) else model

            with torch.autocast(device_type=args.device.split(':')[0], enabled=args.amp):
                if use_probability:
                    pred = model(item, optimizer_step=global_step)
                    loss, metrics = probability_loss(pred, item['points_gt'], probability_cfg, global_step)
                elif use_latent:
                    pred = model(item, gt_volume=item.get('gt_volume'), sample_from='posterior')
                    terms = [elbo_loss(sample, item['points_gt'], info, loss_cfg,
                                       step=global_step, use_local=bool(latent_cfg.get('use_local', False)))
                             for sample, info in zip(pred['sample_predictions'], pred['kl_infos'])]
                    loss = torch.stack([term[0] for term in terms]).mean()
                    metrics = {key: torch.stack([term[1][key] for term in terms]).mean(0)
                               if torch.is_tensor(value) else value for key, value in terms[0][1].items()}
                else:
                    pred = model(item, return_features=teacher is not None, optimizer_step=global_step)
                    loss = volume_nll(pred['points_pred'].float(), item['points_gt'],
                                      loss_cfg.get('reconstruction_loss', 'mse'), loss_cfg.get('volume_sigma', .1))
                    metrics = {'loss': loss.detach(), 'L_volume': loss.detach()}
                    if 'patch_pred' in pred:
                        # auxiliary 3D-SSIM term on the regular K^3 query patch
                        l_ssim = ssim3d_loss(pred['patch_pred'], item['patch_gt'],
                                             window=loss_cfg.get('ssim_window', 7))
                        loss = loss + float(loss_cfg.get('ssim_weight', 0.)) * l_ssim
                        metrics['L_ssim'] = l_ssim.detach()
                        if float(loss_cfg.get('tv_weight', 0.)) > 0:
                            l_tv = tv3d_loss(pred['patch_pred'])
                            loss = loss + float(loss_cfg.get('tv_weight', 0.)) * l_tv
                            metrics['L_tv'] = l_tv.detach()
                    if probability_cfg is not None:
                        # Probability controls retain their diagnostics; plain follows the old path.
                        with torch.no_grad():
                            metrics['mse_raw'] = (pred['points_pred'].float() - item['points_gt']).square().mean()
                loss, metrics = add_structure_loss(pred, loss, metrics, cfg.model)
                if teacher is not None:
                    dense_data = teacher_inputs(item)
                    with torch.no_grad():
                        target = teacher.encode_projs(dense_data)['anchor_feats']
                    sparse_projection = GeometryAdapter(item).project(item['points_gs'])
                    dense_projection = GeometryAdapter(dense_data).project(item['points_gs'])
                    reliability = ((sparse_projection['valid'] & (sparse_projection['uv'].abs() <= 1).all(-1)).any(1)
                                   & (dense_projection['valid'] & (dense_projection['uv'].abs() <= 1).all(-1)).any(1)).float()
                    if 'structure_confidence' in item:
                        reliability = reliability * item['structure_confidence']
                    transfer_loss = feature_distillation(pred['anchor_feats'], target, reliability)
                    loss = loss + transfer_cfg.get('weight', .01) * transfer_loss
                    metrics.update(L_transfer=transfer_loss.detach(), teacher_feature_variance=target.float().var(-1).mean().detach(),
                                   transfer_reliable_fraction=(reliability > 0).float().mean().detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {global_step}')
            metrics['loss'] = loss.detach()
            loss_list.append(float(loss.detach()))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if rank == 0 and global_step % max(1, args.diag_every) == 0:
                log = metrics_to_wandb(metrics, prefix='train')
                log.update({f'encoder/{key}': float(value) for key, value in pred.get('diagnostics', {}).items()})
                log.update(transformer_gradient_norms(raw_model))
                if use_probability:
                    log.update(probability_gradient_norms(raw_model, scaler.get_scale()))
                    log['amp/scale'] = scaler.get_scale()
                log.update(step=global_step, elapsed_seconds=time.monotonic() - started)
                if 'structure_sigma_mm' in item:
                    log.update(structure_sigma_mm=item['structure_sigma_mm'].cpu().tolist(),
                               structure_rho_mm=item['structure_rho_mm'].cpu().tolist())
                with open(os.path.join(save_dir, 'step_metrics.jsonl'), 'a') as handle:
                    handle.write(json.dumps(log) + '\n')
                _wandb_log(args, log, step=global_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            did_update = scaler.get_scale() >= old_scale
            global_step += int(did_update)

            if global_step in validation_schedule and not stop:
                if args.dist:
                    torch.distributed.barrier()
                step_metrics, step_cases = None, None
                if rank == 0:
                    step_metrics, step_cases = eval_one_epoch(raw_model, eval_loader, args.eval_npoint, ignore_msg=True)
                    model.train()
                    record = {'global_step': global_step, 'epoch': epoch, 'metrics': step_metrics,
                              'seconds': time.monotonic() - started}
                    validation_records.append(record)
                    print(f'[val] step {global_step}: ' + json.dumps(step_metrics))
                    with open(os.path.join(save_dir, 'validation_by_step.jsonl'), 'a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    with open(os.path.join(save_dir, f'validation_step_{global_step}.json'), 'w') as handle:
                        json.dump({'global_step': global_step, 'epoch': epoch, 'eval_scale': args.eval_scale,
                                   'eval_npoint': args.eval_npoint, 'metrics': step_metrics, 'cases': step_cases},
                                  handle, indent=2)
                    _wandb_log(args, {f'eval/{key}': value for metric in step_metrics.values()
                                      for key, value in metric.items() if value is not None}, step=global_step)
                states = [None] * (torch.distributed.get_world_size() if args.dist else 1)
                if args.dist:
                    torch.distributed.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    path = os.path.join(save_dir, f'step_{global_step}.pth')
                    if os.path.exists(path):
                        raise FileExistsError(path)
                    torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                                'global_step': global_step, 'rng_by_rank': states,
                                'scaler': scaler.state_dict(),
                                'transfer_initialization_hash': initialization_hash,
                                'sampler_epoch': epoch,
                                'resume_kind': 'epoch_boundary_exact_midepoch_approximate',
                                'config': _cfg_to_dict(cfg), 'args': vars(args)}, path)
                if args.dist:
                    torch.distributed.barrier()

            # latent-only: denser step logs + collapse diagnostics
            if (
                use_latent
                and did_update
                and rank == 0
                and global_step % max(1, args.diag_every) == 0
            ):
                log = metrics_to_wandb(metrics, prefix='train')
                try:
                    diag = compute_collapse_diagnostics(
                        raw_model, item, num_samples=4, eval_npoint=args.num_points
                    )
                    for dk, dv in diag.items():
                        log[f'diag/{dk}'] = dv
                except Exception as e:
                    log['diag/error'] = str(e)
                _wandb_log(args, log, step=global_step)
                with open(os.path.join(save_dir, 'probability_diagnostics.jsonl'), 'a') as handle:
                    handle.write(json.dumps(dict(step=global_step, **log)) + '\n')
        
        if args.local_rank == 0:
            epoch_loss = float(np.mean(loss_list)) if loss_list else float('nan')

            if epoch % 10 == 0:
                print('epoch: {}, loss: {:.4}'.format(epoch, epoch_loss))

            # all experiment groups: epoch train loss
            _wandb_log(args, {
                'train/loss': epoch_loss,
                'train/epoch': epoch,
                'train/lr': float(optimizer.param_groups[0]['lr']),
            }, step=global_step)
            
        if args.dist:
            torch.distributed.barrier()
        evaluation_error = None
        try:
            if rank == 0 and not stop and not validation_schedule and (
                    epoch % 50 == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0)):
                eval_metrics, _ = eval_one_epoch(
                    raw_model,
                    eval_loader, 
                    args.eval_npoint,
                    ignore_msg=True,
                )
                msg = f' --- epoch {epoch}'
                wb_eval = {'eval/epoch': epoch}
                for dst_name in eval_metrics.keys():
                    msg += f', {dst_name}'
                    met = eval_metrics[dst_name]
                    for key, val in met.items():
                        msg += f', {key}: {val}'
                        # e.g. eval/LUNA16/psnr, eval/LUNA16/ssim
                        wb_eval[f'eval/{dst_name}/{key}'] = float(val) if val is not None else None
                        # also flat aliases when a single dataset is used
                        wb_eval[f'eval/{key}'] = float(val) if val is not None else None
                print(msg)
                _wandb_log(args, wb_eval, step=global_step)
        except Exception as error:
            evaluation_error = repr(error)
        if args.dist:
            errors = [evaluation_error]
            torch.distributed.broadcast_object_list(errors, src=0)
            evaluation_error = errors[0]
        if evaluation_error is not None:
            raise RuntimeError(f'Coordinated validation failed: {evaluation_error}')
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        save_epoch = (not validation_schedule) and (
            epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0))
        if save_epoch and not stop:
            states = [None] * (torch.distributed.get_world_size() if args.dist else 1)
            if args.dist:
                torch.distributed.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            if rank == 0:
                path = os.path.join(save_dir, f'ep_{epoch}.pth')
                if os.path.exists(path):
                    raise FileExistsError(path)
                torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                            'global_step': global_step, 'rng_by_rank': states,
                            'scaler': scaler.state_dict(),
                            'transfer_initialization_hash': initialization_hash,
                            'config': _cfg_to_dict(cfg), 'args': vars(args)}, path)
        if stop:
            if rank == 0:
                torch.save({'model': raw_model.state_dict(), 'global_step': global_step,
                            'config': _cfg_to_dict(cfg), 'resume_kind': 'warm_start_only_partial_epoch'},
                           os.path.join(save_dir, f'partial_step_{global_step}.pth'))
            break

    if _wandb_enabled(args):
        wandb.finish()
    if rank == 0:
        with open(os.path.join(save_dir, 'run_summary.json'), 'w') as handle:
            json.dump({'resolved_config': _cfg_to_dict(cfg), 'args': vars(args),
                       'shared_initialization': shared_initialization,
                       'reference_provenance': reference_provenance,
                       'weight_decay_grouping': weight_decay_grouping,
                       'validation_by_step': validation_records,
                       'validation_schedule': validation_schedule,
                       'parent_checkpoint': resume_path or args.warm_start,
                       'train_cases': train_dst.name_list, 'validation_cases': eval_loader.dataset.name_list,
                       'views': view_records, 'optimizer_updates': global_step - initial_step,
                       'global_step': global_step, 'seconds': time.monotonic() - started,
                       'parameters': sum(parameter.numel() for parameter in raw_model.parameters()),
                       'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith('cuda') else None,
                       'status': 'STOPPED_LIMIT' if stop else 'COMPLETED'}, handle, indent=2)
