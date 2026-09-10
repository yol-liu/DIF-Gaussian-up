import os
import shutil
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dst_gs import CBCT_dataset_gs
from models.probabilistic_model import build_model
from utils import convert_cuda, load_config
from evaluate import eval_one_epoch
from losses import elbo_loss, compute_collapse_diagnostics, metrics_to_wandb


try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def _cfg_to_dict(obj):
    '''Recursively convert EasyDict / nested containers for wandb.config.'''
    if isinstance(obj, dict):
        return {k: _cfg_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_dict(v) for v in obj]
    return obj


def _wandb_enabled(args):
    return bool(getattr(args, 'wandb', False)) and _HAS_WANDB and args.local_rank == 0


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
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)

    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--random_views', action='store_true', default=False)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help='enable Weights & Biases logging (default: on)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='disable Weights & Biases logging')
    parser.set_defaults(wandb=True)
    parser.add_argument('--wandb_project', type=str, default='dif-gaussian')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--diag_every', type=int, default=200,
                        help='log latent collapse diagnostics every N optimizer steps')

    args = parser.parse_args()

    if args.dist:
        args.local_rank = int(os.environ["LOCAL_RANK"]) # Make it compatible with different versions of DDP
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.cfg_path)
    latent_cfg = cfg.model.get('latent', {}) if hasattr(cfg.model, 'get') else {}
    use_latent = bool(latent_cfg.get('use_latent', False)) if hasattr(latent_cfg, 'get') else False

    if args.local_rank == 0:
        print(args)
        print(cfg)

        # save config
        save_dir = f'./logs/{args.name}'
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            shutil.copyfile(
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
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
        load_gt_volume=use_latent,  # needed for posterior during C1 training
    )
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=0, # args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='eval',
            num_views=args.num_views,
            out_res_scale=0.5, # low-res for faster evaluation,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- initialize model
    model = build_model(cfg.model)
    if args.resume:
        print(f'resume model from epoch {args.resume}')
        ckpt = torch.load(
            os.path.join(f'./logs/{args.name}/ep_{args.resume}.pth'),
            map_location=torch.device('cpu')
        )
        # strict=False allows warm-start of deterministic ckpt into use_latent=true
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if args.local_rank == 0 and (missing or unexpected):
            print(f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
    
    model = model.cuda()
    if args.dist:
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=use_latent,  # gt_encoder unused at eval
            device_ids=[args.local_rank]
        )
    
    # -- initialize optimizer, lr scheduler, and loss function
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(args.lr_decay, 1 / args.epoch)
    )
    mse_loss = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = args.resume + 1
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=args.resume)

    global_step = 0

    # -- training starts
    for epoch in range(start_epoch, args.epoch + 1):
        if args.dist:
            train_loader.sampler.set_epoch(epoch)

        loss_list = []
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            item = convert_cuda(item)

            raw_model = model.module if isinstance(
                model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
            ) else model

            if use_latent:
                gt_volume = item.get('gt_volume', None)
                pred = raw_model(
                    item,
                    gt_volume=gt_volume,
                    sample_from='posterior',
                )
                loss, metrics = elbo_loss(
                    pred['points_pred'],
                    item['points_gt'],
                    pred['kl_info'],
                    latent_cfg,
                    step=global_step,
                    use_local=bool(latent_cfg.get('use_local', False)),
                )
            else:
                pred = model(item)
                loss = mse_loss(pred['points_pred'], item['points_gt'])
                metrics = {'loss': loss.detach(), 'L_volume': loss.detach()}

            loss_list.append(float(loss.item()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # latent-only: denser step logs + collapse diagnostics
            if (
                use_latent
                and _wandb_enabled(args)
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
            
            if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save(
                    model_state,
                    os.path.join(save_dir, f'ep_{epoch}.pth')
                )

            if epoch % 50 == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):
                eval_metrics, _ = eval_one_epoch(
                    model, 
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
                        msg += ', {}: {:.4}'.format(key, val)
                        # e.g. eval/LUNA16/psnr, eval/LUNA16/ssim
                        wb_eval[f'eval/{dst_name}/{key}'] = float(val)
                        # also flat aliases when a single dataset is used
                        wb_eval[f'eval/{key}'] = float(val)
                print(msg)
                _wandb_log(args, wb_eval, step=global_step)
        
        if lr_scheduler is not None:
            lr_scheduler.step()

    if _wandb_enabled(args):
        wandb.finish()
