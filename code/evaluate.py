import os
import csv
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from datasets.dst_gs import CBCT_dataset_gs
from models.factory import build_model
from utils import convert_cuda, load_config, sitk_save
from checkpoints import weights_from_checkpoint
from evaluation_protocol import volume_metrics, evaluating, predict_volume, resolved_latent_mode
from losses import compute_collapse_diagnostics



def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False,
                   latent_mode=None, num_samples=None, diagnostics_path=None, oracle=False, latent_seed=None):
    results = {}
    metrics = {}
    diagnostics = []
    latent_mode = resolved_latent_mode(model, latent_mode)
    generator = None
    if hasattr(model, 'probability'):
        from probabilistic_metrics import mixture_metrics
        seed = model.probability['evaluation']['seed'] if latent_seed is None else latent_seed
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    metrics_tmp = {key:[] for key in ['psnr', 'ssim', 'unclipped_mse', 'clipping_fraction',
                                     'boundary_error', 'gradient_error']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    with evaluating(model), torch.no_grad():
        for item in loader:
            item = convert_cuda(item, next(model.parameters()).device)

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            image = item['points_gt'].cpu().numpy()
            image = image[0] # W, H, D

            if getattr(model, 'use_latent', False):
                details = predict_volume(model, item, npoint, latent_mode, num_samples, oracle=oracle,
                                         return_details=True, generator=generator,
                                         return_samples=hasattr(model, 'probability'))
                output = details['points_pred']
            else:
                # Release the plain GPU prediction after copying it to CPU, as in the backup.
                details = {}
                output = predict_volume(model, item, npoint)
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            inference_seconds = time.monotonic() - started
            output = output[0, 0].data.cpu().numpy()
            
            output = output.reshape(image.shape)
            if diagnostics_path is not None:
                diagnostics.append(dict(case_id=name, **compute_collapse_diagnostics(
                    model, item, num_samples=num_samples or 4, eval_npoint=npoint)))
            case_metrics = volume_metrics(image, output, item['spacing'][0].cpu().numpy())
            probability_metrics = {}
            if hasattr(model, 'probability'):
                rec = model.probability['reconstruction']
                if rec['likelihood'] == 'gaussian':
                    probability_metrics = mixture_metrics(details['sample_predictions'], item['points_gt'], rec['observation_sigma'])
                probability_metrics['latent_variance_mean'] = float(details['predictive_var_latent'].mean()) if details['predictive_var_latent'] is not None else None
                probability_metrics['observation_variance'] = details['observation_variance']
                probability_metrics['predictive_variance_mean'] = float(details['predictive_var_total'].mean()) if details['predictive_var_total'] is not None else None
            psnr, ssim = case_metrics['psnr'], case_metrics['ssim']

            if not ignore_msg:
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                **case_metrics,
                'latent_mode': latent_mode or model.cfg.get('sampling', {}).get('eval_mode', 'prior_mean'),
                'USES_GT': latent_mode in ('posterior_mean_oracle', 'gt_posterior_mean', 'gt_posterior_sample'),
                'evaluation_scope': ('oracle_dense' if latent_mode.startswith('dense_posterior') else
                                     'oracle_gt' if 'posterior' in latent_mode else 'sparse_deployment'),
                **probability_metrics,
                'view_ids': item['view_ids'][0].cpu().tolist() if 'view_ids' in item else None,
                'inference_seconds': inference_seconds,
                'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if next(model.parameters()).is_cuda else None,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                spacing = item['spacing'][0].cpu().numpy()
                origin = item['origin'][0].cpu().numpy()
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                direction = item['direction'][0].cpu().numpy() if 'direction' in item else None
                sitk_save(save_path, output, spacing=spacing, origin=origin, direction=direction)
                for key in ('predictive_var_latent', 'predictive_var_total'):
                    if details.get(key) is not None:
                        sitk_save(os.path.join(save_dir, f'{name}_{key}.nii.gz'),
                                  details[key][0, 0].cpu().numpy().reshape(image.shape),
                                  spacing=spacing, origin=origin, direction=direction)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key: float(np.mean([value for value in val if value is not None]))
             if any(value is not None for value in val) else None for key, val in dst_met.items()}
        metrics[dst_name] = m

    if diagnostics_path is not None and diagnostics:
        os.makedirs(os.path.dirname(os.path.abspath(diagnostics_path)), exist_ok=True)
        with open(diagnostics_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='eval')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--latent_mode', choices=['prior_mean', 'prior_latent_mean', 'prior_sample', 'prior_mc_mean',
                                                'posterior_mean_oracle', 'dense_posterior_mean', 'dense_posterior_sample',
                                                'gt_posterior_mean', 'gt_posterior_sample'])
    parser.add_argument('--oracle', action='store_true', help='Explicitly enable privileged dense/GT posterior evaluation')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--diagnostics_path', help='Offline USES_GT latent diagnostic CSV')
    parser.add_argument('--case_limit', type=int)
    parser.add_argument('--view_offset', type=int, default=0)
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--save_results', action='store_true', default=False)

    args = parser.parse_args()
    if args.cfg_path is None:
        args.cfg_path = f'./logs/{args.name}/config.yaml'
    
    print(args)

    cfg = load_config(args.cfg_path)
    probability = cfg.model.get('probabilistic')
    effective_mode = args.latent_mode or (probability['evaluation']['estimator'] if probability and probability['mode'] != 'deterministic'
                                          else cfg.model.get('sampling', {}).get('eval_mode', 'prior_mean'))
    if 'posterior' in effective_mode and not args.oracle:
        parser.error('Posterior metrics require --oracle and are excluded from sparse deployment scores')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- dataloader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split=args.split, 
            num_views=args.num_views,
            out_res_scale=args.out_res_scale,
            view_offset=args.view_offset,
            load_gt_volume=('gt_posterior' in effective_mode or effective_mode == 'posterior_mean_oracle' or
                            (args.diagnostics_path is not None and (not probability or probability['mode'] == 'gt_cvae'))),
            load_dense=bool(probability and probability['mode'] == 'dense_np' and
                            (effective_mode.startswith('dense_posterior') or args.diagnostics_path is not None)),
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )
    if args.case_limit is not None:
        eval_loader.dataset.name_list = eval_loader.dataset.name_list[:args.case_limit]

    # -- model, load ckpt
    ckpt_path = args.checkpoint or f'./logs/{args.name}/ep_{args.epoch}.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    print('load ckpt from', ckpt_path)
    
    model = build_model(cfg.model)
    model.load_state_dict(weights_from_checkpoint(ckpt), strict=True)
    model = model.to(args.device)

    # -- output dir
    tag = '{:.2f}x_{}_S{}_seed{}'.format(args.out_res_scale, effective_mode,
        args.num_samples or (probability['evaluation']['num_samples'] if probability else 1), args.seed)
    save_dir = None
    if args.save_results:
        save_dir = f'./logs/{args.name}/results/ep_{args.epoch}/predictions_{tag}'
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        use_tqdm=True,
        ignore_msg=False,
        latent_mode=args.latent_mode, num_samples=args.num_samples,
        diagnostics_path=args.diagnostics_path,
        oracle=args.oracle, latent_seed=args.seed,
    )
    print(metrics)

    # -- save results [csv]
    pred_dir = f'./logs/{args.name}/results/ep_{args.epoch}'
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, f'results_{tag}.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])
    
    csv_file.close()
    with open(os.path.join(pred_dir, f'cases_{tag}.json'), 'w') as handle:
        json.dump(results, handle, indent=2)
    
    # -- save config [args]
    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)
