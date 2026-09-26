"""Bounded, sequential screening of the dense_np probability branch on the T1-nomax backbone.

Mirrors scripts/run_geometry_validation.py: common init seed0, AdamW lr2e-4, 6 views, batch1,
1024 queries, FP32, 2000 updates, 6 eval cases at scale 1.0. Never overwrites a run directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/probability_validation'
CONFIGS = ROOT / 'configs/probabilistic_validation'
ARMS = ['PD0', 'PP0', 'PM0']
COMPARISONS = [('PD0', 'PP0'), ('PD0', 'PM0'), ('PM0', 'PP0')]
STEPS = {'smoke': 10, 'train': 2000}
sys.path.insert(0, str(ROOT / 'code'))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def train(phase):
    REPORT.mkdir(parents=True, exist_ok=True)
    state_path = REPORT / 'execution_status.json'
    state = read(state_path) if state_path.exists() else {'runs': {}}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    steps, val_cases = (STEPS[phase], 1) if phase == 'smoke' else (STEPS[phase], 6)
    schedule = [steps] if phase == 'smoke' else [1000, 2000]
    if phase == 'train' and not all(state['runs'].get(f'probv1-smoke-{arm}-seed0', {}).get('status') == 'PASS'
                                    for arm in ARMS):
        raise RuntimeError('All real-data smoke runs must pass before the main comparison.')
    for arm in ARMS:
        name = f'probv1-{phase}-{arm}-seed0'
        if state['runs'].get(name, {}).get('status') == 'PASS':
            continue
        destination = ROOT / 'logs' / name
        if destination.exists():
            raise RuntimeError(f'Existing run requires inspection, not automatic restart: {destination}')
        command = [sys.executable, 'code/train.py', '--cfg_path', str(CONFIGS.relative_to(ROOT) / f'{arm}.yaml'),
                   '--name', name, '--dst_name', 'LUNA16', '--num_views', '6', '--batch_size', '1',
                   '--num_points', '1024', '--num_workers', '2', '--epoch', '400', '--lr', '2e-4',
                   '--weight_decay', '1e-2', '--lr_decay', '1e-3', '--max_steps', str(steps),
                   '--val_steps', *map(str, schedule), '--manifest_steps', str(steps),
                   '--val_case_count', str(val_cases), '--eval_scale', '1.0', '--eval_npoint', '65536',
                   '--diag_every', '25' if phase == 'train' else '1', '--common_init', 'logs/p1-common-init/seed0.pth',
                   '--seed', '0', '--device', 'cuda', '--no-wandb']
        entry = {'status': 'RUNNING', 'phase': phase, 'arm': arm, 'reserved_updates': steps,
                 'command': command, 'started_at': time.time()}
        state['runs'][name] = entry
        save(state_path, state)
        print(f'START {name}', flush=True)
        with (REPORT / f'{name}.log').open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, seconds=time.time() - entry['started_at'])
        summary = read(destination / 'run_summary.json') if (destination / 'run_summary.json').exists() else None
        passed = result.returncode == 0 and summary and summary['global_step'] == steps
        entry['status'] = 'PASS' if passed else 'FAILED'
        entry['actual_updates'] = summary['global_step'] if summary else None
        if passed:
            entry['metrics'] = read(destination / f'validation_step_{steps}.json')['metrics']['LUNA16']
            reference = read(ROOT / 'logs' / f'probv1-{phase}-PD0-seed0' / 'run_summary.json')
            # Same C IDs, blocks and query digests at every step; D IDs recorded only for dense arms.
            strip = lambda views: [{k: v for k, v in r.items() if k != 'dense_view_ids'} for r in views]
            entry['paired_inputs_all_steps'] = strip(summary['views']) == strip(reference['views'])
            if not entry['paired_inputs_all_steps']:
                entry['status'] = 'FAILED_PAIRED_INPUTS'
        save(state_path, state)
        print(f"END {name}: {entry['status']} {entry.get('metrics', {})}", flush=True)
        if entry['status'] != 'PASS':
            raise RuntimeError(f'Inspect {REPORT / (name + ".log")}')


def posthoc(cases, phase='train', step=2000):
    """Estimator comparison + latent-usage interventions on the trained PP0/PM0 checkpoints."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from datasets.dst_gs import CBCT_dataset_gs
    from evaluate import eval_one_epoch
    from models.probabilistic_model import build_model
    from utils import load_config, convert_cuda
    from checkpoints import weights_from_checkpoint
    out = {}
    plans = {'PP0': [('prior_sample', 1, False), ('prior_latent_mean', 1, False), ('prior_mc_mean', 4, False),
                     ('prior_mc_mean', 8, False), ('dense_posterior_mean', 1, True)],
             'PM0': [('prior_latent_mean', 1, False), ('dense_posterior_mean', 1, True)]}
    for arm, modes in plans.items():
        run = ROOT / f'logs/probv1-{phase}-{arm}-seed0'
        cfg = load_config(run / 'config.yaml')
        model = build_model(cfg.model)
        model.load_state_dict(weights_from_checkpoint(torch.load(run / f'step_{step}.pth', map_location='cpu')), strict=True)
        model = model.cuda()
        loader = DataLoader(CBCT_dataset_gs(dst_name='LUNA16', cfg=cfg.dataset, split='eval', num_views=6,
                                            out_res_scale=1.0, load_dense=True), batch_size=1, shuffle=False)
        loader.dataset.name_list = loader.dataset.name_list[:cases]
        results = {}
        for mode, samples, oracle in modes:
            started = time.monotonic()
            metrics, per_case = eval_one_epoch(model, loader, 65536, latent_mode=mode, num_samples=samples,
                                               oracle=oracle, latent_seed=0)
            key = f'{mode}_S{samples}'
            results[key] = {'metrics': metrics['LUNA16'], 'seconds': time.monotonic() - started,
                            'cases': {c['name']: c for c in per_case['LUNA16']}}
            print(arm, key, json.dumps(metrics['LUNA16']), flush=True)
        # Latent usage at fixed h_C on the eval volumes: z interventions versus latent mean.
        usage = []
        model.eval()
        with torch.no_grad():
            states, items = [], []
            for item in loader:
                item = convert_cuda(item, 'cuda')
                context = model.encode_context(item)
                prior = model.infer_prior(context)
                posterior, _ = model.infer_posterior(item, context)
                states.append((context, prior, posterior))
                items.append(item)
            for index, (context, prior, posterior) in enumerate(states):
                gt = items[index]['points_gt'].reshape(1, 1, -1)
                decode = lambda z: model.decode_chunked(model.state_with_global(
                    model.sample_state(context, prior, mean=True), z), 65536)
                base = decode(prior[0])
                other = states[(index + 1) % len(states)][1][0]
                interventions = {'zero_z': decode(torch.zeros_like(prior[0])),
                                 'shuffled_case_z': decode(other),
                                 'dense_q_mean': decode(posterior[0])}
                generator = torch.Generator(device='cuda').manual_seed(0)
                eps = torch.randn(prior[0].shape, device='cuda', generator=generator)
                interventions['prior_plus_1sigma_draw'] = decode(prior[0] + eps * (.5 * prior[1]).exp())
                mse = lambda p: float((p - gt).square().mean())
                row = {'case': items[index]['name'][0], 'mse_prior_mean': mse(base),
                       'mu_q_minus_mu_p_l2': float((posterior[0] - prior[0]).norm()),
                       'sigma_p_mean': float((.5 * prior[1]).exp().mean()),
                       'sigma_q_mean': float((.5 * posterior[1]).exp().mean()),
                       'kl_q_p': float(__import__('models.latent_np', fromlist=['gaussian_kl']).gaussian_kl(
                           posterior[0], posterior[1], prior[0], prior[1]).sum())}
                for name, prediction in interventions.items():
                    row[f'{name}_l1_vs_prior_mean'] = float((prediction - base).abs().mean())
                    row[f'{name}_mse'] = mse(prediction)
                usage.append(row)
                del base, interventions
            mu_p = torch.cat([s[1][0] for s in states])
            mu_q = torch.cat([s[2][0] for s in states])
        out[arm] = {'estimators': results, 'latent_usage': usage,
                    'between_case_var_mu_p': float(mu_p.var(0, unbiased=False).mean()),
                    'between_case_var_mu_q': float(mu_q.var(0, unbiased=False).mean())}
        del model
        torch.cuda.empty_cache()
    save(REPORT / f'posthoc_{phase}.json', out)


def report():
    import numpy as np
    rows, cases = {}, {}
    for arm in ARMS + ['R0']:
        path = ROOT / (f'logs/probv1-train-{arm}-seed0' if arm != 'R0' else 'logs/geomv1-train-R0-seed0')
        if not (path / 'run_summary.json').exists():
            continue
        summary = read(path / 'run_summary.json')
        validations = {str(step): read(path / f'validation_step_{step}.json') for step in (1000, 2000)}
        rows[arm] = {'parameters': summary['parameters'], 'seconds': summary['seconds'],
                     'peak_memory_gb': summary['peak_memory_gb'],
                     'metrics': {step: v['metrics']['LUNA16'] for step, v in validations.items()}}
        cases[arm] = {v['name']: v for v in validations['2000']['cases']['LUNA16']}
    comparisons = {}
    for a, b in COMPARISONS + [('R0', 'PD0')]:
        if a not in rows or b not in rows:
            continue
        assert cases[a].keys() == cases[b].keys()
        psnr = np.array([cases[b][n]['psnr'] - cases[a][n]['psnr'] for n in cases[a]])
        mse = np.array([cases[b][n]['unclipped_mse'] / cases[a][n]['unclipped_mse'] - 1 for n in cases[a]])
        comparisons[f'{a}->{b}'] = {'delta_psnr': float(psnr.mean()), 'improved_cases': int((psnr > 0).sum()),
                                    'psnr_per_case': psnr.tolist(), 'mean_relative_mse_change': float(mse.mean()),
                                    'paired_inputs': a != 'R0'}
    save(REPORT / 'results.json', {'arms': rows, 'comparisons': comparisons,
         'scope': 'seed0, 2000 updates, 6 eval cases, scale 1.0; screening only. R0 is BN+MSE on a different C sampler: historical, not paired.'})
    print(json.dumps(comparisons, indent=2))
    for arm, row in rows.items():
        print(arm, {k: round(v, 5) if isinstance(v, float) else v for k, v in row['metrics']['2000'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'train', 'posthoc', 'report'], required=True)
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--source', choices=['smoke', 'train'], default='train')
    parser.add_argument('--step', type=int, default=2000)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'report':
        report()
    elif args.phase == 'posthoc':
        posthoc(args.cases, args.source, args.step)
    else:
        train(args.phase)
