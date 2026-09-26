"""Legacy runner for docs/archive/EXPERIMENT_PLAN_20260925_before_recovery.md.

This runner loads common-init weights. It does not implement the fresh-only,
multi-repeat recovery protocol now described in docs/EXPERIMENT_PLAN.md.

Phases (run from the repository root):
  check     build every arm, load its common init (CPU, no training)
  init      create missing common-init files for the requested seeds
  smoke     10 updates per arm, seed 0, 1 eval case (must pass for the current config before train)
  train     full screening runs for --stage/--arms/--seeds (never overwrites a run directory)
  posthoc   estimator/oracle/latent-usage evaluation of trained probability runs
  evidence  latent-usage evidence + oracle gate for the frozen-reference distillation arm
  report    paired comparisons with case x seed bootstrap intervals -> reports/experiment_plan/
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/experiment_plan'
REPORT = ROOT / 'reports/experiment_plan'
PYTHON = sys.executable
sys.path.insert(0, str(ROOT / 'code'))

BASELINE = 'baseline-unet-transformer-meaninit'
CONTROL = 'prob-deterministic-control'
DENSE = 'prob-dense-np'
ARMS = {  # name -> (stage, comparator used in the report)
    BASELINE: (1, None),
    'baseline-unet-transformer-meaninit-replica': (1, BASELINE),
    'encoder-vit-per-view': (1, BASELINE),
    'encoder-hierarchical-window': (1, BASELINE),
    'anchor-fusion-maxpool': (1, BASELINE),
    'anchor-fusion-late-attention': (1, BASELINE),
    'anchor-transformer-max-residual': (1, BASELINE),
    'anchor-transformer-no-residual-maxinit': (1, BASELINE),
    'point-fusion-geometry-attention': (1, BASELINE),
    'view-update-alternating-2d3d': (1, BASELINE),
    CONTROL: (1, BASELINE),
    'prob-gt-cvae': (1, CONTROL),
    DENSE: (1, CONTROL),
    'prob-dense-mean-code': (1, CONTROL),
    'detail-aligned-pixel-coordinates': (2, BASELINE),
    'detail-anatomy-supervision': (2, BASELINE),
    'detail-anatomy-guided-sampling': (2, 'detail-anatomy-supervision'),
    'detail-shared-3d-sampling': (2, 'detail-aligned-pixel-coordinates'),
    'detail-single-scale-features': (2, BASELINE),
    'detail-no-spatial-neighbours': (2, BASELINE),
    'detail-gaussian-nll-loss': (2, BASELINE),
    'detail-gn-standard-sampler': (2, CONTROL),
    'detail-gn-pointchannel-anchor-norm': (2, CONTROL),
    'detail-dense-np-sigma-init-0.05': (2, DENSE),
    'detail-dense-np-beta-x10': (2, DENSE),
    'detail-dense-np-train-samples-4': (2, DENSE),
    'detail-dense-np-local-latent': (2, DENSE),
    'detail-dense-np-multi-context-4-6-8': (2, DENSE),
    'detail-dense-np-gaussian-nll': (2, DENSE),
    'detail-dense-np-frozen-reference-distill': (2, DENSE),
}
# Extra comparisons beyond each arm's own comparator.
EXTRA_COMPARISONS = [('prob-dense-mean-code', DENSE), (BASELINE, 'detail-anatomy-guided-sampling'),
                     (BASELINE, 'detail-shared-3d-sampling'), (BASELINE, 'detail-gn-standard-sampler')]
DISTILL = 'detail-dense-np-frozen-reference-distill'
# Seed 0 with logs/p1-common-init/seed0.pth is the setting of every good earlier result.
DEFAULT_SEEDS = [0]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def config_path(arm):
    return CONFIGS / f'stage{ARMS[arm][0]}' / f'{arm}.yaml'


def config_hash(arm):
    return hashlib.sha256(config_path(arm).read_bytes()).hexdigest()[:10]


def stage2_base():
    path = CONFIGS / 'stage2' / 'BASE.txt'
    return path.read_text().strip() if path.exists() else BASELINE


def comparator(arm):
    stage, target = ARMS[arm]
    # Geometry/loss details compare with whatever structure the Stage-2 configs were built on.
    return stage2_base() if stage == 2 and target == BASELINE else target


def run_name(arm, seed, tag=''):
    return f'plan-{arm}-seed{seed}' + (f'-{tag}' if tag else '')


def smoke_tag(arm):
    return f'smoke-{config_hash(arm)}'


def is_probability(arm):
    import yaml
    model = yaml.safe_load(config_path(arm).read_text())['model']
    return model.get('probabilistic', {}).get('mode', 'deterministic') != 'deterministic'


# ---------------------------------------------------------------- common initialization
def common_init(seed):
    """Seeds 0/42 reuse the files every earlier screening used; others follow the same recipe."""
    legacy = ROOT / f'logs/p1-common-init/seed{seed}.pth'
    return legacy if legacy.exists() else ROOT / f'logs/plan-common-init/seed{seed}.pth'


def create_common_init(seed):
    import random
    import numpy as np
    import torch
    from models.probabilistic_model import build_model
    from utils import load_config
    path = common_init(seed)
    if path.exists():
        return path
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Same recipe as scripts/run_p1.py: the largest multiview arm (T4) defines shared weights.
    reference = build_model(load_config(ROOT / 'configs/multiview/T4.yaml').model).state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': reference, 'seed': seed, 'source_arm': 'T4'}, path)
    return path


def arm_init(arm, seed):
    """Drop tensors whose shape differs for this arm (e.g. single-scale encodings); record which."""
    import torch
    from models.probabilistic_model import build_model
    from checkpoints import weights_from_checkpoint
    from utils import load_config
    source = common_init(seed)
    weights = weights_from_checkpoint(torch.load(source, map_location='cpu'))
    target = build_model(load_config(config_path(arm)).model).state_dict()
    dropped = sorted(name for name, value in weights.items()
                     if name in target and tuple(value.shape) != tuple(target[name].shape))
    if not dropped:
        return source, []
    derived = ROOT / f'logs/plan-common-init/seed{seed}-{arm}.pth'
    if not derived.exists():
        derived.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': {k: v for k, v in weights.items() if k not in dropped}, 'seed': seed,
                    'derived_from': str(source.relative_to(ROOT)), 'dropped': dropped}, derived)
    return derived, dropped


# ---------------------------------------------------------------- training
def resolved_config(arm, seed, tag):
    """Frozen-reference distillation needs the same-seed trained dense NP run as its reference."""
    if arm != DISTILL:
        return config_path(arm)
    import yaml
    reference = ROOT / 'logs' / run_name(DENSE, seed, tag)
    evidence = REPORT / 'evidence' / f'{run_name(DENSE, seed, tag)}.json'
    if not evidence.exists() or read(evidence).get('gate') != 'PASS':
        raise RuntimeError(f'BLOCKED: {evidence} missing or gate not PASS; run --phase evidence first')
    config = yaml.safe_load(config_path(arm).read_text())
    config['model']['probabilistic'].update(
        reference_checkpoint=str(reference / read(evidence)['checkpoint_name']), reference_evidence=str(evidence))
    path = REPORT / 'resolved' / f'{run_name(arm, seed, tag)}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def command(arm, seed, steps, val_cases, tag, smoke):
    init, dropped = arm_init(arm, seed)
    cfg = resolved_config(arm, seed, tag)
    schedule = [steps] if smoke else sorted({steps // 2, steps})
    result = [PYTHON, 'code/train.py', '--cfg_path', str(Path(cfg).relative_to(ROOT)),
              '--name', run_name(arm, seed, tag), '--dst_name', 'LUNA16', '--num_views', '6',
              '--batch_size', '1', '--num_points', '1024', '--num_workers', '2', '--epoch', '400',
              '--lr', '2e-4', '--weight_decay', '1e-2', '--lr_decay', '1e-3', '--max_steps', str(steps),
              '--val_steps', *map(str, schedule), '--manifest_steps', str(steps),
              '--val_case_count', str(val_cases), '--eval_scale', '0.5' if smoke else '1.0',
              '--eval_npoint', '65536', '--diag_every', '1' if smoke else '25',
              '--seed', str(seed), '--device', 'cuda', '--no-wandb']
    if arm != DISTILL:  # distillation initializes from its reference checkpoint instead
        result += ['--common_init', str(init.relative_to(ROOT))]
    return result, dropped


def train(arms, seeds, steps, val_cases, tag, smoke=False, keep='final', min_free_gb=5.):
    """Sequential runs. Smoke runs are tagged with the config hash, so an edited config needs a new smoke."""
    state_path = REPORT / 'execution_status.json'
    state = read(state_path) if state_path.exists() else {'runs': {}}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    for seed in seeds:
        create_common_init(seed)
        for arm in arms:
            run_tag = smoke_tag(arm) if smoke else tag
            name = run_name(arm, seed, run_tag)
            if state['runs'].get(name, {}).get('status') == 'PASS':
                if state['runs'][name].get('config_sha') != config_hash(arm):
                    raise RuntimeError(f'{name} passed with a different config; use a new --tag')
                print(f'skip {name} (PASS)', flush=True)
                continue
            if not smoke and arm != DISTILL:
                if state['runs'].get(run_name(arm, 0, smoke_tag(arm)), {}).get('status') != 'PASS':
                    raise RuntimeError(f'Run --phase smoke for {arm} (current config) first')
            free_gb = shutil.disk_usage(ROOT / 'logs').free / 1e9
            if free_gb < min_free_gb:
                raise RuntimeError(f'Only {free_gb:.1f} GB free under logs/; a run needs ~1 GB and checkpoints '
                                   'corrupt when the disk fills. Free space first.')
            destination = ROOT / 'logs' / name
            if destination.exists():
                raise RuntimeError(f'{destination} exists without PASS: inspect it, then move it away manually')
            cmd, dropped = command(arm, seed, steps, val_cases, run_tag, smoke)
            entry = {'status': 'RUNNING', 'arm': arm, 'seed': seed, 'steps': steps, 'tag': run_tag,
                     'config_sha': config_hash(arm), 'common_init_dropped': dropped, 'command': cmd,
                     'started_at': time.time()}
            state['runs'][name] = entry
            save(state_path, state)
            print(f'START {name}', flush=True)
            log = REPORT / 'logs' / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as handle:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            summary_path = destination / 'run_summary.json'
            summary = read(summary_path) if summary_path.exists() else None
            entry.update(returncode=code, hours=(time.time() - entry['started_at']) / 3600,
                         status='PASS' if code == 0 and summary and summary['global_step'] == steps else 'FAILED')
            if entry['status'] == 'PASS':
                entry['metrics'] = read(destination / f'validation_step_{steps}.json')['metrics']['LUNA16']
                entry['peak_memory_gb'] = summary.get('peak_memory_gb')
                # Screening needs only the final weights; midpoint metrics stay in validation_step_*.json.
                # Smoke runs keep no weights at all.
                removable = [path for path in destination.glob('*.pth')
                             if smoke or (keep == 'final' and path.name != f'step_{steps}.pth')]
                for path in removable:
                    path.unlink()
                entry['pruned_checkpoints'] = sorted(path.name for path in removable)
            save(state_path, state)
            print(f"END {name}: {entry['status']} {entry.get('metrics', {}).get('psnr')}", flush=True)
            if entry['status'] != 'PASS':
                raise RuntimeError(f'Inspect {log}')


# ---------------------------------------------------------------- probability post-hoc
def load_trained(run, steps):
    import torch
    from checkpoints import weights_from_checkpoint
    from models.probabilistic_model import build_model
    from utils import load_config
    cfg = load_config(ROOT / 'logs' / run / 'config.yaml')
    model = build_model(cfg.model)
    model.load_state_dict(weights_from_checkpoint(torch.load(ROOT / 'logs' / run / f'step_{steps}.pth',
                                                             map_location='cpu')), strict=True)
    return cfg, model.cuda().eval()


def posthoc(arms, seeds, steps, val_cases, tag):
    import torch
    from torch.utils.data import DataLoader
    from datasets.dst_gs import CBCT_dataset_gs
    from evaluate import eval_one_epoch
    from models.latent_np import gaussian_kl
    from utils import convert_cuda
    for arm in arms:
        if not is_probability(arm):
            continue
        for seed in seeds:
            run = run_name(arm, seed, tag)
            out_path = REPORT / 'posthoc' / f'{run}.json'
            if out_path.exists() or not (ROOT / 'logs' / run / f'step_{steps}.pth').exists():
                continue
            cfg, model = load_trained(run, steps)
            mean_code = model.probability['mean_code']
            oracle = 'dense_posterior_mean' if model.mode == 'dense_np' else 'gt_posterior_mean'
            plan = ([('prior_latent_mean', 1, False), (oracle, 1, True)] if mean_code else
                    [('prior_sample', 1, False), ('prior_latent_mean', 1, False), ('prior_mc_mean', 4, False),
                     ('prior_mc_mean', 8, False), (oracle, 1, True)])
            loader = DataLoader(CBCT_dataset_gs(dst_name='LUNA16', cfg=cfg.dataset, split='eval', num_views=6,
                                                out_res_scale=1.0, load_dense=model.mode == 'dense_np',
                                                load_gt_volume=model.mode == 'gt_cvae'), batch_size=1)
            loader.dataset.name_list = loader.dataset.name_list[:val_cases]
            result = {'run': run, 'estimators': {}, 'latent_usage': []}
            for mode, samples, is_oracle in plan:
                metrics, cases = eval_one_epoch(model, loader, 65536, latent_mode=mode, num_samples=samples,
                                                oracle=is_oracle, latent_seed=0)
                result['estimators'][f'{mode}_S{samples}'] = {'metrics': metrics['LUNA16'], 'cases': cases['LUNA16']}
                print(run, mode, samples, round(metrics['LUNA16']['psnr'], 4), flush=True)
            # Latent usage at fixed sparse features: interventions on the global code.
            with torch.no_grad():
                states = []
                for item in loader:
                    item = convert_cuda(item, 'cuda')
                    context = model.encode_context(item)
                    prior = model.infer_prior(context)
                    posterior, _ = model.infer_posterior(item, context)
                    states.append((item['name'][0], item['points_gt'].reshape(1, 1, -1), context, prior, posterior))
                for index, (name, gt, context, prior, posterior) in enumerate(states):
                    def decode(z):
                        state = model.state_with_global(model.sample_state(context, prior, mean=True), z)
                        return model.decode_chunked(state, 65536)
                    base = decode(prior[0])
                    variants = {'zero_z': torch.zeros_like(prior[0]),
                                'other_case_z': states[(index + 1) % len(states)][3][0],
                                'posterior_mean_z': posterior[0]}
                    row = {'case': name, 'kl_q_p': float(gaussian_kl(posterior[0], posterior[1], prior[0], prior[1]).sum()),
                           'sigma_p': float((.5 * prior[1]).exp().mean()), 'sigma_q': float((.5 * posterior[1]).exp().mean()),
                           'mse_prior_mean_z': float((base - gt).square().mean())}
                    for key, z in variants.items():
                        prediction = decode(z)
                        row[f'{key}_l1_change'] = float((prediction - base).abs().mean())
                        row[f'{key}_mse'] = float((prediction - gt).square().mean())
                    result['latent_usage'].append(row)
                mu_p = torch.cat([s[3][0] for s in states])
                mu_q = torch.cat([s[4][0] for s in states])
                result['between_case_var_mu_p'] = float(mu_p.var(0, unbiased=False).mean())
                result['between_case_var_mu_q'] = float(mu_q.var(0, unbiased=False).mean())
            psnr = lambda key: result['estimators'][key]['metrics']['psnr']
            result['oracle_gain_db'] = psnr(f'{oracle}_S1') - psnr('prior_latent_mean_S1')
            save(out_path, result)
            del model
            torch.cuda.empty_cache()


def evidence(seeds, steps, val_cases, tag, min_oracle_gain):
    """Reference gate for distillation: measured latent usage AND a positive dense-oracle gain."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from datasets.dst_gs import CBCT_dataset_gs
    from probabilistic_diagnostics import probability_diagnostics
    from transfer import file_hash
    from utils import convert_cuda
    for seed in seeds:
        run = run_name(DENSE, seed, tag)
        posthoc_path = REPORT / 'posthoc' / f'{run}.json'
        if not posthoc_path.exists():
            raise RuntimeError(f'Run --phase posthoc for {DENSE} seed {seed} first')
        cfg, model = load_trained(run, steps)
        loader = DataLoader(CBCT_dataset_gs(dst_name='LUNA16', cfg=cfg.dataset, split='eval', num_views=6,
                                            out_res_scale=1.0), batch_size=1)
        loader.dataset.name_list = loader.dataset.name_list[:val_cases]
        rows = [probability_diagnostics(model, convert_cuda(item, 'cuda'), num_samples=4, eval_npoint=65536)
                for item in loader]
        usage = {key: float(np.mean([row[key] for row in rows])) for key in ('output_change_l1', 'sensitivity_l2')}
        gain = read(posthoc_path)['oracle_gain_db']
        checkpoint = ROOT / 'logs' / run / f'step_{steps}.pth'
        save(REPORT / 'evidence' / f'{run}.json', {
            'checkpoint_name': checkpoint.name, 'checkpoint_sha256': file_hash(str(checkpoint)),
            'case_ids': list(loader.dataset.name_list), 'latent_usage': usage, 'oracle_gain_db': gain,
            'min_oracle_gain_db': min_oracle_gain,
            'gate': 'PASS' if gain >= min_oracle_gain and min(usage.values()) > 1e-8 else 'BLOCKED'})
        print(run, usage, 'oracle gain', round(gain, 4), flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- report
def report(seeds, steps, tag, bootstrap=10000):
    import numpy as np
    rng = np.random.default_rng(0)
    runs = {}
    for arm in ARMS:
        for seed in seeds:
            path = ROOT / 'logs' / run_name(arm, seed, tag) / f'validation_step_{steps}.json'
            if path.exists():
                runs[(arm, seed)] = {c['name']: c for c in read(path)['cases']['LUNA16']}

    def views(arm, seed):
        summary = read(ROOT / 'logs' / run_name(arm, seed, tag) / 'run_summary.json')['views']
        return [{k: v for k, v in r.items() if k != 'dense_view_ids'} for r in summary]

    pairs = [(comparator(arm), arm) for arm in ARMS if comparator(arm)] + EXTRA_COMPARISONS
    rows = []
    for a, b in pairs:
        common = [s for s in seeds if (a, s) in runs and (b, s) in runs]
        if not common:
            continue
        deltas, per_seed, paired = {'psnr': [], 'unclipped_mse': [], 'boundary_error': []}, [], True
        for seed in common:
            cases = sorted(runs[(a, seed)])
            if cases != sorted(runs[(b, seed)]):
                raise RuntimeError(f'{a} vs {b} seed {seed}: different eval cases')
            psnr = [runs[(b, seed)][c]['psnr'] - runs[(a, seed)][c]['psnr'] for c in cases]
            deltas['psnr'] += psnr
            for key in ('unclipped_mse', 'boundary_error'):  # relative change, negative is better
                deltas[key] += [runs[(b, seed)][c][key] / runs[(a, seed)][c][key] - 1 for c in cases]
            per_seed.append(float(np.mean(psnr)))
            paired &= views(a, seed) == views(b, seed)
        psnr = np.array(deltas['psnr'])
        resampled = [psnr[rng.integers(0, len(psnr), len(psnr))].mean() for _ in range(bootstrap)]
        low, high = np.percentile(resampled, [2.5, 97.5])
        rows.append({'baseline': a, 'variant': b, 'seeds': common, 'delta_psnr': float(psnr.mean()),
                     'ci95': [float(low), float(high)], 'per_seed_delta_psnr': per_seed,
                     'improved_cases': int((psnr > 0).sum()), 'cases': len(psnr),
                     'rel_mse_change': float(np.mean(deltas['unclipped_mse'])),
                     'rel_boundary_change': float(np.mean(deltas['boundary_error'])),
                     'paired_inputs': bool(paired)})
    # A case bootstrap ignores run-to-run noise, so a difference must also exceed the replica gap.
    replica = next((r for r in rows if r['variant'].endswith('-replica')), None)
    floor = abs(replica['delta_psnr']) if replica else None
    for r in rows:
        beyond = floor is not None and abs(r['delta_psnr']) > floor
        r['noise_floor_db'] = floor
        r['decision'] = ('NOISE_FLOOR' if r['variant'].endswith('-replica') else
                         'NOISE_FLOOR_PENDING' if floor is None else
                         'PROMOTE' if r['ci95'][0] > 0 and beyond else
                         'REJECT' if r['ci95'][1] < 0 and beyond else 'INCONCLUSIVE')
    arms_table = {}
    for (arm, seed), cases in runs.items():
        arms_table.setdefault(arm, {})[seed] = float(np.mean([c['psnr'] for c in cases.values()]))
    suffix = f'-{tag}' if tag else ''
    save(REPORT / f'results{suffix}.json', {'steps': steps, 'seeds': seeds, 'comparisons': rows,
                                            'arm_psnr_by_seed': arms_table})
    noise = next((r for r in rows if r['variant'].endswith('-replica')), None)
    lines = [f'# Experiment plan results ({steps} updates, tag "{tag}")', '',
             f"Noise floor (baseline vs replica): {noise['delta_psnr']:+.3f} dB, "
             f"CI {noise['ci95'][0]:+.3f}..{noise['ci95'][1]:+.3f}" if noise else 'Noise floor: replica not run yet.', '',
             '| baseline -> variant | seeds | dPSNR | 95% CI | per-seed | improved | rel MSE | rel boundary | paired | decision |',
             '|---|---|---:|---|---|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['baseline']} -> {r['variant']} | {len(r['seeds'])} | {r['delta_psnr']:+.3f} | "
                     f"{r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f} | {' '.join(f'{v:+.2f}' for v in r['per_seed_delta_psnr'])} | "
                     f"{r['improved_cases']}/{r['cases']} | {r['rel_mse_change']:+.1%} | {r['rel_boundary_change']:+.1%} | "
                     f"{'yes' if r['paired_inputs'] else 'no'} | {r['decision']} |")
    lines += ['', '| arm | ' + ' | '.join(f'seed {s}' for s in seeds) + ' |', '|---|' + '---:|' * len(seeds)]
    for arm, by_seed in arms_table.items():
        lines.append(f'| {arm} | ' + ' | '.join(f'{by_seed[s]:.3f}' if s in by_seed else '-' for s in seeds) + ' |')
    (REPORT / f'results{suffix}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def check():
    import torch
    from models.probabilistic_model import build_model
    from train import load_common_init
    from utils import load_config
    for arm in ARMS:
        torch.manual_seed(0)
        model = build_model(load_config(config_path(arm)).model)
        init, dropped = arm_init(arm, 0)
        loaded = load_common_init(model, init)
        print(f"{arm:45s} {sum(p.numel() for p in model.parameters()) / 1e6:6.2f}M shared {loaded['shared_count']:3d} "
              f"own {len(loaded['arm_specific']):3d} dropped {dropped} vs {comparator(arm)}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True,
                        choices=['check', 'init', 'smoke', 'train', 'posthoc', 'evidence', 'report'])
    parser.add_argument('--stage', type=int, choices=[1, 2])
    parser.add_argument('--arms', nargs='*', help='subset of arm names (default: all arms of --stage)')
    parser.add_argument('--seeds', type=int, nargs='*', default=DEFAULT_SEEDS)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--val_cases', type=int, default=12)
    parser.add_argument('--tag', default='', help='suffix for run names, e.g. confirm10k')
    parser.add_argument('--keep_checkpoints', choices=['final', 'all'], default='final',
                        help='final: delete the midpoint checkpoint after a run passes (saves ~0.4 GB per run)')
    parser.add_argument('--min_oracle_gain', type=float, default=0.1, help='dB; distillation gate')
    args = parser.parse_args()
    os.chdir(ROOT)
    selected = args.arms or [arm for arm, (stage, _) in ARMS.items() if args.stage in (None, stage)]
    unknown = set(selected) - set(ARMS)
    if unknown:
        parser.error(f'unknown arms: {sorted(unknown)}')
    if args.phase == 'check':
        check()
    elif args.phase == 'init':
        for seed in args.seeds:
            print(seed, create_common_init(seed).relative_to(ROOT))
    elif args.phase == 'smoke':
        train([a for a in selected if a != DISTILL], [0], 10, 1, '', smoke=True, min_free_gb=2.)
    elif args.phase == 'train':
        train(selected, args.seeds, args.steps, args.val_cases, args.tag, keep=args.keep_checkpoints)
    elif args.phase == 'posthoc':
        posthoc(selected, args.seeds, args.steps, args.val_cases, args.tag)
    elif args.phase == 'evidence':
        evidence(args.seeds, args.steps, args.val_cases, args.tag, args.min_oracle_gain)
    else:
        report(args.seeds, args.steps, args.tag)
