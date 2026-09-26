'''P1 screening runner for reports/multiview_structural_experiment_plan.md.

Phases
  init    build one canonical initialization and report the shared parameters per arm
  train   run the six main arms with identical protocol, own budget ledger, no overwrite
  verify  cross-arm identity of the first N step manifests and of the shared init digest
  report  paired per-case validation metrics, plan thresholds and pilot gates

P1 screens trends and failure modes at 5,000 updates and seed 0 only. It does not
select a structure and makes no formal claim; see the plan's P2/P3 stages.
'''
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
REPORTS = ROOT / 'reports/p1'
TAG = ''
ARMS = ['B0', 'B1', 'T1', 'T2', 'T3', 'T4']
COMPARISONS = [('B0', 'B1'), ('B1', 'T1'), ('B0', 'T1'), ('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4')]
PLAN_THRESHOLDS = {'psnr_db': .2, 'relative_error_drop': .03}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def load_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def source_hash():
    digest = hashlib.sha256()
    paths = sorted(list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'configs/multiview').glob('*.yaml')) +
                   [Path(__file__).resolve()])
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_name(arm, seed):
    return f'p1-{arm}-seed{seed}{TAG}'


def common_init_path(seed):
    return ROOT / f'logs/p1-common-init/seed{seed}.pth'


def init(args):
    '''Canonical initialization: the largest arm's weights at this seed.

    Every parameter shared by name and shape is copied into each arm, so common
    UNet/GS/decoder (and, among T arms, transformer) weights start identical.
    Modules that exist in only one arm are necessarily initialized on their own.
    '''
    import random
    import numpy as np
    import torch
    from models.probabilistic_model import build_model
    from utils import load_config
    path = common_init_path(args.seed)
    if path.exists() and not args.force:
        return {'status': 'PASS', 'common_init': str(path.relative_to(ROOT)), 'note': 'existing initialization kept'}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    reference = build_model(load_config(ROOT / 'configs/multiview/T4.yaml').model).state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': reference, 'seed': args.seed, 'source_arm': 'T4'}, path)
    coverage = {}
    for arm in ARMS:
        torch.manual_seed(args.seed)
        target = build_model(load_config(ROOT / f'configs/multiview/{arm}.yaml').model).state_dict()
        shared = [name for name in target if name in reference and target[name].shape == reference[name].shape]
        coverage[arm] = {'parameters': sum(value.numel() for value in target.values()),
                         'tensors': len(target), 'shared_with_common_init': len(shared),
                         'arm_specific': sorted(set(target) - set(shared))}
    save_json(REPORTS / 'common_init_coverage.json',
              {'common_init': str(path.relative_to(ROOT)), 'seed': args.seed, 'source_arm': 'T4',
               'reference_tensors': len(reference), 'coverage': coverage})
    return {'status': 'PASS', 'common_init': str(path.relative_to(ROOT)),
            'coverage': {arm: value['shared_with_common_init'] for arm, value in coverage.items()},
            'report': 'reports/p1/common_init_coverage.json'}


def training_command(arm, args):
    return [args.python, 'code/train.py',
            '--cfg_path', f'configs/multiview/{arm}.yaml',
            '--name', run_name(arm, args.seed),
            '--dst_name', 'LUNA16', '--num_views', '6',
            '--batch_size', '1', '--num_points', '1024', '--num_workers', str(args.num_workers),
            '--epoch', '400', '--lr', '2e-4', '--weight_decay', '1e-2', '--lr_decay', '1e-3',
            '--max_steps', str(args.steps), '--val_steps', *[str(step) for step in args.val_steps],
            '--manifest_steps', str(args.manifest_steps),
            '--val_case_count', str(args.val_case_count), '--eval_scale', '1.0',
            '--eval_npoint', str(args.eval_npoint), '--diag_every', str(args.diag_every),
            '--common_init', str(common_init_path(args.seed).relative_to(ROOT)),
            '--seed', str(args.seed), '--device', args.device, '--no-wandb']


def train(args, state, persist):
    if not common_init_path(args.seed).exists():
        return {'status': 'BLOCKED_NO_COMMON_INIT', 'reason': 'Run --phase init first'}
    runs = state.setdefault('runs', {})
    ledger = state.setdefault('budget', {}).setdefault(
        f'seed{args.seed}_steps{args.steps}',
        {'updates_limit': args.steps * len(ARMS), 'updates_used': 0, 'seconds_used': 0.})
    for arm in args.arms:
        key = run_name(arm, args.seed)
        if runs.get(key, {}).get('status') == 'COMPLETED':
            continue
        log_dir = ROOT / 'logs' / key
        if log_dir.exists() and any(log_dir.glob('*.pth')) and runs.get(key, {}).get('status') != 'RUNNING':
            runs[key] = {'status': 'SKIPPED_EXISTING', 'reason': f'{log_dir} already holds checkpoints; choose a fresh seed/name'}
            persist()
            continue
        if ledger['updates_used'] + args.steps > ledger['updates_limit']:
            return {'status': 'STOPPED_BUDGET', 'reason': 'P1 ledger exhausted', 'runs': runs, 'budget': ledger}
        command = training_command(arm, args)
        runs[key] = {'status': 'RUNNING', 'command': shlex.join(command), 'reserved_updates': args.steps}
        ledger['updates_used'] += args.steps
        persist()
        started = time.monotonic()
        REPORTS.mkdir(parents=True, exist_ok=True)
        with (REPORTS / f'{key}.log').open('w') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        summary = load_json(ROOT / 'logs' / key / 'run_summary.json')
        actual = (summary or {}).get('optimizer_updates', 0)
        ledger['updates_used'] += actual - args.steps
        ledger['seconds_used'] += elapsed
        runs[key] = {'status': 'COMPLETED' if process.returncode == 0 else 'FAIL',
                     'returncode': process.returncode, 'command': shlex.join(command),
                     'optimizer_updates': actual, 'seconds': elapsed,
                     'peak_memory_gb': (summary or {}).get('peak_memory_gb'),
                     'parameters': (summary or {}).get('parameters'),
                     'log': f'reports/p1/{key}.log'}
        persist()
        if process.returncode != 0:
            return {'status': 'FAIL', 'reason': f'{key} exited {process.returncode}', 'runs': runs, 'budget': ledger}
    return {'status': 'PASS', 'runs': runs, 'budget': ledger, 'quality_claim': False}


def verify(args, state):
    '''Paired-input check: identical cases, views, blocks and query points per step.'''
    summaries, problems = {}, []
    for arm in args.arms:
        summary = load_json(ROOT / 'logs' / run_name(arm, args.seed) / 'run_summary.json')
        if summary is None:
            problems.append(f'{arm}: no run_summary.json')
        else:
            summaries[arm] = summary
    if len(summaries) < 2:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'Need at least two completed arms', 'problems': problems}
    reference_arm = args.arms[0] if args.arms[0] in summaries else sorted(summaries)[0]
    reference = summaries[reference_arm]
    fields = ('cases', 'context_view_ids', 'block_idx', 'points_digest', 'points_gt_digest')
    checked = min(len(summary['views']) for summary in summaries.values())
    for arm, summary in summaries.items():
        if summary['train_cases'] != reference['train_cases']:
            problems.append(f'{arm}: different training case list')
        if summary['validation_cases'] != reference['validation_cases']:
            problems.append(f'{arm}: different validation cases')
        init_report = summary.get('shared_initialization')
        if not init_report:
            problems.append(f'{arm}: no shared initialization report')
        elif init_report.get('common_init_sha256') != (summaries[reference_arm].get('shared_initialization') or {}).get('common_init_sha256'):
            problems.append(f'{arm}: a different common initialization file than {reference_arm}')
        for index in range(checked):
            left, right = summary['views'][index], reference['views'][index]
            for field in fields:
                if left.get(field) != right.get(field):
                    problems.append(f'{arm}: step {index} {field} differs from {reference_arm}')
                    break
    # Every parameter two arms have in common must have come from the same file,
    # so their shared weights are bitwise identical; arm-specific modules cannot be shared.
    coverage = {}
    for left in sorted(summaries):
        report_left = summaries[left].get('shared_initialization') or {}
        full_left = set(report_left.get('shared_parameters', [])) | set(report_left.get('arm_specific', []))
        for right in sorted(summaries):
            if left >= right:
                continue
            report_right = summaries[right].get('shared_initialization') or {}
            full_right = set(report_right.get('shared_parameters', [])) | set(report_right.get('arm_specific', []))
            common = full_left & full_right
            uncovered = sorted(common - set(report_left.get('shared_parameters', [])))
            coverage[f'{left}|{right}'] = {'common_parameters': len(common), 'not_shared': uncovered[:10]}
            if uncovered:
                problems.append(f'{left}/{right}: {len(uncovered)} common parameters were not seeded identically')
    result = {'status': 'PASS' if not problems else 'FAIL', 'reference_arm': reference_arm,
              'arms': sorted(summaries), 'steps_checked': checked,
              'fields': list(fields), 'problems': problems[:40],
              'shared_initialization_coverage': coverage,
              'common_init_sha256': (reference.get('shared_initialization') or {}).get('common_init_sha256'),
              'shared_parameter_count': (reference.get('shared_initialization') or {}).get('shared_count')}
    save_json(REPORTS / 'paired_input_verification.json', result)
    return result


def collect(arm, seed):
    directory = ROOT / 'logs' / run_name(arm, seed)
    records = []
    for path in sorted(directory.glob('validation_step_*.json'), key=lambda item: int(item.stem.split('_')[-1])):
        records.append(load_json(path))
    return records


def branch_usage(arm, seed, val_steps):
    """Did the new branches actually get used, or did weight decay switch them off?

    Reports the residual gate and the output-projection magnitude of each new module
    alongside the recorded residual/base ratios, so a non-zero gradient is never
    mistaken for a branch the network relies on (plan risk 3).
    """
    import torch
    directory = ROOT / 'logs' / run_name(arm, seed)
    steps = [json.loads(line) for line in (directory / 'step_metrics.jsonl').read_text().splitlines()]
    tracked = ['encoder/residual_base_norm_ratio', 'encoder/point/residual_base_norm_ratio',
               'encoder/token_variance', 'gradient/cross_attention', 'gradient/spatial_attention',
               'gradient/point_fusion', 'gradient/structure_head', 'encoder/anatomy_sampling_fraction',
               'train/L_anatomy']
    usage = {'first_step': {}, 'last_step': {}, 'checkpoints': {}}
    for key in tracked:
        if key in steps[-1]:
            usage['first_step'][key] = steps[0].get(key)
            usage['last_step'][key] = steps[-1][key]
    for step in val_steps:
        path = directory / f'step_{step}.pth'
        if not path.exists():
            continue
        weights = torch.load(path, map_location='cpu')['model']
        entry = {name: float(value) for name, value in weights.items()
                 if name.endswith('residual_gate') or name.endswith('fusion.gate')}
        entry.update({f'|W| {name}': float(value.abs().mean()) for name, value in weights.items()
                      if name.endswith('multiview_encoder.output_projection.weight')
                      or name.endswith('point_view_fusion.output_projection.weight')})
        usage['checkpoints'][step] = entry
    return usage


def report(args, state):
    import csv
    import numpy as np
    from research_gates import pilot_gate
    metric_keys = ['psnr', 'ssim', 'unclipped_mse', 'clipping_fraction', 'boundary_error', 'gradient_error']
    per_arm = {arm: collect(arm, args.seed) for arm in ARMS}
    available = [arm for arm in ARMS if per_arm[arm]]
    rows = []
    for arm in available:
        summary = load_json(ROOT / 'logs' / run_name(arm, args.seed) / 'run_summary.json', {})
        for record in per_arm[arm]:
            for dataset, cases in record['cases'].items():
                for case in cases:
                    rows.append({'arm': arm, 'seed': args.seed, 'global_step': record['global_step'],
                                 'epoch': record['epoch'], 'dataset': dataset, 'case_id': case['name'],
                                 'eval_scale': record['eval_scale'], 'eval_npoint': record['eval_npoint'],
                                 'view_ids': case.get('view_ids'),
                                 'inference_seconds': case.get('inference_seconds'),
                                 'peak_memory_gb': case.get('peak_memory_gb'),
                                 'parameters': summary.get('parameters'),
                                 'train_seconds': summary.get('seconds'),
                                 **{key: case.get(key) for key in metric_keys}})
    REPORTS.mkdir(parents=True, exist_ok=True)
    if rows:
        with (REPORTS / 'per_case_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = []
    for baseline, candidate in COMPARISONS:
        if not per_arm[baseline] or not per_arm[candidate]:
            continue
        final = {arm: per_arm[arm][-1] for arm in (baseline, candidate)}
        if final[baseline]['global_step'] != final[candidate]['global_step']:
            comparisons.append({'baseline': baseline, 'candidate': candidate, 'status': 'INCONCLUSIVE',
                                'reason': 'Different optimizer update counts'})
            continue
        packed = {}
        for arm in (baseline, candidate):
            cases = [case for dataset in final[arm]['cases'].values() for case in dataset]
            packed[arm] = {'train_updates': final[arm]['global_step'], 'seed': args.seed, 'loss_protocol': 'mse',
                           'eval_scale': final[arm]['eval_scale'], 'resampling_protocol': 'native',
                           'latent_mode': 'deterministic', 'cases': cases}
        gate = pilot_gate(packed[baseline], packed[candidate])
        deltas = {}
        reference = {case['name']: case for case in packed[baseline]['cases']}
        proposed = {case['name']: case for case in packed[candidate]['cases']}
        for key in metric_keys:
            values = [proposed[name][key] - reference[name][key] for name in reference
                      if reference[name].get(key) is not None and proposed[name].get(key) is not None]
            relative = [(reference[name][key] - proposed[name][key]) / reference[name][key] for name in reference
                        if reference[name].get(key) not in (None, 0) and proposed[name].get(key) is not None]
            deltas[key] = {'mean_delta': float(np.mean(values)) if values else None,
                           'median_delta': float(np.median(values)) if values else None,
                           'cases_improved': int(sum(value > 0 for value in values)) if values else None,
                           'mean_relative_drop': float(np.mean(relative)) if relative else None}
        registered = {
            'psnr_ge_0.2db': bool(deltas['psnr']['mean_delta'] is not None and deltas['psnr']['mean_delta'] >= PLAN_THRESHOLDS['psnr_db']),
            'boundary_drop_ge_3pct': bool(deltas['boundary_error']['mean_relative_drop'] is not None and
                                          deltas['boundary_error']['mean_relative_drop'] >= PLAN_THRESHOLDS['relative_error_drop']),
            'gradient_drop_ge_3pct': bool(deltas['gradient_error']['mean_relative_drop'] is not None and
                                          deltas['gradient_error']['mean_relative_drop'] >= PLAN_THRESHOLDS['relative_error_drop']),
            'ssim_not_worse': bool(deltas['ssim']['mean_delta'] is not None and deltas['ssim']['mean_delta'] >= -.002),
            'unclipped_mse_not_worse': bool(deltas['unclipped_mse']['mean_delta'] is not None and deltas['unclipped_mse']['mean_delta'] <= 0),
        }
        comparisons.append({'baseline': baseline, 'candidate': candidate, 'train_updates': final[baseline]['global_step'],
                            'cases': len(reference), 'deltas': deltas, 'pilot_gate': gate['status'],
                            'plan_registered_thresholds': registered})
    usage = {}
    for arm in available:
        try:
            usage[arm] = branch_usage(arm, args.seed, args.val_steps)
        except Exception as error:
            usage[arm] = {'error': f'{type(error).__name__}: {error}'}
    curves = {arm: [{'global_step': record['global_step'],
                     **{key: record['metrics'][dataset][key] for dataset in record['metrics'] for key in metric_keys}}
                    for record in per_arm[arm]] for arm in available}
    result = {'status': 'PASS' if available else 'NOT_RUN', 'stage': 'P1_SCREENING_ONLY', 'formal_claim': False,
              'seed': args.seed, 'arms': available, 'curves': curves, 'comparisons': comparisons,
              'branch_usage': usage, 'per_case_csv': str((REPORTS / 'per_case_metrics.csv').relative_to(ROOT))}
    save_json(REPORTS / 'p1_results.json', result)
    last_step = max((point['global_step'] for arm in available for point in curves[arm]), default=0)
    case_count = len({row['case_id'] for row in rows}) if rows else 0
    lines = ['# P1 筛查结果（seed %d，%d updates，6 views，eval %d 例）' % (args.seed, last_step, case_count), '',
             '本阶段只筛查趋势与失败模式，不做结构选择，不构成正式结论（计划 P1）。', '',
             '| arm | 参数量 | 训练秒 | 峰值显存 GB |', '|---|---|---|---|']
    for arm in available:
        summary = load_json(ROOT / 'logs' / run_name(arm, args.seed) / 'run_summary.json', {})
        lines.append('| %s | %s | %s | %s |' % (arm, summary.get('parameters'),
                                                round(summary.get('seconds', 0), 1), summary.get('peak_memory_gb')))
    lines += ['', '| arm | step | PSNR | SSIM | unclipped MSE | clip 比例 | 边界误差 | 梯度误差 |', '|---|---|---|---|---|---|---|---|']
    for arm in available:
        for point in curves[arm]:
            lines.append('| %s | %d | %s | %s | %s | %s | %s | %s |' % (
                arm, point['global_step'], *[('%.4g' % point[key]) if point.get(key) is not None else '-'
                                             for key in metric_keys]))
    lines += ['', '| 比较 | ΔPSNR | ΔSSIM | 边界相对下降 | 梯度相对下降 | pilot gate | 计划阈值 |',
              '|---|---|---|---|---|---|---|']
    for entry in comparisons:
        if entry.get('status') == 'INCONCLUSIVE':
            lines.append('| %s→%s | - | - | - | - | INCONCLUSIVE | %s |' % (entry['baseline'], entry['candidate'], entry['reason']))
            continue
        passed = [name for name, value in entry['plan_registered_thresholds'].items() if value]
        lines.append('| %s→%s | %s | %s | %s | %s | %s | %s |' % (
            entry['baseline'], entry['candidate'],
            '%.3f' % entry['deltas']['psnr']['mean_delta'],
            '%.4f' % entry['deltas']['ssim']['mean_delta'],
            '%.3f' % (entry['deltas']['boundary_error']['mean_relative_drop'] or 0),
            '%.3f' % (entry['deltas']['gradient_error']['mean_relative_drop'] or 0),
            entry['pilot_gate'], ', '.join(passed) or '无'))
    lines += ['', '## 新分支是否被真正使用（计划风险 3）', '',
              '| arm | 残差/base 比 首→末 | token 方差 首→末 | cross-attn 梯度 首→末 | gate @1k/3k/5k | \\|W_out\\| @1k/3k/5k |',
              '|---|---|---|---|---|---|']
    for arm in available:
        entry = usage.get(arm, {})
        if 'error' in entry or not entry.get('last_step'):
            continue
        first, last, checkpoints = entry['first_step'], entry['last_step'], entry['checkpoints']
        def trajectory(prefix):
            values = [next((value for name, value in checkpoints.get(step, {}).items() if prefix in name), None)
                      for step in args.val_steps]
            return ' / '.join('-' if value is None else '%.2e' % value for value in values)
        def pair(key):
            if key not in last:
                return '-'
            return '%.2e → %.2e' % (first.get(key, float('nan')), last[key])
        lines.append('| %s | %s | %s | %s | %s | %s |' % (
            arm, pair('encoder/residual_base_norm_ratio'), pair('encoder/token_variance'),
            pair('gradient/cross_attention'), trajectory('residual_gate'),
            trajectory('|W| ') ))
    lines += ['', '每病例配对数值见 `%s`，输入配对核对见 `paired_input_verification.json`，' % result['per_case_csv'],
              '完整分支使用诊断见 `p1_results.json` 的 `branch_usage`。', '']
    (REPORTS / 'p1_report.md').write_text('\n'.join(lines))
    result['report'] = 'reports/p1/p1_report.md'
    return result


def main():
    parser = argparse.ArgumentParser(description='P1 screening for the fixed 6-view structural plan')
    parser.add_argument('--phase', required=True, choices=['init', 'train', 'verify', 'report'])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--val_steps', type=int, nargs='+', default=[1000, 3000, 5000])
    parser.add_argument('--val_case_count', type=int, default=6)
    parser.add_argument('--eval_npoint', type=int, default=65536)
    parser.add_argument('--manifest_steps', type=int, default=100)
    parser.add_argument('--diag_every', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--tag', default='',
                        help='suffix that isolates a round of runs, logs and reports from earlier ones')
    parser.add_argument('--force', action='store_true', help='rebuild the shared initialization; never resets the budget')
    args = parser.parse_args()
    if args.tag:
        if not args.tag.replace('-', '').replace('_', '').isalnum():
            parser.error('--tag must be alphanumeric with - or _')
        globals()['TAG'] = '-' + args.tag
        globals()['REPORTS'] = ROOT / f'reports/p1-{args.tag}'
    REPORTS.mkdir(parents=True, exist_ok=True)
    state_path = REPORTS / 'execution_status.json'
    state = load_json(state_path, {})
    state['source_hash'] = source_hash()
    state['protocol'] = {'views': 6, 'view_ids': [0, 8, 16, 25, 33, 41], 'batch_size': 1, 'num_points': 1024,
                         'precision': 'fp32', 'steps': args.steps, 'val_steps': args.val_steps,
                         'eval_scale': 1.0, 'eval_npoint': args.eval_npoint, 'seed': args.seed,
                         'tag': args.tag, 'run_names': [run_name(arm, args.seed) for arm in ARMS]}
    persist = lambda: save_json(state_path, state)
    started = time.monotonic()
    try:
        if args.phase == 'init':
            result = init(args)
        elif args.phase == 'train':
            result = train(args, state, persist)
        elif args.phase == 'verify':
            result = verify(args, state)
        else:
            result = report(args, state)
    except Exception as error:
        import traceback
        (REPORTS / f'{args.phase}_error.log').write_text(traceback.format_exc())
        result = {'status': 'FAIL', 'reason': f'{type(error).__name__}: {error}'}
    state[args.phase] = dict(result, source_hash=state['source_hash'], elapsed_seconds=time.monotonic() - started)
    persist()
    print(json.dumps(state[args.phase], indent=2, default=str)[:4000])
    if result['status'] in ('FAIL',):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
