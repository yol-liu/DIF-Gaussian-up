"""Bounded, sequential six-arm geometry validation; never overwrite a checkpoint."""
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
REPORT = ROOT / 'reports/geometry_validation'
ARMS = ['R0', 'C0', 'A0', 'A1', 'S0', 'H0']
COMPARISONS = [('R0', 'C0'), ('C0', 'A0'), ('A0', 'A1'), ('C0', 'A1'), ('C0', 'S0'), ('S0', 'H0')]
sys.path.insert(0, str(ROOT / 'code'))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def source_files():
    return sorted(list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'configs/geometry_validation').glob('*.yaml')) + [Path(__file__).resolve()])


def provenance():
    files = {str(path.relative_to(ROOT)): digest(path) for path in source_files()}
    path = REPORT / 'source_manifest.json'
    if path.exists():
        if read(path)['files'] != files:
            raise RuntimeError('Source changed since frozen validation snapshot; use a separate experiment directory.')
        return
    for source in source_files():
        target = REPORT / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    save(path, {'files': files, 'created_at': time.time()})


def initialize():
    import torch
    from models.probabilistic_model import build_model
    from train import load_common_init
    from utils import load_config
    backup = ROOT / 'backups/t1-nomax-best-20260924'
    manifest = read(backup / 'manifest.json')
    for name, record in manifest['files'].items():
        if digest(backup / name) != record['sha256']:
            raise RuntimeError(f'Backup checksum mismatch: {name}')
    results = {}
    common = ROOT / 'logs/p1-common-init/seed0.pth'
    for arm in ARMS:
        torch.manual_seed(0)
        model = build_model(load_config(ROOT / f'configs/geometry_validation/{arm}.yaml').model)
        loaded = load_common_init(model, common)
        if arm == 'H0':
            state = model.state_dict()
            for prefix in ('image_encoder.inc.', 'image_encoder.down.0.', 'image_encoder.up.', 'image_encoder.outc.'):
                assert all(name in loaded['shared_parameters'] for name in state if name.startswith(prefix))
        results[arm] = dict(loaded, parameters=sum(p.numel() for p in model.parameters()))
        del model
    # Frozen checkpoint remains strictly loadable with legacy defaults.
    frozen = torch.load(backup / 'logs/diag-adamw-T1-nomax-lr2e-4/partial_step_2000.pth', map_location='cpu')
    model = build_model(load_config(ROOT / 'configs/geometry_validation/R0.yaml').model)
    model.load_state_dict(frozen['model'], strict=True)
    save(REPORT / 'initialization_verification.json', {'backup_files_verified': len(manifest['files']),
        'backup_checkpoint_strict_load': True, 'arms': results})
    print('Backup checksums, strict checkpoint reload, shared initialization: PASS', flush=True)


def train(phase):
    provenance()
    state_path = REPORT / 'execution_status.json'
    state = read(state_path) if state_path.exists() else {'runs': {}, 'limits': {'smoke': 60, 'train': 12000}}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    steps, val_cases = (10, 1) if phase == 'smoke' else (2000, 6)
    schedule = [10] if phase == 'smoke' else [1000, 2000]
    if phase == 'train':
        if not all(state['runs'].get(f'geomv1-smoke-{arm}-seed0', {}).get('status') == 'PASS' for arm in ARMS):
            raise RuntimeError('All six real-data smoke runs must pass before the main comparison.')
    for arm in ARMS:
        name = f'geomv1-{phase}-{arm}-seed0'
        if state['runs'].get(name, {}).get('status') == 'PASS':
            continue
        destination = ROOT / 'logs' / name
        if destination.exists():
            raise RuntimeError(f'Existing run requires inspection, not automatic restart: {destination}')
        command = [sys.executable, 'code/train.py', '--cfg_path', f'configs/geometry_validation/{arm}.yaml',
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
        reserved = sum(r['reserved_updates'] for r in state['runs'].values() if r['phase'] == phase)
        if reserved > state['limits'][phase]:
            raise RuntimeError('Experiment update budget exceeded')
        save(state_path, state)
        print(f'START {name}: {steps} updates, {val_cases} eval cases', flush=True)
        with (REPORT / f'{name}.log').open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, seconds=time.time() - entry['started_at'])
        summary = read(destination / 'run_summary.json') if (destination / 'run_summary.json').exists() else None
        passed = result.returncode == 0 and summary and summary['global_step'] == steps
        entry['status'] = 'PASS' if passed else 'FAILED'
        entry['actual_updates'] = summary['global_step'] if summary else None
        if passed:
            entry['metrics'] = read(destination / f'validation_step_{steps}.json')['metrics']['LUNA16']
            # All completed arms must have identical manifests, not just the first 20.
            reference_name = f'geomv1-{phase}-R0-seed0'
            reference = read(ROOT / 'logs' / reference_name / 'run_summary.json')
            entry['paired_inputs_all_steps'] = summary['views'] == reference['views']
            if not entry['paired_inputs_all_steps']:
                entry['status'] = 'FAILED_PAIRED_INPUTS'
        save(state_path, state)
        print(f"END {name}: {entry['status']} {entry.get('metrics', {})}", flush=True)
        if entry['status'] != 'PASS':
            raise RuntimeError(f'Inspect {REPORT / (name + ".log")}')


def report():
    import numpy as np
    rows, cases = {}, {}
    for arm in ARMS:
        path = ROOT / f'logs/geomv1-train-{arm}-seed0'
        if not (path / 'run_summary.json').exists():
            continue
        summary = read(path / 'run_summary.json')
        validations = {str(step): read(path / f'validation_step_{step}.json') for step in (1000, 2000)}
        rows[arm] = {'parameters': summary['parameters'], 'seconds': summary['seconds'],
                     'peak_memory_gb': summary['peak_memory_gb'],
                     'metrics': {step: v['metrics']['LUNA16'] for step, v in validations.items()}}
        cases[arm] = {v['name']: v for v in validations['2000']['cases']['LUNA16']}
    comparisons = {}
    for a, b in COMPARISONS:
        if a not in rows or b not in rows:
            continue
        assert cases[a].keys() == cases[b].keys()
        psnr = np.array([cases[b][n]['psnr'] - cases[a][n]['psnr'] for n in cases[a]])
        boundary = np.array([1 - cases[b][n]['boundary_error'] / cases[a][n]['boundary_error'] for n in cases[a]])
        comparisons[f'{a}->{b}'] = {'delta_psnr': float(psnr.mean()), 'improved_cases': int((psnr > 0).sum()),
            'psnr_per_case': psnr.tolist(), 'mean_relative_boundary_drop': float(boundary.mean()),
            'boundary_improved_cases': int((boundary > 0).sum())}
    save(REPORT / 'results.json', {'arms': rows, 'comparisons': comparisons,
        'scope': 'seed0, 2000 updates, 6 eval cases; screening only; all point fusion=max'})
    lines = ['几何结构验证实测结果', '', '统一 AdamW、seed0、2000 updates、固定6 views、6例验证、point max；只作筛查。', '',
             '| arm | PSNR@1k | PSNR@2k | SSIM@2k | 原始MSE | 边界误差 | 梯度误差 | 参数M |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in rows.items():
        m = row['metrics']['2000']
        lines.append(f"| {arm} | {row['metrics']['1000']['psnr']:.4f} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['unclipped_mse']:.6f} | {m['boundary_error']:.6f} | {m['gradient_error']:.6f} | {row['parameters']/1e6:.2f} |")
    lines += ['', 'R0=原nomax复跑；C0=坐标统一；A0=C0+解剖监督；A1=A0+引导采样；S0=C0+共享3D五点采样；H0=S0+层级Transformer。', '',
              '| 比较 | ΔPSNR | 改善病例 | 边界相对下降 |', '|---|---:|---:|---:|']
    for pair, values in comparisons.items():
        lines.append(f"| {pair} | {values['delta_psnr']:+.4f} | {values['improved_cases']}/6 | {values['mean_relative_boundary_drop']:+.2%} |")
    lines += ['', '完整协议见 plan.json；执行命令/全步输入核对见 execution_status.json；源码快照及哈希见 source_manifest.json；原最佳方案备份见 ../../backups/t1-nomax-best-20260924。',
              '共享3D五点采用中心+正四面体，样本数相同但形状/物理支撑不同，不能把差异全部归因于对应关系。',
              'H0是保留卷积浅层和UNet解码器的层级窗口Transformer；新阶段的容量/优化也发生变化，不是纯几何模块消融。',
              '单seed/6例无法作最终结构优劣结论；病例同方向不能代替独立训练重复。']
    (REPORT / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['init', 'smoke', 'train', 'report'], required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'init':
        initialize()
    elif args.phase == 'report':
        report()
    else:
        train(args.phase)
