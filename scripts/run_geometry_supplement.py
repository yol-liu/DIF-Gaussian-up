"""One bounded direct anatomy comparison, after the original six runs finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/geometry_validation'
os.chdir(ROOT)


def save(value):
    path = REPORT / 'supplement_status.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


if __name__ == '__main__':
    name = 'geomv1-train-Araw-seed0'
    destination = ROOT / 'logs' / name
    if destination.exists() or (REPORT / 'supplement_status.json').exists():
        raise RuntimeError('Supplement already exists; inspect it rather than overwrite/restart.')
    state = {'arm': 'Araw', 'status': 'WAITING_FOR_MAIN', 'updates_limit': 2000}
    save(state)
    print('Araw queued; waiting for all six main arms to finish before using the GPU.', flush=True)
    while True:
        ledger = json.loads((REPORT / 'execution_status.json').read_text())
        runs = [ledger['runs'].get(f'geomv1-train-{arm}-seed0', {}) for arm in ['R0', 'C0', 'A0', 'A1', 'S0', 'H0']]
        if any(run.get('status', '').startswith('FAILED') for run in runs):
            state['status'] = 'CANCELLED_MAIN_FAILURE'
            save(state)
            raise RuntimeError('Main experiment failed; supplemental training was not started.')
        if all(run.get('status') == 'PASS' for run in runs):
            break
        time.sleep(10)
    frozen = json.loads((REPORT / 'source_manifest.json').read_text())['files']
    for name_in_snapshot, expected in frozen.items():
        if hashlib.sha256((ROOT / name_in_snapshot).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed: {name_in_snapshot}')
    for path in [Path(__file__).resolve(), ROOT / 'configs/geometry_validation_supplement/Araw.yaml']:
        frozen[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = REPORT / 'supplement_source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (REPORT / 'supplement_source_manifest.json').write_text(json.dumps(frozen, indent=2))
    command = [sys.executable, 'code/train.py', '--cfg_path', 'configs/geometry_validation_supplement/Araw.yaml',
               '--name', name, '--dst_name', 'LUNA16', '--num_views', '6', '--batch_size', '1',
               '--num_points', '1024', '--num_workers', '2', '--epoch', '400', '--lr', '2e-4',
               '--weight_decay', '1e-2', '--lr_decay', '1e-3', '--max_steps', '2000',
               '--val_steps', '1000', '2000', '--manifest_steps', '2000', '--val_case_count', '6',
               '--eval_scale', '1.0', '--eval_npoint', '65536', '--diag_every', '25',
               '--common_init', 'logs/p1-common-init/seed0.pth', '--seed', '0', '--device', 'cuda', '--no-wandb']
    state.update(status='RUNNING', command=command, started_at=time.time(), reserved_updates=2000)
    save(state)
    print('START Araw: 2000 updates; original nomax coordinates + anatomy guidance + point max.', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (REPORT / f'{name}.log').open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    state.update(returncode=result.returncode, seconds=time.time() - state['started_at'])
    state['status'] = 'FAILED'
    if result.returncode == 0:
        summary = json.loads((destination / 'run_summary.json').read_text())
        reference = json.loads((ROOT / 'logs/geomv1-train-R0-seed0/run_summary.json').read_text())
        state['paired_inputs_all_steps'] = len(summary['views']) == 2000 and summary['views'] == reference['views']
        state['actual_updates'] = summary['global_step']
        state['metrics'] = json.loads((destination / 'validation_step_2000.json').read_text())['metrics']['LUNA16']
        if state['paired_inputs_all_steps'] and summary['global_step'] == 2000:
            state['status'] = 'PASS'
    save(state)
    print(json.dumps(state, indent=2), flush=True)
    sys.exit(0 if state['status'] == 'PASS' else 1)
