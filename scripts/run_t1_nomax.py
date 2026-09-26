"""Protected T1-nomax: execute the verified backup in an isolated Python process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'backups/t1-nomax-best-20260924'
MANIFEST_SHA256 = '2e913365c6edb23141a3aa85e5eb253ca5073b48b4c2e05b9db409345ac071ae'
ORIGINAL = 'logs/diag-adamw-T1-nomax-lr2e-4'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_backup():
    manifest = BACKUP / 'manifest.json'
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError('Protected T1-nomax manifest changed; refusing to run')
    files = json.loads(manifest.read_text())['files']
    for name, expected in files.items():
        path = BACKUP / name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Protected T1-nomax backup changed: {name}')
    return len(files)


def make_command(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        raise ValueError('name must be a single run name, without directory separators')
    cfg_path = str(BACKUP / 'configs/multiview/T1-nomax.yaml')
    if args.phase == 'train':
        saved = json.loads((BACKUP / ORIGINAL / 'run_summary.json').read_text())['args']
        values = dict(saved, name=args.name, cfg_path=cfg_path, device=args.device,
                      common_init=str(BACKUP / 'logs/p1-common-init/seed0.pth'))
        if args.max_steps is not None:
            if args.max_steps < 1:
                raise ValueError('max_steps must be positive')
            values['max_steps'] = args.max_steps
    else:
        if args.max_steps is not None:
            raise ValueError('max_steps applies only to training')
        values = dict(name=args.name, cfg_path=cfg_path, device=args.device,
                      checkpoint=str(BACKUP / ORIGINAL / 'partial_step_2000.pth'),
                      split='eval', num_views=6, case_limit=6, out_res_scale=1.,
                      eval_npoint=65536, seed=0)
    entry = BACKUP / 'code' / ('train.py' if args.phase == 'train' else 'evaluate.py')
    # -I ignores PYTHONPATH/user site; the only project code path is the frozen snapshot.
    bootstrap = ('import runpy,sys; entry=sys.argv[1]; sys.argv=sys.argv[1:]; '
                 'sys.path.insert(0,sys.argv.pop(1)); runpy.run_path(entry,run_name="__main__")')
    command = [sys.executable, '-I', '-B', '-c', bootstrap, str(entry), str(BACKUP / 'code')]
    for key, value in values.items():
        if value is None or key == 'wandb':
            continue
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        elif isinstance(value, list):
            if value:
                command.extend(['--' + key, *map(str, value)])
        else:
            command.extend(['--' + key, str(value)])
    if args.phase == 'train':
        command.append('--no-wandb')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'evaluate'])
    parser.add_argument('--name', required=True, help='New output directory under logs/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_steps', type=int, help='Optional short training budget; default is original 2000')
    parser.add_argument('--dry-run', action='store_true', help='Verify all backup files and print the command')
    args = parser.parse_args()
    count = verify_backup()
    command = make_command(args)
    print(f'Protected T1-nomax: {count} backup files verified; frozen BN/MSE/6-view protocol.', flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output = ROOT / 'logs' / args.name
    output.mkdir(exist_ok=False)
    provenance = {'manifest_sha256': MANIFEST_SHA256, 'verified_files': count,
                  'source': str(BACKUP / 'code'), 'command': command,
                  'note': 'Frozen source/protocol; CUDA backward is not guaranteed bitwise deterministic.'}
    (output / 'protected_baseline.json').write_text(json.dumps(provenance, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(command, cwd=ROOT, env=env)
    provenance['exit_code'] = result.returncode
    (output / 'protected_baseline.json').write_text(json.dumps(provenance, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
