import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
REPORTS = ROOT / 'reports'
GPU_DEADLINE = None
SCHEMA = {
    'execution': {'seed', 'total_gpu_hours', 'max_concurrent_gpu_runs', 'smoke_steps', 'overfit_steps',
                  'pilot_steps', 'val_case_count', 'run_formal_training', 'checkpoint_policy', 'device'},
    'data': {'root_dir', 'dataset', 'num_views', 'split'},
    'model': {'base_config', 'checkpoint'},
    'evaluation': {'primary_scale', 'diagnostic_scales', 'resampling_protocol', 'latent_modes', 'mc_samples',
                   'posterior_oracle_diagnostics', 'save_float_predictions_for_selected_cases', 'chunk_size'},
    'research': {'gaussian_changes', 'geometry_arms', 'footprint_samples_per_scale',
                 'use_structure_target_as_encoder_input', 'enable_ray_loss_after_calibration',
                 'initial_rays_per_step', 'deterministic_distillation_before_probabilistic',
                 'probabilistic_training_requires_teacher_distribution', 'pilot_arms'},
}
RESULT_FIELDS = 'arm source_commit_or_diff_hash config_hash parent_checkpoint seed case_id split context_view_ids teacher_view_ids heldout_view_ids train_updates loss_protocol eval_scale resampling_protocol latent_mode psnr ssim boundary_error gradient_error projection_error inference_seconds peak_memory_gb params gpu_hours status reason'.split()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def command_output(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def digest_files(paths):
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def validate_plan(plan):
    if set(plan) != set(SCHEMA):
        raise ValueError(f'Plan sections must be exactly {sorted(SCHEMA)}')
    for section, keys in SCHEMA.items():
        unknown = set(plan[section]) - keys
        if unknown:
            raise ValueError(f'Unknown plan.{section} keys: {sorted(unknown)}')
    if plan['data']['split'] != 'eval' or plan['evaluation']['primary_scale'] != 1.:
        raise ValueError('Research selection uses the validation split at scale=1')
    if plan['execution']['run_formal_training']:
        raise ValueError('This bounded runner does not launch formal training')
    if plan['execution']['max_concurrent_gpu_runs'] != 1:
        raise ValueError('Only one sequential GPU run is supported')
    if plan['research']['use_structure_target_as_encoder_input']:
        raise ValueError('GT cannot be an encoder input')
    if plan['research']['footprint_samples_per_scale'] != 5:
        raise ValueError('This implementation uses five footprint samples per scale')
    for key in ('total_gpu_hours', 'pilot_steps', 'smoke_steps', 'overfit_steps', 'val_case_count'):
        if plan['execution'][key] <= 0:
            raise ValueError(f'execution.{key} must be positive')
    if plan['evaluation']['resampling_protocol'] != 'legacy':
        raise ValueError('Only the named legacy resampling protocol is implemented')


def resolve_arms(plan):
    from config_schema import validate_config
    base = yaml.safe_load((ROOT / plan['model']['base_config']).read_text())
    base['dataset']['root_dir'] = plan['data']['root_dir']
    base['loss'] = {'reconstruction_loss': 'mse'}
    base['model'].setdefault('latent', {})['use_latent'] = False
    base['model'].setdefault('sampling', {}).update(eval_mode='prior_mean', num_samples=plan['evaluation']['mc_samples'])
    arms = {}
    for arm in ['L0', 'B_head', 'B_mask', 'CVAE'] + plan['research']['geometry_arms']:
        config = copy.deepcopy(base)
        config['model']['point_decoder']['legacy_head'] = arm == 'L0'
        config['model']['mask_fov'] = arm not in ('L0', 'B_head')
        if arm == 'L0':
            config['model']['sampling']['eval_mode'] = 'prior_sample'
            config['model']['latent']['local_coordinates'] = 'legacy'
        if arm == 'CVAE':
            config['model']['latent']['use_latent'] = True
        if arm.startswith('G'):
            config['model']['anatomy'] = {'enabled': True, 'arm': arm, 'hidden_dim': 32,
                                          'length_mm': 3.2, 'kappa': 4., 'loss_weight': .01}
            config['dataset']['structure_supervision'] = arm in ('Gaux', 'Ganat')
        validate_config(config)
        path = ROOT / 'configs/research/resolved_arm_configs' / f'{arm}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = str(path.relative_to(ROOT))
    return arms


def scan(plan):
    import torch
    import numpy as np
    import pickle
    from utils import sitk_load
    from datasets.base import CBCT_dataset
    root = Path(plan['data']['root_dir']) / plan['data']['dataset']
    metadata = read_json(root / 'meta_info.json', {})
    dependencies = {}
    for name in ('torch', 'numpy', 'scipy', 'SimpleITK', 'pytorch3d', 'PyYAML', 'pytest', 'easydict', 'scikit-image', 'tigre'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = 'NOT_INSTALLED'
    validation = sorted(metadata.get('eval', []))[:plan['execution']['val_case_count']]
    contract = {'root': str(root), 'exists': root.exists(), 'split_counts': {
        split: len(metadata.get(split, [])) for split in ('train', 'eval', 'test')},
        'validation_cases': validation, 'train_cases': sorted(metadata.get('train', [])),
        'normalization': 'CT=clip(HU,-1000,1024) mapped to [0,1], stored uint8; projections=uint8/255*projs_max/0.2',
        'projector': 'external TIGRE Ax, normalized image integrated using metre geometry; not physical attenuation',
        'axes': 'CT/blocks/query XYZ; detector uv=(rotated Z,rotated Y); radians; index_2d transposes H/W',
        'resampling': 'scipy zoom(order=3,prefilter=False,grid_mode=False); endpoint spacing',
        'ssim': '3D skimage SSIM, 7-voxel window; physical window differs across scales'}
    if metadata:
        contract['geometry_config'] = yaml.safe_load((root / metadata['dataset_config']).read_text())
        case = sorted(metadata['train'])[0]
        with (root / metadata['projs'].format(case)).open('rb') as handle:
            projections = pickle.load(handle)
        image, spacing, origin, direction = sitk_load(str(root / metadata['image'].format(case)), return_direction=True)
        blocks = np.load(root / metadata['blocks_coords'], mmap_mode='r')
        contract['sample'] = {'case_id': case, 'ct_shape': list(image.shape), 'ct_minmax_stored': [float(image.min()), float(image.max())],
                              'spacing_mm': spacing.tolist(), 'origin': origin.tolist(), 'direction': direction.tolist(),
                              'projection_shape': list(projections['projs'].shape), 'projection_dtype': str(projections['projs'].dtype),
                              'projs_max': float(projections['projs_max']), 'angles': projections['angles'].tolist(),
                              'blocks_shape': list(blocks.shape), 'blocks_minmax': [float(blocks.min()), float(blocks.max())]}
    try:
        gpu = command_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'])
    except FileNotFoundError:
        gpu = {'returncode': 127, 'stderr': 'nvidia-smi unavailable'}
    manifest = {'git_root': str(ROOT), 'branch': command_output(['git', 'branch', '--show-current'])['stdout'],
                'head': command_output(['git', 'rev-parse', 'HEAD'])['stdout'],
                'worktree': command_output(['git', 'status', '--short'])['stdout'],
                'entrypoint': 'code/', 'historical_copy': 'code copy/ (untouched)',
                'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                'dependencies': dependencies, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                'gpu_scan': gpu, 'data_contract': contract,
                'checkpoints': [str(path.relative_to(ROOT)) for path in (ROOT / 'logs').glob('**/*.pth')],
                'arms': resolve_arms(plan), 'budget': plan['execution']}
    write_json(REPORTS / 'run_manifest.json', manifest)
    return {'status': 'PASS', 'manifest': 'reports/run_manifest.json', 'cuda_available': torch.cuda.is_available()}


def sanity(plan):
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT / 'code'))
    log = REPORTS / 'sanity.log'
    with log.open('w') as handle:
        result = subprocess.run([sys.executable, '-m', 'pytest', 'code/tests', '-q'], cwd=ROOT,
                                env=environment, stdout=handle, stderr=subprocess.STDOUT)
    return {'status': 'PASS' if result.returncode == 0 else 'FAIL', 'log': str(log.relative_to(ROOT)),
            'result': log.read_text().splitlines()[-1] if log.stat().st_size else 'No test output',
            'two_gpu_ddp': 'UNVERIFIED: no two allocated CUDA devices in this run'}


def profile(plan):
    import torch
    import numpy as np
    from easydict import EasyDict
    from models.probabilistic_model import build_model
    from models.gs_utils import query_gs, knn_search
    torch.set_num_threads(1)
    torch.manual_seed(plan['execution']['seed'])
    config = yaml.safe_load((ROOT / 'configs/research/resolved_arm_configs/B_mask.yaml').read_text())
    config['model']['gs']['res'] = 3
    config['model']['image_encoder']['out_ch'] = 16
    config['model']['point_decoder']['mlp_chs'] = [32, 16, 1]
    model = build_model(EasyDict(config['model'])).eval()
    anchors = torch.stack(torch.meshgrid(*([torch.arange(3) / 3] * 3), indexing='ij'), -1).reshape(1, 27, 3)
    data = {'projs': torch.rand(1, 3, 1, 32, 32), 'angles': torch.tensor([[0., 1., 2.]]),
            'points_gs': anchors, 'points_gs_proj': torch.rand(1, 3, 27, 2) * 1.8 - .9,
            'points': torch.rand(1, 128, 3), 'points_proj': torch.rand(1, 3, 128, 2) * 1.8 - .9}
    captured = []
    hook = model.gs_params_mlp.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    started = time.monotonic()
    with torch.no_grad():
        features = model.encode_projs(data)
        prediction = model.forward_points(features, data)
        covariance = torch.linalg.inv(features['gs_params']['inv'])
        eigenvalues = torch.linalg.eigvalsh(covariance)
        line = torch.tensor([.5, .5, .5]).expand(1, 257, 3).clone()
        line[0, :, 0] = torch.linspace(.1, .9, 257)
        neighbors = knn_search(line, anchors, 3)
        line_features = query_gs(line, anchors, features['gs_params'])
    elapsed = time.monotonic() - started
    hook.remove()
    logits = captured[0][:, :6] * config['model']['gs']['t']
    sigmoid = logits.sigmoid()
    result = {'status': 'PASS', 'scope': 'SYNTHETIC_CPU_RANDOM_INITIALIZATION_ONLY',
              'seconds': elapsed, 'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'resolved_model_config': config['model'], 'sigma_over_normalized_pitch': float(eigenvalues.sqrt().mean() * 3),
              'covariance_min_eigenvalue': float(eigenvalues.min()),
              'max_condition_number': float((eigenvalues[..., -1] / eigenvalues[..., 0]).max()),
              'logit_minmax': [float(logits.min()), float(logits.max())],
              'sigmoid_saturation_fraction': float(((sigmoid < .01) | (sigmoid > .99)).float().mean()),
              'neighbor_change_count': int((neighbors[:, 1:] != neighbors[:, :-1]).any(-1).sum()),
              'max_adjacent_feature_jump': float((line_features[..., 1:] - line_features[..., :-1]).abs().max()),
              'finite': bool(torch.isfinite(prediction).all()), 'full_resolution_time': None,
              'peak_gpu_memory_gb': None, 'selected_gaussian_setting': 'legacy K=3/base centers/unnormalized/sqrt(pitch); no candidate superiority established'}
    write_json(REPORTS / 'gaussian_profile.json', result)
    return result


def scale(plan):
    checkpoint_path = plan['model']['checkpoint']
    if not checkpoint_path or not (ROOT / checkpoint_path).exists():
        return {'status': 'BLOCKED', 'reason': 'No trained checkpoint; S1-S5 real-case comparisons require one'}
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from datasets.dst_gs import CBCT_dataset_gs
    from models.probabilistic_model import build_model
    from checkpoints import weights_from_checkpoint
    from utils import load_config, convert_cuda
    from evaluation_protocol import scale_comparisons
    config = load_config(ROOT / plan['model']['base_config'])
    config.dataset.root_dir = plan['data']['root_dir']
    model = build_model(config.model).to(plan['execution']['device']).eval()
    model.load_state_dict(weights_from_checkpoint(torch.load(ROOT / checkpoint_path, map_location='cpu')), strict=True)
    datasets = [CBCT_dataset_gs(dst_name=plan['data']['dataset'], cfg=config.dataset, split='eval',
                                num_views=plan['data']['num_views'], out_res_scale=value) for value in (1., .5)]
    cases = read_json(REPORTS / 'run_manifest.json')['data_contract']['validation_cases']
    for dataset in datasets:
        dataset.name_list = cases
    chunk = plan['evaluation']['chunk_size']
    rows = []
    stopped = False
    with torch.no_grad():
        for full, coarse in zip(*(DataLoader(dataset, batch_size=1) for dataset in datasets)):
            if GPU_DEADLINE is not None and time.monotonic() >= GPU_DEADLINE:
                stopped = True
                break
            full = convert_cuda(full, plan['execution']['device'])
            coarse = convert_cuda(coarse, plan['execution']['device'])
            if getattr(model, 'use_latent', False):
                state = model.encode_and_sample(full, sample_from='mean')
                predict = lambda data, size: model._decode_chunked(data, state, size)
            else:
                features = model.encode_projs(full)
                def predict(data, size):
                    parts = []
                    for start in range(0, data['points'].shape[1], size):
                        portion = {key: value[..., start:start + size, :] if key in model.registered_point_keys else value
                                   for key, value in data.items()}
                        parts.append(model.forward_points(features, portion))
                    return torch.cat(parts, -1)
            full_output = predict(full, chunk)[0, 0].cpu().numpy().reshape(full['points_gt'].shape[1:])
            coarse_output = predict(coarse, chunk)[0, 0].cpu().numpy().reshape(coarse['points_gt'].shape[1:])
            measured = scale_comparisons(full['points_gt'][0].cpu().numpy(), full_output, coarse_output, full['spacing'][0].cpu().numpy())
            repeated = predict(coarse, max(1, chunk // 2))[0, 0].cpu().numpy().reshape(coarse_output.shape)
            np.testing.assert_allclose(repeated, coarse_output, atol=1e-5, rtol=1e-4)
            measured['S5'] = {'max_abs_chunk_difference': float(np.max(np.abs(repeated - coarse_output)))}
            for comparison, values in measured.items():
                rows.append(dict(case_id=full['name'][0], comparison=comparison, status='PASS', **values))
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['case_id', 'comparison', 'status']
    with (REPORTS / 'scale_diagnosis.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {'status': 'BLOCKED_BUDGET' if stopped else 'PASS', 'cases': cases, 'output': 'reports/scale_diagnosis.csv'}


def pilot(plan, state, save_state):
    import torch
    if not plan['execution']['device'].startswith('cuda') or not torch.cuda.is_available():
        return {'status': 'BLOCKED', 'reason': 'Bounded real-data pilots require an allocated available CUDA device; CPU sanity continues'}
    if state.get('sanity', {}).get('status') != 'PASS':
        return {'status': 'BLOCKED', 'reason': 'Sanity gate has not passed for current inputs'}
    runs = state.setdefault('pilot_runs', {})
    for arm in plan['research']['pilot_arms']:
        config_path = ROOT / 'configs/research/resolved_arm_configs' / f'{arm}.yaml'
        if not config_path.exists():
            raise ValueError(f'Unknown pilot arm {arm}')
        if arm in ('L0', 'CVAE'):
            raise ValueError('Pilot selection currently requires deterministic repaired arms')
        run_key = f'{arm}_{state["input_hash"][:12]}'
        if run_key in runs:
            continue
        remaining = plan['execution']['total_gpu_hours'] - state.get('gpu_hours_consumed', 0.)
        if remaining <= 0:
            return {'status': 'BLOCKED_BUDGET', 'runs': runs}
        seconds = min(remaining * 3600, 1800.)
        name = f'research-{run_key}'
        command = [sys.executable, 'code/train.py', '--cfg_path', str(config_path), '--name', name,
                   '--dst_name', plan['data']['dataset'], '--num_views', str(plan['data']['num_views']),
                   '--batch_size', '1', '--num_workers', '0', '--num_points', '4096', '--no-wandb',
                   '--val_case_count', str(plan['execution']['val_case_count']),
                   '--seed', str(plan['execution']['seed']), '--max_steps', str(plan['execution']['pilot_steps']),
                   '--max_seconds', str(max(1, seconds - 60)), '--device', plan['execution']['device']]
        runs[run_key] = {'status': 'RUNNING', 'command': command, 'reserved_gpu_hours': seconds / 3600}
        state['gpu_hours_consumed'] = state.get('gpu_hours_consumed', 0.) + seconds / 3600
        save_state()
        started = time.monotonic()
        try:
            with (REPORTS / f'{run_key}.log').open('w') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=seconds)
            runs[run_key]['status'] = 'TRAINED_UNEVALUATED' if result.returncode == 0 else 'FAIL'
            runs[run_key]['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            runs[run_key]['status'] = 'STOPPED_BUDGET'
        elapsed = time.monotonic() - started
        state['gpu_hours_consumed'] += elapsed / 3600 - seconds / 3600
        runs[run_key]['elapsed_seconds'] = elapsed
        runs[run_key]['summary'] = read_json(ROOT / 'logs' / name / 'run_summary.json')
        save_state()
        summary = runs[run_key]['summary']
        checkpoints = sorted((ROOT / 'logs' / name).glob('*.pth'), key=lambda path: path.stat().st_mtime_ns)
        remaining = plan['execution']['total_gpu_hours'] - state.get('gpu_hours_consumed', 0.)
        if not summary or not checkpoints or remaining <= 0:
            runs[run_key]['evaluation_status'] = 'BLOCKED_CHECKPOINT_OR_BUDGET'
            continue
        evaluation_seconds = min(remaining * 3600, 1800.)
        evaluation_command = [sys.executable, 'code/evaluate.py', '--name', name, '--epoch', '0',
                              '--cfg_path', str(config_path), '--checkpoint', str(checkpoints[-1]),
                              '--dst_name', plan['data']['dataset'], '--split', 'eval',
                              '--num_views', str(plan['data']['num_views']), '--out_res_scale', '1.0',
                              '--latent_mode', 'prior_mean', '--seed', str(plan['execution']['seed']),
                              '--case_limit', str(plan['execution']['val_case_count']),
                              '--eval_npoint', str(plan['evaluation']['chunk_size']), '--device', plan['execution']['device']]
        runs[run_key]['evaluation_command'] = evaluation_command
        runs[run_key]['evaluation_status'] = 'RUNNING'
        state['gpu_hours_consumed'] += evaluation_seconds / 3600
        save_state()
        evaluation_started = time.monotonic()
        try:
            with (REPORTS / f'{run_key}_eval.log').open('w') as handle:
                evaluated = subprocess.run(evaluation_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           timeout=evaluation_seconds)
            runs[run_key]['evaluation_status'] = 'PASS' if evaluated.returncode == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            runs[run_key]['evaluation_status'] = 'STOPPED_BUDGET'
        state['gpu_hours_consumed'] += (time.monotonic() - evaluation_started - evaluation_seconds) / 3600
        if runs[run_key]['evaluation_status'] == 'PASS':
            result_path = ROOT / 'logs' / name / 'results/ep_0' / f'cases_1.00x_prior_mean_seed{plan["execution"]["seed"]}.json'
            results = read_json(result_path)
            cases = [case for group in results.values() for case in group]
            runs[run_key]['comparison'] = {'train_updates': summary['optimizer_updates'], 'seed': plan['execution']['seed'],
                                          'loss_protocol': 'mse', 'eval_scale': 1., 'resampling_protocol': 'legacy',
                                          'latent_mode': 'prior_mean', 'cases': cases}
            table = REPORTS / 'experiment_results.csv'
            write_header = not table.exists()
            with table.open('a') as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                if write_header:
                    writer.writeheader()
                for case in cases:
                    row = {key: 'NA' for key in RESULT_FIELDS}
                    row.update(arm=arm, source_commit_or_diff_hash=state['input_hash'], config_hash=digest_files([config_path]),
                               parent_checkpoint=summary['parent_checkpoint'], seed=plan['execution']['seed'], case_id=case['name'],
                               split='eval', context_view_ids=json.dumps(case['view_ids']), train_updates=summary['optimizer_updates'],
                               loss_protocol='mse', eval_scale=1., resampling_protocol='legacy', latent_mode='prior_mean',
                               psnr=case['psnr'], ssim=case['ssim'], boundary_error=case['boundary_error'], gradient_error=case['gradient_error'],
                               inference_seconds=case['inference_seconds'], peak_memory_gb=case['peak_memory_gb'],
                               gpu_hours=(elapsed + time.monotonic() - evaluation_started) / 3600,
                               status='PILOT_MEASURED_NOT_FORMAL', reason='Projection metric unavailable: operator uncalibrated')
                    writer.writerow(row)
        save_state()
    from research_gates import pilot_gate
    reference_key = f'{plan["research"]["pilot_arms"][0]}_{state["input_hash"][:12]}'
    reference = runs.get(reference_key, {}).get('comparison')
    gates = {}
    if reference:
        for arm in plan['research']['pilot_arms'][1:]:
            candidate = runs.get(f'{arm}_{state["input_hash"][:12]}', {}).get('comparison')
            if candidate:
                gates[arm] = pilot_gate(reference, candidate, minimum_cases=6)
    return {'status': 'INCONCLUSIVE' if not gates else 'PILOT_EVALUATED',
            'reason': 'Only matched full-resolution validation can promote a pilot; formal training remains disabled',
            'gates': gates, 'runs': runs}


def report(plan, state):
    manifest = read_json(REPORTS / 'run_manifest.json', {})
    for name, fields in [
        ('experiment_results.csv', RESULT_FIELDS),
        ('scale_diagnosis.csv', ['case_id', 'comparison', 'psnr', 'ssim', 'unclipped_mse', 'status', 'reason']),
        ('latent_diagnostics.csv', ['case_id', 'latent_mode', 'USES_GT', 'KL_global_raw', 'KL_global_freebits_penalty',
                                   'prior_variance_g_mean', 'post_variance_g_mean', 'sample_diversity', 'status', 'reason']),
    ]:
        path = REPORTS / name
        if not path.exists():
            with path.open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
    stages = {
        'P0': {'status': state.get('scan', {}).get('status', 'NOT_RUN')},
        'P1': {'status': 'IMPLEMENTED_CPU_TESTED' if state.get('sanity', {}).get('status') == 'PASS' else 'NEEDS_VALIDATION',
               'reason': 'CPU Gloo two-rank smoke passed with a small encoder; real legacy checkpoint and two-GPU validation unavailable'},
        'P2': {'status': state.get('scale', {}).get('status', 'NOT_RUN'), 'reason': state.get('scale', {}).get('reason')},
        'P3': {'status': state.get('profile', {}).get('status', 'NOT_RUN'), 'scope': 'synthetic; retain legacy Gaussian settings'},
        'P4': {'status': 'IMPLEMENTED_NOT_PILOTED', 'reason': 'CUDA unavailable; no CT quality claims'},
        'P5': {'status': 'BLOCKED_CALIBRATION', 'reason': 'Quadrature phantoms implemented; real TIGRE/GT projection matching not validated'},
        'P6': {'status': 'BLOCKED_TEACHER', 'reason': 'No CVAE or validated superior dense checkpoint; paired interface and deterministic feature KD implemented'},
        'P7': {'status': 'BLOCKED_DISTRIBUTIONAL_TEACHER', 'reason': 'P6 evidence and trained distributional teacher absent'},
    }
    state['stages'] = stages
    lines = [
        '# Research execution report', '',
        'Only implementation and synthetic CPU evidence are currently established. No repaired CT baseline, scale-gap attribution, anatomy gain, calibrated ray consistency, or transfer gain is claimed.', '',
        f"Source: {manifest.get('head', 'unknown')}; current input hash: {state['input_hash']}.",
        f"Tests: {state.get('sanity', {}).get('result', 'NOT_RUN')}.",
        f"GPU hours consumed/reserved: {state.get('gpu_hours_consumed', 0.):.6f}; remaining: {max(0, plan['execution']['total_gpu_hours'] - state.get('gpu_hours_consumed', 0.)):.6f}.", '',
        '| Stage | Status | Limitation |', '|---|---|---|',
    ]
    lines.extend(f"| {phase} | {entry['status']} | {entry.get('reason', entry.get('scope', ''))} |" for phase, entry in stages.items())
    lines += ['', '## Evidence and protocol', '',
              '- See code_audit.md, implementation_summary.md, run_manifest.json, sanity.log, ddp_smoke.json and gaussian_profile.json.',
              '- S1=full prediction/full GT; S2=filtered full prediction/filtered GT; S3=direct coarse prediction/coarse GT; S4=coarse point prediction/filtered full prediction; S5=same points/state with different chunks. Real measurements are blocked without a checkpoint.',
              '- Shared Gaussian setting remains base-center KNN, K=3, unnormalized density and sqrt(pitch) sigma. Synthetic diagnostics cannot select a superior trained arm.',
              '- G0/Gfree/Ggeo/Gaux/Ganat share 3D context and two feature levels; support covariance is independent of GS interpolation covariance. T is supervised only in Gaux/Ganat and controls support only in Ganat.',
              '- Physical structure targets use 0.25-scale GT, endpoint-adjusted spacing, sigma=max(spacing) mm and rho=2*sigma; confidence is GT-derived. This is coarse boundary supervision, not evidence of fine vessel topology.',
              '- Ray backend is midpoint quadrature of final scalar CT values with mm ray lengths; it requires mm^-1 attenuation. Stored data instead integrates HU-clipped normalized intensities with metre-based TIGRE geometry and a /0.2 projection scale. Real calibration is BLOCKED, so ray training is disabled.',
              '- CVAE oracle explicitly USES_GT. Paired dense/sparse inputs and frozen teacher feature transfer are separate; no validated teacher exists. P7 is not trained or claimed implemented.',
              '- CSV files with only headers have no measured cases; no zero or invented metrics stand in for missing results. No patient figures are generated without actual predictions.',
              '- Legacy checkpoints require legacy_head=true and mask_fov=false; local legacy latent requires local_coordinates=legacy. Strict synthetic compatibility is tested; actual checkpoint reproduction is unavailable.',
              '- Resume restores full epoch-boundary state including optimizer/scheduler/global step/rank RNG. Partial-step snapshots are explicitly warm-start-only.', '',
              '## Reproduce', '', '```bash', f'PYTHON={sys.executable}']
    for phase in ('scan', 'sanity', 'scale', 'profile', 'pilot', 'report'):
        lines.append(f'$PYTHON scripts/run_research_plan.py --plan configs/research/plan.yaml --phase {phase}')
    lines += ['```', '', '## Next actions', '',
              '1. Restore an allocated CUDA environment, run the two-rank smoke, and train matched B_head/B_mask within the persisted budget.',
              '2. Supply a checkpoint with its exact config, then run S1–S5 and actual CVAE diagnostics on the frozen validation manifest.',
              '3. Calibrate TIGRE against GT and validate a dense teacher before enabling the gated ray/transfer branches.', '']
    (REPORTS / 'final_report.md').write_text('\n'.join(lines))
    return {'status': 'PASS', 'output': 'reports/final_report.md', 'meaning': 'Reporting complete; blocked experiments remain blocked'}


def main():
    global GPU_DEADLINE
    parser = argparse.ArgumentParser(description='Bounded, restartable DIF-Gaussian research runner')
    parser.add_argument('--plan', default='configs/research/plan.yaml')
    parser.add_argument('--phase', required=True, choices=['scan', 'sanity', 'scale', 'profile', 'pilot', 'report'])
    parser.add_argument('--force', action='store_true', help='Rerun this phase; never resets accumulated GPU budget')
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = yaml.safe_load(plan_path.read_text())
    validate_plan(plan)
    REPORTS.mkdir(exist_ok=True)
    state_path = REPORTS / 'execution_status.json'
    state = read_json(state_path, {'gpu_hours_consumed': 0.})
    source_paths = list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    root = Path(plan['data']['root_dir']) / plan['data']['dataset']
    paths = source_paths + [plan_path, ROOT / plan['model']['base_config']]
    paths += [path for path in (root / 'meta_info.json', root / 'config.yaml') if path.exists()]
    if plan['model']['checkpoint'] and (ROOT / plan['model']['checkpoint']).exists():
        paths.append(ROOT / plan['model']['checkpoint'])
    data_stamps = []
    metadata = read_json(root / 'meta_info.json', {})
    for case in sorted(set(metadata.get('train', []) + metadata.get('eval', []))):
        for kind in ('image', 'projs'):
            path = root / metadata[kind].format(case)
            if path.exists():
                stat = path.stat()
                data_stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    runtime = {'python': sys.executable, 'version': sys.version,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    input_hash = hashlib.sha256((digest_files(paths) + json.dumps(data_stamps) + json.dumps(runtime)).encode()).hexdigest()
    for phase in ('scan', 'sanity', 'scale', 'profile', 'pilot'):
        entry = state.get(phase, {})
        if entry.get('input_hash') != input_hash and entry.get('status') == 'PASS':
            entry['status'] = 'STALE'
    state['input_hash'] = input_hash
    previous = state.get(args.phase, {})
    if previous.get('input_hash') == input_hash and previous.get('status') == 'PASS' and not args.force and args.phase not in ('scan', 'report'):
        print(json.dumps({'status': 'CACHED_PASS', 'phase': args.phase}))
        return
    save_state = lambda: write_json(state_path, state)
    started = time.monotonic()
    reservation = 0.
    try:
        if args.phase == 'scale' and plan['execution']['device'].startswith('cuda'):
            reservation = max(0., plan['execution']['total_gpu_hours'] - state.get('gpu_hours_consumed', 0.))
            if reservation <= 0:
                raise RuntimeError('GPU budget exhausted; no new scale run may start')
            state['gpu_hours_consumed'] = state.get('gpu_hours_consumed', 0.) + reservation
            GPU_DEADLINE = started + reservation * 3600
            save_state()
        if args.phase in ('profile', 'pilot'):
            resolve_arms(plan)
        if args.phase == 'pilot':
            if state.get('sanity', {}).get('input_hash') != input_hash:
                result = {'status': 'BLOCKED', 'reason': 'Sanity results are stale or absent'}
            else:
                result = pilot(plan, state, save_state)
        elif args.phase == 'report':
            result = report(plan, state)
        else:
            result = globals()[args.phase](plan)
    except Exception as error:
        result = {'status': 'FAIL', 'reason': f'{type(error).__name__}: {error}'}
    if reservation:
        state['gpu_hours_consumed'] += (time.monotonic() - started) / 3600 - reservation
    state[args.phase] = dict(result, input_hash=input_hash, elapsed_seconds=time.monotonic() - started)
    save_state()
    print(json.dumps(state[args.phase], indent=2))
    if result['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
