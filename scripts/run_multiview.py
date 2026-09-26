import argparse
import gc
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
REPORTS = ROOT / 'reports/multiview'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def source_hash():
    digest = hashlib.sha256()
    paths = list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'configs/multiview').glob('*.yaml')) + [Path(__file__).resolve()]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def synthetic_batch(config, batch=2, views=3, queries=128, image_size=32):
    import torch
    from geometry_adapter import GeometryAdapter
    resolution = config.model.gs.res
    anchors = torch.stack(torch.meshgrid(*([torch.arange(resolution) / resolution] * 3), indexing='ij'), -1).reshape(1, -1, 3)
    data = dict(projs=torch.rand(batch, views, 1, image_size, image_size),
                angles=torch.linspace(.2, 2., views)[None].expand(batch, -1),
                points=torch.rand(batch, queries, 3), points_gs=anchors.expand(batch, -1, -1),
                points_gt=torch.rand(batch, 1, queries),
                volume_extent=torch.tensor([[409.6, 409.6, 409.6]]).expand(batch, -1),
                detector_extent=torch.tensor([[614.4, 614.4]]).expand(batch, -1),
                detector_resolution=torch.tensor([[256., 256.]]).expand(batch, -1),
                DSO=torch.full((batch,), 800.), DSD=torch.full((batch,), 1200.),
                structure_target=torch.diag(torch.tensor([1., 0., 0.])).expand(batch, resolution ** 3, -1, -1),
                structure_confidence=torch.ones(batch, resolution ** 3))
    adapter = GeometryAdapter(data)
    data['points_gs_proj'] = adapter.project(data['points_gs'])['uv']
    data['points_proj'] = adapter.project(data['points'])['uv']
    return data


def check():
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (REPORTS / 'tests.log').open('w') as handle:
        result = subprocess.run([sys.executable, '-m', 'pytest', 'code/tests', '-q'], cwd=ROOT, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT)
    lines = (REPORTS / 'tests.log').read_text().splitlines()
    return dict(status='PASS' if result.returncode == 0 else 'FAIL', result=lines[-1] if lines else 'empty log',
                log='reports/multiview/tests.log')


def synthetic(arms, seed):
    import torch
    from models.probabilistic_model import build_model
    from training_diagnostics import add_structure_loss, transformer_gradient_norms
    from utils import load_config
    torch.set_num_threads(1)
    results = []
    for arm in arms:
        torch.manual_seed(seed)
        config = load_config(ROOT / 'configs/multiview' / f'{arm}.yaml')
        model = build_model(config.model).train()
        data = synthetic_batch(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        records = []
        started = time.monotonic()
        for step in range(2):
            prediction = model(data, optimizer_step=step + 100)
            reconstruction = (prediction['points_pred'] - data['points_gt']).square().mean()
            loss, metrics = add_structure_loss(prediction, reconstruction, {'reconstruction': reconstruction.detach()}, config.model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = transformer_gradient_norms(model)
            assert torch.isfinite(loss)
            assert all(value > 0 for value in gradients.values()), gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            records.append(dict(step=step, loss=float(loss.detach()),
                                losses={key: float(value) for key, value in metrics.items()}, gradients=gradients,
                                diagnostics={key: float(value) for key, value in prediction.get('diagnostics', {}).items()}))
        training_seconds = time.monotonic() - started
        model.eval()
        data = {key: value for key, value in data.items() if key not in ('points_gt', 'structure_target', 'structure_confidence')}
        with torch.no_grad():
            started = time.monotonic()
            full = model(data, is_eval=True, eval_npoint=128)['points_pred']
            inference_seconds = time.monotonic() - started
            chunked = model(data, is_eval=True, eval_npoint=31)['points_pred']
            torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-4)
        result = dict(arm=arm, status='PASS', scope='synthetic CPU, actual configured 12^3 anchors and 128-d tokens; 32x32 projections',
                      parameters=sum(parameter.numel() for parameter in model.parameters()), train_updates=2,
                      training_seconds=training_seconds, inference_seconds=inference_seconds, peak_gpu_memory_gb=None,
                      max_chunk_error=float((full - chunked).abs().max()), gt_free_inference=True,
                      resolved_config=dict(config), steps=records, seed=seed)
        save_json(REPORTS / f'synthetic_{arm}.json', result)
        results.append({key: value for key, value in result.items() if key not in ('resolved_config', 'steps')})
        del model, optimizer, prediction, loss, full, chunked, data
        gc.collect()
    return dict(status='PASS', results=results)


def training_command(arm, name, steps, seconds, seed, device):
    return [sys.executable, 'code/train.py', '--cfg_path', f'configs/multiview/{arm}.yaml', '--name', name,
            '--dst_name', 'LUNA16', '--num_views', '6', '--batch_size', '1', '--num_workers', '0',
            '--num_points', '1024', '--no-wandb', '--diag_every', '1', '--eval_scale', '1.0',
            '--val_case_count', '6', '--seed', str(seed), '--device', device,
            '--max_steps', str(steps), '--max_seconds', str(seconds)]


def smoke(arms, seed, device, state, persist):
    import torch
    if not device.startswith('cuda') or not torch.cuda.is_available():
        return dict(status='BLOCKED_NO_CUDA', reason='Real-data optimizer smoke requires one allocated working CUDA device; synthetic CPU remains available')
    if state.get('check', {}).get('status') != 'PASS' or state['check'].get('source_hash') != state['source_hash']:
        return dict(status='BLOCKED_TESTS', reason='Run --phase check for the current source first')
    runs = state.setdefault('smoke_runs', {})
    for arm_index, arm in enumerate(arms):
        key = f'{arm}-{state["source_hash"][:10]}'
        if key in runs:
            continue
        steps_left = 100 - state.get('real_steps_consumed_or_reserved', 0)
        seconds_left = 1200 - state.get('real_seconds_consumed_or_reserved', 0.)
        if steps_left <= 0 or seconds_left <= 0:
            return dict(status='STOPPED_BUDGET', reason='Cumulative budget is 100 optimizer updates or 1200 seconds, whichever comes first', runs=runs)
        allocated_steps = max(1, steps_left // (len(arms) - arm_index))
        allocated_seconds = seconds_left / (len(arms) - arm_index)
        name = f'multiview-smoke-{key}-seed{seed}'
        command = training_command(arm, name, allocated_steps, max(1, allocated_seconds - 30), seed, device)
        runs[key] = dict(status='RUNNING', command=command, reserved_steps=allocated_steps, reserved_seconds=allocated_seconds)
        state['real_steps_consumed_or_reserved'] = state.get('real_steps_consumed_or_reserved', 0) + allocated_steps
        state['real_seconds_consumed_or_reserved'] = state.get('real_seconds_consumed_or_reserved', 0.) + allocated_seconds
        persist()
        started = time.monotonic()
        try:
            with (REPORTS / f'{key}.log').open('w') as handle:
                process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=allocated_seconds)
            runs[key]['status'] = 'COMPLETED_SMOKE_ONLY' if process.returncode == 0 else 'FAIL'
            runs[key]['returncode'] = process.returncode
        except subprocess.TimeoutExpired:
            runs[key]['status'] = 'STOPPED_WALLTIME'
        elapsed = time.monotonic() - started
        summary = load_json(ROOT / 'logs' / name / 'run_summary.json')
        runs[key]['summary'] = summary
        runs[key]['elapsed_seconds'] = elapsed
        state['real_seconds_consumed_or_reserved'] += elapsed - allocated_seconds
        if summary:
            state['real_steps_consumed_or_reserved'] += summary['optimizer_updates'] - allocated_steps
        persist()
    return dict(status='SMOKE_ONLY', runs=runs, quality_claim=False)


def commands(arms, seed, device):
    lines = []
    for arm in arms:
        lines.append(shlex.join(training_command(arm, f'multiview-{arm}-smoke-seed{seed}', 50, 600, seed, device)))
        for scale in (.5, 1.):
            lines.append(shlex.join([sys.executable, 'code/evaluate.py', '--name', f'multiview-{arm}-smoke-seed{seed}',
                                     '--cfg_path', f'configs/multiview/{arm}.yaml', '--checkpoint',
                                     f'logs/multiview-{arm}-smoke-seed{seed}/partial_step_50.pth', '--split', 'eval',
                                     '--num_views', '6', '--case_limit', '6', '--out_res_scale', str(scale),
                                     '--eval_npoint', '8192', '--seed', str(seed), '--device', device, '--save_results']))
    path = REPORTS / 'commands.txt'
    path.write_text('\n'.join(lines) + '\n')
    return dict(status='PASS', path=str(path.relative_to(ROOT)), note='Evaluation requires the checkpoint actually produced; use the printed run summary if a wall-time limit stops before step 50')


def report(state):
    legacy = load_json(REPORTS / 'legacy_regression.json', {})
    lines = ['# 多视角 Transformer 验证状态', '',
             f'当前源码与配置 hash：`{state["source_hash"]}`。', '',
             '| 项目 | 状态 | 证据/边界 |', '|---|---|---|']
    for phase in ('check', 'synthetic', 'smoke'):
        result = state.get(phase, {})
        status = result.get('status', 'NOT_RUN')
        if result.get('source_hash') != state['source_hash'] and result:
            status = 'STALE'
        lines.append(f'| {phase} | {status} | {result.get("result", result.get("reason", "详见相应 JSON/log"))} |')
    lines += [f'| 旧模式随机 checkpoint | {"PASS" if legacy.get("results") else "NOT_RUN"} | legacy_regression.json；不代表真实已训练 checkpoint |',
              '| CUDA/NCCL、FP16 GPU | 未实测 | 当前 CUDA 不可用 |',
              '| 真实 CT 性能与消融排名 | 未运行 | 没有新 trained checkpoint；合成 smoke 不证明质量提升 |',
              '| Ray consistency | BLOCKED_CALIBRATION | 缺真实 GT/TIGRE/最终 INR 的投影域及量化匹配验证，误开启报错 |',
              '| Sparse–dense 对齐 | BLOCKED_TEACHER | 无合格 dense teacher；数据/调用/梯度与 provenance 门槛有测试 |', '',
              f'真实预算累计 updates（含中断保守预留）：{state.get("real_steps_consumed_or_reserved", 0)}/100；秒：{state.get("real_seconds_consumed_or_reserved", 0.)}/1200。', '',
              '实现、shape/单位、配置和命令说明见 ../multiview_implementation.md。',
              'T2/T4 的 synthetic_*.json 记录真实参数量、CPU 时长、loss、有效采样率、view attention 集中度、token 方差、残差比、footprint 与梯度范数。',
              '完整分辨率 GPU 时间和显存没有测量，不能从小投影 CPU smoke 外推。', '']
    (REPORTS / 'validation_report.md').write_text('\n'.join(lines))
    return dict(status='PASS', output='reports/multiview/validation_report.md')


def main():
    parser = argparse.ArgumentParser(description='Multiview Transformer verification and bounded smoke; default arms T2/T4 only')
    parser.add_argument('--phase', required=True, choices=['check', 'synthetic', 'smoke', 'commands', 'report'])
    parser.add_argument('--arms', nargs='+', choices=['B0', 'B1', 'T1', 'T2', 'T3', 'T4'], default=['T2', 'T4'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--force', action='store_true', help='Rerun checks; does not reset real GPU budget or retry failed training')
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    state_path = REPORTS / 'execution_status.json'
    state = load_json(state_path, {})
    state['source_hash'] = source_hash()
    previous = state.get(args.phase, {})
    inputs = dict(arms=args.arms, seed=args.seed, device=args.device)
    if args.phase in ('check', 'synthetic') and not args.force and previous.get('status') == 'PASS' and previous.get('source_hash') == state['source_hash'] and previous.get('inputs') == inputs:
        print(json.dumps(dict(status='CACHED_PASS', phase=args.phase)))
        return
    persist = lambda: save_json(state_path, state)
    started = time.monotonic()
    try:
        if args.phase == 'check':
            result = check()
        elif args.phase == 'synthetic':
            result = synthetic(args.arms, args.seed)
        elif args.phase == 'smoke':
            result = smoke(args.arms, args.seed, args.device, state, persist)
        elif args.phase == 'commands':
            result = commands(args.arms, args.seed, args.device)
        else:
            result = report(state)
    except Exception as error:
        import traceback
        (REPORTS / f'{args.phase}_error.log').write_text(traceback.format_exc())
        result = dict(status='FAIL', reason=f'{type(error).__name__}: {error}')
    state[args.phase] = dict(result, source_hash=state['source_hash'], inputs=inputs, elapsed_seconds=time.monotonic() - started)
    persist()
    print(json.dumps(state[args.phase], indent=2))
    if result['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
