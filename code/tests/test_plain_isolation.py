"""Plain training must work even when probability modules cannot be imported."""
import os
from pathlib import Path
import subprocess
import sys


def test_plain_entry_points_without_probability_imports():
    root = Path(__file__).resolve().parents[2]
    program = r'''
import importlib.abc
import sys
sys.path.insert(0, 'code')
class BlockProbability(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('probabilistic_') or fullname in (
            'models.probabilistic_model', 'models.dense_probability', 'models.latent_np', 'models.gt_encoder'):
            raise AssertionError('Plain path imported ' + fullname)
sys.meta_path.insert(0, BlockProbability())
import torch
import train
import evaluate
from models.factory import build_model
from models.model import DIF_Gaussian
from utils import load_config
from losses import volume_nll
cfg = load_config('configs/multiview/T1-nomax.yaml')
torch.manual_seed(0)
model = build_model(cfg.model).train()
rng = torch.get_rng_state()
torch.manual_seed(0)
direct = DIF_Gaussian(cfg.model).train()
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, direct.state_dict()[k]) for k, v in model.state_dict().items())
assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()) == 19
assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()) == 6
x = torch.tensor([1., 2.], requires_grad=True)
loss = volume_nll(x, torch.zeros_like(x), 'mse')
loss.backward()
assert torch.equal(x.grad, torch.tensor([1., 2.]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', program], cwd=root,
                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_protected_launcher_rejects_probability_options_and_unsafe_names():
    import argparse
    import importlib.util
    import pytest
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('protected_baseline', root / 'scripts/run_t1_nomax.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(phase='train', name='test-plain', device='cpu', max_steps=5)
    command = module.make_command(args)
    assert '-I' in command and '-B' in command
    assert str(module.BACKUP / 'code/train.py') in command
    assert command[command.index('--cfg_path') + 1] == str(module.BACKUP / 'configs/multiview/T1-nomax.yaml')
    assert '--amp' not in command and '--random_views' not in command
    args.name = '../overwrite'
    with pytest.raises(ValueError):
        module.make_command(args)
    result = subprocess.run([sys.executable, str(root / 'scripts/run_t1_nomax.py'),
                             'train', '--name', 'test', '--cfg_path', 'probability.yaml'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'unrecognized arguments' in result.stderr
