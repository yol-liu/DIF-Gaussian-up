"""Two-rank CPU/Gloo integration of the real probability path with a tiny image encoder."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_probability_repair import TinyEncoder, config, paired_data
from models.probabilistic_model import build_model
from probabilistic_loss import probability_loss
from evaluation_protocol import evaluating


def main():
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    torch.manual_seed(72)
    with patch('models.base.UNet', TinyEncoder):
        model = build_model(config(local=True))
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=.001)
    for step in range(3):
        data = paired_data()
        data['points_gt'] = data['points_gt'] + rank*.02
        output = wrapped(data)
        loss, _ = probability_loss(output, data['points_gt'], model.probability, step+1)
        optimizer.zero_grad(); loss.backward()
        assert model.prior_head.net[-1].weight.grad.abs().sum() > 0
        assert model.posterior_head.net[-1].weight.grad.abs().sum() > 0
        optimizer.step()
        torch.distributed.barrier()
        if rank == 0:
            with evaluating(model), torch.no_grad():
                model(data, is_eval=True, num_samples=2, eval_npoint=5)
        torch.distributed.barrier()
    for parameter in model.parameters():
        other = parameter.detach().clone()
        torch.distributed.broadcast(other, src=0)
        torch.testing.assert_close(other, parameter, atol=0, rtol=0)
    if rank == 0:
        report = {'status': 'PASS', 'backend': 'CPU_GLOO', 'ranks': 2, 'steps': 3,
                  'global_local': True, 'train_samples': 2, 'parameters_identical': True,
                  'scope': 'real probability forward/loss; tiny image encoder; not two-GPU NCCL'}
        Path('reports/probabilistic/ddp.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
