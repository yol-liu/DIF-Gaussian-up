import json
import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import DIF_Gaussian
from training_diagnostics import add_structure_loss, transformer_gradient_norms
from evaluation_protocol import evaluating
from test_multiview_transformer import small_config, synthetic_data, without_gt


def main():
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    torch.manual_seed(26)
    config = small_config('T4')
    model = DIF_Gaussian(config.model)
    wrapped = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
    recorded = []
    for step in range(2):
        data = synthetic_data(seed=rank + 51)
        prediction = wrapped(data, optimizer_step=10)
        reconstruction = (prediction['points_pred'] - data['points_gt']).square().mean()
        loss, _ = add_structure_loss(prediction, reconstruction, {}, config.model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = transformer_gradient_norms(model)
        assert all(value > 0 for value in gradients.values()), gradients
        optimizer.step()
        recorded.append(dict(step=step, loss=float(loss.detach()), gradients=gradients))
        torch.distributed.barrier()
        if rank == 0:
            with evaluating(model), torch.no_grad():
                model(without_gt(data), is_eval=True, eval_npoint=7)
            assert model.training
        torch.distributed.barrier()
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=0)
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
    if rank == 0:
        result = dict(status='PASS', backend='CPU_GLOO', ranks=2, updates=2,
                      scope='actual UNet + T4 Transformer/point fusion/anatomy; small spatial sizes; not CUDA/NCCL',
                      parameters_identical=True, rank0_validation_completed=True, steps=recorded)
        os.makedirs('reports/multiview', exist_ok=True)
        with open('reports/multiview/ddp_smoke.json', 'w') as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
