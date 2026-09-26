import json
import os
import sys

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models.base
from models.probabilistic_model import ProbabilisticDIFGaussian
from losses import elbo_loss
from evaluation_protocol import evaluating
from test_latent import make_cfg, make_data


class TinyEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch
        self.ds_ch = out_ch
        self.layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.SiLU())

    def forward(self, projections):
        batch, views = projections.shape[:2]
        features = self.layer(projections.flatten(0, 1))
        features = features.reshape(batch, views, *features.shape[1:])
        return {'feats': features, 'feats_ds': features}


def main():
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    torch.manual_seed(42)
    models.base.UNet = TinyEncoder
    config = make_cfg(use_latent=True)
    config.gs.t = 1.
    config.latent.free_bits_g = 0.
    config.latent.kl_warmup_steps = 0
    model = ProbabilisticDIFGaussian(config)
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=.01)
    saw_gt_gradient = False
    saw_latent_gradient = False
    for step in range(4):
        data = make_data(B=2, N=17, with_gt_volume=True, seed=100 + rank)
        prediction = wrapped(data, gt_volume=data['gt_volume'], sample_from='posterior')
        assert 'state' not in prediction
        loss, _ = elbo_loss(prediction['points_pred'], data['points_gt'], prediction['kl_info'], config.latent)
        optimizer.zero_grad()
        loss.backward()
        saw_gt_gradient |= any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                               for parameter in model.gt_encoder.parameters())
        saw_latent_gradient |= any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                                   for parameter in model.global_latent.parameters())
        optimizer.step()
        torch.distributed.barrier()
        if rank == 0:
            with evaluating(model), torch.no_grad():
                model(data, is_eval=True, latent_mode='prior_mean', eval_npoint=5)
            assert model.training
        torch.distributed.barrier()
    assert saw_gt_gradient and saw_latent_gradient
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=0)
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
    if rank == 0:
        print(json.dumps({'status': 'PASS', 'backend': 'CPU_GLOO', 'ranks': 2, 'updates': 4,
                          'gt_encoder_gradient': saw_gt_gradient, 'latent_gradient': saw_latent_gradient,
                          'scope': 'real probabilistic forward/loss with small replacement image encoder; not CUDA validation'}))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
