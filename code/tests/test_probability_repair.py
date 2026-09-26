import copy
import json
import math
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from easydict import EasyDict

from test_latent import make_cfg, make_data
from models.probabilistic_model import build_model
from models.latent_np import BoundedGaussianHead, gaussian_kl, LocalLatentGrid
from probabilistic_loss import probability_loss
from probabilistic_config import resolve_probability
from probabilistic_diagnostics import probability_diagnostics
from probabilistic_metrics import mixture_metrics
from probabilistic_reference import attach_reference
from evaluation_protocol import predict_volume
from datasets.dst_gs import CBCT_dataset_gs
from checkpoints import resume_training, rng_state
from transfer import file_hash
from utils import load_config


class TinyEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ds_ch = self.out_ch = out_ch
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.SiLU())

    def forward(self, data, view_mask=None, **kwargs):
        batch, views = data.shape[:2]
        flat = data.flatten(0, 1)
        mask = torch.ones(batch*views, dtype=torch.bool, device=data.device) if view_mask is None else view_mask.bool().flatten()
        value = self.net(flat[mask])
        result = value.new_zeros(batch*views, *value.shape[1:])
        result[mask] = value
        result = result.reshape(batch, views, *result.shape[1:])
        return {'feats': result, 'feats_ds': result}


def config(mode='dense_np', local=False):
    cfg = make_cfg()
    cfg.pop('latent')
    cfg.gs.t = 1
    cfg.probabilistic = EasyDict(resolve_probability({'mode': mode, 'train_samples': 2,
        'latent': {'global_dim': 8, 'use_local': local, 'local_dim': 2, 'grid_res': 2},
        'kl': {'warmup_steps': 1, 'beta_final': .1}}))
    return cfg


def paired_data():
    dense = make_data(B=2, M=4, N=17, K=8, with_gt_volume=True)
    dense['view_ids'] = torch.arange(4).expand(2, -1)
    dense['view_mask'] = torch.ones(2, 4, dtype=torch.bool)
    result = dict(dense)
    for key in ('projs', 'angles', 'points_proj', 'points_gs_proj', 'view_ids', 'view_mask'):
        result['dense_' + key] = dense[key]
        result[key] = dense[key][:, [0, 2]]
    return result


@contextmanager
def tiny_model(mode='dense_np', local=False):
    with patch('models.base.UNet', TinyEncoder):
        yield build_model(config(mode, local))


def test_independent_kl_gradient_and_old_residual_cancellation():
    p = torch.tensor([.2], requires_grad=True)
    q = torch.tensor([1.3], requires_grad=True)
    lv = torch.tensor([math.log(.4)])
    kl = gaussian_kl(q, lv, p, lv).sum()
    gp, gq = torch.autograd.grad(kl, (p, q))
    torch.testing.assert_close(gp, (p-q)/.4)
    torch.testing.assert_close(gq, (q-p)/.4)
    residual = gaussian_kl(p+1.1, lv, p, lv).sum()
    assert torch.autograd.grad(residual, p)[0].abs().item() == 0


def test_bounded_scales_and_amp_kl():
    head = BoundedGaussianHead(3, 2, .01, 2., .5)
    for raw in (-100., 100.):
        with torch.no_grad():
            head.net[-1].bias[2:].fill_(raw)
        _, lv = head(torch.randn(2, 3))
        sigma = (.5*lv).exp()
        assert sigma.min() >= .01-1e-7 and sigma.max() <= 2+1e-7
    q = torch.tensor([2.], dtype=torch.float16, requires_grad=True)
    p = torch.tensor([0.], dtype=torch.float16, requires_grad=True)
    result = gaussian_kl(q, torch.tensor([-9.], dtype=torch.float16), p, torch.tensor([-9.], dtype=torch.float16))
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_joint_reconstruction_and_kl_gradients_multisample():
    with tiny_model(local=True) as model:
        data = paired_data()
        data['dense_projs'] = data['dense_projs'].clone().requires_grad_(True)
        output = model(data)
        assert output['points_pred'].shape == (2, 1, 17)
        assert output['sample_predictions'].shape == (2, 2, 1, 17)
        rec = (output['sample_predictions'] - data['points_gt']).square().mean()
        gradients = torch.autograd.grad(rec, (model.posterior_head.net[-1].weight,
            model.image_encoder.net[0].weight, model.feature_film.net[-1].weight,
            data['dense_projs']), retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        info = output['kl_infos'][0]
        kl = gaussian_kl(info['mu_q_g'], info['logvar_q_g'], info['mu_p_g'], info['logvar_p_g']).sum()
        gradients = torch.autograd.grad(kl, (model.prior_head.net[-1].weight, model.posterior_head.net[-1].weight), retain_graph=True)
        assert all(g.abs().sum() > 0 for g in gradients)
        loss, metrics = probability_loss(output, data['points_gt'], model.probability, step=2)
        loss.backward()
        assert metrics['warmup_beta'] == .1
        assert metrics['KL_l_dimensions'] == 2*2**3
        assert torch.allclose(metrics['L_volume'], torch.stack([
            .5*((sample-data['points_gt'])/.1).square().mean()+math.log(.1*math.sqrt(2*math.pi))
            for sample in output['sample_predictions']]).mean())
        assert not hasattr(model, 'gt_encoder')


def test_gt_mode_is_explicit_and_uses_independent_bounded_heads():
    with tiny_model('gt_cvae') as model:
        data = paired_data()
        out = model(data)
        assert 'mu_q_g' in out['kl_infos'][0]
        with pytest.raises(ValueError, match='oracle'):
            model(data, is_eval=True, latent_mode='gt_posterior_mean')
        model.eval()
        out = model(data, is_eval=True, latent_mode='gt_posterior_mean', oracle=True)
        assert out['oracle']


def test_sparse_information_isolation_dense_sensitivity_and_permutation():
    with tiny_model() as model:
        model.eval()
        data = paired_data()
        changed = copy.deepcopy(data)
        changed['dense_projs'] = torch.rand_like(data['dense_projs'])
        changed['gt_volume'] = torch.randn_like(data['gt_volume'])*100
        changed['points_gt'] *= -100
        with torch.no_grad():
            a = model(data, is_eval=True, latent_mode='prior_mc_mean', num_samples=4)
            b = model(changed, is_eval=True, latent_mode='prior_mc_mean', num_samples=4)
            torch.testing.assert_close(a['points_pred'], b['points_pred'], atol=0, rtol=0)
            context = model.encode_context(data)
            qa, _ = model.infer_posterior(data, context)
            qb, _ = model.infer_posterior(changed, context)
            assert (qa[0]-qb[0]).abs().max() > 1e-7
            permuted = dict(data)
            for prefix, permutation in (('', [1,0]), ('dense_', [3,1,0,2])):
                for key in ('projs', 'angles', 'points_proj', 'points_gs_proj', 'view_ids', 'view_mask'):
                    permuted[prefix+key] = data[prefix+key][:, permutation]
            c = model(permuted, is_eval=True, latent_mode='prior_mc_mean', num_samples=4)
            torch.testing.assert_close(a['points_pred'], c['points_pred'], atol=2e-6, rtol=2e-5)
            qc, _ = model.infer_posterior(permuted, model.encode_context(permuted))
            torch.testing.assert_close(qa[0], qc[0], atol=2e-6, rtol=2e-5)


def test_chunk_contract_sampling_reuse_and_diagnostics_rng():
    with tiny_model(local=True) as model:
        model.eval()
        data = paired_data()
        calls = []
        hook = model.image_encoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            a = model(data, is_eval=True, num_samples=4, return_samples=True, eval_npoint=4)
            assert len(calls) == 1
            b = model(data, is_eval=True, num_samples=4, return_samples=True, eval_npoint=17)
            torch.testing.assert_close(a['sample_predictions'], b['sample_predictions'], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a['points_pred'], a['sample_predictions'].mean(0))
            torch.testing.assert_close(a['predictive_var_latent'], a['sample_predictions'].var(0))
            one = model(data, is_eval=True, latent_mode='prior_sample')
            assert one['predictive_var_latent'] is None
        hook.remove()
        model.train()
        before = torch.get_rng_state().clone()
        state_before = {k:v.clone() for k,v in model.state_dict().items()}
        diag = probability_diagnostics(model, data, num_samples=2, eval_npoint=7)
        assert torch.equal(before, torch.get_rng_state()) and model.training
        assert diag['posterior_available'] and diag['sensitivity_l2'] > 0
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)
        stripped = {k:v for k,v in data.items() if not k.startswith(('dense_', 'gt_')) and k != 'points_gt'}
        diag = probability_diagnostics(model, stripped, num_samples=2, eval_npoint=7)
        assert not diag['posterior_available'] and diag['KL_g_raw'] is None


def test_local_cell_centers_axis_phantom_and_empty_cells():
    local = LocalLatentGrid(2, 2, 1, grid_res=4)
    axis = (torch.arange(4)+.5)/4
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(1,-1,3)
    value = xyz[...,0] + 2*xyz[...,1] + 3*xyz[...,2]
    grid = local.pool_anchors_to_grid(value[:,None], xyz)
    torch.testing.assert_close(local.interpolate_to_anchors(grid, xyz)[...,0], value)
    interior = torch.tensor([[[.3,.4,.6]]])
    torch.testing.assert_close(local.interpolate_to_anchors(grid, interior)[...,0],
                               interior[...,0]+2*interior[...,1]+3*interior[...,2])
    empty = local.pool_anchors_to_grid(torch.ones(1,1,1), torch.zeros(1,1,3))
    assert torch.isfinite(empty).all() and empty.count_nonzero()==1


def test_ambiguous_forward_kl_moment_matching_and_nonlinear_mc():
    mu = nn.Parameter(torch.tensor(.3))
    raw_variance = nn.Parameter(torch.tensor(-1.))
    opt = torch.optim.Adam([mu, raw_variance], lr=.06)
    target_means = torch.tensor([-1.,1.])
    for _ in range(500):
        loss = gaussian_kl(target_means, torch.full((2,), math.log(.04)), mu, raw_variance).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(mu.item()) < 1e-4 and abs(raw_variance.exp().item()-1.04) < 1e-3
    samples = torch.randn(100000, generator=torch.Generator().manual_seed(42))*math.sqrt(1.04)
    assert abs(samples.square().mean().item()-1.04) < .02
    assert mu.square() < 1e-6


def test_mixture_nll_and_coverage_are_voxelwise_unclipped():
    samples = torch.tensor([[[[-1., -1.]]], [[[1., 1.]]]])
    target = torch.tensor([[[0., 2.]]])
    metrics = mixture_metrics(samples, target, .2, voxel_chunk=1)
    logp = torch.distributions.Normal(samples, .2).log_prob(target)
    expected = (-torch.logsumexp(logp,0)+math.log(2)).mean()
    assert abs(metrics['voxelwise_marginal_nll']-expected.item()) < 1e-5
    assert metrics['coverage_90'] == .5


def test_paired_loader_single_read_masks_and_no_gt_deployment():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)/'scan'; root.mkdir()
        meta = dict(train=['one'], eval=['one'], dataset_config='config.yaml', projs='{}.pkl',
                    image='missing_{}.nii.gz', blocks_coords='coords.npy', blocks_vals='{}_{}.npy')
        (root/'meta_info.json').write_text(json.dumps(meta))
        # A fully synthetic scanner configuration, independent of external data.
        import yaml
        geometry = dict(DSD=150, DSO=100, nDetector=[16,16], dDetector=[1,1], nVoxel=[4,4,4],
                        dVoxel=[1,1,1], offOrigin=[0,0,0], offDetector=[0,0], mode='cone', accuracy=.5, filter=None)
        (root/'config.yaml').write_text(yaml.safe_dump({'dataset':{'resolution':[4,4,4]}, 'projector':geometry}))
        raw = np.arange(8,dtype=np.uint8)[:,None,None]*np.ones((8,16,16),dtype=np.uint8)
        with open(root/'one.pkl','wb') as handle:
            pickle.dump({'projs':raw,'projs_max':.4,'angles':np.linspace(0,3,8)},handle)
        np.save(root/'coords.npy', np.random.default_rng(0).random((1,20,3)))
        np.save(root/'one_0.npy', np.arange(20,dtype=np.uint8))
        cfg = EasyDict(root_dir=folder, gs_res=2, paired_views={'dense_count':6,'context_counts':[2,3]})
        train = CBCT_dataset_gs(dst_name='scan',cfg=cfg,split='train',npoint=4,num_views=2,sample_seed=0)
        with patch('datasets.base.pickle.load', wraps=pickle.load) as reader:
            item = train[0]
            assert reader.call_count==1
        ids = item['view_ids'][item['view_mask']]
        assert set(ids) < set(item['dense_view_ids'])
        for i, view in enumerate(ids):
            j = list(item['dense_view_ids']).index(view)
            np.testing.assert_array_equal(item['projs'][i],item['dense_projs'][j])
        deploy = CBCT_dataset_gs(dst_name='scan',cfg=cfg,split='eval',num_views=2,deployment_only=True)
        result = deploy[0]
        assert 'points_gt' not in result and 'dense_projs' not in result and 'gt_volume' not in result


def test_frozen_reference_dependencies_and_one_way_gradients():
    with patch('models.base.UNet', TinyEncoder), tempfile.TemporaryDirectory() as folder:
        ref = build_model(config())
        optimizer = torch.optim.Adam(ref.parameters(), lr=.001)
        data = paired_data()
        output = ref(data)
        loss,_ = probability_loss(output,data['points_gt'],ref.probability,2)
        loss.backward(); optimizer.step()
        cfg = config(); cfg.probabilistic.objective='frozen_reference'
        student = build_model(cfg)
        with pytest.raises(ValueError,match='BLOCKED_REFERENCE'):
            attach_reference(student,None,None)
        path,evidence = Path(folder)/'ref.pth',Path(folder)/'evidence.json'
        torch.save({'model':ref.state_dict(),'global_step':1,'config':{'model':ref.cfg}},path)
        measured = probability_diagnostics(ref,data,num_samples=2)
        evidence.write_text(json.dumps({'checkpoint_sha256':file_hash(path),'case_ids':['synthetic'],
            'latent_usage':{k:measured[k] for k in ('output_change_l1','sensitivity_l2')}}))
        attach_reference(student,str(path),str(evidence))
        loss,_ = probability_loss(student(data),data['points_gt'],student.probability,2)
        loss.backward()
        assert student.prior_head.net[-1].weight.grad.abs().sum()>0
        assert all(p.grad is None for p in student.reference.parameters())
        assert all(not p.requires_grad for n,p in student.named_parameters() if not n.startswith('prior_head.'))
        clone = build_model(cfg); clone.load_state_dict(student.state_dict(),strict=True)


def test_transformer_state_keeps_point_fusion_and_masks():
    from test_multiview_transformer import small_config, synthetic_data
    cfg = small_config('T2')
    cfg.model.pop('latent')
    cfg.model.probabilistic = config().probabilistic
    model = build_model(cfg.model).eval()
    data = synthetic_data(views=4)
    for key in ('projs','angles','view_ids','points_proj','points_gs_proj'):
        data['dense_'+key] = data[key]
        data[key] = data[key][:,[0,2]]
    with torch.no_grad():
        first = model(data,is_eval=True,eval_npoint=7,num_samples=2,return_debug=True)
        second = model(data,is_eval=True,eval_npoint=23,num_samples=2)
    assert 'point_feature_maps' in first['states'][0].features
    torch.testing.assert_close(first['points_pred'],second['points_pred'],atol=2e-5,rtol=2e-4)
    model.train()
    result = model(data)
    loss,_ = probability_loss(result, data['points_gt'], model.probability, 2)
    loss.backward()
    assert model.posterior_head.net[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.point_view_fusion.parameters())


def test_resume_restores_probability_optimizer_step_and_rng():
    with tiny_model() as model:
        data = paired_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        loss,_ = probability_loss(model(data),data['points_gt'],model.probability,2)
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        checkpoint = copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'global_step':3,'epoch':0,'rng_by_rank':[rng_state()],
            'config':dict(model.cfg)})
        expected = model(data)['sample_predictions'].detach()
        torch.manual_seed(999)
        epoch,step = resume_training(model,optimizer,scheduler,checkpoint)
        actual = model(data)['sample_predictions'].detach()
        assert (epoch,step)==(1,3)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)


def test_raw_effective_freebits_and_configured_oracle():
    cfg=resolve_probability({'kl':{'free_bits':.5}})
    z=torch.zeros(2,64)
    output={'sample_predictions':torch.zeros(1,2,1,3),
            'kl_infos':[{k:z for k in ('mu_p_g','mu_q_g','logvar_p_g','logvar_q_g')}]}
    _,metrics=probability_loss(output,torch.zeros(2,1,3),cfg,2000)
    assert metrics['KL_g_raw']==0 and metrics['KL_g_effective']==32
    with tiny_model() as model:
        model.eval()
        model.probability['evaluation']['estimator']='dense_posterior_mean'
        with pytest.raises(ValueError,match='oracle'):
            predict_volume(model,paired_data(),7)
        assert predict_volume(model,paired_data(),7,oracle=True).shape==(2,1,17)


def test_mean_code_control_and_equal_context_dense_sanity():
    with patch('models.base.UNet',TinyEncoder):
        cfg=config(); cfg.probabilistic.mean_code=True
        model=build_model(cfg)
        output=model(paired_data())
        loss,metrics=probability_loss(output,paired_data()['points_gt'],model.probability,2)
        loss.backward()
        assert 'weighted_mean_code_alignment' in metrics
        torch.testing.assert_close(output['sample_predictions'][0],output['sample_predictions'][1])
        data=paired_data()
        for key in ('projs','angles','view_ids','view_mask','points_proj','points_gs_proj'):
            data['dense_'+key]=data[key]
        model.eval()
        model.validate_pair(data,strict=False)
        model(data,is_eval=True,latent_mode='dense_posterior_mean',oracle=True)
        with pytest.raises(ValueError,match='strictly'):
            model(data)


def test_probability_configurations_are_executable():
    for path in Path('configs/probabilistic').glob('*.yaml'):
        load_config(path)


def test_frozen_bn_policy_persists_for_deterministic_control():
    with patch('models.base.UNet',TinyEncoder):
        cfg=config('deterministic'); cfg.probabilistic.normalization='frozen_bn'
        model=build_model(cfg).train()
        assert model.training
        assert all(not module.training for module in model.modules() if isinstance(module,nn.modules.batchnorm._BatchNorm))


def test_transformer_anatomy_posterior_never_reads_ct_auxiliary_targets():
    from test_multiview_transformer import small_config, synthetic_data
    from models.dense_probability import dense_inputs
    cfg=small_config('T4'); cfg.model.pop('latent'); cfg.model.probabilistic=config().probabilistic
    model=build_model(cfg.model).train()
    data=synthetic_data(views=4)
    for key in ('projs','angles','view_ids','points_proj','points_gs_proj'):
        data['dense_'+key]=data[key]
        data[key]=data[key][:,[0,2]]
    dense=dense_inputs(data)
    assert not any(key.startswith('structure_') or key in ('gt_volume','points_gt') for key in dense)
    result=model(data)
    loss,_=probability_loss(result,data['points_gt'],model.probability,2)
    (loss+result['structure_loss']).backward()
    assert model.posterior_head.net[-1].weight.grad.abs().sum()>0


def test_anchor_instance_policy_restores_per_case_anchor_spread_and_chunk_invariance():
    from probabilistic_config import PointChannelNorm
    for mode in ('deterministic', 'dense_np'):
        with patch('models.base.UNet', TinyEncoder):
            cfg = config(mode); cfg.probabilistic.normalization = 'groupnorm_anchor_instance'
            model = build_model(cfg)
        anchor = [m for name, m in model.named_modules() if name.startswith(('gs_feats_mlp', 'gs_params_mlp'))]
        assert any(isinstance(m, nn.InstanceNorm1d) for m in anchor)
        assert not any(isinstance(m, (PointChannelNorm, nn.modules.batchnorm._BatchNorm)) for m in anchor)
        assert any(isinstance(m, PointChannelNorm) for m in model.point_decoder.modules())
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    # Unit per-channel spread across one case's anchors, independent of the other case in the batch.
    data = paired_data()
    model.eval()
    with torch.no_grad():
        feats = model.gs_feats_mlp.layer[:2](torch.randn(2, model.gs_feats_mlp.layer[0].in_channels, 8))
        torch.testing.assert_close(feats.std(-1, unbiased=False), torch.ones_like(feats[..., 0]), atol=1e-2, rtol=0)
        a = model(data, is_eval=True, num_samples=3, return_samples=True, eval_npoint=4)
        b = model(data, is_eval=True, num_samples=3, return_samples=True, eval_npoint=17)
        torch.testing.assert_close(a['sample_predictions'], b['sample_predictions'], atol=2e-6, rtol=2e-5)
        # Latent mean is RNG-free, so batch composition must not change a case's prediction.
        pair = model(data, is_eval=True, latent_mode='prior_latent_mean', eval_npoint=17)['points_pred']
        single = model({k: v[:1] if torch.is_tensor(v) else v for k, v in data.items()}, is_eval=True,
                       latent_mode='prior_latent_mean', eval_npoint=17)['points_pred']
        torch.testing.assert_close(single, pair[:1], atol=2e-6, rtol=2e-5)
