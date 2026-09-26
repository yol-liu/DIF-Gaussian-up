"""Bounded, one-case real evaluation. Smoke checkpoints are not quality/callibration evidence."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from datasets.dst_gs import CBCT_dataset_gs
from models.probabilistic_model import build_model
from evaluate import eval_one_epoch
from probabilistic_diagnostics import probability_diagnostics
from transfer import file_hash
from utils import convert_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probability_checkpoint', required=True)
    parser.add_argument('--baseline_checkpoint', required=True)
    parser.add_argument('--scale', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--max_seconds', type=float, default=480)
    args = parser.parse_args()
    started = time.monotonic()
    report = {'scope':'one validation case, short smoke checkpoints; no quality/calibration claim',
              'scale':args.scale,'seed':0,'cases':[],'evaluations':{}, 'status':'RUNNING'}
    report_path=ROOT/'reports'/'probabilistic'/f'real_eval_scale_{args.scale}.json'
    def save():
        report['seconds']=time.monotonic()-started
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    ckpt=torch.load(args.probability_checkpoint,map_location='cpu')
    cfg=EasyDict(ckpt['config'])
    dataset=CBCT_dataset_gs(dst_name='LUNA16',cfg=cfg.dataset,split='eval',num_views=args.num_views,
                            out_res_scale=args.scale,load_dense=True)
    dataset.name_list=dataset.name_list[:1]
    item=convert_cuda(next(iter(DataLoader(dataset,batch_size=1))),args.device)
    sparse={key:value for key,value in item.items() if not key.startswith('dense_')}
    report['cases']=item['name']
    report['context_view_ids']=item['view_ids'].cpu().tolist()
    report['dense_view_ids']=item['dense_view_ids'].cpu().tolist()
    report['probability_checkpoint_sha256']=file_hash(args.probability_checkpoint)
    report['baseline_checkpoint_sha256']=file_hash(args.baseline_checkpoint)
    model=build_model(cfg.model).to(args.device).eval()
    model.load_state_dict(ckpt['model'],strict=True)
    del ckpt
    settings=[('prior_sample',1),('prior_latent_mean',1),('prior_mc_mean',4),('prior_mc_mean',8),
              ('dense_posterior_mean',1)]
    for mode,count in settings:
        if time.monotonic()-started >= args.max_seconds:
            report['status']='STOPPED_TIME_LIMIT';save();return
        _,cases=eval_one_epoch(model,[item if mode.startswith('dense') else sparse],8192,
            latent_mode=mode,num_samples=count,oracle=mode.startswith('dense'),latent_seed=0)
        label=f'{mode}_S{count}'
        report['evaluations'][label]=cases
        save()
        print(json.dumps({'mode':label,'seconds':report['seconds']}),flush=True)
    # Usage diagnostics on a fixed subset of this same grid, not an expensive second full volume.
    selected=torch.linspace(0,item['points'].shape[1]-1,1024,device=args.device).long()
    probes=dict(item)
    probes['points']=item['points'][:,selected]
    probes['points_proj']=item['points_proj'][:,:,selected]
    probes['dense_points_proj']=item['dense_points_proj'][:,:,selected]
    probes['points_gt']=item['points_gt'].flatten(1)[:,None,selected]
    report['latent_usage_1024_grid_points']=probability_diagnostics(model,probes,num_samples=4)
    del model
    ckpt=torch.load(args.baseline_checkpoint,map_location='cpu')
    baseline=build_model(EasyDict(ckpt['config']['model'])).to(args.device).eval()
    baseline.load_state_dict(ckpt['model'],strict=True)
    _,cases=eval_one_epoch(baseline,[sparse],8192,latent_mode='deterministic')
    report['evaluations']['D0']=cases
    report['status']='COMPLETED_SMOKE_ONLY'
    save()
    print(json.dumps({'status':report['status'],'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    main()
