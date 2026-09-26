# DIF-Gaussian

Yiqun Lin, Hualiang Wang, Jixiang Chen, and Xiaomeng Li. "Learning 3D Gaussians for Extremely Sparse-View Cone-Beam CT Reconstruction." MICCAI 2024. [arXiv](https://arxiv.org/abs/2407.01090)

```
@misc{lin2024learning3d,
      title={Learning 3D Gaussians for Extremely Sparse-View Cone-Beam CT Reconstruction}, 
      author={Yiqun Lin and Hualiang Wang and Jixiang Chen and Xiaomeng Li},
      year={2024},
      eprint={2407.01090},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.01090}, 
}
```

## 1. Installation

torch 1.13, pytorch3d, SimpleITK, easydict

## 2. Data Preprocessing

Please refer to https://github.com/xmed-lab/C2RV-CBCT/tree/main/data for more details.

## 3. Experiments

The active [experiment plan](docs/EXPERIMENT_PLAN.md) specifies 400 complete epochs
per run, three seeds, and validation on all 50 eval cases every 20 epochs. The first
batch contains 24 runs, including the isolated current source and preserved backup,
all trained from scratch. The legacy experiment runner still loads common-init
weights and does not yet implement this protocol.

Research repairs and bounded experiments: see [implementation summary](reports/implementation_summary.md)
and [execution report](reports/final_report.md). Run `python scripts/run_research_plan.py --help`
for the scan/sanity/scale/profile/pilot/report interface. New training uses a linear output
head and MSE; old checkpoints require an explicit legacy configuration and strict loading.

The optional multiview 3D Transformer has B0/B1/T1–T4 configurations under
`configs/multiview/`. See [implementation and commands](reports/multiview_implementation.md)
and [validation status](reports/multiview/validation_report.md).
`python scripts/run_multiview.py --help` lists checks and bounded T2/T4 smoke runs.

For the preserved best **T1-nomax** candidate, use the protected entry point
`python scripts/run_t1_nomax.py train --name t1-nomax-protected-01` in the `torch2-3`
environment. It verifies all 169 backup files and runs the frozen source, configuration,
and common initialization in an isolated Python process; active probability code and
`PYTHONPATH` cannot replace its project modules. `evaluate --name t1-nomax-best-eval-01`
evaluates the preserved best checkpoint on the original six cases. Add `--dry-run` to
verify and inspect a command, or `--max_steps 5` for a short training check. Every run
requires a fresh name. This preserves the historical code/protocol, not bitwise CUDA
training determinism. See [isolation and verification](docs/t1_nomax_isolation.md).

Probability modeling now has a joint **dense_np** path: real nested sparse/dense views,
independent bounded Gaussian heads, and a sparse decoder conditioned only through the
dense latent during training. Deployment uses sparse views only. Configurations D0/G0/P0/P1/Pmean/PL
are in `configs/probabilistic/`; they share an explicit Gaussian reconstruction loss.
See [probability repair report and CLI](reports/probabilistic/repair_report.md) for the
information-flow contract, tests, measured smoke results, and reference requirements.
Old `model.latent` checkpoints keep their legacy GT-CVAE architecture; new training uses
`model.probabilistic.mode: dense_np` (or the separate `gt_cvae` control).
