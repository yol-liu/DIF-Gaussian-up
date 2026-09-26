# 原最佳 T1-nomax 与概率实验隔离

原最佳配置是 UNet + anchor mean-init Transformer/no max residual + point max、BN、MSE、6 views；不是 `probabilistic.mode: deterministic`。后者仍可替换归一化和 loss。

从仓库根目录使用已验证的 `torch2-3` 环境：

```bash
/home/wyin/miniconda3/envs/torch2-3/bin/python scripts/run_t1_nomax.py train --name t1-nomax-protected-01
/home/wyin/miniconda3/envs/torch2-3/bin/python scripts/run_t1_nomax.py evaluate --name t1-nomax-best-eval-01
```

训练默认执行历史 seed0、AdamW lr2e-4、FP32、batch1、1024 queries、2000 updates、1000/2000 步验证、原6例和原公共初始化。评估固定使用备份的 `partial_step_2000.pth`，历史 PSNR 23.083580。

- 启动前校验固定 manifest 自身 SHA256，以及备份全部 169 个文件的大小和 SHA256。失败即停止，不退回当前代码。
- 子进程使用 Python `-I -B`，忽略 `PYTHONPATH` 和 user site，项目导入只指向 `backups/t1-nomax-best-20260924/code`，不生成备份 pycache。
- 不开放替换配置、loss、概率开关、初始化、AMP 或视角采样的参数。仍可通过普通 `code/train.py` / `code/evaluate.py` 运行其他实验。
- 输出写到新 `logs/<name>`，已有目录拒绝覆盖；`protected_baseline.json` 记录执行命令、来源、校验结果和退出码。
- `--dry-run` 只校验和打印；训练可用 `--max_steps 5` 做短跑。短跑仍保留原验证日程，5 步不会触发完整验证。

当前主线入口也做了隔离：`models.factory` 对概率实现按配置延迟导入，旧 `models.probabilistic_model.build_model` 导入方式仍兼容；普通 MSE 路径不加载 latent/KL 实现，不再执行 `mse_raw` 的重复计算；普通评估不保留概率输出字典。概率实验的归一化、损失、诊断和旧 checkpoint 路径仍保留。

冻结入口用于保护历史候选；修改当前共享骨干也不会自动改变它。它依赖已有备份、数据和 Python/CUDA 环境，并未冻结系统运行环境或数据文件。本机实测 CUDA grid_sample backward 不具备逐位确定性，因此隔离不会保证每次从头训练都回到同一 PSNR。

验证记录：[baseline_isolation](../reports/recovery_audit_20260925/baseline_isolation)。
