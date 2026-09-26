# DIF-Gaussian 完整实验计划（结构 → 细节 → 确认）

版本：2026-09-25。适用代码：本仓库当前工作区（`code/`）。配置：`configs/experiment_plan/`。
运行器：`scripts/run_experiment_plan.py`。配置生成器：`scripts/build_experiment_plan.py`。

本计划只做**筛查与确认**，不调超参。所有实验对同一个基线做**只改一处**的配对对照，
先验证大结构（Stage 1），再在胜出结构上验证细节（Stage 2），最后对少数候选做长训练确认（Stage 3）。

---

## 0. 背景：已经知道的事情

| 事实 | 证据 |
|---|---|
| 当前最佳是 `baseline-unet-transformer-meaninit`（原名 T1-nomax）：UNet 编码器 + anchor 几何 Transformer（token 用多视角均值初始化、无 max 残差）+ point max 融合，BatchNorm，MSE | `reports/current_best.json`，2000 步 PSNR 23.08 / 23.10 |
| 这些好结果都来自 **seed 0 + 共享初始化 `logs/p1-common-init/seed0.pth`**；本计划全部沿用这一设置，只跑单 seed | 同上 |
| 同配置复跑会因 GPU 非确定性相差 0.1–0.8 dB，因此设一个与基线完全相同的复跑实验作为噪声地板 | `reports/p1_screening_findings.md` §4 |
| 概率分支（dense NP）在原配置下比确定性对照差 0.77 dB。根因：GroupNorm 策略把 anchor MLP 的 BatchNorm1d 换成逐 anchor LayerNorm → Gaussian 特征跨 anchor 几乎不变 → FiLM 的 z 噪声主导梯度；另外 q 的输入是全局均值，posterior 塌缩（KL≈0.04 nat，oracle 无增益） | `reports/probability_validation/results.md` |
| 已修复：新增 `normalization: groupnorm_anchor_instance`（anchor MLP 用每病例 InstanceNorm），确定性对照 +0.70 dB（6/6） | 同上 |
| Gaussian NLL(σ=0.1) = 50×MSE，配合 grad-clip 1.0 比 MSE 差约 1.1 dB；本计划所有主实验统一用 MSE | 同上 |
| SSIM 在这些设置下随截断比例大幅波动，**决策以 PSNR / 未截断 MSE / 边界误差为准**，SSIM 只记录 | 同上 |

---

## 1. 统一协议（所有实验一致，不得单独修改）

| 项 | 设置 |
|---|---|
| 数据 | LUNA16（C2RV-CBCT 预处理格式），训练集 738 例；数据路径在生成配置时指定（见第 2 节） |
| 输入视角 | 6 views；验证固定 raw IDs `[0,8,16,25,33,41]` |
| 优化 | AdamW，lr 2e-4，weight decay 1e-2（新分支模块免衰减，train.py 默认），StepLR 衰减 1e-3/400 epoch，grad clip 1.0，FP32（不开 AMP），batch 1，每步 1024 query |
| 预算 | Stage 1/2：每个实验 2000 updates；验证在 1000 与 2000 步；主比较只看 **2000 步**，不挑 checkpoint |
| 验证 | 前 12 个 eval 病例，scale 1.0，eval_npoint 65536 |
| 种子与初始化 | **单 seed 0**；所有实验从 `logs/p1-common-init/seed0.pth` 拷贝共享参数，结构独有参数用 seed 0 自行初始化（形状不匹配的张量自动剔除并记录在 `common_init_dropped`） |
| 数据顺序 | 同采样器的实验看到**逐步完全相同**的病例、视角、block、query（报告中 `paired` 一列） |
| 目录 | 每个 run 写入 `logs/plan-<实验名>-seed0[-tag]`；运行器**绝不覆盖**已存在目录 |

`logs/p1-common-init/seed0.pth` 是复现既有好结果的关键文件，换机器时必须一起拷贝。若缺失，运行器会按同一方法
（seed 0 构建 T4 结构）重新生成到 `logs/plan-common-init/seed0.pth`，但不保证与原文件逐位一致。

### 决策规则（运行器自动给出）

对每个比较，取 12 个验证病例的逐例 ΔPSNR（变体 − 基线），对病例做 bootstrap 得 95% 置信区间；
噪声地板 = |基线 vs 复跑| 的 ΔPSNR。

- **PROMOTE（晋级）**：95% CI 下界 > 0，且 |Δ| 大于噪声地板。
- **REJECT（淘汰）**：95% CI 上界 < 0，且 |Δ| 大于噪声地板。
- **INCONCLUSIVE（不确定）**：其他情况。不确定的不晋级；若边界误差/MSE 同向明显改善，在汇报中单独说明。
- 复跑实验未完成前，所有比较标记为 `NOISE_FLOOR_PENDING`，**先跑基线与复跑**。
- 病例 bootstrap 只反映病例间差异，不包含训练随机性；单 seed 的晋级结论必须经 Stage 3 确认。
- 参数量不同的结构（ViT、hierarchical、maxpool）必须同时报告参数量和峰值显存。

---

## 2. 准备与自检（开跑前必须完成）

```bash
PY=python   # 需要 PyTorch 2.x 环境，工作目录为仓库根目录

# 用自己的数据路径生成全部配置（目录下应有 LUNA16 的 projections/blocks/images）
$PY scripts/build_experiment_plan.py --data-root <LUNA16 数据目录>

# 单元测试（约 2 分钟）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest code/tests -q

# 所有实验能构建并加载共享初始化（CPU）
$PY scripts/run_experiment_plan.py --phase check

# 真实数据 GPU 冒烟：每个实验 10 步、1 例，全部 PASS 才能正式训练（约 15 分钟）
$PY scripts/run_experiment_plan.py --phase smoke
```

状态记录在 `reports/experiment_plan/execution_status.json`，每个 run 的日志在 `reports/experiment_plan/logs/`。
中途失败：先看日志，确认原因后**手动**把失败目录移到 `logs/_failed/`，再重跑同一命令（已 PASS 的会自动跳过）。

冒烟只检查“能跑通”：10 步后很多实验的输出被 [0,1] 截断 100%，所以冒烟 PSNR 会一模一样（约 11.82），
这是正常的；看 `unclipped_mse` 和训练 loss 是否因实验而异即可。

冒烟 run 的名字带配置哈希（`plan-<实验名>-seed0-smoke-<hash>`）：配置一旦改动（例如重建 Stage 2 或更换数据路径），
正式训练会要求先为新配置重新冒烟；已 PASS 的正式 run 若配置哈希变了，运行器会拒绝跳过，要求换 `--tag`。

### 磁盘

开跑前确认日志所在磁盘至少有 **50 GB** 空闲。每个正式 run 结束后只保留最终 checkpoint（约 0.45 GB；
`--keep_checkpoints all` 可保留中点），冒烟 run 不保留权重。全计划约 30 个正式 run ≈ 15 GB，确认阶段另计。
运行器在空闲 < 5 GB 时拒绝启动新 run——磁盘写满会损坏 checkpoint 和状态文件。

---

## 3. Stage 1：大结构（先做）

问题：在相同训练协议下，哪一种编码器 / 多视角融合 / 概率建模结构更好？
每个实验相对“比较对象”只改表中一处。“历史线索”为此前 6 例、2000 步的单次结果，仅供参考。

### 1A 图像编码器

| 实验名 | 改动 | 比较对象 | 要回答的问题 | 历史线索 |
|---|---|---|---|---|
| `baseline-unet-transformer-meaninit` | 当前最佳 | — | 基线 | 23.10 |
| `baseline-unet-transformer-meaninit-replica` | 与基线完全相同 | 基线 | **噪声地板** | — |
| `encoder-vit-per-view` | UNet 1/4–1/16 编码器换成逐视角 ViT（从头训） | 基线 | 纯 Transformer 编码器是否更好 | 21.66 |
| `encoder-hierarchical-window` | 换成层级窗口 Transformer（保留卷积 stem 与 UNet 解码器），参数 19.4M | 基线 | 层级 Transformer 编码器 | 22.73（在对齐坐标 + 3D 采样上） |

### 1B anchor 多视角融合

| 实验名 | 改动 | 比较对象 | 要回答的问题 | 历史线索 |
|---|---|---|---|---|
| `anchor-fusion-maxpool` | 去掉 anchor Transformer，anchor 取多视角 max（原 DIF-Gaussian） | 基线 | Transformer 融合带来多少 | 21.82 |
| `anchor-fusion-late-attention` | 旧版晚期跨视角注意力 | 基线 | 旧融合方案 | 5000 步时不如 max |
| `anchor-transformer-max-residual` | Transformer 作为 max 特征上的门控残差，token 从 max 初始化 | 基线 | 保留 max 路径是否有害 | 22.72 |
| `anchor-transformer-no-residual-maxinit` | 无残差，token 从 max 初始化 | 基线 | token 初始化方式（max vs mean） | 22.61 |

### 1C point 融合与 2D/3D 交互

| 实验名 | 改动 | 比较对象 | 要回答的问题 | 历史线索 |
|---|---|---|---|---|
| `point-fusion-geometry-attention` | query 点的视角融合由 max 改为几何注意力 | 基线 | point 级注意力融合 | 5000 步时方向为负 |
| `view-update-alternating-2d3d` | Transformer block 之间 2D 特征图反读 3D token | 基线 | 2D/3D 交替更新 | 22.95 |

### 1D 概率建模（CVAE / dense NP）

概率分支从头训练必须用 GroupNorm（BN 会在 C/D 两次前向之间共享统计），因此先用一个**确定性对照**
把“归一化 + 嵌套采样器”的代价单独量出来，概率实验一律与该对照比较。

| 实验名 | 改动 | 比较对象 | 要回答的问题 |
|---|---|---|---|
| `prob-deterministic-control` | 基线 + `groupnorm_anchor_instance` + 嵌套采样器（D=12 中取 C=6），MSE，无 latent | 基线（采样器不同，非严格配对） | 概率分支所需底座本身损失多少 |
| `prob-gt-cvae` | 条件 VAE：q(z∣C,GT)、p(z∣C)，独立有界 head，全局 z=64 经 FiLM，部署只采 p | 确定性对照 | GT 条件的 CVAE 是否有收益 |
| `prob-dense-np` | dense NP：q(z∣D=12 视角)、KL(q‖p(z∣C=6))，β=2e-5（MSE 尺度），warmup 500 | 确定性对照 | 更多视角作为特权信息是否有收益 |
| `prob-dense-mean-code` | 训练用 q 的确定性均值码，p 对齐该均值（无随机性） | 确定性对照；并与 dense NP 比 | 收益来自特权信息还是随机建模 |

概率实验训练完后**必须**做 post-hoc（第 5 节）：比较 `prior_sample / prior_latent_mean / prior_mc_mean(S=4,8)` 与
oracle（dense 或 GT posterior mean）。**若 oracle 相对 prior_latent_mean 没有增益（< 0.1 dB），说明 latent 没有携带
有用信息，该分支结论为“无效”，不能靠调 β 宣称有效。**

### Stage 1 命令

```bash
# 全部 Stage 1（单卡串行约 7 GPU 小时）
$PY scripts/run_experiment_plan.py --phase train --stage 1

# 多卡时按实验拆分（每张卡设 CUDA_VISIBLE_DEVICES，不同卡跑不同实验）：
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_experiment_plan.py --phase train --arms baseline-unet-transformer-meaninit baseline-unet-transformer-meaninit-replica encoder-vit-per-view encoder-hierarchical-window
CUDA_VISIBLE_DEVICES=1 $PY scripts/run_experiment_plan.py --phase train --arms anchor-fusion-maxpool anchor-fusion-late-attention anchor-transformer-max-residual anchor-transformer-no-residual-maxinit point-fusion-geometry-attention view-update-alternating-2d3d
CUDA_VISIBLE_DEVICES=2 $PY scripts/run_experiment_plan.py --phase train --arms prob-deterministic-control prob-gt-cvae prob-dense-np prob-dense-mean-code

# 概率实验的 post-hoc 评估（每个 run 约 20–30 分钟）
$PY scripts/run_experiment_plan.py --phase posthoc --stage 1

# 汇总（随时可跑，只统计已完成的 run）
$PY scripts/run_experiment_plan.py --phase report
```

> 多个 `--phase train` 并行时共用 `execution_status.json`，每张卡必须跑不同的实验集合，不要同时启动同一个实验。

### Stage 1 → Stage 2 的交接

1. 看 `reports/experiment_plan/results.md`。**只有 PROMOTE 的结构**才能替换基线。
2. 若多个结构都 PROMOTE 且可以组合（例如编码器 + 融合），先不组合；选 ΔPSNR 最大的一个作为 Stage 2 底座，
   组合留到 Stage 3 验证。
3. 用胜出结构重建 Stage 2 配置（Stage 1 配置不变）：
   ```bash
   $PY scripts/build_experiment_plan.py --data-root <LUNA16 数据目录> --stage2-base <胜出的 stage-1 实验名>
   ```
   若没有结构晋级，保持默认底座 `baseline-unet-transformer-meaninit`。
4. 概率实验：若 Stage 1 中 **dense NP 与 GT-CVAE 的 oracle 增益都 < 0.1 dB 且相对确定性对照不是 PROMOTE**，
   Stage 2 的 `detail-dense-np-*` 只作为机制诊断解读，不作为晋级候选。

---

## 4. Stage 2：细节（在 Stage 1 底座上）

### 2A 几何与采样细节

| 实验名 | 改动 | 比较对象 | 要回答的问题 | 历史线索 |
|---|---|---|---|---|
| `detail-aligned-pixel-coordinates` | anchor 初始化与 cross-attention 用同一像素中心约定和 FOV mask | 底座 | 坐标约定统一是否有益 | −0.32 dB |
| `detail-shared-3d-sampling` | 共享 3D 五点采样（中心 + 正四面体，8 mm）；依赖对齐坐标 | 对齐坐标实验（也与底座比） | 多视角对应的一致采样 | 相对对齐坐标 +0.10 |
| `detail-anatomy-supervision` | 解剖结构张量辅助监督，采样不变 | 底座 | 辅助监督 | ≈0 |
| `detail-anatomy-guided-sampling` | 解剖监督 + 预测张量引导采样 | 解剖监督实验（也与底座比） | 引导采样的额外作用 | −0.15 |
| `detail-single-scale-features` | anchor token 只读 1/16 特征图 | 底座 | 多尺度是否必要 | 未跑 |
| `detail-no-spatial-neighbours` | 空间注意力窗口 1（不读邻近 anchor） | 底座 | 空间邻域交互是否必要 | 未跑 |

### 2B 训练目标与归一化细节

| 实验名 | 改动 | 比较对象 | 要回答的问题 |
|---|---|---|---|
| `detail-gaussian-nll-loss` | 底座改用 Gaussian NLL(σ=0.1) | 底座 | 损失尺度 × 梯度裁剪的影响（预期约 −1 dB） |
| `detail-gn-standard-sampler` | 确定性对照改回原 6 视角采样器 | 确定性对照；也与基线比 | 拆分“嵌套采样器”和“GroupNorm”各自的代价 |
| `detail-gn-pointchannel-anchor-norm` | 确定性对照用旧 GroupNorm 策略（anchor 用逐点 LayerNorm） | 确定性对照 | 复核 anchor 归一化修复（此前 +0.70） |

### 2C 概率分支细节（比较对象均为 `prob-dense-np`）

| 实验名 | 改动 | 要回答的问题 |
|---|---|---|
| `detail-dense-np-sigma-init-0.05` | latent 初始标准差 0.5 → 0.05 | 早期 FiLM 噪声的影响 |
| `detail-dense-np-beta-x10` | KL 权重 2e-5 → 2e-4 | KL 强度与塌缩 |
| `detail-dense-np-train-samples-4` | 每步 4 个 latent 样本 | 训练梯度方差 |
| `detail-dense-np-local-latent` | 增加局部 latent 网格（4³×16） | 空间化 latent 能否携带病例信息 |
| `detail-dense-np-multi-context-4-6-8` | 训练 context 视角数 4/6/8 混合 | 视角数泛化 |
| `detail-dense-np-gaussian-nll` | 原修复方案：Gaussian NLL + β=1e-3 | 与旧配方对照 |
| `detail-dense-np-frozen-reference-distill` | **条件实验**：冻结训练好的 dense NP 作为参考，只训练 prior head，KL(stopgrad q_ref‖p) | 单向分布蒸馏 |

蒸馏实验的前置条件由运行器强制检查：先对 `prob-dense-np` 做 post-hoc，再运行 `--phase evidence`。
只有当 oracle 增益 ≥ 0.1 dB 且解码器对 z 的敏感度非零时，evidence 才会是 `gate: PASS`，
否则该实验标记 BLOCKED，**不要绕过**。

几何/损失类细节建在 Stage 2 底座上，报告中它们的比较对象自动取 `configs/experiment_plan/stage2/BASE.txt`
记录的实验；归一化与概率类细节始终建在原基线骨干上，与 Stage 1 的 `prob-*` 实验比较。

### Stage 2 命令

```bash
$PY scripts/run_experiment_plan.py --phase smoke --stage 2      # 重建过配置后会自动为新配置重新冒烟
$PY scripts/run_experiment_plan.py --phase train --arms detail-aligned-pixel-coordinates detail-shared-3d-sampling detail-anatomy-supervision detail-anatomy-guided-sampling detail-single-scale-features detail-no-spatial-neighbours detail-gaussian-nll-loss detail-gn-standard-sampler detail-gn-pointchannel-anchor-norm
$PY scripts/run_experiment_plan.py --phase train --arms detail-dense-np-sigma-init-0.05 detail-dense-np-beta-x10 detail-dense-np-train-samples-4 detail-dense-np-local-latent detail-dense-np-multi-context-4-6-8 detail-dense-np-gaussian-nll
$PY scripts/run_experiment_plan.py --phase posthoc --stage 2

# 蒸馏（条件实验）
$PY scripts/run_experiment_plan.py --phase evidence
$PY scripts/run_experiment_plan.py --phase train --arms detail-dense-np-frozen-reference-distill
$PY scripts/run_experiment_plan.py --phase report
```

---

## 5. 概率分支的 post-hoc 与指标含义

`--phase posthoc` 对每个概率 run 输出 `reports/experiment_plan/posthoc/<run>.json`：

| 字段 | 含义 | 怎么看 |
|---|---|---|
| `prior_sample_S1` | 一次随机 prior 采样重建 | 只作参考 |
| `prior_latent_mean_S1` | f(μ_p)，确定性近似 | 与 MC mean 对照 |
| `prior_mc_mean_S4/S8` | S 个 prior 样本重建的平均，**概率模型的主要点预测** | 与确定性对照比较用这个 |
| `dense_posterior_mean_S1` / `gt_posterior_mean_S1` | **oracle**，用了部署时拿不到的信息 | 只用于判断 latent 有没有用，不能当部署成绩 |
| `oracle_gain_db` | oracle − prior_latent_mean | < 0.1 dB：latent 没带来信息 |
| `voxelwise_marginal_nll`、`coverage_50/90` | 逐体素混合分布 NLL 与区间覆盖率 | 有限样本诊断，不能在验证集上调 σ |
| `latent_usage` 中 `other_case_z_l1_change` | 换成别的病例的 z，输出变化 | ≈0 表示 z 没有病例信息 |
| `zero_z_l1_change` | z 置零的输出变化 | 远大于 `other_case_z` 说明 z 只是噪声 |
| `between_case_var_mu_p/q` | μ 在病例间的方差 | ≈0 表示塌缩 |

---

## 6. Stage 3：确认（只对 Stage 1/2 中 PROMOTE 的 1–3 个候选）

- 每个候选与当前基线（以及可组合时的组合版本）、基线复跑各跑 **10000 updates、验证 24 例**，仍用 seed 0 与同一共享初始化：
  ```bash
  $PY scripts/run_experiment_plan.py --phase train --arms baseline-unet-transformer-meaninit baseline-unet-transformer-meaninit-replica <候选...> --steps 10000 --val_cases 24 --tag confirm10k
  $PY scripts/run_experiment_plan.py --phase report --steps 10000 --tag confirm10k
  ```
- `--tag` 保证与筛查 run 分开目录。确认阶段同样按 PROMOTE 规则判断，通过后再更新 `reports/current_best.json`。
- 组合（如编码器 + 融合）只在确认阶段做，必须同时跑各单项与组合。

---

## 7. 时间预算（单张 RTX 3090，串行）

| 部分 | 时间 |
|---|---:|
| Stage 1 确定性实验（10 个；ViT / 交替更新约 35 分钟，其余约 15 分钟） | ≈3.3 h |
| Stage 1 概率实验（4 个，约 30–45 分钟） | ≈2.3 h |
| Stage 1 post-hoc | ≈1.5 h |
| Stage 2 确定性实验（9 个） | ≈2.3 h |
| Stage 2 概率实验（7 个，含 4 样本实验约 60 分钟） | ≈5 h |
| Stage 3（每个候选 10000 步 ≈ 1–1.5 h） | 视候选数 |

多卡时按第 3 节的方式按实验分卡；同一张卡不要同时跑两个训练（会互相拖慢，但不影响结果正确性）。

---

## 8. 汇报内容

每个阶段结束后提交：

1. `reports/experiment_plan/results.md` 与 `results.json`（运行器生成，不要手改）。
2. 比较表：每一行的 ΔPSNR、95% CI、改善病例数、相对 MSE / 边界误差变化、是否严格配对、决策。
3. 噪声地板（基线 vs 复跑）数值。
4. 概率实验：post-hoc 估计量表、oracle 增益、latent 使用诊断。
5. 每个实验的参数量、峰值显存、训练时长（`execution_status.json` 里有）。
6. 失败 / 异常：失败 run 的日志路径与原因；任何 NaN、loss 爆炸。
7. 结论只按决策规则写：PROMOTE / REJECT / INCONCLUSIVE，不写“略有提升”之类的措辞。

---

## 9. 注意事项 / 常见坑

- **不要改协议参数**（步数、lr、seed、病例数、视角）去“救”某个实验；要改就作为新的明确命名实验提出来。
- **不要覆盖目录**；运行器拒绝覆盖是有意的。
- 同配置复跑也会有 0.1–0.8 dB 差异（GPU 非确定性），这正是复跑实验的用途；先完成基线与复跑再解读其他结果。
- `anchor-fusion-*`、`encoder-*` 的参数量与基线不同；结论要写“结构 + 容量”的变化，不是纯结构。
- 概率实验的验证默认用 `prior_mc_mean`（S=8），因此训练中验证比确定性实验慢。
- `prob-deterministic-control` 与基线的采样器不同（嵌套 D=12 的偏移范围更小），这两者的比较不是严格配对；
  `detail-gn-standard-sampler` 专门用来拆开这一项。
- `detail-single-scale-features` 的 `multiview_encoder.scale_encoding` 形状与共享初始化不同，运行器会自动
  从初始化中剔除该张量并记录在 `common_init_dropped`，属正常。
- GT-CVAE 训练需要读下采样 GT 体数据，IO 稍慢，属正常。
- 旧版 GT-CVAE（`configs/latent_stage_g*.yaml`，residual posterior）只兼容 max 融合骨干，且有已知的均值梯度抵消问题，
  **本计划不跑**；CVAE 路线以 `prob-gt-cvae`（修复版）代表。

---

## 10. 实验名速查

Stage 1（14）：`baseline-unet-transformer-meaninit`、`baseline-unet-transformer-meaninit-replica`、
`encoder-vit-per-view`、`encoder-hierarchical-window`、`anchor-fusion-maxpool`、`anchor-fusion-late-attention`、
`anchor-transformer-max-residual`、`anchor-transformer-no-residual-maxinit`、`point-fusion-geometry-attention`、
`view-update-alternating-2d3d`、`prob-deterministic-control`、`prob-gt-cvae`、`prob-dense-np`、`prob-dense-mean-code`。

Stage 2（16）：`detail-aligned-pixel-coordinates`、`detail-shared-3d-sampling`、`detail-anatomy-supervision`、
`detail-anatomy-guided-sampling`、`detail-single-scale-features`、`detail-no-spatial-neighbours`、
`detail-gaussian-nll-loss`、`detail-gn-standard-sampler`、`detail-gn-pointchannel-anchor-norm`、
`detail-dense-np-sigma-init-0.05`、`detail-dense-np-beta-x10`、`detail-dense-np-train-samples-4`、
`detail-dense-np-local-latent`、`detail-dense-np-multi-context-4-6-8`、`detail-dense-np-gaussian-nll`、
`detail-dense-np-frozen-reference-distill`。

旧名对照：基线 = T1-nomax / R0，maxpool = B0，late-attention = B1，max-residual = T1，no-residual-maxinit = T1-nores，
point-fusion = T2 的 point 部分，alternating = T1-nomax-alt，vit = T1-nomax-vit，hierarchical = H0 的编码器，
aligned = C0，shared-3d = S0，anatomy-supervision / guided = Araw 拆分，dense-np = P0，mean-code = Pmean，
gt-cvae = G0，local-latent = PL，distill = P1。
