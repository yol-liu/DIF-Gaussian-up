# DIF-Gaussian 完整训练实验安排（学生执行版）

更新：2026-09-26。本版按照**完整epoch、完整验证集、仅seed0**安排实验，取代此前以2000/10000步为主的排查计划。旧文件保留在[排查版归档](archive/EXPERIMENT_PLAN_20260925_recovery_steps.md)和[更早的结构筛查计划](archive/EXPERIMENT_PLAN_20260925_before_recovery.md)。本次只更新安排，未启动训练。

**正式训练统一400个完整epoch，本轮只跑seed=0。** 已有2000步参考结果都来自seed0，本轮保持同一seed与之对应；**seed=1、2的探索暂不进行**，不列入本轮工作量。首批必须完成8组×seed0=8次完整训练；扩展实验也使用同样的400 epoch和seed0。短步数测试仅用于检查程序能否运行，不能用短跑结果替代完整实验或直接淘汰结构。

## 1. 已有结果如何使用

- 原无概率T1-nomax在2000步时有PSNR 23.083580 / 23.100019 / 23.117404的历史记录。
- 隔离后的当前源码在**不读取任何权重文件**的条件下，从seed0随机初始化训练2000步，得到PSNR **22.889211**、SSIM **0.616085**。初始化状态与历史起点相同，未导入概率模块。
- 旧最佳权重在当前代码上可以精确复评；GPU backward存在非确定性，已验证只开确定性标志不够。
- 这些结论用于确定基线和排查风险，**不能说明任一结构在完整训练后优于或劣于另一结构**。历史23.1不是400 epoch实验的性能上限，也不是其必须精确复现的数值。

依据：[从零训练报告](../reports/recovery_audit_20260925/fresh_isolated_reproduction/result.md)、[隔离验证](../reports/recovery_audit_20260925/baseline_isolation/result.md)、[版本差异](../reports/recovery_audit_20260925/current_vs_best/analysis.md)。

## 2. Epoch的定义及统一训练协议

### 2.1 一个完整epoch是什么

目前LUNA16训练集738例；单卡、batch size=1、无梯度累积时：

| 实际完成的epoch | optimizer updates | 用途 |
|---:|---:|---|
| 1 | 738 | 遍历一次训练病例；不是遍历每个病例的所有体素 |
| 20 | 14760 | 第一次正式全eval验证节点 |
| 100 | 73800 | 阶段学习曲线检查，不提前结束正式实验 |
| 200 | 147600 | 中期比较节点，不替代400 epoch结论 |
| 300 | 221400 | 后期学习曲线检查 |
| **400** | **295200** | **每个正式run必须完成的预算** |

每个病例仍随机抽取block和1024个query，因此“完整epoch”表示病例级完整遍历，不表示完整体数据监督。原2000步约2.71 epoch，原10000步约13.55 epoch。

**当前代码的执行注意：** `code/train.py` 使用 `range(start_epoch, args.epoch + 1)`，从0起会多跑一轮；同时 `--epoch` 参与LR调度。为保持400 epoch调度并精确完成400轮，当前入口使用 **`--epoch 400 --max_steps 295200`**。这里max_steps是完整epoch预算的精确截止，不是短步数筛查。不要只写 `--epoch 400`，也不要随意改成399而改变LR调度。

按此命令正常完成时，现有 `run_summary.json` 会记为 `STOPPED_LIMIT`。验收应核对 `optimizer_updates = global_step = 295200`、最终验证和checkpoint均已保存，而不是要求状态字符串为 `COMPLETED`。达到预算后代码还会进入一个无更新的终止循环，可能打印 `epoch: 400, loss: nan` 并多调用一次scheduler；这条空循环日志不能单独判为训练NaN，须检查实际更新中的loss与异常记录。该终止循环不改变模型权重。

上述换算只适用于当前738例、单卡、batch1。不得为提速私自修改batch、病例数、DDP进程数或梯度累积。若协议确需改变，按实际 `len(train_loader)` 重算，并重跑对应对照。

### 2.2 所有主实验固定的设置

| 项目 | 设置 |
|---|---|
| 数据 | LUNA16：train738 / eval50 / test100；根目录 `/media/wyin/Expansion/Data/LUNA16/C2RV-CBCT/data` |
| 输入 | 6真实投影，固定raw IDs `[0,8,16,25,33,41]`；不启用random_views或view dropout |
| 输入分辨率/输出评价 | 维持现有数据处理；评价scale=1.0，query chunk=65536 |
| 起点 | 每次从随机初始化开始，Python/NumPy/Torch seed统一为0 |
| 权重约束 | 不传common_init、warm_start、resume、resume_path；不加载历史权重、公共初始化文件、teacher/reference |
| 优化器 | AdamW，lr=2e-4；共享骨干weight decay=0.01，融合模块wd=0；记录参数分组 |
| 调度 | 保持原StepLR：每epoch乘 `(1e-3)^(1/400)`；400次衰减后LR约2e-7 |
| 训练 | FP32、batch1、queries1024、workers2、global grad clip=1.0；不开AMP、不开DDP |
| 长度与seed | **400完整epoch × seed 0**（seed1/2暂不跑）；无按验证成绩提前停止 |
| Plain基线 | UNet，anchor mean-init geometry Transformer，无max residual，point max，原BN、MSE、12³ anchors |
| Plain关闭项 | probabilistic字段不存在、旧latent关闭；anatomy/view_update/teacher/ray关闭；legacy坐标、detector采样 |
| 来源与输出 | 每批冻结源码/config/env/data清单；run名包含arm、seed、400ep、版本；不覆盖任何旧run |

### 2.3 初始化公平性和隔离要求

训练禁止读权重文件，可沿用审计中的 `torch.load` / `torch.serialization.load` 禁止加载检查。保存初始254项状态或相应新结构的状态哈希；原T1-nomax seed0的参考SHA256为 `021d532c1d5d5ddf51cdd32402f04f982b3d0280a99f41217a77ee319281fe7f`。

同seed不同结构不一定得到相同共有参数。必须核对共有参数哈希并记录差异；不一致时不宣称“严格仅结构变化”。需要纯结构归因时，用独立模块RNG从随机数重新生成相同共有初始化，并对双方统一重跑；不读旧权重文件。

当前plain运行保持概率模块零导入。备份源码自身会导入旧factory，但必须确认未建立latent、未替换BN、未使用概率loss；不要为了满足零导入而修改指定备份。概率实验放在独立源码/配置/日志中，不回写冻结plain。

## 3. 验证、保存、选模型和汇报频率

1. **每20个完整epoch验证一次完整50例eval**：E20、40、…、400，共20次。每次同时给出原6例子集和完整50例两张表，不把二者均值混用。不能让部分arm只评6例、另一部分评50例后直接排名。
2. E100、200、300、400提交阶段曲线和累计耗时。到E100表现较差仍继续到E400；NaN、数据错误、OOM等工程失败立即记录并排查，不把失败结果静默删除。
3. 主比较使用**E400最后模型**；补充表使用20个相同eval节点中的best-eval PSNR模型。所有arm采用相同规则，不按病例或指标挑checkpoint。
4. 每20 epoch保存模型及完整实验记录。当前入口会保存full checkpoint，约0.4GB/次，400epoch约8GB/run；加最终partial和日志，先按**10–12GB/run**预留空间。若实现分级存储，再保留每20epoch纯权重、last/best完整checkpoint，不能声称现有入口已经自动做到。
5. 只允许训练结束后加载**该run自己新产生的checkpoint**做离线评价；不得把它用作另一个run的起点。当前resume不能保证精确中断恢复，本轮不把拼接续训冒充完整从零run；如需支持中断恢复，先验证epoch边界的模型/优化器/调度/RNG/loader状态恢复。
6. 首epoch记录全部738步输入摘要；长程使用逐epoch输入摘要/流式日志，避免把295200条大字典一直堆在内存。现有CLI可用 `--manifest_steps 738`，此时只证明首epoch配对，不能声称400epoch所有输入都已核查；全程流式摘要是待补的记录能力。
7. 每个run记录：train loss、LR、梯度与裁剪情况、BN统计、Gaussian跨anchor特征方差、PSNR/SSIM/unclipped MSE/clip比例/边界和梯度误差、时长与显存。额外诊断保持双方一致，不增加随机forward污染训练。

## 4. 第一批必做：核心比较与机制消融

以下8组各跑seed0，**每个run训练400完整epoch**。配置为当前可参考路径；运行前核对解析后的配置差异，冻结到实验目录。

| ID | 实验 | 配置/源码参考 | 直接比较对象 | epoch/run | seed | runs |
|---|---|---|---|---:|---|---:|
| M0 | 隔离后T1-nomax | `configs/multiview/T1-nomax.yaml`，当前隔离源码 | 主基线 | 400 | 0 | 1 |
| M1 | 指定备份T1-nomax，从零训练 | `backups/t1-nomax-best-20260924/code`及同名配置 | M0：版本差异 | 400 | 0 | 1 |
| M2 | 无anchor Transformer，max融合 | `configs/multiview/B0.yaml` | M0：整个新融合结构 | 400 | 0 | 1 |
| M3 | 旧late geometry attention | `configs/multiview/B1.yaml` | M2及M0：融合路线 | 400 | 0 | 1 |
| M4 | 无残差，token改为max初始化 | `configs/multiview/T1-nores.yaml` | M0：mean/max初始化 | 400 | 0 | 1 |
| M5 | max-init上添加max residual | `configs/multiview/T1.yaml` | **M4**：残差作用 | 400 | 0 | 1 |
| M6 | spatial window由3改1 | `configs/experiment_plan/stage2/detail-no-spatial-neighbours.yaml` | M0：邻域交互 | 400 | 0 | 1 |
| M7 | feature scales只保留1/16 | `configs/experiment_plan/stage2/detail-single-scale-features.yaml` | M0：多尺度作用 | 400 | 0 | 1 |

**合计：8 runs，3200 model-epochs，2,361,600 optimizer updates。** M0的seed0结果在后续各表复用，不重新训练或复制它的权重作为变体初始化。

学生执行顺序：

- 第一组提交：M0、M1，共2次完整训练，先建立两版本完整学习曲线，并与已有seed0 2000步结果对应。
- 第二组提交：M2–M5，共4次完整训练，回答max、late attention、mean-init、max residual的作用。
- 第三组提交：M6、M7，共2次完整训练，完成邻域和多尺度消融。
- seed1、2暂不安排；是否补跑待本轮seed0结果出来后另行决定，届时须对同一批对照统一补齐，不能只给表现好的模型补seed。

单seed下arm间差异同时包含结构变化与GPU非确定性。若需量化同seed GPU波动，**可选**让M0、M1的seed0各重复1次，每次仍400epoch，共2 runs，作为本轮唯一的噪声参考；这不是用几次2000步短跑替代核心训练。备份是训练后保存的候选快照，不假定其每一行等于历史最佳训练当时源码。

## 5. 第二批：几何、局部融合与编码器扩展

保持M0所有未列出的设置不变。每组仍为**400epoch × seed0**，不因已有2000步成绩差而提前否定。

| ID | 实验 | 配置参考 | 必需直接对照 | epoch/run | runs |
|---|---|---|---|---:|---:|
| G0 | point geometry attention | `configs/experiment_plan/stage1/point-fusion-geometry-attention.yaml` | M0 point max | 400 | 1 |
| G1 | 2D/3D交替view update | `configs/multiview/T1-nomax-alt.yaml` | M0 view_update=false | 400 | 1 |
| G2 | 仅anatomy辅助监督，legacy坐标 | `configs/experiment_plan/stage2/detail-anatomy-supervision.yaml` | M0 | 400 | 1 |
| G3 | anatomy监督+引导采样 | `configs/experiment_plan/stage2/detail-anatomy-guided-sampling.yaml` | **G2**，再与M0比 | 400 | 1 |
| G4 | aligned像素坐标及mask | `configs/experiment_plan/stage2/detail-aligned-pixel-coordinates.yaml` | M0 legacy | 400 | 1 |
| G5 | aligned基础上shared3d采样 | `configs/experiment_plan/stage2/detail-shared-3d-sampling.yaml` | **G4**，再与M0比 | 400 | 1 |
| E0 | per-view ViT encoder | `configs/multiview/T1-nomax-vit.yaml` | M0 UNet | 400 | 1 |
| E1 | hierarchical window encoder | `configs/experiment_plan/stage1/encoder-hierarchical-window.yaml` | M0 UNet | 400 | 1 |

本批新增8 runs；核心加本批共16 runs。E0/E1的容量不同，报告参数量、吞吐和显存，不将结果全部归因于attention形式。配置生成器可能带入不同底座，必须实际检查：只换encoder的实验不能同时启用aligned/shared3d。

必要的后续归因实验也按400epoch、seed0，不进行无计划组合：

- G0若有收益：query保留位置但移除GS feature，decoder仍保留GS插值，区分attention与3D条件贡献（开关待实现）。
- G3若有收益：相同support trace的各向同性对照，区分方向性与半径变化（待实现）；不能用kappa=0冒充等尺度对照。
- G4若有变化：拆像素中心映射和有效性mask，两因素分别测试（待实现），避免把两个改动当作一个原因。
- 组合模型必须保留单项和组合的同预算对照；只按预先选定的组合补实验，不穷举所有开关。

## 6. 第三批：独立概率路线

这一批在plain基线冻结后单独进行，不能影响M0/M1。原plain使用BN；概率底座使用什么归一化和采样必须显式标注，`mode: deterministic`不是原plain。统一MSE，概率超参数先固定，避免边做结构比较边调整beta/sigma。

| ID | 实验 | 配置参考 | 直接比较对象 | epoch/run | seed / runs |
|---|---|---|---|---:|---|
| P0 | GN+anchor IN，原6-view采样，无latent | `configs/experiment_plan/stage2/detail-gn-standard-sampler.yaml` | M0：归一化代价 | 400 | 0，1 |
| P1 | 相同归一化，嵌套D12/C6采样，无latent | `configs/experiment_plan/stage1/prob-deterministic-control.yaml` | P0：采样代价 | 400 | 0，1 |
| P2 | P1的anchor IN改PointChannelNorm | `configs/experiment_plan/stage2/detail-gn-pointchannel-anchor-norm.yaml` | P1：anchor归一化 | 400 | 0，1 |
| P3 | dense mean-code | `configs/experiment_plan/stage1/prob-dense-mean-code.yaml` | P1及P4 | 400 | 0，1 |
| P4 | dense NP，global latent | `configs/experiment_plan/stage1/prob-dense-np.yaml` | **P1**：随机latent净效应 | 400 | 0，1 |
| P5 | GT条件CVAE | `configs/experiment_plan/stage1/prob-gt-cvae.yaml` | P1及P4 | 400 | 0，1 |
| P6 | dense NP增加local latent | `configs/experiment_plan/stage2/detail-dense-np-local-latent.yaml` | **P4** | 400 | 0，1 |

新增7 runs。plain、归一化、采样和latent影响分开报告。P分支现有代码默认beta=2e-5、warmup=500 optimizer updates、global dim64、initial sigma0.5等，应在开跑前从配置导出完整参数表并冻结；这些内部步数参数不是训练终止预算，不能随着各arm自行改动。

P4/P5/P6在E100/E200/E400以及best-eval checkpoint做离线诊断：prior sample、prior latent mean、prior MC mean S4/S8、oracle posterior、KL、跨病例latent方差、换病例z和zero-z。主部署点预测统一prior MC mean S8；oracle用额外信息，只能作为诊断。概率验证成本高，须实测，不能直接套plain的时间估计。

oracle增益不足0.1 dB只能说明本实验没有观察到足够利用，不能直接证明信息为零。最终也必须比较概率模型与M0的部署质量和成本。冻结参考蒸馏需要读取训练权重，不在本轮“不复用权重”安排内；旧Stage G不作为新的主实验。

## 7. 优化稳定性实验：作为单独课题，不阻塞完整训练

已有GPU非确定性现象成立。可以在运行完整模型实验的同时安排算子级验证，但不要求学生先彻底消除非确定性才开始主实验。

- 正确性预检：固定features/grid/upstream，同进程重复20次、独立进程3次，检查前向与feature/grid梯度；再做真实模型10步状态对照。这些是工程测试，不进入epoch实验主表。
- 如实现了真正可重复的grid_sample backward，先验证误差、边界与性能，再将新算子与原算子各跑**400epoch × seed0**；可复用协议一致的M0结果，不能只用20步一致性宣布质量更好。
- 如需测试lr、clip或query，分别设置lr=1e-4、clip=5、queries4096的单因素实验，每个均400epoch×seed0。当前clip为硬编码，需先实现开关。只在主线完成或有明确诊断依据时选择这些附加实验，不同时改三项。
- queries4096会改变训练样本和计算量，必须报告总query预算，不能称等计算量比较。结构效果和超参数效果分表。

## 8. 学生可使用的当前入口示例

旧 `scripts/run_experiment_plan.py` 会加载公共初始化，旧 `scripts/run_t1_nomax.py train` 也默认读取初始化文件；都不能直接作为本轮从零训练命令。`scripts/train.sh` 默认是latent Stage G和DDP，也不是本计划主线。

以下是**现有CLI支持的M0、seed0、400个完整epoch**示例，在仓库根目录及已冻结的当前源码上执行。源码冻结/哈希记录、零加载检查和长程流式manifest可由统一外层runner补齐；这些增强尚不能冒充已经实现。

```bash
PY=/home/wyin/miniconda3/envs/torch2-3/bin/python
val_steps=()
for ((exp_epoch=20; exp_epoch<=400; exp_epoch+=20)); do
    val_steps+=("$((exp_epoch * 738))")
done

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 "$PY" code/train.py \
    --name full-M0-T1nomax-seed0-400ep-v1 \
    --cfg_path configs/multiview/T1-nomax.yaml \
    --dst_name LUNA16 --num_views 6 \
    --batch_size 1 --num_points 1024 --num_workers 2 \
    --epoch 400 --max_steps 295200 \
    --lr 2e-4 --weight_decay 1e-2 --lr_decay 1e-3 \
    --val_steps "${val_steps[@]}" --val_case_count 50 \
    --eval_scale 1.0 --eval_npoint 65536 \
    --manifest_steps 738 --diag_every 25 \
    --seed 0 --device cuda --no-wandb
```

本轮只用 `--seed 0`，不跑seed1/2；M1将入口与配置指向备份。其他arm只改配置、run名，保持其余协议一致。绝不添加common_init/warm_start/resume。运行前做一次短smoke检查日志的初始化和模型类型；smoke用独立名字，其权重不用作正式训练起点。

上述验证步数为14760、29520、…、295200，对应**实际完成**20、40、…、400轮。程序中的0-based epoch标签会比实际完成轮数小1，提交表统一使用completed_epochs，不直接照抄标签。

## 9. 结果分析、预算与学生交付

### 9.1 固定的分析要求

- E400全50例eval主表：各arm的seed0结果；同时给出逐病例配对差值、SSIM/MSE/边界/梯度和成本。病例不是独立训练重复，病例bootstrap不代替跨seed分析。
- 另外提交best-eval表、E20至E400曲线、原6例子集表。可以与已有seed0 2000步结果对照学习曲线，但不能与历史6例2000步结果直接排名并宣称结构进步。
- 本轮只有seed0，所有结论标注为“单seed结果”。arm间差距若小于同seed重复run（第4节可选M0/M1重复）的波动，不下结论；单seed好结果不作为更换主基线的充分证据。
- 400epoch是统一充分训练预算，不是保证收敛。若E300–400仍明显持续改善，报告“预算内未收敛”；可另立**600epoch、相同对照双方、seed0、从零训练**的补充实验，预先写清LR调度，不只给表现好的模型追加训练。
- 模型和选checkpoint规则冻结后评估test100；test不参与选参。主test表采用seed0依同一eval规则选择的checkpoint，并清楚标明选择规则。

### 9.2 计算与磁盘预算

最近一次2000步用时约678秒，其中约450秒为训练、约228秒为两次6例验证。按此粗估，单个plain run的400epoch训练约19小时，加20次完整50例验证约5小时，连同IO/日志预留为**约25–30 GPU小时/run**。这是估算，第一条完整run完成后更新；ViT、point attention和概率模型可能明显更慢。

| 批次 | 模型组数 | seed | epoch/run | runs | 累计model-epochs |
|---|---:|---:|---:|---:|---:|
| 第一批核心M0–M7（必做） | 8 | 0 | 400 | 8 | 3200 |
| 第二批G0–G5、E0–E1 | 8 | 0 | 400 | 8 | 3200 |
| 第三批P0–P6 | 7 | 0 | 400 | 7 | 2800 |
| 全部列出矩阵 | 23 | 0 | 400 | **23** | **9200** |

第一批预计200–240 GPU小时，即单张3090约8–10天；4张同档GPU各自单卡跑不同run，理想并行约2–3天，实际看IO。可选的M0/M1同seed重复另加2 runs。不要改成DDP后仍沿用上述epoch/update换算。比较arm尽量分配到相同GPU型号并记录设备，避免硬件与模型完全绑定。

现有保存方式按10–12GB/run估算，第一批需80–96GB，加已有记录建议预留至少100GB；若当前磁盘不足，先落实新结果存储或经核验的checkpoint分级保留策略，不删除历史最佳备份、不盲目启动8轮。全部23轮按原保存方式约230–276GB，尚不含大量预测体积；完整体积只导出规定checkpoint与病例。

本计划给出完整工作量和先后批次，不要求所有批次同时启动。学生先交第一批完整结果；扩展按资源顺序推进，所有正式行都保持完整400epoch，不能换成2000步结论。

### 9.3 每个run及每批提交清单

- [ ] 开跑前：arm/seed/epoch/配置diff/源码与环境哈希/初始化方式/显存与磁盘预算。
- [ ] 完成400epoch、295200次更新；核对训练病例数，提交run_summary与最终checkpoint。
- [ ] E20至E400共20个完整50例eval结果、原6例子集结果、规定checkpoint及选择记录。
- [ ] E100/E200/E300/E400曲线检查；参数量、吞吐、耗时、显存、失败/重试说明。
- [ ] 每组提交seed0结果和逐病例比较，标注为单seed结论；seed1/2本轮不跑。
- [ ] 概率组另交latent利用及oracle诊断；oracle不算部署指标。
- [ ] 主表冻结后提交test100、统一显示窗切片及误差图。

任何异常停止都记为未完成或失败，用新名字重跑，保留原失败原因。达到295200次更新并通过上述验收的正常 `STOPPED_LIMIT` 不属于异常停止。不得把部分epoch、旧权重评估、加载checkpoint继续训练混入“从零完成400epoch”的表格。
