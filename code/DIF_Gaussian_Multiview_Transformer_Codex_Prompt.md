# Codex 实现任务：DIF-Gaussian 多视角几何 Transformer

请直接在当前 DIF-Gaussian 工作仓库中实现下面的方案，完成必要验证并交付可运行代码。不要只输出分析、计划、伪代码或没有接入训练的模块。先扫描实际代码；下面的文件名和接口是已审阅版本的依据，若当前代码有变化，以实际代码为准并说明差异。

目标仓库：https://github.com/yol-liu/DIF-Gaussian-up 。如果当前目录已经有该仓库，使用现有 checkout。保留用户的未提交修改，遵守适用的 AGENTS.md。确认实际训练入口使用的是 `code/` 还是其他目录；不要同时修改旧的 `code copy/` 副本。

## 1. 目标和本轮范围

任务是改进 generalizable sparse-view CT/CBCT reconstruction：构建多尺度 2D stem、共享 3D tokens、几何约束的跨视角 attention、局部 3D attention 组成的混合 encoder，并加入可消融的解剖结构引导采样。共享 3D tokens 复用现有 Gaussian anchors，最终仍由 Gaussian 特征场和 INR 重建 CT。

本轮必须完整实现：

1. 几何 Transformer 主干和 Gaussian 分支接入。
2. 使用共享 3D 信息的高分辨率查询点融合。
3. 解剖结构预测、GT 结构监督、下一层采样调整，以及完整的训练损失接入。
4. 配置开关、旧模式兼容、必要的数值测试、可执行的消融入口和实现报告。

投影一致性和 sparse–dense teacher 对齐作为独立扩展，在主干通过验证后处理。具备真实数据与标定条件时实现并做 smoke test；若缺少必要信息，明确记录阻塞原因，默认关闭并在误开启时给出清晰错误。不能用返回零的损失或空函数冒充完成。已有 latent neural process 本轮默认关闭，不重写其 prior/posterior/KL 设计。

研究假设是改善多视角融合及空间结构恢复，不预设 PSNR、SSIM 会提升，也不把新 encoder 当作 test scale 差异的直接解释。

## 2. 扫描现有实现并记录接口

优先读取：

- `models/unet.py`、`models/base.py`、`models/model.py`、`models/view_fusion.py`。
- `models/gs_utils.py`、`models/utils.py`、`models/point_decoder.py`。
- `models/probabilistic_model.py`、latent/GT encoder，以及损失实现。
- `datasets/base.py`、`datasets/dst_gs.py`、`datasets/geometry.py`。
- `train.py`、`evaluate.py`、配置加载器、默认及 geometry_attention 配置、训练脚本。

已审阅版本的关键事实需要现场确认：

- UNet 输入 `[B,V,1,H,W]`，内部合并 B、V 独立编码；返回 `feats` 和 `feats_ds`。默认 `out_ch=128`、`ds_ch=1024`，瓶颈为约 1/16 分辨率。
- `DIF_Gaussian.encode_projs` 产生 `[B,1024,K]` 的 `anchor_feats`，供 Gaussian 两个 MLP 及可选 latent 分支使用。
- `GeometryAwareViewFusion` 在每个 anchor 上独立查询 view tokens，多层主要更新 query；没有 anchor 之间的空间 attention。
- `forward_points` 将 pixel-aligned 特征与 Gaussian 插值特征拼接；默认 decoder 输入是 `128+128`。
- `Recon_base.forward` 在 eval 时只 encode 一次，再按点分块解码；训练和概率 wrapper 可能走不同路径。
- `points_gs` 当前由 `np.mgrid / gs_res` 生成，不是默认的端点 linspace 或体素中心网格。必须验证 reshape 顺序及插值坐标。
- 几何实现包含角度反号、z 翻转、轴顺序、DSO/DSD 和物理尺寸；不要自行套用通用相机约定。

先写简短接口说明并立即进入实现。小型设计选择自行决策，缺少数据时继续完成不依赖数据的部分。

## 3. 保持可比较的基线

默认配置仍走旧路径。增加新模式，例如：

```yaml
model:
  anchor_fusion: geometry_transformer
  point_fusion: geometry_attention
  multiview_transformer:
    d_model: 128
    num_heads: 4
    num_blocks: 2
    feature_scales: [4, 8, 16]
    samples_per_scale: 5
    spatial_window: 3
    dropout: 0.0
    anchor_chunk_size: 256
    point_chunk_size: 8192
    residual_init: 0.001
    anatomy:
      enabled: false
  latent:
    use_latent: false
```

这是目标配置语义，需按项目现有配置系统落地；所有键必须实际生效，不能静默忽略未知模式。默认继续使用 `gs.res=12`，dataset 的 gs_res 必须一致。内存相关值可按实测调整，记录原因。

保留原 `max` 和 `geometry_attention` 路径。旧 UNet 的已有参数名及返回键尽量不变；新配置需要额外返回多尺度特征。先复用现有 UNet 作为 2D stem，不进行纯 ViT 替换，也不同时修改 Gaussian KNN、权重归一化、中心偏移逻辑或 latent loss。

核查已知基线问题：PointDecoder 最后一层判断可能错误地保留 BN/激活；训练验证可能固定 scale=0.5。需要修复时使用独立、明确的兼容选项或独立改动，保留 legacy 复现路径。所有用于比较的新旧模型应用同一组修复；不能把基线修复收益归到 Transformer。新主干与基线使用相同 reconstruction loss 和数据设置。

## 4. 几何与数据契约

实现集中管理的几何 adapter，负责 normalized xyz、物理坐标、投影 grid、射线方向和 Jacobian 的转换。禁止在模块间重复写略有不同的坐标公式。

要求：

- 从数据集真实 projector 配置读取 `nVoxel/dVoxel/nDetector/dDetector/DSO/DSD` 及实际需要的标定参数。训练和 eval 都可访问物理几何，不从 GT 输出尺寸猜测扫描范围。
- 3D 位置编码使用经过明确尺度归一化的物理坐标；射线编码至少含方向、距离和探测器位置。角度 sin/cos 可补充，不能作为完整标定的替代。
- 明确代码注释中的 W/H 与 PyTorch/grid_sample 的 H/W、grid x/y 是否一致。明确 align_corners 及每个 feature level 的 pixel-to-grid 映射。
- 先保持与现有投影函数兼容，使用非对称 phantom 与已知位置检查坐标。如果发现旧几何错误，独立修正并对所有比较模型统一应用。
- 视角数可以变化；任意重排投影和对应标定后，eval 输出应在浮点容差内不变。不要使用固定 view index embedding。
- 支持 view mask、各 offset 的 FOV mask、无效深度/奇异投影 mask。某点所有采样无效时要有有限值 fallback，不能 softmax 全负无穷或留下 NaN。
- view mask 要在 anchor 与 point 两条分支一致生效。训练 masked views 不能通过共享 BatchNorm 统计影响有效视角：优先在 stem 前仅打包有效视角再散回；评估测试同时覆盖改变 masked input 的情况。
- 投影缓存只能在几何、角度、查询坐标/分辨率相同的条件下复用；不要依赖首个样本的角度缓存。

## 5. 多视角 Transformer 主干

建议新增 `models/multiview_encoder.py`，具体拆分服从仓库风格。

### 5.1 多尺度 2D features

在不破坏旧 UNet 接口的前提下暴露 1/4、1/8、1/16 特征，用各尺度独立 1×1 projection 映射到 d_model。高分辨率 `feats` 继续保留，防止粗 3D 网格成为唯一细节通路。

### 5.2 共享 3D tokens

K 与现有 anchors 数一致。初始化由 anchor 物理位置编码和经过 masked 聚合的图像证据构成；不要只给每个固定 anchor 一个不可迁移的可学习 ID。

每个 block 包含：

1. 几何局部采样和跨视角 cross-attention。
2. anchor 网格中的局部空间 self-attention。
3. pre-LayerNorm、残差和 FFN。

对 anchor x_k，构造：

```text
t[i,k,s,m] = W_s F_i^s(project_i(x_k) + delta[i,k,s,m])
             + geometry_embedding[i,k,s,m] + scale_embedding[s]
z_cross[k] = z[k] + CrossAttention(z[k], t[:,k,:,:], valid_mask)
z_next    = SpatialTransformer(z_cross, physical_relative_positions)
```

第一次采样用中心加四个邻域点，偏移尺度以对应特征图像素定义再正确转成 grid 坐标。跨视角、尺度、局部样本一起作为 keys/values。记录 attention mask 的维度与语义。

空间 attention 使用窗口或局部邻接，加入物理空间相对位置；通过 shifted windows 或重叠邻接让连续 blocks 跨窗口传播。不要构造 K×K 全局 attention 或 `(VHW)×(VHW)` 矩阵。边界 padding 必须 mask。

当前最小方案允许 2D maps 在 blocks 之间保持固定，反复更新共享 3D tokens 和采样位置；不要声称实现了 3D→2D 双向更新。后续若要更新 2D maps，应作为独立扩展。

### 5.3 接回 Gaussian 分支

把最终 `[B,K,d_model]` tokens 经 adapter 映射回原来的 `[B,ds_ch,K]` `anchor_feats`，再交给现有 Gaussian heads。可以用 masked max 的旧特征作为残差基底，但必须记录新分支与基底的范数比。

残差采用小的非零初值，避免整个新 trunk 首步无梯度。若采用零初始化最后一层，必须解释并验证后续步骤梯度恢复，不能把首步 trunk 零梯度误判为完成。不要同时把残差 gate 和 projection 都置零，形成持续死分支。

不要将 anatomy sampling covariance 与 Gaussian feature interpolation covariance 混为同一个参数；本轮先分开。

## 6. 高分辨率 point 分支

新增可切换的 geometry_attention point fusion：

1. 先计算现有 Gaussian 插值特征 `interp_feats`，每个 query point 得到共享 3D 条件。
2. 从各视角高分辨率 `feats` 采样该点的 pixel-aligned features。
3. 用 `interp_feats` 和物理位置构造 query，用各视角特征及射线编码构造 keys/values，输出 `[B,out_ch,N]`。
4. 与原 `interp_feats` 拼接，保持 point decoder 输入通道不变。

可采用 masked max + 小残差以稳定训练。这里一个 query 只关注对应位置的 V 个 view tokens，初版不加多尺度多偏移，控制全分辨率推理开销。按 query points 分块，不跨 query points 做 attention。采用逐点 LayerNorm 等不依赖 chunk 中其他点的归一化。

编码每个 volume 只执行一次；不得在每个 query chunk 内重新编码投影、重新预测 anatomy 或重建 anchor tokens。合理缓存视图 feature projections，避免每个 chunk 重复整张特征图上的线性变换。

## 7. 解剖结构引导采样：完整实现，独立开关

在第一轮具有 3D 空间交互的 tokens 后增加结构 head。构造 PSD、trace 归一化的结构张量，例如预测 L 后：

```text
T_hat = L L^T / (trace(L L^T) + eps)
Sigma_support = ell^2 * inverse(I + kappa * T_hat) + sigma_floor^2 * I
```

ell、sigma_floor 用 mm，给出合理有界参数化；不能直接把无界矩阵当 covariance。初版可固定 ell/kappa 并在配置显式给出或依据 anchor 物理间距推导，说明单位。

训练 GT：

```text
g      = physical_gradient(G_sigma * CT_gt)
S_star = G_rho * (g g^T)
T_star = S_star / (trace(S_star) + eps)
L_anatomy = sum(c_star * ||T_hat - T_star||_F^2) / (sum(c_star) + eps)
```

sigma/rho 以物理尺度定义并转换到各轴 voxel units。梯度考虑 spacing、轴排列、物理坐标变换；GT tensor 在与预测一致的坐标系表达。confidence c_star 来自 GT 结构能量并设稳定阈值/上界，不能让模型自由把 loss 权重降到零。

GT supervision 仅在训练使用，可缓存低分辨率结构标签，避免每一步加载/处理完整 CT。结构头的输入始终来自投影特征，不允许把 GT tensor 或 GT volume 输入主 encoder。没有 GT 的 inference forward 必须可运行。支持仅 anatomy supervision 和 supervision+sampling 两种配置以做消融。

把 Sigma_support 通过当前视角物理投影 Jacobian 转为探测器/feature grid 的二维 covariance：

```text
U[i,k,s] = J[i,k,s] Sigma_support[k] J[i,k,s]^T + eps I
```

明确 J 的输出单位，并用有限差分验证。通过稳定的 2×2 Cholesky 等得到中心和 ±两个支撑方向采样点，限制最小/最大 footprint；每个 offset 单独检查 FOV。让下一 block 使用这些采样点。warmup 从固定邻域逐渐过渡到 anatomy offsets，记录实际 footprint 与梯度，防止实现了 head 却未改变采样。

coarse anchors 能表示的结构有限，不宣称 12³ tokens 已恢复细小血管拓扑。本轮不加入器官分割网络、clDice 或对衰减场施加 Eikonal loss。

## 8. 两个后续一致性扩展

先完成上述主干及 anatomy 的真实训练/推理接入，再处理以下扩展；两者分别默认关闭。

### 8.1 最终输出的投影一致性

检查原始投影是强度、对数线积分还是归一化/量化结果，检查 CT 输出与衰减系数的映射。已审阅数据涉及 uint8、projs_max 等字段，不能直接假设模型值就是 mm^-1。

若单位、投影 preprocessing、扫描几何可恢复，实现分块射线采样，对最终 INR 衰减输出做线积分，并与同一数据域的投影比较。编码结果必须复用；训练可随机采少量 rays，执行先验独立的均匀物体/球体 phantom 检验及梯度检验。全部组织沿射线贡献都要计入，不使用不透明表面 alpha-compositing。

只在物理坐标、路径长度、数值积分与输入标度验证后启用。若这些信息缺失，报告具体缺失项及最小所需数据，不使用可学习缩放偷偷吸收未核实的标定错误。Gaussian 当前存的是特征，不能把它们直接解析积分当作投影。

### 8.2 sparse–dense 对齐

先做确定性的空间表示对齐，不引入新的 KL。dense teacher 必须来自同一训练对象更多真实视角的训练/验证流程，不能把同一 sparse input 复制多遍称为 dense。

对齐固定物理 anchors 的 normalized features 或共享结构输出；避免在随机漂移的独立特征基底之间直接做无意义 MSE。优先相同语义/对齐初始化的 encoder，若 teacher 独立训练则使用有明确防塌缩约束的 adapter 或关系/结构目标。teacher stop-gradient 并采用 eval 行为；同时保留重建监督。

只在可靠区域/结构上对齐，记录 teacher 重建质量、feature 方差和 student reconstruction。不要强迫不同稀疏子集拥有完全相同的不确定性。teacher 仅训练可见，测试不得访问 dense views/GT，也不能从验证或测试对象构造训练 teacher。没有可用 teacher 时完成数据与调用接口的可执行验证，但明确质量收益尚未评估，不用随机 teacher 跑损失充当有效实验。

## 9. 训练整合和配置

新增模块必须进入 optimizer、device/AMP 管理、DDP 和保存/加载路径。辅助损失所需输出经统一 forward 返回；训练调用 DDP wrapper 的 forward，不绕过 wrapper 单独调用 module.encode_projs 来反传。

anatomy loss 接入当前实际 loss 入口；报告每项 loss、权重和梯度量级。所有新 loss 默认在基线配置关闭，anatomy 实验配置显式开启。不要自动把 MSE 换成 Laplace，或同时改变采样/归一化使比较失效。

至少提供下列配置或 runner arms：

- B0：max baseline。
- B1：现有 late anchor geometry_attention。
- T1：新的共享 3D Transformer，point 分支仍 max，anatomy 关闭。
- T2：T1 + 新 point fusion。
- T3：T2 + anatomy supervision，固定 sampling。
- T4：T3 + anatomy-guided sampling。
- ray consistency、teacher alignment 各自为单独追加 arm，只有依赖验证后才运行。

默认运行目标为 T2 和 T4 的有限 smoke test，不自动启动整个消融矩阵。所有 arm 使用一致的数据划分、训练样本、优化器、训练预算、loss、seed 和公共 bug fixes，并报告参数量与显存/时间差异。

eval 配置显式支持 scale=0.5 和 1.0。分别报告各自原生网格指标；另提供在同一参考网格/同一低通目标上的诊断，不把跨目标的 PSNR 差直接归因于模型。区分 D(pred_1) 对 D(GT_1) 与 pred_0.5 对同一 D(GT_1)；坐标重采样需按实际物理定义，不能只 resize 后声称完全对齐。高频改善至少附一致显示窗下的切片、边缘/梯度误差或高频误差。

## 10. 必须通过的必要验证

这些测试用于几何、mask、梯度、兼容性等真实风险；避免只重复实现公式的测试。

1. 旧模式兼容：新功能关闭时，同一权重和输入的 eval 输出与修改前参考一致；检查实际 checkpoint key 兼容性。若当前没有 checkpoint，明确区分随机权重回归测试与真实 checkpoint 测试。
2. 几何正确性：非对称点/phantom，至少两个角度；轴顺序、符号、grid_sample 对齐；新 projector 对既有 projector 的对照及独立物理检查；Jacobian 与有限差分一致。
3. view permutation：eval 中同步重排所有 per-view 张量，输出保持数值一致。
4. masks：改变 masked view 像素不会影响有效输出；支持可变 V、单视角、部分 FOV、某点全部无效；无 NaN/Inf。说明全样本无有效视角的约定，训练 sampler 不产生这种样本。
5. 梯度连通：reconstruction loss 能训练 cross-view attention、spatial attention、point fusion；anatomy loss 能训练结构 head；开启 guided sampling 后，采样路径的梯度能返回结构 head。检查不存在 .detach、缓存、零 gate 造成的长期断路。
6. anatomy correctness：均匀 phantom、平面边界及各向异性 spacing；检查 PSD、方向、物理尺度和有限值。明确 inference 无 GT 可以运行。
7. chunk 一致性：相同 eval 输入用不同 query chunk size，输出一致；空间 attention 分块不应截断跨 chunk 的邻域关系。AMP/GPU 比较采用说明过的容差。
8. 训练/eval smoke：最小 synthetic batch 的 forward/backward/optimizer step；有真实数据和 GPU 时再运行有限真实步骤，记录 loss、时间、峰值显存和中间诊断。DDP 环境可用时验证新参数通过 wrapper 获得同步梯度；不可用则记录未实测。

诊断至少包括：各尺度采样有效率、attention 是否塌到单一视角、新残差/基底范数比、3D token 方差、anatomy loss、footprint 范围及关键模块梯度范数。attention 可视化只是诊断，不当作可靠性真值。

## 11. 执行预算和交付

代码实现和小型测试持续完成，不停在计划阶段。默认不启动长时间训练、全数据 sweep 或后台任务。真实 smoke 的保守默认上限为一个 GPU、累计 100 个 optimizer steps 或 20 分钟墙钟时间，先到者停止；若用户已有明确预算，按用户预算执行。测试预算到达后交付代码与启动命令，不因此省略实现。

没有 PyTorch/GPU/数据时，完成能够执行的静态检查及测试代码，清楚标注未运行的检查和运行条件；不能把语法通过写成 forward/backward 通过。不要为了 smoke 安装/下载大型模型或 CT 数据。保持仓库现有 checkpoint 策略，不删除或覆盖用户结果。

最终交付：

1. 可运行代码、新配置和必要测试。
2. 与实际 CLI 对应的训练、eval 和消融命令；不要编造不存在的参数。
3. 一份简洁实现报告：改动文件、张量 shape/单位、实际通过的测试、未运行项、依赖阻塞及性能实测。
4. 清楚区分「已实现」「已测试」「待长训练验证」「因明确依赖未实现」。不要把待实现模块列为完成。

请从检查当前 git 状态、有效源码路径和调用链开始，然后按几何 adapter → 多尺度 features → 3D Transformer → point fusion → anatomy → 验证/扩展的顺序执行。
