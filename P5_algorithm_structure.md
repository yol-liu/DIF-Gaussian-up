# P5 算法结构梳理：GT-Conditioned CVAE（步骤 → 代码映射）

> 用途：帮助梳理算法结构（非论文用途）。每一步对应到具体代码位置（文件、函数、行号）与关键实现片段。
> 模型类：`DenseProbabilisticDIFGaussian`（`mode='gt_cvae'`），构造入口 [factory.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/factory.py#L6-L18)。
> 运行配置：[prob-gt-cvae.yaml](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml)；
> 概率模块默认值/校验：[probabilistic_config.py](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_config.py#L18-L82)。

---

## 0. 算法总览

```
训练（gt_cvae，S=1 个后验样本）                                 部署（prior_mc_mean，N=8 个先验样本）
─────────────────────────────────────────────                 ─────────────────────────────────────
C(6视图) ──► UNet E ──► 几何Transformer ──► ψ(C) ──► prior_head → p(z|C)
                                                        │
Y(GT体) ──► GTEncoder ─► h_Y ∈R³² ─┐                    (部署时丢弃 posterior & GTEncoder)
                                   ▼
                        ψ(C)⊕h_Y ──► posterior_head → q(z|C,Y)
                                   │
                 z ~ q(z|C,Y)（重参数化）
                                   │
              FiLM(z) 调制 G 特征 ──► query_gs/点解码 ──► x̂⁽ᵠ⁾
                                   │
    L = L_vol⁽ᵠ⁾ + β_t·KL[q‖p]（free-bits 0.5，β线性升温到 2e-2）
```

源码级流程（`forward`）：[dense_probability.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L245-L337)。

---

## 1. 训练主链（每一步对应代码）

### 步骤 1：多视角 UNet 编码器 E

- **做什么**：把每个视角的 2D 投影图编码为多尺度特征图。
- **入口**：[dense_probability.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L123-L130) `encode_context()` → [model.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L125-L131) `encode_projs()` → `self.image_encoder(data['projs'])`。
- **网络**：[unet.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/unet.py#L7-L91) `UNet(in_ch=1, out_ch=128)`；`ds_ch=1024`（[L12](file:///data1/workspace/DIF-Gaussian-up-new/code/models/unet.py#L12)）；下采样到 1/16 后 U 型还原到 1×；`return_pyramid` 时同时输出 stride {4,8,16} 金字塔 [L89-L90](file:///data1/workspace/DIF-Gaussian-up-new/code/models/unet.py#L89-L90)。
- **关键代码**：

```python
encoder_output = self.image_encoder(data['projs'], view_mask=..., return_pyramid=True)
# result: feats[B,V,128,H,W], feats_ds[B,V,1024,H/16,W/16], pyramid{4,8,16}
```

- **备注**：输入前 `prepare_views()`（view dropout，配置为 0，[model.py#L97-L123](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L97-L123)）；`context_inputs()` 会剥离 `gt_/dense_/teacher_` 和 `points_gt`，保证前置分支看不到特权信息（[dense_probability.py#L15-L18](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L15-L18)）。

### 步骤 2：锚点 token 初始化（token_init）

- **做什么**：在 K=12³=1728 个 3D 锚点（res=12 均匀网格，[dst_gs.py#L18-L20](file:///data1/workspace/DIF-Gaussian-up-new/code/datasets/dst_gs.py#L18-L20)）上，用 `feats_ds` 通过像素对齐采样 + max 融合得到初始锚点特征 `p_feats [B,1024,K]`。
- **代码**：[model.py#L143-L158](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L143-L158)（`coordinate_mode='legacy'`（默认）走 `query_view_feats` 分支；`token_init='max'`）：

```python
p_feats = query_view_feats(view_feats=encoder_output['feats_ds'],
                           points_proj=anchor_coordinates,  # points_gs_proj
                           fusion=token_init, mask_fov=self.mask_fov, ...)
```

### 步骤 3：几何感知多视角 Transformer（锚点融合）

- **做什么**：对 K 个锚点 token 做 3D 空间注意力 + 跨视角 2D 特征交叉注意力，得到几何感知的锚点特征。
- **入口**：[model.py#L161-L162](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L161-L162) → `MultiViewGeometryEncoder.forward` [multiview_encoder.py#L282-L341](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L282-L341)。
- **子模块**（初始化 [L160-L197](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L160-L197)）：
  - Token 化：`base_projection + position_encoding(3D绝对坐标)` [L292-L293](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L292-L293)。
  - 每层 block：从多尺度 2D 特征采样观测 token（`_observations` [L222-L250](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L222-L250)，含 ray/scale 编码）→ `LocalCrossAttention`（[L24-L48](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L24-L48)）→ `GeometryBlock.finish`：±1 窗口的 `SpatialLocalAttention` + FFN（[L83-L98](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L83-L98)、[L51-L80](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L51-L80)）。
  - 输出投影：`output_norm + output_projection` → `[B,1024,K]`；`residual_base=false`（配置）时直接作为最终锚点特征（[L328-L332](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py#L328-L332)）。
- **配置**：`d_model=128, num_heads=4, num_blocks=2, feature_scales=[4,8,16], spatial_window=3, anchor_chunk_size=256`（[prob-gt-cvae.yaml#L18-L32](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L18-L32)）。

### 步骤 4：上下文池化 ψ(C)

- **做什么**：把锚点特征池化 + 视角描述子拼接为条件向量。
- **代码**：[dense_probability.py#L128](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L128)：

```python
pooled = torch.cat([features['anchor_feats'].mean(-1),   # [B,1024]（K 维均值）
                    observation_descriptor(clean)], -1)  # [B,7]  →  pooled ∈ R^1031
```

- **observation_descriptor**（[L36-L58](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L36-L58)）：7 个置换不变数字 = log1p(视角数)、圆形覆盖率、sin/cos 一阶角度矩、sin2/cos2 二阶矩、平均合力长度。

### 步骤 5：GT 编码器 h_Y = GTEncoder(Y)（训练专用）

- **做什么**：把 GT 体编码为全局描述子，只在后验分支使用。
- **代码**：[dense_probability.py#L158](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L158) → `GTEncoder` [gt_encoder.py#L15-L64](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gt_encoder.py#L15-L64)。
- **网络**：`Conv3d(1→16)→GN→SiLU→Conv3d(16→16,stride2)→GN→SiLU→Conv3d(16→32,stride2)→GN→SiLU`（[L24-L34](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gt_encoder.py#L24-L34)）；全局描述子 `feat_volume.mean(dim=(2,3,4))` [L52](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gt_encoder.py#L52) ∈ R³²。

### 步骤 6：先验头 p(z|C) 与后验头 q(z|C,Y)

- **做什么**：两层独立的有界高斯头，输出对角高斯 `(μ, 2·log σ)`。
- **构建**：[dense_probability.py#L91-L95](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L91-L95)：
  - `prior_head = BoundedGaussianHead(channels+7, 64)`（输入 1031）
  - `posterior_head = BoundedGaussianHead(channels+7+32, 64)`（输入 1063）
- **BoundedGaussianHead**：[latent_np.py#L39-L57](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L39-L57)：

```python
mu, raw = self.net(features).chunk(2, -1)                    # Linear(·→128)→SiLU→Linear(128→2d_z)
sigma = sigma_min + (sigma_max - sigma_min) * raw.sigmoid()  # σ∈[0.001,3]，初始化 σ=0.5（L48-L51 设定 bias）
return mu, 2 * sigma.log()
```

- **先验/后验推理**：[dense_probability.py#L132-L133](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L132-L133)（`infer_prior`）、[L150-L164](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L150-L164)（`infer_posterior`：`pooled = cat([context.pooled, gt_global_feat], -1)`）。

### 步骤 7：采样 z（重参数化）

- **做什么**：训练时从 q 采样 1 个 z（`train_samples=1`）。
- **代码**：[dense_probability.py#L256-L260](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L256-L260)（`mode='gt_posterior_sample'`, `count=1`）→ `sample_state` [L172-L201](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L172-L201)：

```python
mu, lv = posterior if posterior is not None else prior
z_g = mu if mean else reparameterize(mu, lv, generator)   # L177
```

- **reparameterize**：[latent_np.py#L16-L20](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L16-L20)：`mu + eps*std`。
- **备注**：`use_local=False`（配置）⇒ 无局部潜变量，conditioning 就是全局 `z_g ∈ R⁶⁴`。

### 步骤 8：FiLM 调制（z → 高斯特征）

- **做什么**：`f' = (1+γ(z))⊙f + β(z)`，把 z 注入高斯特征。
- **代码**：[dense_probability.py#L198](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L198) → `FeatureFiLM` [latent_np.py#L270-L301](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L270-L301)。结构：`Linear(64→128)→SiLU→Linear(128→256)`，最后层零初始化（[L285-L286](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L285-L286)）→ 初始即恒等；随后按 `film_init_std=0.001` 重置权重（[dense_probability.py#L106](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L106)）。
- **FiLM 作用对象**：`gs_params['feats']`（每锚点高斯特征 [B,128,K]），来自 `gs_feats_mlp`。

### 步骤 9：3D 高斯图元 + 点解码器 g∘f_GS

- **做什么**：把锚点特征转成 3D 高斯参数，用高斯插值查询任意 3D 点并解码为体素值。
- **高斯参数**（[model.py#L172-L225](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L172-L225)）：
  - `gs_feats_mlp: [1024→256→128]` → feats `[B,128,K]`（L172）
  - `gs_params_mlp: [1024→256→10]`，除以 `t=1/512`（L186-L188）→ offsets/s/r
  - offsets：`sigmoid` 限定在 `±p_dist*0.5`（L191-L193）；s：`sigmoid` 映射到 `((1-k)d,(1+k)d)`，k=0.3（L196-L198）；r：四元数（L201）
  - `build_covariance` + `det/inv/log_det`（L204-L210；[gs_utils.py#L77-L99](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gs_utils.py#L77-L99)）
- **点查询与解码** `forward_points`（[model.py#L227-L256](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py#L227-L256)）：
  1. 查询点像素对齐特征 + max 融合 `p_feats [B,128,N]`（L231-L237）
  2. `query_gs`（[gs_utils.py#L102-L215](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gs_utils.py#L102-L215)）：KNN(k=3) 选最近高斯 → 高斯密度加权求和 → `interp_feats [B,128,N]`
  3. `cat([p_feats, interp_feats], 1)` → `[B,256,N]`（L252）→ `point_decoder`（[point_decoder.py#L6-L33](file:///data1/workspace/DIF-Gaussian-up-new/code/models/point_decoder.py#L6-L33)，channels `[256,256,64,16,1]`，残差拼接）→ `[B,1,N]`

### 步骤 10：损失 L = L_vol⁽ᵠ⁾ + β_t·KL[q‖p]（+ w_prior·L_vol⁽ᵖ⁾）

- **入口**：[train.py#L414-L416](file:///data1/workspace/DIF-Gaussian-up-new/code/train.py#L414-L416) → `probability_loss` [probabilistic_loss.py#L8-L62](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_loss.py#L8-L62)。
- **重建项**：

```python
reconstruction = volume_nll(samples, target[None], rec['likelihood'], rec['observation_sigma'])  # L11
# volume_nll（losses.py#L18-L36）：likelihood='mse' ⇒ diff.square().mean()；σ=0.1 用于 gaussian/laplace
```

- **KL 项**（只有全局组 'g'）：`gaussian_kl`（[latent_np.py#L23-L36](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L23-L36)）→ 逐维 free-bits 下限 `clamp_min(0.5)` → sum → batch mean（L20-L26）；β 线性升温：

```python
beta = kl_cfg['beta_final'] * kl_warmup_scale(step, kl_cfg['warmup_steps'])   # L42
# kl_warmup_scale（losses.py#L97-L100）：min(1, t/500)；β_final=2e-2
```

- **先验重建项**（配置 `prior_reconstruction_weight=0` ⇒ 默认关闭，[L51-L57](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_loss.py#L51-L57)）；**需要先验样本时**在 [dense_probability.py#L314-L319](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L314-L319) 生成：`sample_state(context, prior, mean=True)` + decode。
- **总损失**：`loss = reconstruction + weighted`（L50）。
- **实际配置**：[prob-gt-cvae.yaml#L67-L76](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L67-L76)：`likelihood=mse, observation_sigma=0.1, beta_final=2e-2, warmup_steps=500, free_bits=0.5`。

### 步骤 11：优化循环

- [train.py#L463-L485](file:///data1/workspace/DIF-Gaussian-up-new/code/train.py#L463-L485)：`scaler.scale(loss).backward()` → `clip_grad_norm_(1.0)` → `scaler.step(optimizer)`。
- AdamW：`lr=2e-4, weight_decay=1e-2`（共享骨干），融合模块（`multiview_encoder` 等）wd=0（L309-L323）；StepLR `gamma=(1e-3)^(1/400)`（L324-L328）。

---

## 2. 部署链（inference，N=8 MC）

- **入口**：[evaluate.py#L212-L222](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluate.py#L212-L222) → `evaluation_protocol.predict_volume`（[evaluation_protocol.py#L26-L46](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluation_protocol.py#L26-L46)）→ 模型 `forward(is_eval=True)`。
- **模式选择**：[dense_probability.py#L262-L272](file:///data1/workspace/DIF-Gaussian-up-new/code/dense_probability.py#L262-L272)：`estimator='prior_mc_mean'`（配置），`count = num_samples = 8`；非 `prior_mc_mean` 时 `count=1`。**GTEncoder/后验头在此完全不用**。
- **采样循环（Welford 在线均值/方差）**：[L296-L312](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L296-L312)：

```python
for index in range(count):
    state = self.sample_state(context, prior, posterior=None, mean=False, generator=gen)  # 基于 p(z|C)
    prediction = self.decode_chunked(state, eval_npoint)   # 分块解码（eval_npoint=1e5）
    # Welford: mean_prediction += delta/(index+1); m2 += delta*(prediction-mean)
variance = m2 / (count - 1) if count > 1 else None        # L320
observation_var = rec['observation_sigma']**2 if rec['likelihood']=='gaussian' else None  # L322
predictive_var_total = variance + observation_var         # L325（likelihood='mse' ⇒ 无观测噪声项）
```

- **随机种子固定**：`generator = manual_seed(evaluation.seed=0)`（[L293-L294](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py#L293-L294)）。
- **指标**：[evaluation_protocol.py#L53-L75](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluation_protocol.py#L53-L75)（PSNR/SSIM 等）；`evaluate.py` 中 `predict_volume` 对 [S,B,1,N] 样本取均值 [L41-L42](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluation_protocol.py#L41-L42)。
- **注意（与论文差异）**：论文式 (8) 假设 Gaussian 似然并加 σ_obs²；实际配置 `likelihood='mse'` ⇒ 代码中 `observation_variance=None`，`predictive_var_total` 也只含潜变量样本方差。

---

## 3. 关键超参速查（prob-gt-cvae.yaml）

| 项 | 值 | 代码位置 |
|---|---|---|
| 潜变量维度 d_z | 64 | [yaml#L61-L66](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L61-L66) |
| σ 范围 / 初值 | [0.001, 3.0] / 0.5 | [latent_np.py#L42-L51](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py#L42-L51) |
| β_final / warmup / free-bits | 2e-2 / 500 步 / 0.5 nats | [yaml#L71-L76](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L71-L76) |
| train_samples | 1 | [yaml#L60](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L60) |
| 部署估算器 / N / seed | prior_mc_mean / 8 / 0 | [yaml#L77-L80](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L77-L80) |
| 锚点数 K / res | 1728（12³ 均匀网格）/ 12 | [yaml#L46](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L46)、[dst_gs.py#L18-L20](file:///data1/workspace/DIF-Gaussian-up-new/code/datasets/dst_gs.py#L18-L20) |
| UNet 输出 / 深层通道 | out_ch=128 / ds_ch=1024 | [unet.py#L12-L38](file:///data1/workspace/DIF-Gaussian-up-new/code/models/unet.py#L12-L38) |
| point_decoder 通道 | [256,256,64,16,1] | [yaml#L8-L14](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L8-L14) |
| 先验重建权重 w_prior | 0（关闭） | [probabilistic_config.py#L21](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_config.py#L21) |
| 视角数：训练 C / 数据集 dense | 6（`context_counts`）/ 48 | [yaml#L85-L89](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml#L85-L89) |

---

## 4. 文件地图（按算法角色）

| 角色 | 文件 |
|---|---|
| 概率模型主类 / forward / 采样 / Welford | [dense_probability.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/dense_probability.py) |
| 骨干：UNet 编码、锚点 token、Transformer、GS 参数、点解码 | [model.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/model.py)、[unet.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/unet.py)、[multiview_encoder.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/multiview_encoder.py) |
| 3D 高斯：协方差构建、KNN 查询 | [gs_utils.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gs_utils.py) |
| 概率组件：reparameterize / KL / BoundedGaussianHead / free-bits / FiLM | [latent_np.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/latent_np.py) |
| GT 编码器 | [gt_encoder.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/gt_encoder.py) |
| 损失 | [probabilistic_loss.py](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_loss.py)、[losses.py](file:///data1/workspace/DIF-Gaussian-up-new/code/losses.py) |
| 训练循环 / 评估 | [train.py](file:///data1/workspace/DIF-Gaussian-up-new/code/train.py)、[evaluate.py](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluate.py)、[evaluation_protocol.py](file:///data1/workspace/DIF-Gaussian-up-new/code/evaluation_protocol.py) |
| 概率配置解析 / 默认值 | [probabilistic_config.py](file:///data1/workspace/DIF-Gaussian-up-new/code/probabilistic_config.py) |
| 模型工厂 | [factory.py](file:///data1/workspace/DIF-Gaussian-up-new/code/models/factory.py) |

> 注：`models/probabilistic_model.py` 是遗留的 C1 分层潜变量版本（GlobalLatent/LocalLatentGrid），非当前 P5 模型，梳理时勿混淆。

---

## 5. 逐步参数详解：每个参数的作用与训练中的合理变化

> 值均来自 [prob-gt-cvae.yaml](file:///data1/workspace/DIF-Gaussian-up-new/configs/experiment_plan/stage1/prob-gt-cvae.yaml)（P5 实际运行配置）。"合理变化"指训练全程该参数或其对应指标应呈现的正常走势，"监控项"指可在 wandb/`step_metrics.jsonl` 中查看的对应指标。

### 5.1 步骤 1｜多视角 UNet 编码器 E

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `in_ch` | 1 | 输入为单通道灰度投影图 | 固定 |
| `out_ch` | 128 | 细特征通道数：`feats [B,V,128,H,W]`，同时是 FiLM 调制的目标维度 | 固定；监控锚点特征 `anchor_feats` 的方差是否脱离初始化（太小说明退化） |
| `ds_ch` | 1024 | 最深特征（1/16 分辨率）通道数，锚点 token 的初始维度 | 固定 |
| `feature_scales` | [4,8,16] | 金字塔下采样倍数，供 Transformer 做多尺度观测 | 固定 |
| 梯度范数 | — | 共享骨干是最大梯度来源 | `gradient/shared_encoder` 应明显大于融合/概率组件（真实日志 step0：1216 vs 其他 0.2~192） |

### 5.2 步骤 2｜锚点 token 初始化

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `gs_res`（数据集侧 `gs_res`） | 12 | 3D 均匀网格分辨率，锚点数 K=12³=1728（[dst_gs.py#L18-L20](file:///data1/workspace/DIF-Gaussian-up-new/code/datasets/dst_gs.py#L18-L20)） | 固定；K 越大空间分辨率越高、但 Transformer 与高斯图元开销越大 |
| `token_init` | max | 多视角像素对齐特征的最大池化融合：对遮挡鲁棒、置换不变 | 固定 |
| `mask_fov` | true | 只聚合落在视野内（FOV）的投影，避免视野外噪声 | 固定 |
| `anchor_chunk_size` | 256 | 1728 个锚点分块喂给 Transformer，控制显存 | 固定 |

### 5.3 步骤 3｜几何感知多视角 Transformer

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `d_model` | 128 | 注意力内部维度（Q/K/V 投影） | 固定；监控 `gradient/cross_attention`、`gradient/spatial_attention` 是否持续获得梯度（真实日志 step0 为 69/61，随训练应保持非零、量级远小于 backbone） |
| `num_heads` | 4 | 多头注意力头数 | 固定 |
| `num_blocks` | 2 | 堆叠的几何注意力 block 数，越大跨视角/空间交互越强 | 固定 |
| `feature_scales` + `samples_per_scale` | [4,8,16] + 5 | 每个 block 从多尺度 2D 金字塔采样观测 token 的数量 | 固定 |
| `spatial_window` | 3 | 3D 局部空间注意力窗口（±1 锚点），限制计算量 | 固定 |
| `residual_base` | false | Transformer 输出直接作为锚点特征（不做残差） | 固定（关） |
| `dropout` | 0.0 | 注意力 dropout | 固定 |
| `coordinate_mode` | legacy | 走 `query_view_feats` 采样分支 | 固定 |

### 5.4 步骤 4｜上下文池化 ψ(C)

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| 均值池化 | — | `anchor_feats.mean(-1)`：把 1728 个锚点特征压成 1024 维全局向量 | 固定 |
| `observation_descriptor` | 7 维 | log1p(视角数)、圆形覆盖率、sin/cos 与 sin2/cos2 角度矩、平均合力长度——置换不变的视角质量描述 | 固定 |
| 输出维度 | 1031 | 先验头输入 = 1024 + 7 | 监控 `gradient/sparse_conditioning`（真实日志 step0 仅 0.004，量级极小属正常，因为它只是前端的输入汇总） |

### 5.5 步骤 5｜GTEncoder（训练特权路径，部署不用）

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `out_ch` | 32 | 特征体积通道数；空间均值池化后得到 `h_Y∈R³²` | 固定 |
| 下采样 | 两次 stride-2 | GT 体压到 1/4 分辨率，控制计算量 | 固定 |
| 梯度 | — | 只走后验路径 | `gradient/posterior` 应非零（真实日志 step0 0.20，量级小于 backbone 正常） |

### 5.6 步骤 6｜先验/后验头（BoundedGaussianHead）

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `global_dim` | 64 | 潜变量维度 d_z | 固定 |
| `sigma_min` | 0.001 | σ 下限：防止后验坍缩到零方差（数值稳定） | 固定 |
| `sigma_max` | 3.0 | σ 上限：防止先验过度发散 | 固定 |
| `initial_sigma` | 0.5 | 初始化 σ（通过 bias 设定），训练从温和方差起步 | 固定 |
| 头结构 | Linear(·→128)→SiLU→Linear(128→128) | 两分支各输出 μ[R⁶⁴] 与 raw；`σ=σ_min+(σ_max-σ_min)·sigmoid(raw)` | 固定 |

> **核心监控指标（loss.py 每步输出）**：
> - `train/p_g_std`（先验 σ 均值）：应缓慢变化、保持在 0.5 附近（真实日志 0.500→0.531），**不得逼近 3.0**；
> - `train/q_g_std`（后验 σ 均值）：应**随训练下降**（后验利用 GT 越变越确定，真实日志 0.500→0.231）；
> - `q_g_scale_near_lower / q_g_scale_near_upper`：σ 撞击上下界的比例，**健康训练应始终接近 0**。

### 5.7 步骤 7｜重参数化采样

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `train_samples` | 1 | 每例每步从 q 采 1 个 z | 固定；S=1 时重建损失即单样本估计 |
| `mean_code` | false | 关：用随机采样而非后验均值 | 固定 |

### 5.8 步骤 8｜FeatureFiLM 调制

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `film_init_std` | 0.001 | 线性层零初始化后按 0.001 重置权重：γ,β 初始≈0，潜变量只做"小扰动" | 固定；`gradient/film` 应远小于 backbone（真实日志 step0 16 vs 1216），否则潜变量在覆盖主干而非微调 |
| FiLM 结构 | Linear(64→128)→SiLU→Linear(128→256) | 从 z 生成每通道 γ,β，`f'=(1+γ)⊙f+β` | 固定 |

> **意图**：让 z 只调制 128 维高斯特征而不重学整个视觉主干——这是"条件 VAE 缝合在确定性骨干上"的关键。

### 5.9 步骤 9｜3D 高斯图元 + 点解码

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `t` | 512 | `gs_params = mlp输出 / t`：把 MLP 原始 logits 缩回合理尺度 | 固定 |
| `res` | 12 | 高斯锚点网格 = 锚点数 K=1728 | 固定 |
| `o_scaling` | 1.0 | offset 幅度：offset∈[−p_dist/2, +p_dist/2]，p_dist=1/12 | 固定 |
| `s_scaling` | 0.3 | 高斯 s 宽度范围 `(1±k)·√p_dist`，k=0.3 | 固定 |
| `knn_k` | 3 | 查询点取最近 3 个高斯做密度加权插值 | 固定；`gradient/decoder` 应保持非零（真实日志 step0 192） |
| `shifted_center_knn` | false | KNN 基于初始锚点（legacy，而非偏移后中心） | 固定 |
| `normalize_weights` | false | 未归一化的高斯密度求和（legacy） | 固定 |
| `mlp_chs`（point_decoder） | [256,64,16,1] | 解码 MLP（输入拼接后 256 维，残差拼接） | 固定 |
| `eval_npoint` | 1e5 | 推理时分块解码的查询点数 | 固定 |

### 5.10 步骤 10｜损失

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `likelihood` | mse | 实际训练用 MSE 重建（`volume_nll` = 均方误差） | 监控 `train/L_volume`：应快速下降后平台收敛（真实日志 9.66→0.001） |
| `observation_sigma` | 0.1 | 仅在 gaussian/laplace 似然下作为观测噪声 σ；**mse 下不参与损失** | 固定 |
| `beta_final` | 2e-2 | KL 权重上限 | 监控 `train/warmup_beta` |
| `warmup_steps` | 500 | β 线性升温步数：`β_t=β_final·min(1,t/500)` | 观察前 500 步 β=0（纯重建阶段）、之后线性到 0.02 |
| `free_bits` | 0.5 | 每维 KL 下限（nats）：`KL_eff=Σmax(kl_d,0.5)` | 监控 `train/KL_g_effective` vs `train/KL_g_free_bits_floor`(=64×0.5=32) |
| `free_bits_order` | per_case | 先按 case 平均再取 free-bits | 固定 |
| `prior_reconstruction_weight` | 0 | 先验重建项关闭（部署分支不参与训练） | 固定 |

### 5.11 步骤 11｜优化循环

| 参数 | 值 | 作用 | 训练中的合理变化 / 监控 |
|---|---|---|---|
| `--lr` | 2e-4 | AdamW 学习率 | 固定 |
| `--weight_decay` | 1e-2 | 权重衰减（融合模块/概率模块不计衰减） | 固定 |
| `--no_decay_modules` | multiview_encoder, point_view_fusion, anchor_view_fusion | 这些分支 wd=0，避免融合分支在招募前被衰减到零 | 固定 |
| `--lr_decay` / StepLR | 1e-3 / 每 epoch `γ=(1e-3)^{1/400}` | 总衰减 1000×，每 epoch ×≈0.983 | 观察 lr 单调下降 |
| grad clip | 1.0 | 全局梯度裁剪 | `gradient/all_finite` 恒为 true |

---

## 6. 训练诊断：怎样算"合理的优化方向"

> 以下数值取自真实训练日志 [logs/DIFG-6v_1031_1655/step_metrics.jsonl](file:///data1/workspace/DIF-Gaussian-up-new/logs/DIFG-6v_1031_1655/step_metrics.jsonl)（gt_cvae 全程约 14.8 万步），可作为"健康曲线"的参照。

### 6.1 关键指标时间线（真实数据）

| step | β | L_volume (MSE) | KL_g_raw | KL_g_effective | p_g_std | q_g_std | 说明 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 9.657 | 2.8 | 32.06 | 0.500 | 0.500 | 随机初始化；β=0 纯重建 |
| 1.4k | 0.02 | 0.003 | 11.6 | 32.00 | 0.514 | 0.364 | β 升温完毕，重建已基本拟合 |
| 7.2k | 0.02 | 0.004 | 11.4 | 32.00 | 0.521 | 0.344 | 后验 σ 持续收缩 |
| 36.8k | 0.02 | 0.003 | 17.6 | 32.00 | 0.527 | 0.290 | 真实 KL 稳步上升 |
| 110.8k | 0.02 | 0.001 | 28.1 | 32.00 | 0.531 | 0.231 | 接近但未超过 free-bits 地板 |
| 147.8k | 0.02 | 0.001 | 28.3 | 32.00 | 0.531 | 0.231 | 收敛平台期 |

### 6.2 合理解读（对照上面的表）

1. **阶段一（step<500，β=0）**：只有重建梯度，**先验头梯度恒为 0**（`gradient/prior=0.0`，日志 step0 可见）——先验完全不动，模型先学会"用 z+解码器重建 CT"。期望：`L_volume` 从十位数骤降到 ~0.003。
2. **阶段二（500~1000 步，β 线性升温）**：KL 项逐步激活，把 q(z|C,Y) 拉向 p(z|C)。期望：`q_g_std` 开始下降、`KL_g_raw` 开始上升。
3. **阶段三（β=0.02 固定，长期）**：重建与 KL 达到平衡。健康的画像：
   - `L_volume` 低位平台（~0.001 量级，不再震荡上升）；
   - `q_g_std` 单调收缩到某个平台（0.2~0.3），**不撞击 0.001 下限**；
   - `p_g_std` 基本持平（0.5 附近），**不逼近 3.0**；
   - `KL_g_raw` 缓慢上升后趋于平台；
   - `KL_g_effective` ≈ 地板 32（当前真实 KL/维 0.44 < free_bits 0.5，说明 KL 约束目前主要来自"地板"，潜变量自由度还很大）。

### 6.3 红旗信号与调参方向

| 现象 | 含义 | 建议 |
|---|---|---|
| `q_g_scale_near_lower` > 0（后验 σ 撞 0.001） | 后验坍缩成确定性编码，先验 p 失去覆盖，部署时从 p 采样会严重错位 | 增大 `free_bits`（≥1.0）或增大 `beta_final`、`initial_sigma`；检查 GTEncoder/后验是否过强 |
| `p_g_std` 逼近 `sigma_max`=3.0 或 `KL_g_raw` 远超地板 | 先验过度发散 / q-p 差距过大，部署采样方差巨大 | 增大 `beta_final` 或缩短 `warmup_steps`；必要时打开 `prior_reconstruction_weight` 给先验直接的监督 |
| `L_volume` 平台不降/震荡 | 解码器过拟合或 FiLM 过强（film 梯度接近 backbone） | 调低 `film_init_std`、检查 `gradient/film : gradient/shared_encoder` 比例 |
| `KL_g_effective` 长期 = 地板且 `q_g_std` 几乎不变 | KL 没有真正约束，后验可能只是"记住了 GT" | 提高 `beta_final` 或降低 `free_bits`，让真实 KL 参与约束 |
| `gradient/all_finite=false` 或 loss 非有限 | 训练崩溃 | 检查 AMP 尺度、σ 边界、梯度裁剪 |

### 6.4 部署侧验收

- 部署 `prior_mc_mean`：只从 p(z|C) 采 N=8 个 z，取均值体积与逐点方差（Welford）。**比值**：`predictive_var_total` 应明显大于 0 且空间结构合理（高不确定性通常在稀疏视角覆盖差的区域）；
- `likelihood='mse'` ⇒ 部署重建方差中**没有**观测噪声 σ_obs² 项（论文式(8) 是 Gaussian 假设，代码按 mse 实现，见第 2 节备注）；
- 最终验收：验证集 PSNR/SSIM 随训练单调上升后收敛；若收敛后仍明显低于确定性 backbone（baseline），优先怀疑潜变量后验/先验未对齐（见 6.3 前两行）。