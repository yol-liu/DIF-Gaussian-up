# DIF-Gaussian Progressive Training Guide

目标：从 **可复现 baseline** 出发，**一次只加一个改动**。  
确认收益后再往上叠。

数据路径：

```text
root_dir: /media/wyin/Expansion/Data/LUNA16/C2RV-CBCT/data
```

---

## 先看清楚：哪些实验开了 latent？

**结论：Stage A–F 全部是确定性模型（`use_latent: false` / 不开 latent）。**  
**只有 Stage G（以及之后的 Stage L）才开 global latent。**

| Stage | 实验 | geometry attention | GS 改动 | view dropout | **latent Stage G** |
|-------|------|--------------------|---------|--------------|--------------------|
| **A** | baseline | 否（max） | 否 | 否 | **否** |
| **B** | stable GA | 是（residual） | 否（legacy） | 否 | **否** |
| **C** | + shifted KNN | 是 | 仅 shifted | 否 | **否** |
| **D** | + knn_k | 是 | shifted + k | 否 | **否** |
| **E** | + normalize | 是 | + normalize | 否 | **否** |
| **F** | + view dropout | 是 | 继承最优 | 是 | **否** |
| **G** | + global latent | 继承 B（或当前最优确定性） | 继承该确定性设置 | 通常仍关 | **是（global only）** |
| **L** | + local latent | 继承 G | 继承 G | 可选 | 是（global+local） |

推荐入口 cfg：

| Stage | cfg | latent? |
|-------|-----|---------|
| A | `configs/default.yaml` | 否 |
| B | `configs/geometry_attention_stable.yaml` | 否 |
| G（推荐） | `configs/latent_stage_g_stable.yaml` | **是** |
| G（激进，不推荐先跑） | `configs/latent_stage_g.yaml` | 是，但叠了太多改动 |

> 旧的 `latent_stage_g.yaml` = 激进 GA + 激进 GS + view_dropout + latent，**不要当作第一个 latent 实验**。

---

## 0. 环境与公共命令

```bash
cd /data/Codes/CT-Reconstruction/DIF-Gaussian
source scripts/env.sh
conda activate torch2-3
```

评估（ep100 / ep200 / ep400）：

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluate.py \
  --name <EXP_NAME> \
  --cfg_path <CFG> \
  --dst_name LUNA16 \
  --num_views 6 \
  --epoch <EPOCH> \
  --split eval
```

记录表：

| stage | name | cfg | latent? | ep100 SSIM | ep200 SSIM | ep400 SSIM | 备注 |
|-------|------|-----|---------|------------|------------|------------|------|

判定：

- 明显低于上一阶段（例如 SSIM 掉 >0.01 且 ep200 仍追不上）→ **停，先修**。
- 持平或更好 → 可进下一阶段。
- 每个阶段用 **新的 `--name`**。

---

## 两条线（不要混）

```text
确定性线（先跑完关键）：
  A → B → (可选 C/D/E/F)

概率线（在确定性稳住之后再开）：
  G = 在「当前最好的确定性设置」上只打开 global latent
  L = 在 G 成功后再打开 local latent
```

**不要**在 A/B 还没跑通时直接跑 latent。  
**不要**把 latent 和 normalize / view_dropout 第一次一起开。

---

## Stage A — Baseline（无 latent）

```bash
CUDA_VISIBLE_DEVICES=0 python code/train.py \
  --name DIFG-base-6v \
  --cfg_path ./configs/default.yaml \
  --dst_name LUNA16 \
  --num_views 6 \
  --batch_size 4 \
  --epoch 400 \
  --random_views
```

通过：稳定、接近你已知 base（ep100 ~0.81）。

---

## Stage B — Stable Geometry Attention（无 latent）

只加 residual attention；GS 与 base 相同。

```bash
CUDA_VISIBLE_DEVICES=0 python code/train.py \
  --name DIFG-GA-B-stable-6v \
  --cfg_path ./configs/geometry_attention_stable.yaml \
  --dst_name LUNA16 \
  --num_views 6 \
  --batch_size 4 \
  --epoch 400 \
  --random_views
```

通过：ep100 接近 base；不要再出现大落差。  
B 不稳 → **不要进 C，更不要进 G**。

---

## Stage C / D / E / F — 仍全部无 latent

按需叠加（每次只改一项）：

| Stage | 只改什么 | 建议 name |
|-------|----------|-----------|
| C | `shifted_center_knn: true` | `DIFG-GA-C-shifted-6v` |
| D | `knn_k: 5`（再试 8） | `DIFG-GA-D-knn5-6v` |
| E | `normalize_weights: true` | `DIFG-GA-E-norm-6v` |
| F | `view_dropout: 0.1` | `DIFG-GA-F-vd01-6v` |

复制方式示例：

```bash
cp configs/geometry_attention_stable.yaml configs/geometry_attention_C_shifted.yaml
# 编辑后只改 gs.shifted_center_knn: true
```

训练命令模板（把 name / cfg 换掉即可）：

```bash
CUDA_VISIBLE_DEVICES=0 python code/train.py \
  --name <EXP_NAME> \
  --cfg_path <CFG> \
  --dst_name LUNA16 \
  --num_views 6 \
  --batch_size 4 \
  --epoch 400 \
  --random_views
```

---

## Stage G — Global Latent（第一次开 latent）

**目的**：在已经稳住的确定性模型上，**只打开** `latent.use_latent: true`（`use_local: false`）。

### 推荐：G-stable（叠在 Stage B 上）

配置：`configs/latent_stage_g_stable.yaml`  
= Stage B 的 GA/GS 设置 + global latent。

```bash
CUDA_VISIBLE_DEVICES=0 python code/train.py \
  --name DIFG-LatentG-on-B-6v \
  --cfg_path ./configs/latent_stage_g_stable.yaml \
  --dst_name LUNA16 \
  --num_views 6 \
  --batch_size 4 \
  --epoch 400 \
  --random_views
```

可选：从确定性 B 的权重 warm-start（若 `train.py` 支持 resume/ckpt；用你们现有的加载方式即可）：

```bash
# 示例：若使用 --resume，从 B 的 epoch 继续 / 加载
# 具体以 code/train.py 的参数为准；关键是先有健康的确定性 ckpt
```

**通过标准**：

- 体积重建不低于 Stage B（允许略慢收敛）。
- KL 不全塌成 0（看训练日志里的 latent diagnostics）。
- 不要一上来就比 B 差很多。

### 若你已经跑完 C/D 且更好

不要死用 G-stable；应：

```bash
cp <你当前最好的确定性cfg> configs/latent_stage_g_from_best.yaml
```

然后只改：

```yaml
latent:
  use_latent: true
  use_local: false
```

其余（attention / GS / dropout）保持与「最好确定性」完全一致。

### 不要先跑的：激进 `latent_stage_g.yaml`

它同时开了：激进 attention + knn8 + normalize + view_dropout + latent。  
仅作「一次叠太多会怎样」的对照，**不是课程默认**。

---

## Stage L — Local Latent（G 成功后再说）

在 Stage G 的 cfg 上改：

```yaml
latent:
  use_latent: true
  use_local: true
```

单独开实验名，例如 `DIFG-LatentL-6v`。  
**G 没过关前不要开 L。**

---

## 推荐实际跑法（最少集合）

如果时间紧，最少跑这 3 个：

1. **A** `default.yaml` — 无 latent  
2. **B** `geometry_attention_stable.yaml` — 无 latent  
3. **G** `latent_stage_g_stable.yaml` — **有** global latent  

C/D/E/F 是确定性消融；有余力再跑。  
L 在 G 成功后再跑。

顺序图：

```text
A (base, no latent)
 └─ B (stable GA, no latent)
     ├─ C → D → E → F   (optional deterministic ablations, no latent)
     └─ G (global latent on B, or on best of C–F)
         └─ L (local latent)
```

---

## 不要一上来就跑

| 配置 | 原因 |
|------|------|
| 旧激进 `geometry_attention.yaml` | 一次改太多，无 latent 已经不稳 |
| 激进 `latent_stage_g.yaml` | latent + 激进 GA/GS/dropout 叠在一起 |
| A/B 未过就开 latent | 无法判断是 attention 问题还是 latent 问题 |
| normalize + view_dropout + latent 一起开 | 尺度 + 正则 + KL 同时动 |

---

## 多卡（可选）

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --master_port 2026 \
  --nproc_per_node 2 \
  code/train.py \
    --name <EXP_NAME> \
    --cfg_path <CFG> \
    --dst_name LUNA16 \
    --num_views 6 \
    --batch_size 4 \
    --epoch 400 \
    --random_views \
    --dist
```

注意：参数是 `--random_views`（不是 `--random_view`）。

---

## 一句话

- **A–F：都没有 latent。**  
- **G：第一次加 latent（global only）。**  
- 先把确定性（尤其 B）稳住，再开 G；不要用激进 `latent_stage_g.yaml` 当第一枪。
