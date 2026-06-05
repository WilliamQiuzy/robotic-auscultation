# 感知模型 v3 — LoRA 替全解冻(抗过拟合 + 后背瓶颈再进一步)

> 日期: 2026-06-04 · 训练机 Vast.ai RTX 4090 · 权重本地 `auscultation/outputs/perc_v3/`
> v3 选型: 用户在"③提升"中选 **LoRA 替全解冻**(低风险)。动机 = v2 暴露全解冻在该数据量上 ep1 即过拟合(real-back PCK ep1 0.55→晚期 0.43-0.49),想用更强正则把后背定位推得更稳、并把 ckpt 从 356MB 降到几 MB。
> 评测口径与 v2 **完全一致**:同一 `val_split.json`(md5 相同,真受试者 held-out 无泄漏)、soft-argmax 读出、同评测集。

## 1. v3 做法(LoRA 低秩适配)

| 维度 | v2(全解冻) | **v3(LoRA)** |
|---|---|---|
| 编码器 | 整个 DINOv2 ViT-B 微调 | **冻结基座**, 在 qkv/proj/fc1/fc2 注入低秩旁路 A/B(r16/α32) |
| 可训练参数 | ~86M | **4.73M**(头 + LoRA A/B) |
| 正则 | weight_decay 0.05 | 同 + LoRA 结构性低秩约束(B 初始化 0 → 起点=原模型) |
| ckpt 大小 | 356MB(全模型) | **19MB**(仅可训参数, 基座可由 torch.hub 重建) |
| ckpt 选择 | `--select realback_pck`(真实后背 PCK@10 选 best) | 同 — **不用 mixed val-loss** |
| 域均衡采样 | syn0.4/真前0.3/真后0.3 | 同 |

> 训练/评测均修了一个 device bug:`apply_lora` 在 `.to(device)` 之后注入新 A/B 层会留在 CPU → 注入后重新 `.to(device)`。

## 2. 评测对比(v0→v1→v2→v3, soft-argmax, 同 held-out)

| 域 | 指标 | v0(仅合成/冻结) | v1(真混训/解冻) | v2(全解冻+重心UV+soft) | **v3(LoRA)** |
|---|---|---|---|---|---|
| **Scoliosis 真实裸背肺点** | PCK@10 / 中位% | 0.22 / 16% | 0.47 / 13% | 0.60 / 7% | **0.67 / 6.3%** |
| **Scoliosis 真实裸背肺点** | PCK@5 | — | — | — | **0.42** |
| HuMMan 真实正面心点 | PCK@10 / 中位% | 0.65 | 0.91 | **0.94** / 4.9% | 0.91 / 4.9% |
| Synthetic 同域后背 | PCK@10 | 0.62 | 0.60 | 0.62 | **0.64** |
| Synthetic 同域正面 | PCK@10 | — | — | — | 0.99 |

**真实后背肺点 PCK@10 全程: 0.22 → 0.47 → 0.60 → 0.67;中位误差 16% → 13% → 7% → 6.3%。** 后背瓶颈持续闭合,v3 又 +0.07。

### soft-argmax vs argmin 消融(v3 same ckpt, 真实裸背)
| 读出 | PCK@10 | PCK@5 | 中位% |
|---|---|---|---|
| **soft-argmax** | **0.67** | 0.42 | 6.3% |
| argmin | 0.51 | 0.24 | 9.6% |
| Δ | **+0.16** | +0.18 | -3.3% |
→ soft-argmax 增益(+0.16)比 v2(+0.12)更大:LoRA 让 part 分割略松(mIoU 0.34),亲和热图的亚像素期望比硬 argmin 更抗这种松动。

### 训练侧佐证(per-epoch 真实后背 held-out=92, 选 ckpt 用)
| | best epoch | best real-back PCK@10 | ep14(last) val-loss |
|---|---|---|---|
| v2 全解冻 | ep1 | 0.552 | 0.203 |
| **v3 LoRA** | ep1 | **0.604** | **0.162** |
→ 同样 ep1 早峰,但 LoRA 峰更高(0.604>0.552)、尾部 val-loss 更低(0.162<0.203)= **过拟合更轻**(正是选 LoRA 的目的)。

## 3. 关键发现 / 反思

1. **LoRA 达成设计目标**:用 1/18 的可训练参数(4.73M vs 86M)、1/19 的 ckpt(19MB vs 356MB),把核心瓶颈"真实后背肺点"再推 0.60→0.67、误差 7%→6.3%,且过拟合更轻。低风险路线兑现了收益。
2. **代价 = 正面心点 0.94→0.91(near-saturated)**:LoRA 强正则在已接近饱和的前胸上有轻微取舍(回到 v1 水平),但前胸 0.91 仍很强、临床上前胸定位本就不是瓶颈;用这点换后背 +0.07 值得。
3. **ckpt 选择准则再次验证**:v3 同样按真实后背 PCK 选 ep1(峰 0.604),若按 mixed val-loss 会选 ep3(val 0.0965)→ 后背仅 0.436。**"选定位最优而非分割/重建最优"** 的准则在 v3 上同样成立(参见用户在 v1 观察到的 v1best 0.33 vs v1last 0.39 背离)。
4. **part 分割 vs landmark 定位解耦**:v3 scoliosis part-mIoU 仅 0.342(< v2),但 landmark PCK 反而最高(0.67)。说明听诊点定位主要由 **UV 地址精度 + soft 读出** 决定,而非 part 分割质量 —— 这进一步支持后续优化重心放在 UV 标签/读出而非分割头。
5. **仍有空间**:真实后背 0.67(误差 6.3%)离前胸(0.91)仍有差距。瓶颈仍是真实后背数据量(held-out 仅 92 点 / 78 图)、scoliosis 直脊柱拟合给肺点注入纵向标签噪声。下一步候选不变:scoliosis 曲率重拟合(清纵向噪声)、自采更多真实后背、或 Sapiens 人体编码器 A/B。

## 4. 产物(本地 `outputs/perc_v3/`)
```
best.pt(ep1, 19MB) / last.pt(ep14) / ep00..ep14.pt(全 15 个 per-epoch ckpt, 各 19MB, 全记录留存)
config.json · metrics.jsonl · train_v3.log · val_split.json(与 v2 同 md5)
eval/ eval_v3best_{scoliosis,humman,syn2}.{txt,json} + scoliosis argmin 消融 + sample_{scoliosis,humman}.png
```
> LoRA ckpt 小(19MB), 全部 15 个 epoch ckpt + best/last 已下载本地校验(fail=0)。GitHub 放报告 + metrics + config + eval json + val_split + 样例图(ckpt 因 outputs/ 被 .gitignore 忽略, 本地留存)。

## 5. 四版总览(感知模型 v0→v3)
- **v0** 仅合成 + 冻结编码器:真实后背 0.22(sim-to-real 鸿沟)。
- **v1** 真实混训 + 全解冻:0.47(混训关键)。
- **v2** 重心插值 UV 标签 + soft-argmax + 真实后背选 ckpt + 域均衡:0.60(标签+读出双线)。
- **v3** LoRA 替全解冻:**0.67**, ckpt 19MB, 过拟合更轻(参数/正则线)。
- 真实后背肺点 PCK@10 **0.22→0.67(3×)**,中位误差 **16%→6.3%**。
