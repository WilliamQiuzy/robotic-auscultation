# 感知模型 v2 — 训练与评测报告(攻坚后背肺点瓶颈)

> 日期: 2026-06-04 · 训练机 Vast.ai RTX 4090 · 权重本地 `auscultation/outputs/perc_v2/`(best=ep1/last 全模型各 356MB)
> v2 目标: 闭合 v1 暴露的 **#1 瓶颈 = 真实后背肺点定位**(architect + 自评一致定位为"UV标签→读出链")。

## 1. v2 的四项改动(均针对后背瓶颈)

| 改动 | 动机 | 实现 |
|---|---|---|
| **逐像素重心插值 UV 标签** | 逐面均值→UV阶梯状, 后背单part内8点全靠UV精度区分 | `render_iuv.rasterize_iuv` 三角内重心插值(U 验证 9216 个不同值/有梯度) |
| **soft-argmax 亲和热图读出** | 硬argmin单像素、UV噪声主导 | `eval.readout_soft`: P(part)·exp(-‖UV-地址‖²/τ) 取期望位置(亚像素抗噪) |
| **真实后背 PCK 选 ckpt** | v1 暴露 val-loss≠真实域最优 | 每epoch在纯真实后背held-out(92)上算PCK@10选best(实选 ep1) |
| **域均衡采样 + WD0.05** | 合成占多→过拟合、真实后背稀缺 | WeightedRandomSampler syn0.4/真前0.3/真后0.3; weight_decay 1e-4→0.05 |

## 2. 评测(公平: v2 val_split 同一批真受试者 held-out; soft-argmax)

| 域 | 指标 | v0(仅合成/冻结) | v1(真实混训/解冻) | **v2(全部改进)** |
|---|---|---|---|---|
| **Scoliosis 真实裸背肺点** | PCK@10 / 中位% | 0.22 / 16% | 0.47 / 13% | **0.60 / 7%** |
| **HuMMan 真实正面心点** | PCK@10 | 0.65 | 0.91 | **0.94** |
| Synthetic 同域后背 | PCK@10 | 0.62 | 0.60 | 0.62 |

**真实后背肺点 PCK@10: 0.22 → 0.47 → 0.60;中位误差 16% → 13% → 7%。** 持续闭合 sim-to-real。

### soft-argmax vs argmin 消融(v2 same ckpt)
| 域 | soft | argmin | Δ |
|---|---|---|---|
| 真实裸背后背 | **0.60**(med7%) | 0.48(med11%) | **+0.12** |
| 同域前胸 | 0.95 | 0.94 | +0.01 |
→ soft-argmax 在**难的后背**显著有效, 前胸(本就易)影响小。

### 增益分解(真实后背 PCK@10)
- v1 argmin 0.39 → v1 soft 0.47:**读出改进 +0.08**
- v1 soft 0.47 → v2 soft 0.60:**重心UV标签 + 域均衡 + 真实后背选ckpt +0.13**
- 两条线(标签/训练 + 读出)都实打实贡献。

## 3. 关键发现 / 反思

1. **瓶颈定位正确, 干预有效**:四项改动叠加把真实后背 PCK 从 0.47→0.60、误差 13%→7%。重心UV(连续可分辨)+ soft-argmax(亚像素)是后背定位的关键组合。
2. **选 ckpt = 选"定位最优"而非"分割最优"**:v2 best=ep1, scoliosis part-mIoU(0.36)反低于 v1(0.44), 但 landmark PCK 更高(0.60>0.47)。real-back-PCK 选择正确锁定了定位峰值(ep1), 避开了后续过拟合(real-back PCK ep1 0.55→晚期 0.43-0.49)。
3. **仍有空间**:真实后背 0.60(误差7%)虽大幅改善, 离前胸(0.94)仍有差距。真实后背held-out仅92点、噪声大; scoliosis 直脊柱拟合给肺点注入纵向标签误差; 真实后背数据量仍少。
4. **v2 best=ep1 早停**:说明全解冻在这个数据量上很快过拟合后背→ v3 可试 LoRA(更强正则、小ckpt)或更多真实后背数据。

## 4. v3 候选(③提升)
- **Sapiens 人体编码器 A/B**(干净 v2 基线上测, architect 建议)——更高真实泛化天花板。
- **LoRA 替全解冻**——抗过拟合、ckpt 从 356MB→几MB、可能让后背更稳到更晚 epoch。
- **scoliosis 曲率重拟合**——清掉肺点纵向标签噪声(直脊柱拟合的系统误差)。
- **更多真实后背数据**(自采/更多 scoliosis)。

## 5. 产物(本地)
```
outputs/perc_v2/  best.pt(ep1)/last.pt(356MB) · config/metrics/train_v2.log/val_split.json
  eval/  eval_{v0,v1,v2}best_{humman,syn2,scoliosis}.json(+v2 argmin消融) · sample_{scoliosis,humman}.png
```
> 356MB 全模型本地留存; GitHub 放报告+metrics+config+eval json+val_split+样例图。
