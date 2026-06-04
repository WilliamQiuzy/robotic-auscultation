# 感知模型 v1 — 训练与评测报告(sim-to-real 闭合)

> 日期: 2026-06-04 · 训练机: Vast.ai RTX 4090 · 权重本地: `auscultation/outputs/perc_v1/`(best/last 全模型 356MB + metrics + log + val_split + eval/)
> v1 目标: 闭合 v0 暴露的 **sim-to-real gap**(v0 仅合成训练, 真实域差)。

## 1. v1 相对 v0 的改动

| | v0 | **v1** |
|---|---|---|
| 训练域 | 仅 synthetic | **HuMMan(真实)+ Scoliosis(真实)+ Synthetic** 混训 |
| 编码器 | DINOv2 冻结(训 2.4M) | **解冻 DINOv2 全量微调(88.9M)**(头 lr 3e-4 / 编码器 3e-5) |
| 数据 | 旧增强(含 F4 极近裁剪 / 强退化) | **去 F4(只留 F1/F2/F3)+ mild 退化**(护低分辨率真实图) |
| train/val 切分 | ⚠️ **泄漏**(同人不同视角跨集, val 乐观) | **真受试者切分**(p*/syn*/scol, 重叠=0, val 可信) |

## 2. 评测(公平: 真受试者 held-out, v0/v1 都未训过这些人)

| 域 | 指标 | v0(仅合成/冻结) | **v1 best**(ep4) | **v1 last**(ep14) |
|---|---|---|---|---|
| **HuMMan 真实正面** | mIoU / 心点 PCK@10 | 0.19 / 0.30 | **0.54 / 0.90** | 0.54 / 0.92 |
| **Scoliosis 真实裸背** | mIoU / 肺点 PCK@10 | 0.16 / 0.19 | 0.44 / 0.33 | **0.44 / 0.39** |
| Synthetic 同域(参考*) | mIoU / 心点 PCK@10 | 0.82 / 0.98 | 0.88 / 0.99 | 0.93 / 0.99 |

*v0 训过这些 synthetic 生成体(旧F4版), 故 v0-synthetic 略乐观; HuMMan/Scoliosis 对 v0 是干净跨域。
心点误差(% 图宽): v0 真实正面 15.4% → v1 **4.0%**。肺点(真实背): v0 27.6% → v1 best 21.5% / last **15.3%**。

## 3. 关键发现

1. **sim-to-real gap 大幅闭合(v1 方向被证实)**:
   - 真实正面**心点 PCK@10 0.30 → 0.90**(误差砍到 4% 图宽)——真实正面基本可用。
   - 真实裸背**肺点 PCK@10 0.19 → 0.39,mIoU 0.16 → 0.44**——明显改善, 但仍是最弱项。
2. **肺点/后背是瓶颈**:真实后背数据少(HuMMan 后背仅 91 + Scoliosis 1118)+ 肺点跨度大 + 逐面均值 UV(粗)。下一步: 精确重心 UV(GPU pytorch3d)、更多真实后背数据。
3. **checkpoint 选择 ≠ val-loss 早停**:val-loss 最优的 best(ep4)在**真实裸背**上反而**不如 last(ep14)**(PCK 0.33 vs 0.39)。混合 val-loss 在 ep4 触底后升(过拟合到合成占多的 val), 但真实域定位 last 更好。**结论: 应按真实域 landmark PCK 选 checkpoint, 而非 mixed val loss。**
4. **泄漏修复的意义**:v0"同域 PCK 0.94"是泄漏乐观值; v1 真受试者 held-out 给出诚实数(真实正面 0.90, 真实背 0.39)。

## 4. 下一步(v2 候选)

- **按真实域指标选 ckpt / 早停**(而非 mixed val loss);
- **精确 UV 标签**(逐像素重心, 修后背弱);
- **更多真实后背数据**(自采 / 更多 scoliosis);
- **Sapiens 人体编码器**(进一步抬真实泛化);
- 真实/合成**采样配比**与正则(缓过拟合)。

## 5. 产物(本地)

```
outputs/perc_v1/
├── best.pt / last.pt        # 全模型(编码器+头), 各 356MB(解冻必存全模型)
├── config.json / metrics.jsonl / train_v1.log / val_split.json
└── eval/  eval_{v0best,v1best,v1last}_{humman,synthetic,scoliosis}.json + dump_*/ 定性图
```
> best/last 全模型 356MB 不入 git(本地留存); GitHub 放本报告 + metrics + config + eval json + val_split + 样例图。
