# 感知模型 v0 — 训练与评测报告

> 日期: 2026-06-02 · 训练机: Vast.ai RTX 4090(实例 39167186)· 评测机: RTX 3060(38791825)
> 全部产物本地: `auscultation/outputs/perc_v0/`(权重 ep00–19 + best/last、metrics.jsonl、train_v0.log、eval/)

## 1. 配置(可复现)

| 项 | 值 |
|---|---|
| 任务 | 单图 RGB → 稠密 IUV(DensePose 制式: part I 0–24 + U,V) |
| 编码器 | **DINOv2 ViT-B/14(冻结)** —— 可插拔, 下一步换 Sapiens |
| 头 | 轻量卷积解码器(part 25类 + UV 2通道), 可训练 **2.4M** 参数 |
| 损失 | part 交叉熵 + 前景 masked Smooth-L1(UV) |
| 训练数据 | **synthetic 25600**(train 23552 / val 2048, 按 subject 划分防泄漏) |
| 分辨率/批 | 518² / bs16 / 20 epoch / AdamW 3e-4 cosine |
| 用时 | ~642s/epoch, 共 **~3.6h** |
| 收敛 | val loss 0.117(ep0)→ **0.0596(ep19)**, `[done]` |

## 2. 评测结果

评测标准: ① 稠密 part mIoU + 像素准确率 + 前景 UV-L1; ② **★听诊点定位**(从预测 IUV 按各点标准 (I,U,V) 地址反查 → 与 GT 比, 像素误差 / 占图宽% / PCK@τ); ③ 跨域 gap。

| 数据集 | part mIoU | pix-acc | UV-L1 | 前胸心点(med px / %img / PCK@10%) | 后背肺点(med / %img / PCK@10%) |
|---|---|---|---|---|---|
| **syn-val**(同域) | **0.800** | 0.975 | 0.048 | **16.4px / 3.2% / 0.94** ✅ | 65.3px / 12.6% / 0.45 |
| **scol-real**(跨域真实裸背) | 0.124 | 0.765 | 0.120 | —(无前胸) | 175.7px / 33.9% / 0.13 ❌ |

定性图: `eval/syn_overlays/`(同域: 分块干净、心点红≈绿) · `eval/scol_overlays/`(跨域: 分块杂乱、点散落)。

## 3. 关键发现(反思, 不只结果)

1. **同域、前胸心点 = 成功**。PCK@10% 0.94、中位误差仅 3.2% 图宽 —— 证明"稠密 IUV → 按地址反查听诊点"这条**机制成立**, 模型在合成域学到了可用的心脏定位。
2. **后背肺点明显弱于前胸**(同域 12.6% vs 3.2%)。原因推测: ① 后背躯干块面积大、UV 变化平缓 + 标签是**逐面均值 UV(粗)**→ 反查分辨率低; ② 8 个肺点跨度大。**改进**: UV 用逐像素重心(GPU pytorch3d 渲染精确标签), 或给 UV 更高权重 / 后背单独评估。
3. **★sim-to-real gap 是头号问题**。同域 mIoU 0.80 → 真实 0.12, 后背点 65→176px。**纯 synthetic + 冻结 DINOv2 不迁移到真实照片**。这正是下一步要打的点:
   - **(a) 训练加入真实域**: scoliosis(已有 1118)纳入训练, 拉 HuMMan 真实多视角;
   - **(b) 换 Sapiens 人体预训练编码器**: 3 亿真人图预训练, 对真实图泛化远好于通用 DINOv2(调研已证 Sapiens > DINOv2 于人体稠密任务);
   - **(c) 更强域随机化 / 真皮纹理**(BEDLAM 已用, 可加背景替换、更激进光照);
   - **(d) 解冻骨干微调**(v0 只训了头 2.4M; 24GB 机可解冻 0.3B)。

## 4. 结论

v0 验证了**整条链路**(数据→稠密IUV训练→听诊点反查→评测)可用, 且**同域心点定位达到可用精度**。瓶颈非常清晰 = **真实域泛化**。下一版优先级: **Sapiens 编码器 + 真实数据混训 + 解冻微调**, 预期主要改善 scol-real 的跨域指标。

## 5. 产物清单(本地, 供审核)

```
outputs/perc_v0/
├── best.pt / last.pt / ep00–ep19.pt   # 全 20 epoch 权重(可审计)
├── config.json                         # 训练配置
├── metrics.jsonl                       # 每 epoch 指标历史
├── train_v0.log                        # 完整训练日志
└── eval/
    ├── eval_syn-val.json / eval_scol-real.json
    ├── syn_overlays/  scol_overlays/    # 定性叠加图(GT绿 vs 预测红)
    └── EVAL_REPORT.md (本文件在 outputs/perc_v0/)
```
