# `ausc/models/` —— 感知模型(已实现)

**感知模块**的网络定义。设计契约见 [`docs/model_architecture.md`](../../docs/model_architecture.md) ①感知模块。
现状与诚实评估见 [`docs/PERCEPTION_STATUS.md`](../../docs/PERCEPTION_STATUS.md);架构/评测细节见
[`docs/landmark_head_and_eval.md`](../../docs/landmark_head_and_eval.md)。

> **状态:已实现并投产。** 全部网络代码在单文件 [`iuv_model.py`](./iuv_model.py)(~175 行)。
> 生产模型 = **`perc_lmh_v4`**(landmark 热图头 + 结构化读出),对人手金标 **PCK@10 ≈ 96%**
> (旧 IUV-地址法 v7 仅 57%)。早期计划的多文件拆分(backbone/iuv_head/point_readout/…)未采用,
> 改为一个紧凑模块,职责见下表。

## 实际文件

| 符号(在 `iuv_model.py`) | 内容 |
|---|---|
| `DinoV2Encoder` / `build_encoder` | 视觉编码器:DINOv2 ViT-B/14(`torch.hub`),patch=14,默认**冻结**;`build_encoder` 预留换 Sapiens 的接口 |
| `LoRALinear` / `apply_lora` | LoRA 低秩适配(默认 r16/a32):冻结 DINOv2 基座,只训注意力/MLP 的低秩旁路 A/B + 头(强正则 + ckpt 小,替代全解冻) |
| `IUVHead` | 稠密 IUV 预测头:卷积解码器 → 逐像素 `(part I∈{0..24}, U, V)`(25 类 part logits + 2 通道 UV,sigmoid) |
| `LandmarkHead` | **听诊点热图头**:卷积解码器 → `K` 个点各一个热图通道,推理 soft-argmax 出 `(x,y)`。**直接受 2D 点监督,绕开有偏 V 场**;各点独立通道 → 前胸/后背天然解耦(不互相稀释) |
| `DenseIUVModel` | 组装:`encoder` + `IUVHead`(`head`) + 可选 `LandmarkHead`(`landmark_head`,`n_landmarks>0` 启用)。`forward`(只出 IUV,推理兼容)/`forward_with_landmarks`(一次编码同出 IUV+热图,训练)/`predict_landmark_heatmap`(推理读点) |
| `iuv_loss` | part 交叉熵(全像素) + UV masked Smooth-L1(仅前景 part>0) |
| `landmark_heatmap_loss` | 听诊点热图损失:`sigmoid(logits)` 与高斯目标的 masked MSE(仅在场+可见通道上算) |

> 输入:第三人称 RGB `[H×W×3]`(单视角,RGB-only)。
> 输出:稠密 IUV `[H×W×3]` + 听诊点 `{name:(x,y,conf)}`(landmark 头读出) + region(front/back)。

## 双头设计(为什么两条头都留)

- **`LandmarkHead` = 生产读点路径**。听诊点直接从专用热图通道读出,**不再**在 `(I,U,V)` 地址上反查。
  金标证明 IUV-地址法被 HMR2 标签锁死在 57%;换直接监督的 landmark 头即破到 96%。
- **`IUVHead` 仍保留**,作为体表参数化坐标图(body-surface map),供将来 Phase 3 的**声音闭环 / World Model
  主动探索**用——沿 `(I,U,V)` 在体表流形上挪点(见架构文档「IUV 作为闭环探索的体表坐标系」)。
  生产训练时它**冻结**(warm-start 自 `perc_v7`),保证后背 IUV 不退步。

> 训练入口见 [`../training/`](../training)(`train.py` 的 `--landmark_head/--freeze_iuv/--init_from` 路径)
> 与生产用的纯人手标训练 [`scripts/train_landmark_human.py`](../../scripts/train_landmark_human.py);
> 读出库 [`scripts/landmark_readout.py`](../../scripts/landmark_readout.py);部署 [`scripts/deploy_landmark.py`](../../scripts/deploy_landmark.py)。
> landmark 头的**固定通道顺序**(后背肺 10 + 前胸心 5 = 15)定义在 [`ausc/tasks.py`](../tasks.py) `LANDMARK_ORDER`。
