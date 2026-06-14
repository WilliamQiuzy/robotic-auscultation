# `ausc/models/` —— 模型代码(占位, 待实现)

**感知模块**网络定义放这里。设计契约见 [`docs/model_architecture.md`](../../docs/model_architecture.md) ①感知模块。

## 计划文件

| 文件 | 内容 |
|---|---|
| `backbone.py` | 视觉编码器(ViT / HRNet / CNN)→ 特征图 |
| `iuv_head.py` | 稠密 IUV 预测头:逐像素 `(part I∈{0..24}, U, V)`(分类 part + 回归 U,V) |
| `point_readout.py` | 从预测 IUV 读出听诊点:匹配目标顶点 `(I,U,V)` 的像素 → 2D 点;`infer_side` 定前/后 |
| `perception.py` | 组装 backbone + iuv_head + readout 的完整感知模型 |
| `losses.py` | part 交叉熵 + U,V 回归(masked) + 点位辅助损失 |

> 输入:第三人称 RGB `[H×W×3]`(单视角, RGB-only)。
> 输出:稠密 IUV `[H×W×3]` + 听诊点 `{name:(x,y,conf)}` + region(front/back/none)。
> 监督标签由 `ausc/datasets/` 生成的 `manifest.jsonl + iuv/` 提供。
