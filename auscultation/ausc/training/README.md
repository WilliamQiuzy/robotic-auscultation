# `ausc/training/` —— 感知模块训练 / 评测(已实现)

感知模块的数据加载 / 训练循环 / 评测。模型定义见 [`../models/`](../models)。

> **状态:已实现。** 训练 IUV 头 + landmark 头的通用循环在 [`train.py`](./train.py);
> 生产模型 `perc_lmh_v4`(纯人手标 landmark 头)由 [`scripts/train_landmark_human.py`](../../scripts/train_landmark_human.py)
> 产出(人手标种子 + 自训练放量)。现状见 [`docs/PERCEPTION_STATUS.md`](../../docs/PERCEPTION_STATUS.md)。

## 实际文件

| 文件 | 内容 |
|---|---|
| `dataset.py` | `AuscIUVDataset`:读 `manifest.jsonl`,加载 img + iuv(+可选 landmark 高斯热图);**按受试者切 train/val/test**(`split_manifest(s)` / `subject_key`,防"同人不同视角"泄漏);多数据集混训;可选在线部署取景增强(`geom_aug`:上背半身裁切 + 受试者小远缩放) |
| `train.py` | 训练循环。冻结编码器训 IUV 头 / `--lora` 低秩微调 / `--unfreeze` 全解冻;`--landmark_head`+`--freeze_iuv`+`--init_from` = 暖启动 v7、冻结 IUV、只训 landmark 头;域分组均衡上采样(合成/真实前/真实后,真实后背稀缺提权);按**真实后背 PCK@10** 选 best ckpt;可选权重 EMA;`config.json`+`metrics.jsonl` 审计 |
| `eval.py` | IUV-地址 soft-argmax 读出的听诊点定位评测(PCK,内部 ckpt 选择用);`landmark_addresses` 解析目标顶点 `(I,U,V)` |
| `eval_realback_landmarks.py` | landmark 头在真实后背上的 PCK 评测(热图读出 vs GT 点) |
| `infer_overlay.py` | 推理可视化:`center_square` 取景 + 点/IUV 叠图 |

> 训练期增强**已在数据生成阶段烘焙**进数据集(`ausc/augment.py` + `datasets/`),`train.py` 不再二次增强;
> 唯一在线增强是 `dataset.py` 的部署取景 `geom_aug`(默认关)。早期计划的 `transforms.py` / `config.py` 未单列。

## 数据约定
- 输入数据集由 `python scripts/gen_dataset.py --dataset all --out data/dataset` 产出。
- 多数据集混采:HuMMan + synthetic(稠密 IUV 监督)为主;scoliosis(后背 in-domain)。
- 切分必须**按受试者**(`id` 前缀:`<seq>` / `scol_<stem>` / `syn<NNNNN>`),不可按帧;`train.py` 启动时断言无受试者泄漏并落 `val_split.json`。

## 两条训练路径(都已实现)
1. **通用循环 `train.py`** —— 产出早期 IUV-地址模型(`perc_v0..v7`)与 landmark 头实验(`perc_v9a`,冻结 IUV、只训 landmark 头)。
2. **生产路径 `scripts/train_landmark_human.py`** —— 纯人手标后背 10 点直接监督 landmark 头(暖启动+冻结 v7 IUV 骨干),
   配结构化读出 → 生产模型 `perc_lmh_v4`(金标 PCK@10 ≈ 96%)。这是金标证明 IUV-地址法仅 57% 后的"治本"路径。
