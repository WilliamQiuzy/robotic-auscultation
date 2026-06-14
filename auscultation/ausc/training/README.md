# `ausc/training/` —— 训练代码(占位, 待实现)

感知模块的数据加载 / 训练循环 / 评测放这里。

## 计划文件

| 文件 | 内容 |
|---|---|
| `dataset.py` | `torch.utils.data.Dataset`:读 `manifest.jsonl`,加载 img + iuv,**按受试者切 train/val/test**(防泄漏) |
| `transforms.py` | 训练期在线增强钩子(复用 `ausc/augment.py` 的 5 种核心,随机组合) |
| `train.py` | 训练循环(混合 HuMMan + synthetic 预训练 → scoliosis/in-domain 微调) |
| `evaluate.py` | 评测:IUV part mIoU、U/V 误差、听诊点定位误差(像素/分位)、前后判别准确率 |
| `config.py` | 超参 / 路径(读 `configs/*.yaml`) |

## 数据约定
- 输入数据集由 `python scripts/gen_dataset.py --dataset all --out data/dataset` 产出。
- 多数据集混采:HuMMan + synthetic(稠密 IUV 监督)为主;scoliosis(点监督, `iuv=null`)做后背 in-domain。
- 切分必须**按受试者**(`id` 前缀:`<seq>` / `scol_<stem>` / `syn<NNNNN>`),不可按帧。
