# `ausc/audio/` —— 听诊声学质量核心包（脚手架, 待实现)

声学**质量模型 Q** 的代码:质量代理 / 编码器封装 / 质量头 / 数据加载。
**范围:只做后背肺音 (lung/respiratory),不碰心音。** License 路线 = 研究/非商用。
策略与数据/模型清单见 [`docs/acoustic_quality_strategy.md`](../../docs/acoustic_quality_strategy.md);
整机接口见 [`docs/model_architecture.md`](../../docs/model_architecture.md) ③。

> **状态:脚手架。** 本模块刚起步(对应路线图 S0)。下面是计划的文件 + 接口契约;
> 肺音数据集/预训练模型已调研登记([`scripts/audio/datasets_registry.yaml`](../../scripts/audio/datasets_registry.yaml)),
> 下载脚本已就绪([`scripts/audio/download_audio_datasets.sh`](../../scripts/audio/download_audio_datasets.sh))。
> 先有数据 + 自监督弱标(S1)再落 Q v0(S2),避免无数据空写训练代码。

## 计划文件

| 文件 | 内容 |
|---|---|
| `proxies.py` | **自监督质量代理**(零/低人工弱标 teacher,肺音):`Grooby` 多级 SQI(GPL,取肺部分)、呼吸相/气流检出(HF_Lung 风格)、SNR/削波/静音比/突发噪声、心音污染惩罚(肺点串到的心跳残余) |
| `encoders.py` | **冻结编码器封装**(可插拔):**OPERA**(呼吸专用,首选)/ HeAR(健康声)/ AuscultaBase(听过听诊器肺音)/ WavLM(噪声感知先验)。统一 16kHz 重采样 + 窗口适配 |
| `quality_head.py` | **轻量质量头**:线性/小 MLP → `Q∈[0,1]` + 不确定性(MC-dropout/集成)。对标感知的"冻结骨干 + 轻头" |
| `dataset.py` | 读音频 `manifest.jsonl`(schema 见策略文档 §5),多数据集混采,**按受试者切分**(防泄漏);弱标 + 校准标双通道 |
| `train.py` | Q 训练:弱标(代理)主监督 + 质量金标(SPRSound 'Poor Quality')校准;按金标选 best |

## 数据约定

- **外部开源肺音**(预训练/校准):由 [`scripts/audio/download_audio_datasets.sh`](../../scripts/audio/download_audio_datasets.sh) 下到 `data/audio/<dataset>/`。
- **自采同步肺音**(部署域):遥操作 session 与 RGB-D/力/语言**时间戳对齐**录,`task=lung`,后背肺点(R1..L5),manifest 一条音频一行(schema 见策略文档 §5)。
- **质量金标**:仅 **SPRSound 'Poor Quality'** 是现成肺音质量标;其余靠 `proxies.py` 自动补弱标。
- 切分**按受试者**,不可按帧/按片段。

## 设计要点(为什么这么搭)

- **冻结编码器 + 轻头**:与感知 DINOv2+LoRA 同思路——数据少,别全调编码器;吃音频基础模型先验。**OPERA 是呼吸音专用 foundation model**,肺音首选;WavLM 作噪声感知先验。
- **自监督代理为主、人标为辅**:Q 主要从 SNR/呼吸周期可检测性/气流这些**声学物理**学"什么算好",对没见过的差音频也判差 → 能泛化触发探索(架构 ③ 的命门)。
- **不确定性是一等输出**:Q 低或不确定性高 → 触发复核/主动探索,与感知翻转一致性置信门同构。
- **研究/非商用路线**:OPERA(CC-BY-NC)、AuscultaBase、Grooby(GPL)等均可直接用;发表时遵守各自 license。
