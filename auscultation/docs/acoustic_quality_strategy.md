# 听诊声学质量模块（Acoustic Quality, 模块 ③ 的 Q 基石）

> 本文是声学质量模块的**策略 + 数据 + 设计**总纲,对应 [`model_architecture.md`](./model_architecture.md) **③ 音频驱动主动搜索**里的**音频质量模型 Q**。
> 机器可读的数据集/模型清单见 [`scripts/audio/datasets_registry.yaml`](../scripts/audio/datasets_registry.yaml);下载脚本 [`scripts/audio/download_audio_datasets.sh`](../scripts/audio/download_audio_datasets.sh);代码落点 [`ausc/audio/`](../ausc/audio)。
> 全部数据集/模型/license 事实在 2026-06-15 经联网核验,标 ⚠️ 的需人工再确认。

## 0. 这个模块在做什么

听诊的关键不只是"把听诊器放对位置",更是"**放上去之后听到的声音到底能不能用**"。本模块训练一个**声学质量模型 Q**:给一段电子听诊器录到的**心音 (PCG) 或肺音**,输出**质量分 + 不确定性**——这段录音是否清晰到能用于诊断。

Q 在系统里有两个用途(与感知模块一脉相承:先做 bootstrap 监督,再涨成闭环):
- **近期(数据采集期)**:作为**采集质检闸门**。遥操作采数据时,实时判断每个听诊点录到的声音是否合格,差的当场重录/重贴——保证我们自采的动作数据带的是干净音频,而不是事后才发现一堆废录音。
- **远期(Phase 3 闭环)**:作为**声音闭环的奖励信号 Q**(架构 ③)。"声音差→在邻域主动探索找更好声窗"这个行为,目标函数就是 `max Q`;Q 在每个状态都有定义,所以搜索能泛化到没见过的失败(详见 [`model_architecture.md`](./model_architecture.md) 「自发搜索:为什么用 Q 而非模仿」)。

> 与感知模块的对称:**感知**回答"听诊器该放哪"(空间),**Q** 回答"放这儿听得清不清"(声音)。两者都先用开源数据 + 自监督代理做 bootstrap,再在自采数据上对齐部署域。

## 1. 两条数据轨道（和感知一样,绝不混)

| | **外部开源音频轨道** | **自采同步音频轨道** |
|---|---|---|
| 学什么 | "什么声音算好/算差"的通用先验 + 质量校准标 | 我们这把听诊器、这套接触在真实部署域的声音分布 |
| 来源 | 公开心音/肺音数据集 + 预训练音频模型 | 遥操作采集时,与 RGB-D/力/语言**时间戳对齐**同步录的听诊器音频 |
| 规模 | 万级录音,便宜,马上能下 | 百~千级,贵(受 IRB/真人限制) |
| 角色 | 预训练 + 自监督代理 + 质量金标校准 | 部署域 in-domain 对齐 + 真实质量分布 + 最终评测 |
| 风险 | 设备/人群域差(儿科多、设备杂) | 量小,需自监督代理补标 |

下面 §2/§3 是**外部轨道**(本次重点:收集开源声音 + 预训练模型);§5 是**自采轨道**(同步采集规范)。

## 2. 开源声音数据集（重点:带"质量/噪声标"的最值钱）

判一段录音"好不好",最值钱的是**本身就带信号质量/噪声标注**的数据集——它直接给监督目标,不用我们瞎猜。下面按器官分,⭐=有现成质量标的主锚。完整字段见 registry。

### 2.1 心音 (PCG)

| 数据集 | 规模 | 设备 | 质量标? | 获取 | 优先级 |
|---|---|---|---|---|---|
| ⭐ **CinC 2016** (PhysioNet) | 3126 条 @2kHz | 多来源混合 | **有**:每条 cardiologist 评 good/poor + clean/noisy 测试划分 + `unsure` 标 | 实名认证(ODC-BY, 可商用) | **anchor** |
| ⭐ **CirCor DigiScope** (CinC 2022) | 5272 条 @4kHz | **DigiScope 电子听诊器**(与我们同类) | **弱**:murmur=Present/Absent/**Unknown**(Unknown≈不可判≈差音质) | 实名认证(ODC-BY) | **anchor** |
| ⭐ **ZCHSound** (浙大儿科) | 1259 人 | 电子听诊器 | **有**:现成 **clean vs noisy** 划分 | ⚠️ 访问未证实(走 IEEE 论文/作者) | anchor(若可得) |
| **EPHNOGRAM** | 69 条 @8kHz | 数字听诊器 + 环境噪声参考麦 | **有**:`low quality` 标(电极脱落/饱和) | 实名认证(ODC-BY) | support |
| **PASCAL** (2011) | A176/B656 | A: iPhone; B: DigiScope | **有**:A 的 `Artifact` 纯噪声类 + B 的 noisy 子集 | ⚠️ 官方站不稳,用镜像 | support |
| Yaseen 2018 | 1000 条 @8kHz | 混合 | 无(仅干净) | GitHub 开放 | support(仅正样本) |

### 2.2 肺音 / 呼吸音

| 数据集 | 规模 | 设备 | 质量标? | 获取 | 优先级 |
|---|---|---|---|---|---|
| ⭐ **SPRSound** (上海儿医) | 2683 录/9089 事件 | **Yunting II 电子听诊器** | **有**:record 级 **'Poor Quality' 类**(187 条) | GitHub 开放(**CC-BY-4.0, 可商用**) | **anchor** |
| **ICBHI 2017** | 920 录/~5.5h | **Littmann 3200 / Meditron 电子** + AKG 麦 + 声学听诊器 | 无(噪声未标);有 crackle/wheeze + 呼吸周期 | Kaggle 镜像开放 | anchor(设备多样性) |
| **HF_Lung_V1** | 9765×15s @4kHz | Littmann 3200 + 8 麦阵列;**ICU/呼吸机** | 无;但 ICU 段=真实高噪声硬负例 | GitLab(CC-BY-4.0) | support |
| **KAUH** | 112 人/336 文件 | Littmann 3200(蓝牙) | 无(干净参考分布) | Mendeley(CC-BY-4.0) | support |
| **RespiratoryDatabase@TR** | 77 人/12+4 通道 | 多通道数字听诊器 | 无;多通道→SNR 一致性特征 | Mendeley(CC-BY-4.0) | support |

### 2.3 SSL 预训练补充(非听诊器,仅自监督预训练)
- **Coswara** / **COUGHVID**:手机麦录的呼吸/咳嗽,CC 授权。**不进有监督质量集**(设备域差太大),只用于呼吸音自监督预训练扩量。

> **三个主锚记住**:**SPRSound**(肺音质量标)、**CinC 2016**(心音质量标)、**CirCor DigiScope**(设备最像我们)。其余作多样性/负样本/SSL。

## 3. 预训练音频模型（frozen embedder + 轻头,对标视觉的 DINOv2+LoRA）

策略与感知一致:**冻结一个强音频编码器当特征提取器,只训一个轻量质量头**。下面按"能不能 ship"分,⚠️=license 雷区。

| 模型 | 预训练内容 | license / 可商用 | 角色 |
|---|---|---|---|
| **HeAR** (Google 健康声学) | 313M×2s 健康声(咳嗽/呼吸,**无心音**) | gated(接受 HAI-DEF 条款)/ **可商用**(有医疗器械豁免条款) | **肺/呼吸音质量最强可商用 frozen embedder** |
| **WavLM** (微软) | 语音 + **去噪/重叠** | CC-BY-SA-3.0 / 可商用(需署名+相同共享) | ⭐**质量代理首选先验**:预训练含噪声→embedding 已编码 SNR/干扰 |
| **BEATs** (微软) | AudioSet SSL | **MIT** / 可商用 | license 最干净的强通用 backbone |
| **CardioPHON** (LIH) | 6 个心音集 SSL(BYOL-A) | ⚠️ 未证实 | ⭐**与本任务最贴合**:心音 SSL **+ 质量评估头**。首选心音 teacher/对照 |
| **AuscultaBase** | 心+肺+肠 40k 录(含 CinC/CirCor) | ⚠️ 含糊(部分 NC) | **唯一听过心音的已放权重模型**;license 含糊 |
| **OPERA** (剑桥) | ~404h 呼吸音(无心音) | ⚠️ **CC-BY-NC-4.0 非商用** | 最强 ungated 呼吸编码器,**仅研究/蒸馏,不可 ship** |
| PANNs/CNN14 · YAMNet · CLAP | AudioSet / 文本-音频 | MIT / Apache-2.0 / 可商用 | 轻量 baseline;CLAP 可用文本探针("noisy/clipped")零样本打质量分 |

> **license 雷区(产品方向必看)**:**OPERA 非商用**、**HeAR gated**、**WavLM 是 CC-BY-SA 不是 MIT**、SQI 工具多为 **GPL-3.0**、**NISQA 权重非商用**。详见 §6。

## 4. Q 模型设计（自监督代理 + 轻头 + 少量校准标）

延续架构 ③ 组件 A 的设计:**Q 主要靠自监督声学代理 + 演示隐式标签,只用极少人工校准**。

### 4.1 自监督质量代理(零/低人工,作弱标 teacher)
- **心音 (PCG)**:
  - `pyPCG`(Tang 2021 SQI 套件,GPL-3.0):autocorr 峰、包络标准差、**周期性分数**、峰度、谱熵——分割无关。
  - **S1/S2 可检测性**:用 Springer HSMM 分割 / `BioSPPy.signals.pcg` 出 S1·S2,看检出**数量 + 置信 + 周期规则性**。
  - SNR(Giordano:`SNR=20·log10(A_S/4σ_N)`,≥14dB 为物理锚)。
- **肺音**:
  - `Grooby` 5 级 SQI(GPL-3.0,心肺通用):功率比 + 自相关周期性 + SNR。
  - **呼吸相检出**(HF_Lung 风格):无吸/呼相 → "无气流/无呼吸"质量代理。
  - 心音污染惩罚:肺音通道里残留心跳能量越大、贴合越差。
- **通用模板**:`torchaudio SQUIM`(参考-free 估 STOI/PESQ/SI-SDR)、`NISQA` 多维质量(noisiness≈运动伪影,discontinuity≈听诊器掉接触)——**借架构/思路,分数需在体声上重校**(都是语音训练的)。

### 4.2 校准标(少量真标,对齐代理与"人耳判好坏")
- **肺音**:SPRSound 'Poor Quality' vs 可用 → 直接训质量头。
- **心音**:CinC 2016 good/poor + `unsure`;ZCHSound clean/noisy(若可得)。
- **演示隐式标**(自采轨道):医生/操作者**停留久=好、快速离开/重录=差**。

### 4.3 架构(冻结 + 轻头)
```
听诊器音频窗口(+ 力/呼吸相位上下文)
   → [冻结编码器]  心音:CardioPHON/AuscultaBase(teacher) → 自训可商用学生
                   肺音:HeAR / BEATs(可商用)  · WavLM(噪声感知先验)
   → [轻量质量头(线性/小 MLP)]  → Q∈[0,1] + 不确定性(MC-dropout/集成)
监督 = 自监督代理弱标(主) + 质量金标校准(SPRSound PQ / CinC unsure) + 演示隐式标
```
- 输入约定:统一重采样(各集 2k–44.1k 混杂),16kHz mono 起步;窗口对齐编码器(HeAR 2s / OPERA 8s / WavLM 任意)。
- **不确定性是一等输出**:Q 低**或**不确定性高 → 触发"需复核/主动探索",与感知的翻转一致性置信门同构。

## 5. 自采同步采集规范（音频如何与遥操作数据对齐）

采集动作数据时,听诊器音频**和 RGB-D/力/本体/语言严格时间戳对齐**同步录(架构里 VLA 和 Q 都吃音频)。约定:

- **硬件**:末端电子听诊器 = **ThinkLabs One**(见 [`mechanical_sensor/`](../mechanical_sensor));音频走 3.5mm/USB 声卡进采集机,与力反馈(Hall 位移)、相机共用同一时钟。
- **格式**:WAV,**≥16kHz**(建议 16k mono 起步,设备支持则 44.1k 留档),16-bit;录原始波形 + 落采集时戳。
- **每点 ~10s 窗口**:对应架构里"每点录音 ~10s"的速度瓶颈——这正是 Phase 3 用 World Model 想缓解的。
- **元数据(每段音频随录)**:`task`(heart/lung)、听诊点名(APETM / R1..L5)、目标力 & 实测力、呼吸/心动相位(若可得)、操作者动作标签、是否重录。
- **目录 + manifest**:与感知的 `manifest.jsonl` 同制式,一条音频一行:
  ```json
  {"id": "...", "wav": "audio/<sess>/<point>.wav", "task": "lung",
   "point": "R3", "fs": 16000, "dur_s": 10.2, "force_n": 4.1,
   "quality_label": null, "quality_proxy": {"sqi": 0.0, "snr_db": 0.0},
   "session": "...", "t_start_ms": 0}
  ```
  - `quality_label` 留给医生/演示隐式标;`quality_proxy` 由 §4.1 自监督代理自动填。
- **采集即质检**:每段录完即时跑 §4.1 代理,差的当场标红重录(Q 的近期用途)。

> 这条轨道与感知的自采裸背图采集**并行**进行——同一次遥操作 session 同时产出:对齐的图像(给感知)+ 动作轨迹(给 VLA)+ 音频(给 Q)。

## 6. 许可证决策（产品 vs 研究,先定再训）

这是真正的岔路,**模型选择取决于此**:

- **若走临床/商用产品**:用**可商用**栈——编码器 HeAR(gated 但允许商用)/ BEATs(MIT)/ PANNs(MIT)/ CLAP;质量头**我们自训**;质量金标 CinC(ODC-BY)/ SPRSound(CC-BY)。**OPERA / AuscultaBase / NISQA 权重 / pyPCG·Grooby(GPL)** 仅作**研究 teacher 蒸馏**,产出我们自己 license 干净的学生模型,**不直接进产品**。
- **若先做研究/原型**:上述全部可直接用(含 OPERA),最快出结果,日后再蒸馏出商用版。

> **默认建议**:按**产品路线**搭骨架(可商用编码器 + 自训头),把 OPERA/AuscultaBase/CardioPHON 当 teacher——既不挡商用,又吃到体声专用先验。如确认纯研究,放开即可。⚠️ GPL-3.0 的 SQI 工具(pyPCG/Grooby/Springer)用于**离线生成弱标**没问题(不连接、不分发),进在线产品前再确认。

## 7. 路线图

```
S0  收集开源数据 + 模型   下 §2 开放集 + 拉 §3 可商用编码器权重(本周可做)
   │                      registry + download 脚本已就绪; PhysioNet/HeAR 走认证
S1  自监督代理弱标         pyPCG/Grooby/分割 跑出 SQI/S1S2/呼吸相 → 弱质量标
   │
S2  Q v0(冻结+轻头)        编码器冻结 + 轻头, 弱标训 → SPRSound PQ / CinC unsure 校准+评测
   │
S3  自采同步音频对齐        遥操作 session 同步录音(§5) → 部署域 in-domain 微调 + 演示隐式标
   │
S4  接入 Phase 3 闭环       Q 作奖励: 声音差→邻域主动探索(架构 ③); 进阶声学 World Model
```

## 8. 代码落点

| 路径 | 内容 |
|---|---|
| [`scripts/audio/datasets_registry.yaml`](../scripts/audio/datasets_registry.yaml) | ★数据集 + 模型 + 质量工具登记表(access/license/质量标/优先级) |
| [`scripts/audio/download_audio_datasets.sh`](../scripts/audio/download_audio_datasets.sh) | 开放集下载脚本;认证/申请集打印手动指引 |
| [`ausc/audio/`](../ausc/audio) | 声学质量核心包(质量代理 / Q 模型 / 数据加载,见其 README) |
| [`docs/model_architecture.md`](./model_architecture.md) ③ | Q 在整机里的接口(声音闭环奖励) |

> 一句话:**Q 回答"这段听诊音质量够不够诊断"。先用开源数据(SPRSound/CinC 质量标)+ 可商用音频编码器 + 自监督 SQI 代理 bootstrap;采动作数据时同步录对齐音频做部署域对齐;最终长成 Phase 3 声音闭环的奖励信号。**
