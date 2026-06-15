# 听诊声学质量模块（Acoustic Quality, 模块 ③ 的 Q 基石）

> **范围:只听后背肺音 → 本模块只做肺音 (respiratory/lung sound)。不收心音、不做心音质量。**
> 与项目既定范围一致(感知侧已 lock 到 posterior-lung)。
> 本文是声学质量模块的**策略 + 数据 + 设计**总纲,对应 [`model_architecture.md`](./model_architecture.md) **③ 音频驱动主动搜索**里的**音频质量模型 Q**。
> 机器可读清单见 [`scripts/audio/datasets_registry.yaml`](../scripts/audio/datasets_registry.yaml);下载脚本 [`scripts/audio/download_audio_datasets.sh`](../scripts/audio/download_audio_datasets.sh);代码落点 [`ausc/audio/`](../ausc/audio)。
> **License 路线 = 研究/非商用**:OPERA / AuscultaBase 等 CC-BY-NC 模型可**直接使用**(只需发表时遵守署名/非商用)。
> 全部数据集/模型/license 事实在 2026-06-15 经联网核验,标 ⚠️ 的需人工再确认。

## 0. 这个模块在做什么

听诊的关键不只是"把听诊器放对位置",更是"**放上去之后听到的肺音到底能不能用**"。本模块训练一个**声学质量模型 Q**:给一段电子听诊器(ThinkLabs One)录到的**后背肺音**,输出**质量分 + 不确定性**——这段录音是否清晰到能用于诊断。

Q 在系统里有两个用途(与感知模块一脉相承:先做 bootstrap 监督,再涨成闭环):
- **近期(数据采集期)**:作为**采集质检闸门**。遥操作采数据时,实时判断每个肺听诊点录到的声音是否合格,差的当场重录/重贴——保证我们自采的动作数据带的是干净音频。
- **远期(Phase 3 闭环)**:作为**声音闭环的奖励信号 Q**(架构 ③)。"声音差→在邻域主动探索找更好声窗"这个行为,目标函数就是 `max Q`;Q 在每个状态都有定义,所以搜索能泛化到没见过的失败(详见 [`model_architecture.md`](./model_architecture.md) 「自发搜索:为什么用 Q 而非模仿」)。

> 与感知模块的对称:**感知**回答"听诊器该放在后背哪个肺点"(空间),**Q** 回答"放这儿听得清不清"(声音)。两者都先用开源数据 + 自监督代理做 bootstrap,再在自采数据上对齐部署域。

## 1. 两条数据轨道（和感知一样,绝不混)

| | **外部开源肺音轨道** | **自采同步肺音轨道** |
|---|---|---|
| 学什么 | "什么肺音算好/算差"的通用先验 + 质量校准标 | 我们这把听诊器、这套后背接触在真实部署域的肺音分布 |
| 来源 | 公开肺/呼吸音数据集 + 预训练音频模型 | 遥操作采集时,与 RGB-D/力/语言**时间戳对齐**同步录的听诊器音频 |
| 规模 | 万级录音,便宜,马上能下 | 百~千级,贵(受 IRB/真人限制) |
| 角色 | 预训练 + 自监督代理 + 质量金标校准 | 部署域 in-domain 对齐 + 真实质量分布 + 最终评测 |

下面 §2/§3 是**外部轨道**(本次重点:收集开源肺音 + 预训练模型);§5 是**自采轨道**(同步采集规范)。

## 2. 开源肺音数据集（重点:带"质量/噪声标"的最值钱）

判一段录音"好不好",最值钱的是**本身就带信号质量/噪声标注**的数据集——直接给监督目标。⭐=有现成质量标的主锚。完整字段见 registry。

| 数据集 | 规模 | 设备 | 质量标? | 获取 | 优先级 |
|---|---|---|---|---|---|
| ⭐ **SPRSound** (上海儿医) | 2683 录/9089 事件(repo 含 BioCAS 多年扩充) | **Yunting II 电子听诊器** | **有**:record 级 **'Poor Quality' 类**(187 条) | GitHub 开放(CC-BY-4.0) | **anchor** |
| **ICBHI 2017** | 920 录/~5.5h | **Littmann 3200 / Meditron 电子** + AKG 麦 + 声学听诊器 | 无(噪声未标);有 crackle/wheeze + 呼吸周期 | Kaggle/官方镜像 | anchor(设备多样性) |
| **HF_Lung_V1** | 9765×15s @4kHz | Littmann 3200 + 8 麦阵列;**ICU/呼吸机** | 无;ICU 段=真实高噪声硬负例;含吸/呼相标 | GitLab(CC-BY-4.0) | support |
| **KAUH** | 112 人/336 文件 | Littmann 3200(蓝牙) | 无(干净参考分布) | Mendeley(CC-BY-4.0,⚠️浏览器下) | support |
| **RespiratoryDatabase@TR** | 77 人/12 通道肺 | 多通道数字听诊器 | 无;多通道→SNR 一致性特征 | Mendeley(CC-BY-4.0,⚠️浏览器下) | support |
| Coswara / COUGHVID | 万级呼吸/咳嗽 | 手机麦(**非听诊器**) | 无 | Zenodo/GitHub(CC) | ssl(仅自监督预训练) |

> **主锚 = SPRSound**(唯一现成肺音质量标 'Poor Quality')。**ICBHI** 给设备多样性(4 种设备,4–44.1kHz 混杂→显式重采样,防过拟合单一听诊器)。**HF_Lung ICU 段**作真实高噪声硬负例。其余作干净参考 / SSL 预训练。

## 3. 预训练音频模型（frozen embedder + 轻头,对标视觉的 DINOv2+LoRA）

策略与感知一致:**冻结一个强音频编码器当特征提取器,只训一个轻量质量头**。**研究路线下全部可直接用**(含 CC-BY-NC)。

| 模型 | 预训练内容 | license | 角色 |
|---|---|---|---|
| ⭐ **OPERA** (剑桥) | ~404h **呼吸音**(breathing/cough/lung), CoLA 对比 + MAE | CC-BY-NC-4.0 | **首选肺音编码器**:呼吸音专用 foundation model,与本任务最贴合 |
| **HeAR** (Google 健康声学) | 313M×2s 健康声(咳嗽/呼吸) | gated(HAI-DEF) | 强健康声 frozen embedder,肺/呼吸音备选;需接受条款 |
| **AuscultaBase** | 心+肺+肠 40k 录, 对比编码 | mixed/NC | 听过真实听诊器肺音;研究可用 |
| **WavLM** (微软) | 语音 + **去噪/重叠** | CC-BY-SA-3.0 | ⭐**质量代理首选先验**:预训练含噪声→embedding 已编码 SNR/干扰 |
| **BEATs** / PANNs / CLAP / YAMNet | AudioSet / 文本-音频 | MIT / Apache-2.0 | 通用 backbone(BEATs 最强);CLAP 可用文本探针("noisy/clipped")零样本打质量分 |

> 路线:**OPERA 当肺音主编码器**(呼吸专用),WavLM 当噪声感知先验;两者各取所长(OPERA 给呼吸语义,WavLM 给"干不干净")。BEATs/CLAP 作通用对照 + 零样本质量探针。

## 4. Q 模型设计（自监督代理 + 轻头 + 少量校准标）

延续架构 ③ 组件 A:**Q 主要靠自监督声学代理 + 演示隐式标签,只用极少人工校准**。

### 4.1 自监督质量代理(零/低人工,作弱标 teacher)—— 肺音
- **`Grooby` 多级 SQI**(心肺通用,取肺部分):功率比 + 自相关周期性 + SNR → 5 级质量。
- **呼吸相 / 气流检出**(HF_Lung 风格 + 呼吸周期检测):无吸/呼相、无气流 → "无呼吸信号"质量代理。
- **SNR / 削波 / 静音比 / 运动伪影**:基础声学统计(librosa/scipy),低 SNR、削波、长静音、突发噪声 → 差。
- **心音污染**(肺音里残留心跳能量):后背肺点也会串到心音,残留心跳能量大可作干扰惩罚(用心肺源分离的残差度量)。
- **通用模板**:`torchaudio SQUIM`(参考-free 估 STOI/PESQ/SI-SDR)、`NISQA` 多维质量(noisiness≈运动伪影,discontinuity≈听诊器掉接触)——**借架构,分数需在肺音上重校**(语音训练的)。

### 4.2 校准标(少量真标,对齐代理与"人耳判好坏")
- **SPRSound 'Poor Quality' vs 可用** → 直接训质量头(唯一现成肺音质量金标)。
- **演示隐式标**(自采轨道):操作者**停留久=好、快速离开/重录=差**。

### 4.3 架构(冻结 + 轻头)
```
后背肺音窗口(+ 力/呼吸相位上下文)
   → [冻结编码器]  OPERA(呼吸专用) / HeAR(健康声)  · WavLM(噪声感知先验)
   → [轻量质量头(线性/小 MLP)]  → Q∈[0,1] + 不确定性(MC-dropout/集成)
监督 = 自监督代理弱标(主) + SPRSound 'Poor Quality' 校准 + 演示隐式标
```
- 输入约定:统一重采样(各集 4k–44.1k 混杂),16kHz mono 起步;窗口对齐编码器(OPERA 8s / HeAR 2s / WavLM 任意)。
- **不确定性是一等输出**:Q 低**或**不确定性高 → 触发"需复核/主动探索",与感知翻转一致性置信门同构。

## 5. 自采同步采集规范（肺音如何与遥操作数据对齐）

采集后背肺听诊动作数据时,听诊器音频**和 RGB-D/力/本体/语言严格时间戳对齐**同步录。约定:

- **硬件**:末端电子听诊器 = **ThinkLabs One**(见 [`mechanical_sensor/`](../mechanical_sensor));音频走 3.5mm/USB 声卡进采集机,与力反馈(Hall 位移)、相机共用同一时钟。
- **格式**:WAV,**≥16kHz**(建议 16k mono 起步,设备支持则 44.1k 留档),16-bit。
- **每点 ~10s 窗口**:对应架构里"每点录音 ~10s"的速度瓶颈——Phase 3 用 World Model 想缓解。
- **元数据(每段音频随录)**:`task=lung`、后背肺点名(R1..L5)、目标力 & 实测力、呼吸相位(若可得)、操作者动作标签、是否重录。
- **目录 + manifest**:与感知 `manifest.jsonl` 同制式,一条音频一行:
  ```json
  {"id": "...", "wav": "audio/<sess>/<point>.wav", "task": "lung",
   "point": "R3", "fs": 16000, "dur_s": 10.2, "force_n": 4.1,
   "quality_label": null, "quality_proxy": {"sqi": 0.0, "snr_db": 0.0},
   "session": "...", "t_start_ms": 0}
  ```
  - `quality_label` 留给演示隐式标;`quality_proxy` 由 §4.1 自监督代理自动填。
- **采集即质检**:每段录完即时跑 §4.1 代理,差的当场标红重录(Q 的近期用途)。

> 这条轨道与感知的自采裸背图采集**并行**:同一次遥操作 session 同时产出对齐的图像(给感知)+ 动作轨迹(给 VLA)+ 后背肺音(给 Q)。

## 6. 许可证（研究/非商用路线）

项目走**研究/非商用**,故模型选择**不受商用限制**:OPERA(CC-BY-NC)、AuscultaBase、NISQA 权重、Grooby(GPL)等**均可直接使用**。唯一注意:**发表时**遵守各自的署名(CC-BY)/ 非商用(NC)/ copyleft(GPL/CC-BY-SA)条款。HeAR 仍需登录接受 HAI-DEF 条款(研究允许)。如将来转商用,再回头换可商用栈 + 自训头蒸馏(见 git 历史)。

## 7. 路线图

```
S0  收集开源肺音 + 模型   下 §2 肺音开放集 + 拉 OPERA/HeAR/WavLM 权重 ← 进行中(RunPod)
S1  自监督代理弱标         Grooby SQI / 呼吸相检出 / SNR → 弱质量标
S2  Q v0(冻结+轻头)        OPERA/HeAR 冻结 + 轻头, 弱标训 → SPRSound 'Poor Quality' 校准+评测
S3  自采同步肺音对齐        遥操作 session 同步录音(§5) → 部署域 in-domain 微调 + 演示隐式标
S4  接入 Phase 3 闭环       Q 作奖励: 声音差→邻域主动探索(架构 ③); 进阶声学 World Model
```

## 8. 代码落点

| 路径 | 内容 |
|---|---|
| [`scripts/audio/datasets_registry.yaml`](../scripts/audio/datasets_registry.yaml) | ★肺音数据集 + 模型 + 质量工具登记表(access/license/质量标/优先级) |
| [`scripts/audio/download_audio_datasets.sh`](../scripts/audio/download_audio_datasets.sh) | 开放肺音集下载脚本;Mendeley/认证集打印手动指引 |
| [`ausc/audio/`](../ausc/audio) | 声学质量核心包(质量代理 / Q 模型 / 数据加载,见其 README) |
| [`docs/model_architecture.md`](./model_architecture.md) ③ | Q 在整机里的接口(声音闭环奖励) |

> 一句话:**Q 回答"这段后背肺音质量够不够诊断"。先用开源肺音(SPRSound 质量标)+ 呼吸专用编码器 OPERA + 自监督 SQI 代理 bootstrap;采动作数据时同步录对齐肺音做部署域对齐;最终长成 Phase 3 声音闭环的奖励信号。只做肺音,不碰心音。**
