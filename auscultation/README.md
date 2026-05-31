# 单臂机器人自动化听诊（Robotic Auscultation）

用 6-DOF 协作臂实现**全自动化**听诊。区别于行业内常见的硬编码（hard-coded）方案，我们用 **VLA / World-Action Model** 训练机器人自主完成听诊。

> 参考文献见 [`reference/`](./reference)：Tsumura 2022（解剖图谱 + 弹簧力控）、Zhu/Hauser 2022（点云配准 + 声音质量贝叶斯优化闭环）、Lopes 2023（深度学习定点）。
>
> **工程流程详解见 [`docs/engineering.md`](./docs/engineering.md)**（每个 Phase 的细节 + 数据双轨道 + 自动标注方案）。

## 我们与现有工作的本质区别（这是项目的立身之本）

现有所有工作——包括最强的 Zhu/Hauser——都把听诊降维成「**(某种方式)找到一个 3D 点 → 把听诊器摆过去**」。那个"点"就是瓶颈：它用一个手工接口把感知和动作切开了。即便 Zhu 做了声音闭环，也仍然锚定在配准出来的先验点、在固定网格里搜、声音评估器是外挂黑盒、流程脚本化逐区域跑。

> ⚠️ 如果我们也走「找点 → 执行」，那不管点是人标的、网络预测的还是配准来的，**本质上就是 Lopes / Tsumura**。差异化必须在问题定义层面，而不是工程增量（"能坐着""更鲁棒"都不算）。

真实临床听诊不是"摸某个点"，而是医生**边听边动**：听到异常→沿声音探索、左右对比、配合呼吸、调力求清晰。我们把听诊重新定义为一个**以声音为实时输入、语言可指令、无需解剖配准的多模态感觉运动策略**。

| | 差异化 | 评价 |
|---|---|---|
| ❌ | 能坐着 / 更鲁棒 | 工程增量，不算贡献 |
| A | **无配准泛化**：像素+语言直接出动作，不需人体网格/解剖图谱/逐人手标点；一个模型通吃心肺、胸背、可指令（"听二尖瓣"/"对比左右下肺"） | foundation-model 路线 |
| B | **声音进策略闭环**：听到异常就主动探索、双侧对比——行为层面的新东西（注：与 Zhu 部分重叠，单独不够） | 需与 A/C 合并 |
| C | **World Model 主动感知**：学声音随位置/力/呼吸的动力学，规划"最大诊断信息"的下一步，超越标量贝叶斯优化 | World-Action Model 的正当用途 |
| D | **(激进) 声音自监督定点**：奖励=声音质量，系统从声音本身学出"该在哪听"，翻转"人找点、机器执行" | 最有论文相 |

**一句话：他们做「定位+执行」的专用管线；我们做一个「会自己边听边找」、语言可指令、无需配准的通用物理检查智能体。**

## 工作流程（4 Phase）

### Phase 0 — 硬件与标定（前置，必须先行）
- 3D 打印电子听诊器-机械臂**末端连接件**。
- 集成**力反馈**（参考 Tsumura 弹簧被动机构，解耦位置控制与接触力，保证安全）。
- 手眼标定、坐标系对齐、力信号标定、**工作空间可达性分析**（决定坐/躺、听胸/听背）。

### Phase 1 — 多模态数据采集（含标注）
- **主从臂遥操作**演示完整听诊动作，同步采集 **RGB-D + 力 + 音频 + 语言指令**，时间戳严格对齐。
- 听诊点位标注作为 **bootstrap 监督信号**，而非最终的硬接口（目标是逐步让定点变成声音驱动的涌现行为）。

### Phase 2 — 模仿学习策略（VLA）
- 训练**语言条件 VLA**，观测空间含**视觉 + 力 + 音频 + 本体感知**。
- 学的不是"到坐标"，而是"**逼近 → 接触 → 稳到目标力 → 听诊 → 按声音探索**"这套接触-力技能。
- 先在**人体模型/假人**上采集降低 IRB 压力，再迁移真人。

### Phase 3 — 声音闭环 / World Model（核心创新）
- 训练声音质量评估器；**用声音做闭环**：声音差就在邻域主动微调（差异化 B）。
- 进阶：学**多模态世界模型**预测声场，规划信息增益最大的下一步（差异化 C/D），缓解"每点录音 ~10s"的速度瓶颈。

### Phase 4 — 部署评测
- 指标：跨体型/坐姿成功率、定位误差、接触力误差、**声音质量**、安全（最大力/碰撞）。
- 基线对比：硬编码 / Lopes 几何定点 / Zhu 贝叶斯优化。

## 感知训练数据（三个数据集 + 数据处理，估算）

详见 [`docs/datasets_and_processing.md`](./docs/datasets_and_processing.md)；展示图见 [`data/_viz_humman/`](./data/_viz_humman)（`_SHOWCASE_humman/synthetic/scoliosis.png`）。

| 数据集 | 可用原图（过滤后，前胸+后背） | 每原图处理→训练图 | 训练图小计 |
|---|---|---|---|
| **HuMMan**（受控影棚+SMPL真值） | ~12,600（前~5.5k + 后~7.1k；后背多视角则~23k） | 1规范 + ~6 随机组合增强（`--n_aug 6`）| ~88,000（多视角后背~160k）|
| **Scoliosis**（真实裸背，仅后背，HMR2.0拟合标） | 86（干净竖幅；188中剔除102张X光拼图；标签已存 `data/scoliosis_back/labels_hmr2.pkl`）| 1规范 + ~14 随机组合增强（`--n_aug 14`，珍贵→榨满）| ~1,200+ |
| **Synthetic**（pyrender 纯仿真，体型/肤色人种/坐站可控）| 受试者数可调（如 400）× 站坐 × 前后 × F1/F2/F3 三档取景 | 每 base + ~3 随机增强 | ~19,000（400受试者时）|

- **可用原图合计 ≈ 17,700 张**（HuMMan后背用多视角则 ~28,000）。
- **可训练图合计 ≈ 15 万张**（增强倍数由 `--n_aug` 控 + ~25%负样本；HuMMan后背多视角则 ~23 万）。
- 增强 = **5 种核心**（光照 / 手臂+听诊器遮挡 / 水平翻转[仅后背] / 小角度对称旋转±8/±15 / 尺度裁剪）的**随机组合**，非固定枚举；旋转严格 ±θ 对称（Trossen 臂）。详见 [`docs/datasets_and_processing.md`](./docs/datasets_and_processing.md) §六。
- 主力预训练量来自 HuMMan + Synthetic，Scoliosis 作后背 in-domain 微调，**听诊点定义=前胸4(Tsumura)/后背8(Lopes)，唯一权威定义在 `ausc/humman_label.define_ausc_vertices`**。

## 代码结构

```
auscultation/
├── ausc/                      # 核心 Python 包(感知数据 + 模型)
│   ├── humman_label.py        # 听诊点定义(前4/后8, SMPL顶点, 唯一权威) + 投影 + infer_side
│   ├── render_iuv.py          # 稠密 IUV 标签渲染
│   ├── frame_filter.py        # 姿态/朝向过滤(front/back/reject + 弯腰检测)
│   ├── augment.py             # 5 种核心增强算子
│   ├── humman_io.py           # HuMMan 视频 IO
│   ├── datasets/              # ★ 三数据集批量生成器
│   │   ├── common.py          #   标签 schema / manifest / 5种增强随机组合
│   │   ├── humman.py          #   HuMMan → 稠密IUV+点(前胸/后背+负样本)
│   │   ├── scoliosis.py       #   裸背 → 后背8点(点监督, 用 labels_hmr2.pkl)
│   │   └── synthetic.py       #   pyrender纯仿真(平滑Phong+多人种肤色+坐/站+F1/F2/F3取景, 需GPU+EGL)
│   ├── models/                # ☐ 感知模型(占位, 见 models/README.md)
│   └── training/              # ☐ 训练循环(占位, 见 training/README.md)
├── scripts/
│   ├── gen_dataset.py         # ★ 统一生成入口: --dataset {humman,scoliosis,synthetic,all}
│   ├── qc/                    # 质检/展示(qc_render/qc_back/spotcheck/sample_gallery/estimate_usable)
│   ├── gpu/                   # GPU 脚本(label_scoliosis_hmr2_gpu / download_scoliosis)
│   └── _deprecated/           # 废弃探索代码(COCO + DensePose 旧路线), 勿用
├── configs/data_gen.yaml      # 生成配置
└── data/dataset/, outputs/    # 生成数据集 / 训练产物(不入库)
```

**批量生成**：`python scripts/gen_dataset.py --dataset all --out data/dataset`(三数据集各出子目录)。
真实图打标(scoliosis)的 HMR2.0 SMPL 拟合已离线完成并存入 `data/scoliosis_back/labels_hmr2.pkl`，无需再开 GPU。

## 现状

- [ ] 0. 硬件模块（连接件 + 力反馈）+ 标定
- [ ] 1. 多模态遥操作数据采集 + 听诊点标注
- [ ] 2. 语言条件 VLA 模仿学习
- [ ] 3. 声音闭环 / World Model 主动感知
- [ ] 4. 部署评测 vs 基线
