# 单臂机器人自动化听诊（Robotic Auscultation）

用 6-DOF 协作臂实现**全自动化**听诊。区别于行业内常见的硬编码（hard-coded）方案，我们用 **VLA / World-Action Model** 训练机器人自主完成听诊。

> 参考文献见 [`reference/`](./reference)：Tsumura 2022（解剖图谱 + 弹簧力控）、Zhu/Hauser 2022（点云配准 + 声音质量贝叶斯优化闭环）、Lopes 2023（深度学习定点）。
>
> **文档索引**:[`docs/engineering.md`](./docs/engineering.md)(工程流程/每 Phase 细节)、[`docs/model_architecture.md`](./docs/model_architecture.md)(三模块结构 + Target Hand-off 接口)、[`docs/datasets_and_processing.md`](./docs/datasets_and_processing.md)(感知数据)、[`docs/training.md`](./docs/training.md)(训练流程 + 反馈方案:BC→chunking→HG-DAgger→audio-RL)。

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

详见 [`docs/datasets_and_processing.md`](./docs/datasets_and_processing.md)；增强总览图见 [`data/_viz_synthetic/`](./data/_viz_synthetic)（`AUG_humman_all / AUG_scoliosis_all / AUG_synthetic_all.png`，以及 scoliosis 临床复核图 `SCOL_doctor_check.png`）。

实测数(锁定管线)：

| 数据集 | 可用原图 | 每原图→训练图 | 训练图小计 |
|---|---|---|---|
| **HuMMan**（受控影棚+SMPL真值） | **592**（前 579 + 后 13；每序列取 1 最正帧。后背少因本机仅下 part_01 Kinect）| 1规范 + 6 随机组合增强 + ~25%负样本（`--n_aug 6`）| **≈ 4,292** |
| **Scoliosis**（真实裸背，仅后背，HMR2.0拟合标） | **86**（干净竖幅，标签 `data/scoliosis_back/` + 重拟合 `scol_refit.pkl`）| 1规范 + 12 随机组合增强（`--n_aug 12`，珍贵→榨满）| **1,118** |
| **Synthetic**（pyrender 纯仿真，体型/肤色人种/坐站可控）| **6,400**（400 受试者 × 站/坐 × 前/后 × F1–F4 四档取景）| 每 base + 3 随机增强（`--n_aug 3`）| **25,600** |

- **可用原图合计 ≈ 7,078 张**；**可训练图合计 ≈ 31,000 张**（增强倍数由 `--n_aug` 控）。
- 增强 = 一套算子的**随机组合**（几何：透视/±8°对称旋转 + 后背水平翻转换L/R；外观：临床光照/色相/高光/阴影；遮挡：机械臂从右下够向听诊点；退化：运动模糊/JPEG/噪声），取景 F1远→F4近为关键增强。详见 [`docs/datasets_and_processing.md`](./docs/datasets_and_processing.md) §六。
- 主力预训练量来自 HuMMan + Synthetic，Scoliosis 作后背 in-domain 微调。**听诊点按任务定义(心→前胸 / 肺→后背)，且因标签全由 SMPL 顶点合成、可自由增删点位**——当前 **心 5 区(APETM, Bates')/肺 8 点(Lopes)**;解剖定义在 `ausc/humman_label.define_ausc_vertices`，任务→点集路由在 `ausc/tasks.py`。

## 代码结构

```
auscultation/
├── ausc/                      # 核心 Python 包(感知数据 + 模型)
│   ├── humman_label.py        # 听诊点【解剖定义】(SMPL顶点, 唯一权威: 心前胸5/肺后背8) + 投影 + infer_side
│   ├── tasks.py               # Perception→Action 任务路由: task(heart/lung)→点集+听诊顺序(为将来 task 参数预留)
│   ├── render_iuv.py          # 稠密 IUV 标签渲染
│   ├── frame_filter.py        # 姿态/朝向过滤(front/back/reject + 弯腰检测)
│   ├── augment.py             # 增强算子库(几何/外观/遮挡-机械臂/退化)
│   ├── humman_io.py           # HuMMan 视频 IO
│   ├── datasets/              # ★ 三数据集批量生成器
│   │   ├── common.py          #   标签 schema / manifest / 增强随机组合编排(augment_once)
│   │   ├── humman.py          #   HuMMan → 稠密IUV+点(前胸/后背+负样本)
│   │   ├── scoliosis.py       #   裸背 → 后背肺点(稠密IUV, 用 scol_refit.pkl)
│   │   └── synthetic.py       #   pyrender纯仿真(BEDLAM真皮+坐/站+前后+F1–F4取景, 需GPU+EGL)
│   ├── models/                # ☐ 感知模型(占位, 见 models/README.md)
│   └── training/              # ☐ 训练循环(占位, 见 training/README.md)
├── scripts/
│   ├── gen_dataset.py         # ★ 统一生成入口: --dataset {humman,scoliosis,synthetic,all}
│   ├── qc/                    # 质检/展示: aug_showcase(X-OR增强总览) / scol_doctor_check(临床复核)
│   │                          #   / draw_architecture + build_architecture_pptx(结构图) / spotcheck / estimate_usable
│   ├── gpu/                   # GPU 脚本(label_scoliosis_hmr2_gpu / refit_scoliosis / download_scoliosis)
│   └── _deprecated/           # 废弃探索代码(COCO+DensePose旧路线 + 旧QC qc_render/qc_back/sample_gallery), 勿用
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
