# 数据集与数据处理手册（给训练师）

本文件记录**我们用了哪些数据集**、**每个数据集怎么处理成可训练标签**，供训练同学理解与复现。配合 [model_architecture.md](./model_architecture.md)（模型结构）与 [engineering.md](./engineering.md)（工程细节）看。

> 任务：训练**感知模块**——从第三人称相机图像预测**稠密身体表面对应(DensePose-IUV)**，并**按任务**读出听诊点（心→前胸心点 / 肺→后背肺点）。点集是**任务条件化**的, 且因标签全由 SMPL 顶点合成, 点的数量/位置可自由定义。

---

## 一、数据集清单

| 数据集 | 位置 | 内容 | 角色 | 备注 |
|---|---|---|---|---|
| **HuMMan**（point 子集） | `data/humman/` | 907 序列：`smpl_params`(SMPL 参数) + `point_cameras`(10 Kinect+1 iPhone 的 K/R/T) + `iphone_color`(907 mp4) + `point_kinect_color_part_01`(453 序列×10 相机) | **感知预训练主力**（渲染稠密 IUV 标签） | 受控影棚、穿衣、RGB-D、有 SMPL 真值。iPhone 只拍正面；后背用 Kinect 后向相机。part_02(~454 序列)未下 |
| **SMPL 模型** | `data/smpl/SMPL_NEUTRAL.pkl` | SMPL 中性人体模型(6890 顶点) | 由 SMPL 参数算顶点 → 投影/渲染 | v1.1.0，已去 chumpy 转纯 numpy |
| **DensePose UV 映射** | `data/densepose_coco/uv_data/UV_Processed.mat` | SMPL 顶点 → DensePose (I,U,V) | 渲染稠密 IUV 标签的基础 | 7829 DP 顶点 / 13774 面 / 24 部位 |
| **脊柱侧弯裸背** | `data/scoliosis_back/` | 193 张真实**裸背**图 + COCO 背部分割 | **后背/肺 in-domain 微调 + 肤色/人种多样性** | 真实皮肤、含深肤色、躯干近景 |
| 音频（计划复用） | — | CirCor(PhysioNet 心音)、ICBHI(呼吸音)、HLS-CMDS(Littmann CORE 心肺) | 声音质量评估器预训练 | Stage3 音频闭环用 |
| 自采临床集（计划） | — | 机器人第三人称 RGB-D，裸/病号服，真实临床取景 | in-domain 微调 + 验证 | 最贴部署分布 |

> ❌ 已评估并废弃：**COCO/DensePose-COCO**（开放世界、稀疏手标→点位塌缩、与临床域差大）。

---

## 二、数据处理流程（HuMMan → 可训练标签）

完整流水线：`scripts/gen_dataset.py` → `ausc/datasets/`（输出 `manifest.jsonl + img/ + iuv/`）。各步骤：

### 1. 听诊点定义（`ausc/humman_label.py: define_ausc_vertices`）—— **任务条件化, 可自由扩展**
按**任务**定义点集, 每个点 = 一个固定 SMPL 顶点索引(渲染进 IUV/投影成 2D 点):
- **心音任务 / 前胸**（Tsumura 标准, 当前 4 点, 可扩到 6–8）：主动脉瓣 / 肺动脉瓣 / 三尖瓣 / 二尖瓣（+ 可加 Erb 点等）。
- **肺音任务 / 后背**（Lopes 标准, 当前 8 点）：L1-L4 / R1-R4（4 层双侧）。
> **点集是自由的设计参数**:所有监督标签都由 SMPL 顶点**渲染合成**(三数据集一致, 无人工标注), 想改点的数量/位置 → 在 `define_ausc_vertices` 换顶点 + 重跑生成器即可, 零标注成本。
- 定义为 **SMPL 固定顶点**（按解剖比例从模板选，经临床医生审阅微调）。标定：+x=患者左，-x=患者右，躯干核心 |x|<0.16。
- ⚠️ 体表近似（SMPL 无肋骨），区域级精度；精确落点靠 Stage3 音频闭环。

### 2. 稠密 IUV 标签渲染（`ausc/render_iuv.py`）
- SMPL 参数 → 6890 顶点 → 按 `UV_Processed.mat` 光栅化成逐像素 `(part I∈0..24, U, V)`。
- 本机 CPU 用 cv2 painter 算法（U,V 取每面均值，**粗**）；⚠️ **全量精确连续 UV 建议 GPU + pytorch3d**。

### 3. 前胸/后背判别（`ausc/humman_label.py: infer_side`）
- 稠密 IUV 躯干块 **part2=前胸 / part1=后背**，谁的像素多算谁。实测极悬殊（前 87209:770 / 后 544:45278）。
- 标注时也用 `frame_filter.classify_frame`（SMPL 躯干法向 vs 相机）路由 front→4 心 / back→8 肺。

### 4. 姿态/朝向过滤（`ausc/frame_filter.py`）
- 朝向：chest 法向相机系 z，|z|>0.6 判 front/back，否则侧身 reject。
- 弯腰：脊柱(neck-pelvis)在相机系若大幅指向相机轴(tilt>0.5) → 前缩/下蹲 reject。
- 作用：剔除侧转、弯腰、下蹲等动作帧（HuMMan 是动作数据集，必须过滤）。

### 5. 转正（内联于 `ausc/datasets/humman._base_sample`）
- 按 SMPL 脊柱方向(pelvis→neck)把图像/IUV/点旋转到躯干竖直，并裁到躯干 bbox。

### 6. 数据增强（算子库 `ausc/augment.py` + 编排 `ausc/datasets/common.augment_once`）—— 针对第三人称固定相机 + 实验室 WidowX/Trossen 桌面侧装臂

> 原则：增强必须匹配**部署分布**，盲目堆砌（如大角度旋转）反而有害。**训练时随机组合这 5 种**生成每张，不枚举。

| 手段 | 模拟部署变化 | 要点 |
|---|---|---|
| ① 光照/对比/色调/轻噪声 | 临床灯光、肤色、相机差异 | 抗光照 |
| ② **手臂+听诊器遮挡** | Trossen 臂+听诊器必然遮住躯干 | **最关键**；灰杆+听诊器圆头，标签保留 |
| ③ 水平翻转 + 换 L/R 标签 | 人体近似双侧对称→数据翻倍 | `LR_SWAP` + IUV 对称 |
| ④ **小角度旋转（严格 ±θ 对称）** | 病人轻微侧倾/相机轻微 roll | **仅 ±8/±15，左右成对**；固定机位不放大角度 |
| ⑤ **取景 F1远/F2中/F3近**（定死，非随机） | 第三人称相机的取景尺度（部署核心） | 见下「取景分类」；以听诊区居中 |
| (源头) betas 体型 / 肤色人种 / 姿态(坐/站) | 体型/人种/姿态多样 | synthetic 在渲染源头加 |

**❌ 已评估剔除**：大角度旋转(±30+，固定机位不会出现)、垂直翻转/倒置、枚举式重组合(改随机组合)、随机尺度抖动(改成定死的 F1/F2/F3 三档)。

#### 取景分类（framing taxonomy，部署相机看到的"哪一部分身体"）
感知相机在临床会经历三档取景尺度，**这是增强里最关键、已定死的一项**：

| 档 | 内容 | 躯干占比 | 部署对应 |
|---|---|---|---|
| **F1 远景** | 头+躯干+(坐姿椅) | ~45% | 病人就座/进入，宽视野 |
| **F2 中景** | 肩→腰，听诊区居中 | ~75% | 主听诊取景 |
| **F3 近景** | 胸/背填满+margin | ~92% | 机械臂作业近景 |

- **synthetic**：`ausc/datasets/synthetic.FRAMINGS` = 三档**相机距离**（真透视重渲染）。
- **HuMMan / scoliosis**（2D真实图）：`common.FRAMINGS_2D` = 按听诊点跨度三档**裁剪**。
- × {心任务=前胸点集 / 肺任务=后背点集} × {坐 / 站}。**每 base → 1规范 + ~n_aug 随机叠加(光照/遮挡/对称旋转/翻转[仅后背])**。

### 7. 负样本（`ausc/datasets/humman.py`，~25%）
| 类型 | 内容 |
|---|---|
| limb | 裁到四肢/头（有身体非躯干） |
| bg | 背景角落（无人） |
| side | 过滤器 reject 的侧身帧（回收） |
全部 `region=none, is_valid=false, points={}`。

### 8. 标签 Schema（`manifest.jsonl` 每行）
```json
{"id","image":"img/..jpg","iuv":"iuv/..png",
 "region":"front|back|none","is_valid":true/false,
 "side_conf":0.0-1.0, "points":{"name":{"xy":[x,y],"visible":true}}}
```
- `iuv` PNG：R=part(0..24)，G=U×255，B=V×255。
- 正样本：front(4 点)/back(8 点)。负样本：region=none，空点。

---

## 三、代码结构

**核心包 `ausc/`**
| 模块 | 作用 |
|---|---|
| `ausc/humman_label.py` | **听诊点定义（按任务: 心前胸/肺后背, SMPL顶点, 唯一权威, 可自由增删）** + SMPL投影 + `infer_side`前后判别 |
| `ausc/render_iuv.py` | 渲染稠密 IUV 标签 |
| `ausc/frame_filter.py` | 姿态/朝向过滤（front/back/reject + 弯腰检测） |
| `ausc/augment.py` | 5 种核心增强算子（裁剪/光照/翻转/手臂遮挡/旋转） |
| `ausc/humman_io.py` | HuMMan 视频 IO（`find_kinect_videos`） |
| `ausc/datasets/common.py` | 标签 schema + manifest 写入 + **5种增强的随机组合 `augment_once`** |
| `ausc/datasets/humman.py` | HuMMan → 稠密IUV + 前胸/后背点 + 负样本 |
| `ausc/datasets/scoliosis.py` | 裸背 → 后背肺点 + 稠密IUV（用重拟合 `scol_refit.pkl`） |
| `ausc/datasets/synthetic.py` | **pyrender 纯仿真**：平滑Phong + 连续程序化皮肤(多人种) + 坐/站 + F1/F2/F3取景 + 稠密IUV。⚠️ 需 GPU+EGL(`PYOPENGL_PLATFORM=egl`) |
| `ausc/models/`, `ausc/training/` | ☐ 感知模型 / 训练循环（占位, 见各自 README） |

**入口与工具 `scripts/`**
| 脚本 | 作用 |
|---|---|
| `scripts/gen_dataset.py` | **统一生成入口**：`--dataset {humman,scoliosis,synthetic,all} --out <dir>` |
| `scripts/gpu/label_scoliosis_hmr2_gpu.py` | 真实图打标：HMR2.0 单目SMPL拟合（GPU，已离线跑完存盘） |
| `scripts/gpu/download_scoliosis.py` | Roboflow 裸背集下载 |
| `scripts/qc/*` | 质检/展示（`qc_render/qc_back/spotcheck/sample_gallery/estimate_usable`） |

> 用法示例：`python scripts/gen_dataset.py --dataset all --out data/dataset`（三数据集各出子目录）。
> 冒烟测试：`--dataset humman --limit 3`。配置默认值见 `configs/data_gen.yaml`。
> ⚠️ **已废弃代码隔离在 `scripts/_deprecated/`**（COCO实验 + DensePose旧路线 + 旧点位定义ausc_points；旧统一流水线build_dataset已被 `datasets/`+`gen_dataset` 取代）。**勿用**，见该目录 README。

---

## 四、规模与体积（估算）

三数据集（可用原图 → ×~7处理 → 训练图）：
| 数据集 | 可用原图(前+后) | 训练图 |
|---|---|---|
| HuMMan | ~12,600(后背多视角~23k) | ~88k(~160k) |
| Scoliosis(仅后背) | 86 | ~600 |
| Synthetic(按需) | ~5,000 | ~35k |
| **合计** | **~17,700(~28k)** | **~15万(~23万)** |
- HuMMan原始帧(未去重)：前胸~66k/后背~86k(×后向相机~212k)；上表为去近重复后。
- +~25%负样本；384² 训练分辨率全量 ≈ 16–35 GB。

---

## 五、已知限制 / 训练师注意

1. **深度**：感知网络 **RGB-only**；深度仅推理时几何用（机器人 RGB-D 传感器测，2D→3D），不进网络。
2. **U,V 精度**：CPU 版 U,V 粗（每面均值）；全量用 **pytorch3d** 出精确连续 UV。
3. **域差**：HuMMan 穿衣/影棚 → 真实是裸皮/临床背景。靠 **scoliosis 裸背集 + 自采集 in-domain 微调** + 背景替换补。
4. **切分**：train/val/test 必须**按受试者**切，不能按帧（防泄漏）。
5. **平衡**：前胸/后背样本平衡；负样本 ~15-25%。
6. **真实图打标**：真实图（如 scoliosis）无 SMPL 参数 → 用 **HMR2.0 单目 SMPL 拟合**（`scripts/gpu/label_scoliosis_hmr2_gpu.py`，已离线跑完存 `labels_hmr2.pkl`），投影固定顶点得点位。HMR2.0 只给 verts2d 无稠密 IUV → scoliosis 为**点监督**（`iuv=null`），稠密 IUV 由 HuMMan+synthetic 提供。
