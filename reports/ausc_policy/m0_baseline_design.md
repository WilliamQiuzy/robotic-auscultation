# M0 状态基线设计审计

日期：2026-09-29。状态：**设计已形成，官方 ACT 纯状态路径已按源码确认，尚未实现、训练或闭环；M0 未通过。** 本次只读本地代码及小型 JSON/Markdown，并按后续授权读取官方 LeRobot GitHub 源码；未访问 SSH、GPU、Isaac、模型权重或大型数组，未安装依赖。唯一新增文件为本报告。

## 1. 建议与实际可确认范围

首选候选是 **LeRobot 官方 ACT 的无图像、无 VAE 小配置**：19 维学生状态、原始弧度绝对关节输出、10 步预测/5 步执行，只学习到预接触交接区的 L2 转移。L3 使用共同的既有 skill/executor。这样首先排除图像编码器、扩散采样和 VLA 适配因素，且不再复现 41 维、首步执行、全阶段端到端关节策略。

**可以：同一官方 commit 原生支持无图像 ACT，但须同时提供两个正确的状态键。** 指定的本地 `results/diagnostics/molmoact2_feasibility_v1/lerobot/` 是 MolmoAct2 局部源码快照，没有 ACT/DP。后续按根任务授权读取了其已记录的 LeRobot commit `eb7a54c6ff4983fc86075442bf4fa61503e7b5d1` 的官方文件：`ACTConfig.validate_features` 允许没有 images、但有 ENV feature。使用 `observation.state` 为 STATE7，`observation.environment_state` 为 ENV12；只给 STATE19 会被拒绝。[官方配置](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/configuration_act.py#L151-L153)、[feature 类型与识别](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/configs/policies.py#L120-L141)。

模型只在有图像时构建 backbone，环境状态有独立投影；无图像时用 ENV 取 batch size。**不要省略 STATE**：即使 `use_vae=false`，零 latent 仍访问 `batch[OBS_STATE].device`。这里的 ENV 只是 API 类型，内容必须是可部署跟踪估计，不是教师真值。无需假零图或自写 ACT。虽无视觉计算，文件顶层仍 import torchvision，因此仍需兼容的 torch/torchvision 和 einops 安装。[官方模型构造与 forward](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/modeling_act.py#L303-L433)。

“最便宜”是本设计的工程选择，尚无本机速度/显存比较结果。以下字段已与该 commit 的源码核对；配置实例化、运行依赖及 forward/backward 尚未实测。

## 2. M0 与 M1 的范围及拟用配置

方案 §9 明确要求 M0 训练小基线；§0 又要求下一次 GPU 训练前完成 M0/M1。合理执行顺序是：允许有限的 M0/M1 管线拟合诊断，禁止继续旧契约的全量 VLA 训练及其 Isaac 验收。M0 ACT 优先 CPU 运行，可完全避免文字上的 GPU 次序冲突；若实际需要 GPU，仅作为预先限定预算的小基线诊断，并由当前 GPU 所有者调度。它不是重新放行全量 MolmoAct2。当前 native TRAIN32 仅结束既有诊断并按新指标解释，不自动扩展为 full16 VLA。

| 项目 | M0：先打通最小闭环 | M1：完整契约比较 |
|---|---|---|
| 状态 | 19 维，无 RGB、q̇、Hall、力、真值 | 同一学生状态；可配置 ψ，仍不超过 20 维 |
| 动作 | 7 维原始绝对关节弧度 | 目标坐标 TCP 相对位置3 + 旋转6D + ψ1；块起点相对关节7；绝对关节7 |
| 标签时刻 | t+0.05 到 t+0.50 s | t+0.05 到 t+1.00 s |
| H / E | 10 / 5，0.5 s 块执行 0.25 s | 主配置 20 / 10；以后比较 16 / 8、30 / 15 |
| ACT | `n_obs_steps=1`, `chunk_size=10`, `n_action_steps=5`, `temporal_ensemble_coeff=null` | 仅相应修改 H/E 和输出宽度；不做时间集成 |
| 小模型 | `dim_model=256`, `n_heads=4`, `dim_feedforward=1024`, `n_encoder_layers=2`, `n_decoder_layers=1`, `use_vae=false`, `dropout=0.0` | 三臂使用相同容量和预算 |
| 归一化 | 外部一次 raw-unit quantile adapter；原生处理器对已归一化数值设 identity | 各动作表示重新拟合自己的 TRAIN 统计；无有界 logit |
| 起点 | 真实记录的 TRAIN 起点/恢复起点；经典准备段与学习段分开记 | 明确定义非奇异 staging；自由空间 IK 可行化 |
| L2 终点 | 能让既有 L3 实际接管的名义预接触状态 | 多身体实测 ≥95% 接管区域后，冻结其容差 |

以下是可供实现直接采用的构造入口，**本审计没有执行它**。参数属于本项目建议值；类型和字段依据官方 `configuration_act.py`、`configs/types.py`。

```python
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

cfg = ACTConfig(
    input_features={
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        "observation.environment_state": PolicyFeature(FeatureType.ENV, (12,)),
    },
    output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    normalization_mapping={
        "STATE": NormalizationMode.IDENTITY,
        "ENV": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    },
    n_obs_steps=1, chunk_size=10, n_action_steps=5,
    dim_model=256, n_heads=4, dim_feedforward=1024,
    n_encoder_layers=2, n_decoder_layers=1,
    use_vae=False, dropout=0.0, temporal_ensemble_coeff=None,
    vision_backbone="resnet18", pretrained_backbone_weights=None,
    optimizer_lr=1e-4, optimizer_weight_decay=1e-4,
    device="cpu", use_amp=False, push_to_hub=False,
)
policy = ACTPolicy(cfg)  # fresh initialization; no from_pretrained
```

无图像时仍保留 `vision_backbone="resnet18"`，因为配置校验检查该字符串，但模型不会实例化它；`pretrained_backbone_weights=None` 排除误触视觉权重下载。`device="cpu"` 必须在构造后重新断言，避免调用端不慎传空值让通用配置自动选择 GPU。[配置字段与校验](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/configuration_act.py#L76-L153)、[通用设备选择](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/configs/policies.py#L77-L87)。

训练初值：seed 61、float32、batch 32、AdamW(lr=1e-4, betas=(0.9,0.999), weight_decay=1e-4)、梯度范数上限 10、恒定学习率。先以固定 32 个 TRAIN L2 观测做 512 次更新的管线拟合；随后**从同一随机初始化重新开始**，对全部可用 TRAIN L2 窗口训练 2,000 次更新，均匀抽 episode，再均匀抽该 episode 的合格窗口。存 step 0/250/500/1000/2000；首次名义闭环只用预声明的 step 2000，不在试过若干闭环后挑最好的一次。固定 32 不额外成为正式拟合的 warm start。每个 checkpoint 记录实际 draw 数和 unique windows，不将 64,000 draws 称为“遍历了全部帧”。非有限数立即结束并报告；不因不及格而自动延长或调参。

固定 32 纯拟合时不加状态噪声；full-TRAIN 小基线时以 seed 61 的独立 RNG、概率 0.1 将**归一化后的七维 q token 一起置零**，其余状态保持，推理不置零。真实 q 仍单独保留供动作解码、执行器和安全检查使用。标签及块相对关节锚点不得使用被置零的 q。该概率是明确的建议初值，不是验证过的最优值。

注意频率约束：20 Hz 执行下，E=5 的 ACT 查询率为 4 Hz，E=10 为 2 Hz。方案 §3 的“策略 5–10 Hz 出块”和 §5.3 的“20 步执行一半”无法同时逐字满足。建议以 §5.3 的实际执行比例为准，保持 20 Hz 新观测/安全反馈和 240 Hz 内环；不能为凑 5–10 Hz 查询而每次丢弃尚未执行的前缀。

## 3. 数据适配必须从实际行语义出发

历史 full-data audit 确认的库存是 16 TRAIN episodes / 24,397 个 H10 窗口；另有 R5 validation 1,345 个窗口，L5 是已查看过的历史 test。TRAIN 是 original8 加五条 yaw derivative 加三条 R4 recovery-source；R5/L5 不属于这 16 条。历史所有阶段窗口数不能冒充本次 L2 的训练量。

原始派生代码 `sim/auscultation_pi05_v6/export_joint_dataset.py:162` 的 `eligible_pair`/`derive_joint_actions` 已将观测 t 配到 **following frame 的 q_command**：50 ms，物理行差 12，含两个端点共 13 行。因此：

```text
observation = state[s]
labels[h]   = raw_actions[s+h]  # 已经是 t+(h+1)*0.05 的指令
```

不能再做 `actions[s+1:s+H+1]`。`times[s+h]` 是观测时间，不是该 action 的未来执行时间。dataset row、原始 frame index、physics row 必须保留不同列。每个窗口验证同 episode、target、有效性、动作标签对应的 phase、无 recovery、连续 50 ms，且窗口最后一个标签也不能跨边界。现有 `training_data.full_chunks(..., horizon=H)` 支持参数化 H，但调用者仍须应用 cross-boundary label 排除，不能只看相邻 `times`。

M0 的 L2 不能直接使用全部 FREE/APPROACH：旧 FREE/APPROACH 分界是观察到的 90 mm 距离（`phase_contract.py:13`），APPROACH 会继续进入接触准备；新 L2 应止于预接触交接，后续由 L3 执行。先借已有原始 physics/subphase 记录定位真正的预接触端点，再冻结 L2 segment。历史 FREE 6,162 + APPROACH 2,523 = 8,685 个窗口只是**切 L2 前的上界**；按新 handoff 边界与 H 重建后必须重新计数。不能伪造 20/30 步的尾部、跨接触拼接或重复末帧来维持旧样本数。每条 TRAIN episode 无可用 L2 窗口时要显式列出，不能默默删掉。

M0 19 维输入按以下固定顺序构造：

1. `observation.state[7]`：`state41[:7]` 的实测 q。
2. `observation.environment_state[12]`：TCP 在**估计目标坐标系**的位置3、旋转前两列6、目标置信度1、目标 ID1、阶段 ID1。

现有 `task_observation.encode_task_state` 保存的后九维 `state41[32:41]` 是 **目标在 TCP 系** 的位姿，不能直接改标签名声称是 TCP 在目标系。重建 R、p 后显式求逆：`R_inv=R.T; p_inv=-R.T@p`，然后按列优先顺序保存旋转前两列。先验证正交性/旋转行列式，不能默默修补坏数据。原始 `observations.STATE_FEATURES` 中 confidence 是 index 22。离散目标按 `R1,L1,R2,L2,R3,L3,R4,L4,R5,L5` 固定整数 0…9，再映射到 [-1,1]；FREE/APPROACH 映射 -1/+1。这是一项明确的紧凑类别编码选择，不是连续解剖距离。ACT 没有语言提示，因此不重复放提示词。

该小状态使用现有 RGB-D 注册目标估计，并非从真值重算。它继承“首次注册后按相机身体变换运输”的限制；本次不能宣称已接入新的持续视觉跟踪。Hall 可继续服务底层传感/安全以及既有 TCP 定义，但不成为 L2 独立特征；q̇ 只给匀速评估基线与底层运动学检查。

每个适配样本保留只供追溯/评估的 `episode_id, source_row, original_frame, physics_row, target, phase, observation_time, action_times, uncorrupted_q, raw_labels, source_kind`。训练 batch 仅包含输入特征、`action[B,H,7]` 和官方需要的全 false `action_is_pad[B,H]`；推理 batch 绝不能带 action/labels。M0 不需要视频解码或第二套 LeRobot 视频数据转换。先用 lossless 小型 row adapter；以后再按方案统一到 LeRobot v2 存储。

### 归一化与损失

从**本次合格的 TRAIN L2 窗口**拟合各连续维 q01/q99，统计分母为每个合格 window×horizon 一次，不能跟随训练 draw 的 episode 均衡重加权；状态每个 window origin 一次。动作各关节跨 horizon 合并，不沿用旧 phase×horizon 的有界潜变量统计。M1 的 delta 动作必须减块起点同一个实测 q；TCP 臂按该观测的估计目标坐标系表达未来 commanded TCP，并固定该块参考系，不使用未来目标真值。

采用 `z=2*(x-low)/(high-low)-1`，反变换恰好一次，默认不 clamp。置信度 [0,1]、旋转6D 分量 [-1,1] 和离散 ID 用手设范围；q 与绝对 q 标签若 q99−q01 <0.002 rad，以分位数中点扩到 0.002 rad；相对位置若小于 0.005 m，同样扩到 0.005 m。记录这些建议下限，并在审计中标出哪些维触发。真正恒定且无识别作用的维在 M1 可删；不要悄悄删掉目标/阶段标识。M1 的 Δq、TCP、ψ 按各自单位生成统计，不能复制绝对 q 的统计。

训练前输出各维原始 min/max、q01/q99、扩展后的范围、投影到 [-1,1] 时的**潜在**裁剪比例及 256 桶占用。ACT 自身不需要离散 token；该直方图是与后续 VLA 对齐的信息分辨率诊断，不是 ACT 输入量化。真实执行的裁剪次数应为零；若官方 processor 内还有第二次 normalization/clamp，先修 adapter 配置。禁止用“切换 MEAN_STD”规避检查。

官方 `ACTPolicy.forward` 已确认：按 `action_is_pad` 对所有有效 horizon×action 求平均 L1；`use_vae=false` 时只返回该损失，无 KL、无 ODE。默认 `predict_action_chunk` 返回整块；`select_action` 原生只排入前 E 步并依次 pop。[官方 loss 与队列](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/modeling_act.py#L87-L150)。报告同时给 normalized loss 与反归一化的整块物理误差。确定性 ACT 的 20×16 重复推理检验确定性；Molmo/DP 则真正更换采样种子，不能用 ACT 的零采样方差代替成功证据。

最小适配可直接调用官方 policy：外部 adapter 一次归一化，训练使用 `policy(batch)`；推理使用 `policy.predict_action_chunk(batch)` 获取 [1,H,7]，一次反归一化并保留整个 raw chunk供审计，再由新执行队列取前 E 步。这样无需改官方模型或把其队列与物理队列混用。若使用官方 pre/post pipeline，ACT factory 委托给通用处理器，identity mapping 对 STATE/ENV/ACTION 均直接返回原张量，不再缩放；必须验证 reload 后仍 identity。[ACT processor](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/processor_act.py#L26-L46)、[identity 实现](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/processor/normalize_processor.py#L297-L299)。

## 4. 复用安全与执行器，避免把旧模型门槛变成依赖

推荐在新 `sim/ausc_policy/` 中写薄 controller 组合层，复用现有模块而不复制新一套 vN 目录：

| 保留的实际组件 | 用法与限制 |
|---|---|
| `pi05_v6/execution.py:5` `JointTrajectory` | 240 Hz 连续 Ruckig 状态；新目标不重置 q/v/a；每个 50 ms action 执行 12 个 physics ticks |
| `skill_v1/controller.py:115` collision/`_tick` 链 | 每个实际 command 和 observed q 的碰撞、姿态、连续差分门保持；名字里有旧版本不改变其物理用途 |
| `expert_layout_v1/controller.py:233` `_violations` | 保持 Hall <5.5 N、物理总接触 ≤6 N、当前软件 stroke ≤6 mm、禁止部件接触检查；不能把机械 7.25 mm 行程误当允许放宽的软件门 |
| `skill_v1/executor.py` `NormalExecutor` | 固定法向曲线、法向速度/加速度/jerk、有界撤离与 endpoint stop |
| `pi05_v6/observed_supervisor.py` | 复用 ADC/FK/跟踪观测驱动的阶段判断；PhysX 只供独立物理评估，不作 L2 输入 |
| `skill_v1/models/state_bc_v4_e2000/` | 本地存在既有 checkpoint + manifest，可作为固定 L3 候选；本次只确认文件/元数据存在，未加载或重新确认其闭环质量 |

固定两种 L2 比较时使用**同一个 L3 actor、同一 limits、同一手交接过程**。最便宜的 executor 初检用 skill_v1 解析 Hall 法向策略；将其标成解析 L3。正式 M0 学生名义复现可以接既有冻结 MLP，但需先在这次名义起点验证它能工作。若冻结 L3 失败，保留组件级定位结果，不能换解析 L3 后继续声称“冻结 MLP 系统已通过”。首选既有 state_bc_v4_e2000；不要因更高版本号自动换成尚未重新审查的 state_feedback9。

旧 `LearnedController` 不适合直接继承为新策略：它硬编码 41 维 observation binding、读 RGB、执行 `raw[0]`、禁止 normal actor fallback，并把 CONTACT/HOLD/RETRACT 也交给学习关节策略（`learned_controller.py:25–84`）。新 L2 只替换 `_move_to_prepose` 的提案来源；到交接区后由独立 L3 接管，且记录控制权。解析准备、最后回 home 和 L3 动作不得计作学习转移。

**handoff 是实际缺口。** `NormalExecutor.__init__` 要求最后三个 retained command 相等；`FixedNormalCurve.build` 从 retained q 建曲线，并逐点要求位置误差 ≤20 µm、角误差 ≤0.0002 rad。因此“距离目标 5 mm 以内”绝不等于能直接接管。到候选交接区后先用连续执行器停车、检查 measured speed/跟踪误差，再尝试构建从当前姿态出发的曲线。若需要一个有界 pose alignment 动作，要预声明、单独计入整形距离及耗时；不允许偷偷走专家全转移轨迹替学生完成任务。M0 可先复现名义预接触姿态，随后测小范围交接网格；在没有该测量前，§6.3 的“位置误差 <交接容差 1/3”只能标未可判定，不能替它填一个臆定容差。

自由段动作建议采用现有 gentle acquisition 的 joint limits：速度 0.175 rad/s、加速度 0.5 rad/s²、外层 jerk 8 rad/s³；内部 Ruckig jerk 可明确采用已存在的 7.99999 数值余量。L3 inward 0.002×0.7 m/s、outward 0.004 m/s，normal acceleration 0.02、jerk 0.2，保持当前限值。全局 speed_scale=0.7 必须与选定的 16 episode dynamics 元数据对齐，不能借此混合另一套 armature/solver 设置。记录 runtime actual limits，不复用旧 `execution_contract()` 中 `maximum_open_loop_s=.05` 和 `first_raw_action` 的过时描述。

无需复用旧 pi05 checkpoint selection admission、服务 schema、425 文件 hash 台账、版本改名证明或 frozen-runtime 全目录。新运行只记录代码 commit、一个配置 hash、官方 ACT revision、数据 shard/episode 清单及其现有身份、归一化文件 hash、checkpoint hash、seed、结果与失败原因；物理安全门和数据血缘仍保留。不能篡改旧文件来让旧 validator 接受新模型，也不能把旧失败重命名成通过。

## 5. 时钟、前缀与 split 边界

`vla_v2/chunks.py` 的旧 `ChunkGate` 将 `max_prefix_s=.1`、`max_observation_age_s=.15` 固化在检查逻辑中；它会拒绝 E=5 的 0.25 s 前缀，更不支持 E=10 的 0.5 s。新 gate 应分开管理：

- **观测时钟**：推理前新观测必须满足原有数据年龄/Hall 时限；整个推理期间冻结 simulation ticks，另记真实 wall inference time，不把暂停时长加进 simulation observation age。
- **计划时钟**：块内 label/action index 0…E−1 在 t+0.05…t+E×0.05 执行，不能反复发 h0。前缀期限按 E×0.05 明确配置，不把旧观测重新盖新时间戳。
- **安全反馈时钟**：整个前缀继续取 20 Hz 跟踪/状态和 240 Hz 物理/Hall；任何目标 revision、阶段/控制权变化、丢目标、stale feedback 或禁止接触立即清队列并触发规定的 sim 故障处置。原观测年龄的 plan 语义与当前安全反馈年龄不同，不能仅把全部 stale 阈值改成 0.5 s。
- **队列边界**：episode reset、target ID/revision 改变、L2→L3 handoff、L3→L2 新转移均 reset ACT 队列与本地 prefix。无 temporal ensemble；不容许上一点残余动作流入下一点。

TRAIN16 及其恢复祖先保持同组。R5、旧 L5 都是 development-known；不重新随机切 overlapping windows，不将它们加入统计拟合或伪称新的 final heldout。M0 训练场景复现仅回答“管线是否能闭环”；同一身体上的不同点不提供 unseen-body 泛化证据。M1 三动作臂保持相同可比较原始窗口起点，若 H 不同先明示交集/删减数量。未来独立身体/姿态/纹理 heldout 在数据生成前冻结；M2 oracle/DAgger 不得吸收这些 heldout 的失败状态作训练。

## 6. 最小名义闭环与 M0 退出证据

先在同一 nominal scene 用解析 L2 + 共同 L3 跑一个 R4 组件检查，确认 simulator、关节/Hall/碰撞与接管路径工作，再用固定 ACT checkpoint 和同样起点/目标替换 L2。接着覆盖 TRAIN16 的每个源起点；至少各一次并完整记录失败，建议 nominal 原始八点各三次便于发现运行噪声。R5/L5 可以以后作为已知回归，不能用它们的成功弥补 TRAIN 点失败。连续十点属于后续序列评估，不能作为这次单身体 M0 的泛化门。

M0 退出要求：预声明的 nominal TRAIN 场景套件中，学习 L2 能到实际可接管状态；共同 L3 完成工程力带内连续 3 s hold 和 verified withdrawal；未发生禁止接触/力/行程/命令动力学越界，且没有专家 L2 接管替学生完成。报告每个起点/目标成功数、总数及置信区间；有限重复只用于 nominal reproducibility，不声称总体成功率 ≥90%。若一部分失败，报告部分复现而非 M0 通过。大规模 ≥100 heldout transfer/checkpoint 和 200–300 种子化验收留在方案后续阶段。

最低记录集合，无需每次生成巨大审计包：

| 层 | 必须留下的数值 |
|---|---|
| 每次 ACT 查询 | episode/target/phase/revision，sim observation time、wall start/end，输入19及其原始来源，H 个 raw physical proposals，实际执行 E、打断原因，checkpoint/config/stats 身份 |
| 每个 20 Hz waypoint | 原始提案、整形后目标、实际送入 Ruckig 的目标、measured q/TCP、command q、观测时间、shape/reject 类型及幅度；IK 臂加残差、ψ、投影距离 |
| 240 Hz 独立 trace | command/observed q、原始差分速度/加速度/jerk、最小碰撞间隙、Hall/stroke、各部件 PhysX force；既有 trace 可复用，不重复写第二份 |
| 控制权/结果 | L2 起止、handoff 前后 q/v/a、曲线建成/失败、L3 actor、continuous hold、force band fraction/peak、加载滑移、verified release、timeout/drift/failure 类别 |
| 汇总 | L2 到达率和预接触 mm/°误差、proposal 整形/拒绝率、执行前缀分布、L2/L3 各自耗时、每点及套件成功数；hold/constant-velocity 两基线的 full-block 与 h0 误差 |

历史 `FRONT_GROUPS=(diaphragm, lip, contact_rim)` 合计可满足旧 front-contact 审计；它不能证明膜片实际承力。单独记录 diaphragm force 是否非零，维持 S2 “膜片承接接触”的未验收状态，不能把旧 rim-contact 条件包装成传感器孪生通过。原始 measured dynamics 与整形 command dynamics 分开报告。

## 7. 还缺什么；只需做哪些验证

1. **官方 ACT 本地依赖/运行**：官方 state-only、字段、L1、identity、queue 已在线读源码确认；本地仍缺完整包。准备同 commit 的标准依赖后，以合成 19 维 observation 做 CPU shape/forward/backward/label-free inference，一并确认无 backbone、无权重下载、无 GPU 自动选择和正确 save/reload；这尚未运行。
2. **16 episode 原始小状态/动作重建入口**：历史 receipts 和远端 provenance 不是本地 arrays 已可读的证明。本次未加载或验证 source arrays；必须由已有 source manifest 解析、streaming 读取，重建 L2 边界与 H=10/20/30 合格计数。无需先搬所有 RGB。
3. **动作表示缺口**：M0 absolute q 可从原动作直接获得。M1 的目标系 TCP 要统一未来 commanded FK 的 tool/compression 定义；ψ 要实现并验证同专家低肘分支一致的冗余角提取/IK 约束，现有 41 维数据没有已审计 ψ 列。不可把任意某个关节角当 ψ。现有 `pose_ik` 能用作起点，尚不等于新目标系主臂已实现。
4. **仿真资产/依赖**：本地 `candidate_b_plans`、camera JSON、URDF、env.py 以及既有 skill checkpoints 路径存在；这不验证运行机器的 USD/meshes/Isaac/PhysX、Ruckig 0.12.2、PyTorch、NumPy/SciPy、官方 LeRobot 环境可用。按既定机器分工执行；不要为 Mac 缺 Isaac 修同事的环境。本次未探测远端资源。
5. **小而有辨识力的适配测试**：t+50 ms 哨兵标签防重复 shift；旋转/平移求逆与 round trip；三动作臂 decode round trip；近常数/超 quantile 数据不隐式 clamp；label-free inference；L2/L3/target 边界不能拼 chunk；E 次消耗必须为 h0…h(E−1)；phase/revision 改变清队列；paused sim 与 wall clock 分离；q dropout 不改变真实动作锚点。
6. **executor 与 handoff 回归**：同一 Ruckig q/v/a 跨 waypoint/跨 query，前缀期间碰撞/force feedback 不中断，handoff 保证 stationary history；从几何附近但不能构曲线的反例必须明确失败。到实际 Isaac 后做解析对照、ACT train-scene 套件、独立 trace 复核与正常 shutdown 记录。

最短顺序是：准备已确认支持 state-only 的官方 ACT 依赖 → CPU adapter/归一化/时序小测 → 固定32整块拟合诊断 → TRAIN16 L2 有限拟合 → 解析对照/ACT 名义 Isaac 组件闭环 → 才判 M0。接着 M1 三动作臂与 1 s 块，再进入多样化数据；不再恢复旧契约全量训练。本报告不将任何尚未执行的一步标记为完成。

## 本次主要代码依据

- 采纳方案：`docs/superpowers/plans/2026-09-29-ausc-diverse-data-policy-redesign.md` §3、§5、§6、§9。
- 历史 lineage/timing：`results/diagnostics/molmoact2_full_data_audit_v1/REPORT.md` §1–3、§5；其“保留旧 normalization/旧 h0 selection/下一步 full16 native”建议已被新方案替代，本报告只复用历史事实。
- `sim/auscultation_pi05_v6/export_joint_dataset.py` `eligible_pair`, `derive_joint_actions`；`training_data.py` `full_chunks`；`task_observation.py` `encode_task_state`；`phase_contract.py` `precontact_phase`。
- `sim/auscultation_vla_v2/observations.py` `STATE_FEATURES`/`encode`；`chunks.py` `ChunkGate`。
- `sim/auscultation_pi05_v6/execution.py` `JointTrajectory`；`learned_controller.py` `_infer_step`；`observed_supervisor.py`；`dynamics_contract.py`。
- `sim/auscultation_skill_v1/controller.py`、`executor.py`、`curve.py`、`config.py`；`sim/auscultation_expert_layout_v1/controller.py`、`phase_manager.py`。
- `results/diagnostics/molmoact2_feasibility_v1/sources.json`（证明下载范围仅包含 Molmo 文件，不证明 ACT 支持）。
- 官方固定 revision 源码入口：上文链接的 `src/lerobot/policies/act/{configuration_act,modeling_act,processor_act}.py`、`src/lerobot/configs/{types,policies}.py`、`src/lerobot/processor/normalize_processor.py`；均为 `huggingface/lerobot@eb7a54c6ff4983fc86075442bf4fa61503e7b5d1`。未读取第三方实现。
