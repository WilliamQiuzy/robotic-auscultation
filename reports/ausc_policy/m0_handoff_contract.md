# M0：L2 → L3 名义交接契约与 capture-region 组件测量

本报告是 2026-09-29 的源码核查和实验设计；未运行 Isaac、加载模型或测得接管成功率。依据 [M0 baseline design §4–6](/Users/williamqiu/Desktop/Harvard/WAM/reports/ausc_policy/m0_baseline_design.md:108)。读取时仓库 HEAD 为 `93e9b2bcb9b6e9614b77028b4e0cfbacfffb780d`；并行中的 runtime 工作不在本文完成范围内。

**现有 L3 是沿固定法向曲线运行的控制器，不是任意邻近姿态的收敛控制器。** ACT 的关节弧度误差不能直接换算成可接管的 TCP 容差；现有代码也没有证明“距离目标 5 mm 内即可接管”。应分别测量直接交接和经过显式、有限局部对齐后的交接。M0 的最终成功仍要求共同 L3 实际完成连续 3 s HOLD、验证撤离以及全部原有安全门。

## 1. 已存在的交接条件，不可混为一个容差

| 检查层 | 源码实际要求 | 对 L2 终点的含义 |
| --- | --- | --- |
| 已批准 nominal prepose | 本次读取的 R1/L1…R5/L5 十份计划均为 `free_approach_start_gap_m=0.025`；自由间隙 3 mm、接触非接触件间隙 0.3 mm、关节边采样步长 0.015 rad。[R4 plan](/Users/williamqiu/Desktop/Harvard/WAM/sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_R4.json:112) | 名义目标是 25 mm prepose；同文件的 15 mm `free_approach_end_gap_m` 不是当前交接成功门。父控制器实际读取 start gap，并设置本点 phase gate。[controller](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/controller.py:462) |
| APPROACH → CONTACT | 当前点为 normal gap 25±2 mm、切向误差 ≤3 mm、工具轴角误差 ≤5°、实测 TCP speed <0.02 m/s。[配置](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/phase_manager.py:38)、[transition](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/phase_manager.py:234) | 只是阶段转换条件，既不保证构曲线成功，也不证明关节指令已经静止。 |
| 固定法向曲线入口 | 保留 `retained_q`，不先投影到理想起点；默认位置误差 ≤20 µm、完整 SO(3) 角误差 ≤0.0002 rad。之后整条插值曲线还须通过 FK、姿态和碰撞预检。[curve](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/curve.py:64) | 入口位置门实际限制相对法向线的切向误差，详见下文；完整旋转门包括工具轴绕自身的 roll，严于只看轴夹角的 phase gate。 |
| NormalExecutor 初始化 | 真实最后三条 command 为 3×7；最后一条与曲线保留端点逐元素相同，三条相邻差的最大值 ≤1e-10 rad。[executor](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/executor.py:36) | 需要真正执行停车过程及静止 command 历史。`history=None` 的通用默认会重复端点；真实 handoff 禁止利用该默认制造静止历史。skill 当前调用已传实际最后三行。[调用点](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:175) |
| 自由段执行 | 每条 command 与执行后的 measured q 均检查 mesh；`contact=False` 使用 3 mm、禁止 patient contact；contact 模式才使用 0.3 mm。[collision](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:115) | 局部对齐仍是自由段，不能提前进入 contact 模式借较宽的接触权限完成它。 |
| 真实物理安全 | 禁止部件各自 >0.5 N 拒绝，总接触力 >6 N 拒绝，实际 compression >6 mm 拒绝，理想 Hall ≥5.5 N 的独立物理审计仍保留。[physical gate](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/controller.py:233) | 不用 PhaseConfig 中较大的通用 7.25 mm 行程值覆盖既有 6 mm 物理门；也不以 actor 输入替代独立物理审计。 |

令 `T_c=kin.tcp(q_command, compression=0)`，批准的目标位置、法向为 `p*, n`。代码先计算 `g_max=(p_c-p*)·n`，再校验 `‖p_c-(p*+n·g_max)‖`。因此在单位法向下，**20 µm 是入口的切向距离门，不是以 25 mm prepose 为中心的 20 µm 球**。沿法向的偏差会改变曲线最大 gap，但仍受到曲线域、APPROACH gate、撤离端点和碰撞门限制。不能由此声称法向任意偏差可接管。[投影与保留端点](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/curve.py:68)、[插值校验](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/curve.py:85)

还须区分两个 TCP 和两个 task frame：构曲线使用 command q、零 compression FK、批准的静态 task；实测监督使用 measured q、Hall displacement FK、初始登记后由 RGB-D body transform 搬运的 task。后者的角度又只是工具轴夹角。二者是不同统计量，不能直接互换或把其中一个重命名为另一个。[TCP 定义](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/kinematics.py:153)、[observed metrics](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/observed_supervisor.py:88)

`force_derived_domain` 的 reserve 来自静态 Hall 模型；源码明确 `perturbation_envelope_certified=False`。它不是扰动捕获域，也没有证明动态力、伺服滞后或跟踪误差能被控制器吸收。[domain](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/curve.py:8)

## 2. 实际调用链与尚缺的接线

现有 skill controller 的父链为 `skill.ExpertController → curobo.ExpertController → NormalWithdrawalMixin / CandidateController → LowElbowController → layout.ExpertController`。这些父类提供静态世界检查、posture 包装、逐 tick 物理与碰撞门。现有 `_move_to_prepose` 只记录 plan 后调用完整专家转移，尚不是学习 L2。[skill parent/hook](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:4)、[cuRobo parent](/Users/williamqiu/Desktop/Harvard/WAM/sim/curobo_expert_v1/controller.py:20)、[candidate parent](/Users/williamqiu/Desktop/Harvard/WAM/sim/expert_quality_20260924/candidate_controller.py:14)、[posture wrapper](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/low_elbow_controller.py:14)

`ObservedContactSupervisorMixin` 在父 `_tick` 中关闭理想 Hall 的 phase 更新，随后采集真实 sensor/Hall，计算相邻 240 Hz 样本的 measured TCP speed，再更新 phase。其初始化用当前真实样本 `dt=0`，没有虚构 settle 时间；若缺少连续样本导致 speed 不可用，会明确拒绝。[速度与初始化](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/observed_supervisor.py:119)、[tick](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/observed_supervisor.py:229)

**单独加这个 mixin 不会自动把 L3 actor 输入改成实测值。** skill 默认 `desired_normal_velocity` 仍读 `state['hall_N']`，默认 `_skill_observation` 调用真值 `metrics`；它们和 phase supervisor 是不同接线。[默认动作与输入](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:81)、[真值 metrics](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/controller.py:227)

同样，plain skill loop 的 `released` 仍直接读理想 Hall/compression，并据此触发 executor endpoint stop。仅替换 actor features 仍不完整；需要明确使用 measured release 判据，或复用已含该修改的实测 loop。理想量可以留在独立物理审计，不能继续决定这种正常执行分支。[plain release](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:203)、[measured release hook](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/collect_demo.py:77)、[实际调用](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/collect_demo.py:215)

可复用的实测输入范例是 `ObservedTeacher.observe_actor/_skill_observation`，其中 Hall 为校准 ADC 的 load estimate，位姿为登记 task + measured q；其 analytic actor 又使用 `measured_gentle_v1`，并非原始 skill 默认 actor。`ObservedTeacher.__init__` 明确禁止注入 `action_provider`，所以它也不是直接接冻结 MLP 的现成容器。[teacher 限制及 analytic actor](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/collect_demo.py:47)、[实测特征](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/collect_demo.py:81)

冻结 MLP 的现有 `StudentController` 提供 metadata/action contract 校验、History 和预测记录，但继承的是普通 skill controller。新组合必须明确覆盖 actor observation 来源、实测 supervisor 初始化和 history 生命周期，不能让 MRO 偶然选择理想输入。使用 `state_bc_v4_e2000` 时，需在 nominal 控制组重新验证这次输入语义与 checkpoint 契约，而不是仅凭文件存在认定 L3 可用。[student](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/student.py:81)、[M0 固定 L3 选择](/Users/williamqiu/Desktop/Harvard/WAM/reports/ausc_policy/m0_baseline_design.md:121)

最小待补接口为：学习 `_move_to_prepose`；显式 handoff receipt；使用相同真实历史的停稳交接；共享实测 L3 observation/release hooks；明确的 owner/phase 切换及队列清除；组件 grid runner。当前 skill `run.worker` 可以注入 controller factory 并复用 certified collision、source binding、audit 与正常 shutdown，但其现成命令只选择 expert/student，未提供这些 ACT/grid 功能。[worker](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/run.py:103)、[factory](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/run.py:136)

## 3. L2 成功如何计分

必须保存以下三个不同结果，不能以几何到达率替代闭环成功率：

1. **L2 原始到达**：记录最后一个学习提案、实际停下的 command/measured q、task frame、normal/tangent/full-rotation error、servo lag 和时间。停止触发使用预声明的 observed 条件；不能事后选择最接近目标的一帧。
2. **可接管入口成立**：安全观测新鲜且目标/episode/revision 一致；自由段安全门全程通过；真实 command 停稳；observed supervisor 初始化有效；固定法向曲线及 NormalExecutor 按原阈值接受真实 retained command。记录原始到达与是否使用局部对齐。该条件仍不能单独叫 L2→L3 成功。
3. **实际 L2→L3 成功**：在同一真实状态连续执行共同 L3，完成 4±0.5 N 带内、切向 ≤2 mm、轴角 ≤5°、speed <0.005 m/s 的 HOLD；先满足原有 0.3 s settle，再连续 3 s，失效会清零。随后满足 release 确认、停在 retained prepose，并通过现有独立 HOLD/release 审计。当前 25 mm plan 对应 phase retract clearance 20 mm、load <0.2 N 连续 0.15 s；skill 还要求 endpoint final hold。读取实际 config 与 trace，不另写宽松成功函数。[HOLD](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/phase_manager.py:246)、[release](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/phase_manager.py:205)、[skill final gate](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:241)

原框架随后 `_return_home` 的成功可另报整体 episode 结果；home、解析准备、局部对齐与 L3 均不得计作学生学习转移距离。原始 ACT 到达失败、对齐辅助后成功，应报告为“ACT + 有界对齐”成功，不能写成 ACT 直接命中 L3 入口。

禁止以三条相同 command 推断机械臂已经静止。停车应沿正在使用的同一个 Ruckig q/v/a 状态，将目标速度/加速度归零并实际执行，直到产生真实静止 command 历史；同时记录 measured dq、TCP speed、tracking lag 和 Hall。既有 phase gate 的 0.02 m/s 不是新的停稳认证阈值。若增加 measured settle gate，其数值必须先从解析 nominal 控制组冻结，不能随 ACT 结果放宽。[连续 Ruckig](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/execution.py:5)

父 `_hold` 是真实 physics tick 的重复 command，可用于已经安全停下之后的观察；父 `_joint_segment` 从 rest-to-rest quintic 起步，不能直接替代正在运动的 Ruckig 停车。初始化新的 `JointTrajectory` 会将 current velocity/acceleration 清零，也不能用于伪造 handoff 连续性。[hold/segment](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/controller.py:317)、[Ruckig 初始化](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/execution.py:15)

## 4. 是否需要 bounded local alignment

目前只能回答：**工程上很可能需要，源码不能证明每个 ACT 终点都需要，也不能证明对齐一定可行。** 关节 mrad 误差与严格 full-rotation/切向入口门存在明显接口风险，但关节误差到 TCP 的映射受构型、误差方向和 servo lag 影响。先保存 ACT 真实终点，然后做下列成对组件测试。

| 分支 | 操作 | 成功归属 |
| --- | --- | --- |
| D0 直接交接 | 同一终点有界停车，保留实际 q，不额外纠正几何；按原门构曲线并运行 L3。 | ACT 直接入口；停车所造成的额外位移单独记录。 |
| D1 局部对齐交接 | 相同终点和状态条件下，用预先冻结预算的局部位置/姿态修正，经连续 Ruckig、全部自由段门后停稳，再运行相同 L3。 | ACT + 显式有限局部对齐。D0 失败仍保留在 D0 分母。 |

局部对齐只允许从当前 retained q 出发的小范围 IK/姿态修正，目标仍是同一个批准且登记的 prepose。对齐器的最大 TCP 平移、完整旋转、各关节总位移、路径长度、时间、重试次数必须在组件测量后、ACT 套件评价前冻结。超过预算、无解或门失败就结束该分支，不调用 home/staging、canonical `transfer_q` 尾段、cuRobo 全转移、专家插值路径或 loaded_q 来完成剩余 L2。loaded_q 仅保留其既有“声明 nominal 法向域”的用途。

这里还存在 task-frame 一致性检查：现有 curve 用静态 approved frame，实测输入用 registered/transported frame。对齐器若按后者求解，必须记录两者差值并以原 curve gate 实测结果判断；不能把 approved target 改为当前 TCP、把 tracked target 重置为真值或静默投影 retained q。移动身体时重建曲线属于后续设计，不能从本次静态 nominal 组件测试推导。

对齐期间 `_contact_active=False`，保持 3 mm/free collision 和原姿态约束；skill `_contact_cycle` 会在 preflight 前设置 contact flag，所以对齐须在调用它之前完成。handoff 后不得再运行学习前缀。沿用 `prefix` 的 episode/target/revision/phase/owner 失效语义；FREE→APPROACH 清队列后可按新 phase 重查询，而真正 safety fault 按现有故障处置停止，不能用专家补救掩盖失败。[contact flag](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:141)、[失败处置](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:253)、[prefix](/Users/williamqiu/Desktop/Harvard/WAM/sim/ausc_policy/prefix.py:1)

## 5. 最小可测 nominal capture-region 试验

以下是待采纳和冻结的组件试验提案，不是已运行的配置，也不是通过阈值。先用 R4 单个名义姿态定位接口；随后扩展到 TRAIN16 的每个源起点。接触 actor、传感条件、动力学、speed_scale=0.7、所有原有物理门在 D0/D1 和解析/ACT L2 对照之间保持一致。

**A. 中心点控制组。** 用已有批准路径将试验夹具带到名义 prepose，实际停稳、采集真实 sensor/history，再开始组件测试计时。专家准备只用于建立组件初态，明确排除在学习 L2 成功统计之外。先验证选定的解析 measured L3；正式使用冻结 MLP 时，再对该固定 MLP 单独验证中心点。若中心点构曲线或 L3 已失败，保留故障证据并先定位，不用 ACT 误差解释环境/接线问题。建议各 3 次 fresh episode 作运行稳定性检查，3/3 不能解释为已证明 ≥95% 成功率。

**B. 几何轴向筛查，最小 35 个节点。** 坐标基为同一 nominal task 的两个切向轴和法向轴；旋转使用声明坐标系的完整旋转向量。取中心，加下表的单轴正负偏移；固定同一 IK branch/seed 与 posture profile，并记录实际实现的 FK 扰动，不能只记录要求的偏移。

| 轴 | 待测偏移幅值 | 节点数 |
| --- | --- | --- |
| 两个切向平移轴 | ±0.01、±0.02、±0.1 mm，各轴单独扰动 | 12 |
| 法向平移轴 | ±0.5、±2 mm | 4 |
| 三个旋转轴 | ±0.0001、±0.0002、±0.001 rad，各轴单独扰动 | 18 |
| 中心 | 0 | 1 |

这些数值只是跨越现有入口门的探针，**不预设其可行或通过**。在 20 µm/0.0002 rad 边界，IK 数值误差必须进入实际误差列；不得调整目标掩盖它。先按原 FK/posture/collision/curve 检查筛查全部节点；IK 不可实现、几何失败、构曲线失败分别记录，不能从分母静默删除。理论门内也可能因整条曲线或碰撞预检失败。

**C. 实际动力学。** 对几何可准备的节点，以受检自由段命令真实到达并取得 history/sensor，分别运行 D0、预算冻结后的 D1。同一节点若 D0 不能构曲线，记录 direct failure 后仍可在单独 fresh episode 测 D1；不得沿失败物理状态继续偷偷修复。每条分支都需要实际 L3 HOLD 与撤离，curve preflight pass 只是一列。先按全局坐标冻结相同准备方式，再记录实际 attained command/measured state；两分支若初态差异超出预声明准备误差，不能叫同态成对比较。

**D. 真实 ACT 终点和动态边界。** 上述静止轴向截面不覆盖真实 ACT 的多轴联合误差、冗余关节构型、残余速度、跟踪滞后。必须额外把 ACT TRAIN16 实际终点映射到相同误差坐标，按预声明的中心/边界/联合方向补测，至少保留原始实际终点各一次。用真实连续停车产生入口，不把 dq 人工设零。少量组合角点与固定种子的联合扰动可验证轴向结果是否能组合；在此之前只能称为“测过的轴向截面”，不能把各轴最大成功值相乘成已验证 6D capture box。

输出每节点与每来源起点的尝试数、D0/D1 成功数、失败原因以及二项置信区间。组件网格用于确定名义预算与可测边界；有限名义重复不能代替方案后续多身体 ≥95% 接管域测量，更不能支持未见姿态或移动身体的泛化结论。

## 6. 运行依赖、最小记录及不可省略的审计

可组合的现有入口如下；本报告不实现新的 controller，也不复制新的 v 目录。

| 接口 | 复用内容 | 新调用方必须补齐 |
| --- | --- | --- |
| `skill.run.worker` / `run_sequence.main(controller_class=...)` | env/kin/checker 装配、selected plans、certified collision、episode/source capture、shutdown receipt。[入口](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/run.py:168) | 明确 controller factory 与 grid case 参数；加载的 checkpoint/limits/config 都要写 receipt。旧 CLI 未自动获得 ACT 或 grid 能力。 |
| skill `_move_to_prepose` | 单个目标和 plan 的交接位置。[hook](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_skill_v1/controller.py:71) | 用学习 L2 替换来源；按成功定义保存到达、停车、D0/D1 和 ownership。 |
| `ObservedContactSupervisorMixin` | 校准 Hall、登记 task、逐 tick 实测速度、phase 初始化和审计。 | sensor 实例及标定、body/task registration、measured L3 actor hooks；不能只混入 class 就认为都已完成。 |
| `FixedNormalCurve.build` + `NormalExecutor` | 原 curve/posture/collision 预检、固定法向有界执行、真实三 command 入口。 | approved frame、真实 retained q/history、plan loaded_q/domain、limits、collision callback；全部来自同一 case snapshot。 |
| `JointTrajectory` + `ActionPrefix` | 240 Hz 连续状态和 H10/E5/50 ms 目标提案调度。 | 单实例连续 q/v/a、每 tick 安全门、handoff 清队列；命令/观测时钟不得互相替代。 |

新观测年龄继续使用 0.15 s，Hall 0.05 s；feedback 0.05 s。它们来自既有 [task observation](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/task_observation.py:19) 和 [feedback contract](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_vla_v2/chunks.py:38)。E5 的计划年龄不会放宽它们。连续 sensor tick、Hall decode、base/task-frame 绑定和独立物理审计不能在停车或 handoff 时暂停后假盖新时间戳。

每 case 只需保留一个原始 240 Hz trace，并生成精简 receipt/汇总，至少包括：

- checkpoint/config/source/输入 plan hash、case/episode/target/revision、初态准备方式、actual sensor 条件和所有 runtime limits；D0/D1 与 L3 actor 的明确名称。
- 学习结束、停车开始/结束、对齐开始/结束、曲线预检、owner 切换、HOLD/release 的 sim time 和 physics row；同时保留 wall inference time。
- 入口的 command q/measured q/dq、真实最后三 command、Ruckig q/v/a、command 零 compression FK 和 measured Hall FK、static/registered task 差值；normal/tangent/full SO(3)/axis-angle error 分列。
- 对齐实际平移/旋转、各关节总位移、轨迹长、耗时、预算与是否用完；每条门的 pass/fail、原始明确失败原因；不记录一个模糊“没到目标”。
- 实际 continuous HOLD、release、原始有限差分动态峰值、最低间隙、Hall/load/compression、各接触部件 PhysX force 与禁止接触。

**front-group 力不是膜片力。** 现有 `FRONT_GROUPS` 包含 diaphragm、lip、contact_rim；observed supervisor 的独立 HOLD front 审计使用过滤后的组和，因此旧 rim/front-group 成功不能被转述为膜片受力成功。新报告必须单列三者；若需要证明 diaphragm loading，要从实际 per-component trace 得到证据。[groups](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_expert_layout_v1/controller.py:21)、[front audit](/Users/williamqiu/Desktop/Harvard/WAM/sim/auscultation_pi05_v6/observed_supervisor.py:247)

建议汇总故障分类：初态准备/IK 不可实现、静态 frame 不一致、自由碰撞/posture、输入 stale/target revision、停车/历史、curve FK/interpolation、curve collision/domain、L3 actor contract/输出、HOLD、release、物理 force/stroke、原始 command 动力学。若某 gate 失败，保留失败，不以扩大阈值或专家转移补完。

本轮只做源码读取、十份小型 plan 参数检查和本文写入；没有运行接管组件测试。后续完成条件仍是实际 runtime 接线、名义 capture 测量、固定 ACT TRAIN 场景闭环及独立 trace 复核；本报告不宣布 M0 完成。

### 本轮关键文件指纹

| 文件 | SHA-256 |
| --- | --- |
| `sim/auscultation_skill_v1/curve.py` | `1091117623a7cdaaa0bee5c638ff5457c859f8a2a3130ae2f482f6223f653161` |
| `sim/auscultation_skill_v1/executor.py` | `a92e3f6c5704a2c5704e59b054899de4577a30b3babb955aedc4bd41cf24d9b9` |
| `sim/auscultation_pi05_v6/observed_supervisor.py` | `c493be864050f479c21c112cf68e76aa377f8d101419416ded245df767371ed4` |
| `sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_R4.json` | `a5350dc5f5da21c5e476264da927636ea3c2dbf3c7a8a77d08898c4561a0e53f` |
