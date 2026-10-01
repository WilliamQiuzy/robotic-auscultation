# R2 修正与完整十点物理复测：v10

2026-10-01。**R2 已修正；名义人体上从 home 连续完成十点并返回 home，十点物理检查全部通过。** 退出时资源守护发生进程采样误判，因此原运行总状态仍为失败。下面分别报告完整物理证据和未正常收尾的运行状态，不把二者混为一个 PASS。

完整实际录像：625 秒，640×360、10 fps，未经拼接；批准的相机和人体摆位保持不变。

![完整十点实际 Isaac 录像](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/retainer_tenpoint_feedback_v10/actual_fresh_expert.mp4)

## R2 的问题及修正

旧 v8 在 Hall 换算力约 3.65 N 时开始计算三秒保持，但当时真正的法向接触力只有约 3.38 N，尚未达到 3.5 N 下限。Hall 表示沿传感器滑动轴的载荷，物理审计使用法向接触冲量的模长和；R2 的工具轴与接触法向有夹角，摩擦又贡献了轴向载荷。两者不能直接相减后作为固定零偏去校正。

修正将 **Hall 稳定计时范围收紧到 3.9–4.1 N，稳定后才计满三秒**；瞬态离开范围则重新计时。Hall 公式、标定和 4 N 目标保持不变。最终仍按原始法向接触力 **3.5–4.5 N 连续至少三秒**验收。接触真值只用于独立诊断，不输入控制策略。

本轮 R2 的全部 720 个 HOLD 物理样本：

| 测量 | 结果 |
|---|---:|
| Hall 平均换算力 | 3.982024 N |
| 实际法向接触力平均 / 最低 | 3.695715 / 3.619629 N |
| 法向接触力在传感器轴上的平均投影 | 3.533774 N |
| 摩擦力在传感器轴上的平均投影 | 0.422501 N |
| 两者合计轴向平均 | 3.956275 N |
| Hall 原始力平衡模型减合计轴向的平均残差 | 0.026084 N |
| 法向接触力连续达标时间，旧 v8 → 本轮 v10 | 2.883333 → **4.000000 s** |

这支持“力的定义、方向与摩擦造成主要差异”的诊断。剩余惯性及求解器残差未单独辨识，不能解释为真实 Hall 标定误差。当前较窄计时带在本名义场景有效，尚不能保证随机人体或任意接触姿态都合格。

![R2 力反馈与十点保持时间对照](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/retainer_tenpoint_feedback_v10/tenpoint_force_comparison.png)

图中曲线为 20 Hz 摘要；验收使用全部 240 Hz 样本。左下图是轴向投影，不应拿它直接代替右下图的法向力模长门槛。

## 完整十点结果

按 `home → L1 → R1 → L2 → R2 → L3 → R3 → L4 → R4 → L5 → R5 → home` 连续执行，中间 **0 次 reset**。

| 点 | 实际法向力连续达标时间 | HOLD 法向力最低值 | 本点区间观测 J2 最小软件余量 | 接触 / 释放 |
|---|---:|---:|---:|---|
| L1 | 4.7125 s | 3.8966 N | 7.0000° | 通过 / 通过 |
| R1 | 4.7333 s | 3.9003 N | 19.1599° | 通过 / 通过 |
| L2 | 4.7000 s | 3.8953 N | 19.1600° | 通过 / 通过 |
| R2 | **4.0000 s** | **3.6196 N** | 32.2647° | 通过 / 通过 |
| L3 | 4.5292 s | 3.8261 N | 17.2059° | 通过 / 通过 |
| R3 | 4.8917 s | 3.9925 N | 25.8225° | 通过 / 通过 |
| L4 | 4.7000 s | 3.9058 N | **9.0559°** | 通过 / 通过 |
| R4 | 4.6125 s | 3.8455 N | 9.0559° | 通过 / 通过 |
| L5 | 4.7208 s | 3.9020 N | **2.8396°** | 通过 / 通过 |
| R5 | 4.7208 s | 3.9023 N | 2.8396° | 通过 / 通过 |

每点 HOLD 均为 720 个样本，即三秒；上表“连续达标”还包括紧邻 HOLD 的达标片段，所以超过三秒。J2 余量按各点实际连续区间（含进入交接、接触、撤离）逐物理步统计，不与旧单点 home 路线的约 0.20° 值混用。L4/L5 的命令余量分别为 9.0553°、2.8389°。

本轮 C 前缘承担接触，凹入约 1 mm 的 B 膜片力为零。非允许接触部件各力通道均为零；原生 C 前缘位置审计无违规记录。独立审计重新计算原始冲量，未重新构建所有接触点的几何分类。法向接触力峰值 4.08057 N；计入摩擦的合力向量峰值 4.38646 N。保持原有 6 N 力上限、Hall 5.5 N、压缩 6 mm 及全部碰撞、释放和姿态要求。

两个最终回 home 段均完成，全部 240 Hz 命令通过原网格检查。末态卸力为零，关节最大绝对位置仅 1.4043e-7 rad。全部 150,003 个物理样本复算法向冲量，与原记录最大差 8.88e-16 N；无记录到的物理或姿态违规。

## 收尾告警：保留失败状态

原生 `fresh_expert_result.json` 和 `COMPLETE.json` 完全一致，均为 `success=true / physical_validation=true`，十点物理结果全部通过。但外层 owner 的 `completed=false`、`stage.passed=false`，原独立综合审计也保留 `passed=false`。

| UTC 时间 | 原始证据 |
|---|---|
| 22:59:12.625 | 完整物理 trace 写完 |
| 22:59:19.516 | 完整录像写完 |
| 22:59:25.312 / 25.661 | 原生结果与 COMPLETE 写完 |
| 22:59:27.387 | 最后一次采样：进程组只剩父进程，但 GPU 列表仍包含自己的 Isaac PID 1708266 |
| 22:59:27.443 | 原资源 guard 将该 PID 判成组外 GPU 进程，向父进程发送 SIGINT |

此前所有资源采样均将该 GPU PID 归入本次进程组，历史身份还记录了相同的 `start_ticks=48973294`。没有记录到另一个项目的 GPU PID；退出时进程列表与 GPU 列表不一致是当前证据支持的解释，具体驱动延迟或非原子采样原因尚未进一步区分。最终进程组已清空、GPU 已释放。

**没有修改 guard、覆盖原收据或放宽原综合审计。** 单独的 `sealed_physics_review.json` 核验完整物理证据早于告警，记录 `physical_evidence_passed=true`，同时明确 `owner_completed=false / original_aggregate_passed=false / training_eligible=false`。它证明物理序列完成，不代表任务干净退出；批量生成前仍需修复并回归验证这个退出判定问题。

## 运行成本、版本和证据

- 仿真时长 **625.0125 秒**；整轮墙钟 **6251.692896 秒（104.19 分钟）**，包含最后异常收尾。控制器阶段约 **9.793 秒墙钟 / 1 秒仿真**，包含规划、物理、安全检查和录像；不能视为 Isaac Lab 并行吞吐量。
- 新配置及代码已版本化。112 项相关 CPU 回归通过；v8 九个实际交接起点与 v9 失败起点共十个 CPU 全网格规划复核均通过，随后才运行本次完整物理序列。
- 保留失败历史：首次 R2 修正因诊断变量覆盖收据而中止，已用回归修复；v9 在前四点通过后，于 L3 运动前遇到搜索与复核共用 30 秒的预算问题。v10 将原搜索 30 秒和完整复核 30 秒分开，所有原检查与间隙阈值保留。v9 不能计为十点通过。
- 物理运行提交：`88f7a249a344c3ebceb4178fa761f57886dd14eb`；冻结目录 `/home/william/wam-vla-v2/policy-r2-feedback-20261001-v3` 的 1,529 个源文件再次逐 blob 核验。
- Run：`retainer-tenpoint-feedback-20261001-v10`。原始数据保留于 Linux `/home/william/wam-vla-v2/runs/ausc_policy/retainer-tenpoint-feedback-20261001-v10`；Mac 只导出并逐 SHA 校验八个摘要、图表、视频文件，共 6,174,621 字节，未同步约 2.96 GB 的原始 trace。
- Config SHA256：`176ad1502e87eb6a00ef6f5b5f035a469057451efba840c0967a57cf054de14a`。
- Geometry SHA256：`d4ef5726096055c5cdbbb409825d853d610ec865a7bd32e7135c39ec7653a571`；批准的十份计划哈希见媒体收据。
- Trace SHA256：`81d21c92d7e3745af83325d68871bdc1afd085949a70267740cd57c17ebbac54`。
- Video SHA256：`753e65b879db23bfbd0866c54a3d2c2e5735a49359d9704df66aceddf84c795d`。

可检查：[逐点原始冲量复核摘要](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/retainer_tenpoint_feedback_v10/summary.json)、[独立封存证据复核](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/retainer_tenpoint_feedback_v10/sealed_physics_review.json)、[媒体、运行状态和计划收据](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/retainer_tenpoint_feedback_v10/media_and_terminal_receipt.json)、[R2 定向诊断](/Users/williamqiu/Desktop/Harvard/WAM/reports/ausc_policy/r2_feedback_diagnosis_20261001.md)。

此次通过的是**单一名义人体的专家物理动作**，不是训练后的 VLA、跨人体泛化、声学或实机验收。下一阶段仍需退出流程修复、M2 身份×坐姿×座位的 CPU 扫描、可查询专家与 DART、小资格批及至少 100 条随机场景成功率 ≥95% 的门，再进入数据与学习模型验证。本次未启动训练；定时续接继续暂停。
