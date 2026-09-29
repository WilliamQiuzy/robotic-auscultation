# 听诊策略实验

以 `docs/superpowers/plans/2026-09-29-ausc-diverse-data-policy-redesign.md` 为当前方向。
M0 尚未通过。旧 41 维、只执行首步、单身体 VLA 全量训练已停止。

这里集中放新实验代码；变体由 `configs/ausc_policy/` 控制，历史目录只读。
Mac 用于开发、纯软件检查和查看网页。GPU 推理、训练与 Isaac 均在 Linux；
每次只有一个 GPU owner。没有真实机械臂入口。

## 入口

```sh
python -m sim.ausc_policy.diagnostics --config configs/ausc_policy/m0_diagnostics.json
python -m sim.ausc_policy.web --port 8774
python -m sim.ausc_policy.run_index --runs /home/william/wam-vla-v2/runs/ausc_policy
# 仅 Linux，由指定 Molmo Python 环境执行；每次唯一 run-id。
python -m sim.ausc_policy.run --config configs/ausc_policy/m0_seed_probe.json --run-id m0-seed512-20260929
```

GPU owner 复用既有进程组/资源保护器与共享 GPU 锁。每个 run 保存配置、
代码 commit、来源 shard、种子、进度、结果和进程退出证据；没有自动重试、
自动续训或隐含下游 GPU 作业。旧 checkpoint 的诊断使用原始适配器读取，
不改写历史结果。32 条拟合和 20×16 采样都只是训练样本诊断。
`run_index` 从已有收据生成一张 `summary.csv`；作业完成与 M0/闭环通过是不同列。

## 工程边界

`sim/` 第一方 Python、shell、必要配置以及批准的相机/座位 USDA 覆盖层进入 Git；
人体数据、模型、完整二进制 USD、实验数组与录像留在数据/结果目录，不进 Git。
`configs/ausc_policy/scene_assets.json` 绑定三个覆盖层及其两份直接二进制依赖的大小与 SHA256。
离线复现覆盖层须按清单将二进制资产放回对应相对路径；实际 Isaac 入口继续显式读取
批准的 `asset_root` 与已有完整 manifest，不改用缺少依赖的 Git 目录。
代码仓库不等于完整资产备份：
Linux 已有资产与结果保留，资产定位/哈希从历史 manifest 读取。
不得上传 SMPL/HuMMan 原始数据或绕过它们的使用许可。

网页实验博客在 `docs/experiment_blog/`。正文保留问题、方法、数值、局限与
下一步；视频只使用真实 Isaac 录像。服务只绑定回环地址并开放固定清单，
不会把项目根目录、凭证、患者资产或模型暴露成网站。

## M0 当前执行与解释

状态 ACT 使用官方 LeRobot（源码版本固定）的有限预算基线，19 维估计状态，
块相对关节动作，H10/E5、20 Hz。它是诊断用小基线，不是已完成的多模态 VLA。
固定 TRAIN16 checkpoint2000 的实际 CPU 重载已经通过；探针仍是训练样本。

单 R4 D0 入口为 `m0_act_d0.json` → `isaac_run` → `isaac_case`。
唯一 owner 先完成资产与 native FCL/containment 构造检查，再启动 CPU ACT 服务和
同进程组的 Isaac 子进程。ACT L2 不执行专家转移；到达候选入口后沿同一 Ruckig
真正停车，先检查保留指令位姿是否满足固定曲线的 20 µm / 0.0002 rad 完整位姿要求，
合格后才调用实测解析 L3。粗候选、停车、曲线入口合格、L3 开始与完成分别记录。
入口不合格时在构造曲线及接触运动前拒绝；没有局部对齐或专家动作替代。

- `completed`：进程、日志、退出和资源清理完成。
- `closed_loop_evaluated`：实际发起了策略闭环。
- `case_passed`：该名义单点的真实动作及独立审计通过。
- `m0_passed`：此单点入口始终为 false；里程碑不能凭单个作业完成判定。

首次 D0 在运动前因旧静态碰撞器的隐式资产目录失败，已保留原失败目录；
后续修复仅显式传入同一份已验证几何，未放宽安全或姿态约束。

`video --run RUN` 仅在真实执行/清理完成后，从已校验的实际相机 RGB 帧编码录像。
Linux CPU 编码，不重新渲染场景；按仿真时间正常速度播放，推理暂停不计入影片。
网页只展示有意义的结论和实际 Sim 片段。

### M0 reference velocity diagnostic

The first actual D0 ended before contact (269 queries / 69.354 s). Its terminal
zero-velocity Ruckig targets introduced lag even on correct teacher labels:
after each independent segment's first second, mean command-vs-target TCP error
was 6.503 mm, versus 0.558 mm with adjacent-label velocity. Neither CPU FK
result is a contact or collision check.

`m0_executor_probe_quadratic5.json` declares one fixed five-point quadratic
local derivative estimator at the original target times. `m0_act_d0_quadratic5.json`
uses the same checkpoint and physics with that hint. This is 50 ms reference
tracking, not a guarantee of exact waypoint arrival. Keep raw secant excess
flags and shaped-command error; do not silently clip, retime labels, reset
current q/v/a, or use an expert L2 fallback. Explicitly zero terminal velocity
on stopping, and discard hints when the query context changes. Run the native
teacher probe before the policy-driven Isaac variant. All existing physical,
collision, observation and derivative checks remain mandatory.

The quadratic hint's actual first command crossed a joint lower bound before
execution. `m0_motion_preview_probe.json` reproduced that native CPU failure:
factors 1/.5/.25 failed; .125 passed all 12 posture ticks, without changing the
live state. `m0_act_d0_preview.json` then ran actual Isaac and failed at 59.7 s,
still in FREE: tangential error 246.57 mm, and a proposed command violated the
unchanged 3 mm clearance. Only its first hint was scaled. This candidate is
not a successful fix, demonstration, or new training source.

`m0_act_path_probe.json` diagnoses every qualified original R4 TRAIN window and
reproduces the two saved online query sets on CPU with the same fixed weights.
It has no optimizer or simulator. Nearest TRAIN state distances are descriptive
support measures, never oracle labels or proof of generalization.

`m0_act_robot_feedback_probe.json` preserves the inferred task frame and changes
one measured joint by ±0.005/±0.02 rad, recomputing the relative TCP from the same
FK. It uses the same 16 R4 APPROACH anchors and checkpoint as the target-shift
probe, at most 480 deterministic CPU inferences and zero optimizer updates.
URDF-invalid inputs are counted and excluded without clipping. Output-versus-input
perturbation ratios are local proposal sensitivity, not executed recovery or
stability guarantees. No collision/posture feasibility or oracle label is implied.
