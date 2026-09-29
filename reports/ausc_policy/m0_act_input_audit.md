# ACT D0 R4：训练到运行输入契约核查

**结论：在已核查的保存证据中，未发现明确的坐标、目标姿态、state19 编码或时间契约不一致。** 这不证明 ACT 闭环策略或执行器正确，也没有定位模型漂移的唯一成因。

范围是失败运行 `m0-act-d0-r4-20260929-assetfix`，以及 prepared manifest 绑定的原始 `R4_observed_armature02_v1`。本次未加载模型、未执行仿真、未改变控制器或物理阈值。仅读取远端四个指定小缓存，共 **351,992 bytes**；缓存只在内存中解析，没有拉取原始大图像或完整大 trace。

## 可直接验证的证据

| 项目 | 本次结果 | 含义 |
| --- | --- | --- |
| 运行输入重建 | **269/269** 条 state19 逐元素一致，最大差 **0.0** | 从该 query 的 frame、measured q、Hall packet、同帧 body transform 和登记 task 重走 `encode_task_state → small_state`，与实际发给 ACT 的 state19 对照。 |
| 原始训练输入重建 | **1515/1515** 行 state41 逐元素一致，最大差 **0.0** | 用原始 R4 frames/body/task registration 重建，逐行比较 SHA 绑定的 prepared `state.npy`。没有只检查两个最近邻或少数样本。 |
| 目标位置、完整姿态与 task-in-body | 训练与运行四项 transform 的最大逐元素差均为 **0** | `initial_task_position`、`initial_task_rotation`、初始 body transform、`task_pose_in_body_reference` 完全一致。没有发现目标旋转 transpose、反向法向或目标点替换。 |
| FK/TCP/Hall 依赖 | 批准计划、URDF、kinematics、tool frames、Hall 解码/标定/force calibration 实现的训练原始快照、运行快照和本地文件 SHA 一致 | 排除了这些已绑定文件的版本漂移；TCP 都是 Hall compression 修正的同一膜片定义。 |
| 阶段 | 运行与训练 L2 的单向 FREE→APPROACH 重建均无不一致 | 运行 query 为 FREE 138、APPROACH 131；原 R4 有效 H10 起点为 FREE 334、APPROACH 107，共441。 |
| query/frame 时钟 | 所有 query 与其 frame 的年龄为 **0 s**；Hall 年龄约 **4.1667 ms** | 没有将旧帧重盖时间；Hall 在既有 50 ms 门内。 |
| 执行标签时钟 | dispatch/label 相对块起点最大误差 **1.85e-13 s**；每条 completed action 恰好 **12 ticks** | 标签仍为 +50…+500 ms，前缀实际使用索引0…4；没有观察到反复执行 h0 或整 tick 错位。 |
| 训练时间绑定 | cache time 与原始 frame 完全一致；下一标签 frame 间隔约 **50 ms** | 两个已保存原 R4 探针（start71、464）的 state41、state19 和 reference 也分别逐元素匹配四个小缓存。 |
| 配置与标准化 | 运行 receipt 的 saved config、normalization、action contract、policy config 四个 SHA 全匹配 | 仓库配置文件与 saved configuration 仅 JSON 序列化字节不同，解析后字典完全相同，不能把这两个不同文件哈希说成参数漂移。 |

源观测中的 task pose 是“目标在 TCP 系”；`small_state` 显式求逆后得到“TCP 在目标系”，位置为 `-R.T @ p`，旋转保存前两列。训练缓存和运行均经过这条相同路径。269 条实际 query 的精确重建也覆盖 target ID、phase 与 confidence 编码。

原始训练首个登记样本在 0.050000 s，本次在 0.004167 s；采样起始时刻确有差别，但两者绑定的初始 body transform 与批准 task 完全相同。绝对 sim time 不是 state19 特征；不能仅凭这约45.8 ms的起始差认定输入存在时间错位。

## 末尾几何：工具轴角与完整姿态必须分开

最后可用 RGB-D/Hall frame 是 **1387，sim time 69.350000 s**；它早于碰撞 tick 69.354167 s约4.17 ms，因此下列量不冒充精确碰撞 tick 的 pose。

| 同一末帧重算指标 | 结果 |
| --- | ---: |
| 法向 gap | 9.982439 mm |
| 切向误差 | **26.564085 mm** |
| 工具轴夹角 | **8.633797°** |
| 完整 SO(3) 姿态误差（含 roll） | **19.285271°** |

因此“8.6°”只描述工具轴角，不能转述为完整姿态误差。二者差异不是编码错误，而是指标定义不同。

运行全程，登记 task 随 RGB-D body tracking 的位置变化最大 **0.046351 mm**、姿态变化最大 **0.012315°**；末帧分别为 0.040406 mm / 0.011350°。记录到的 task 漂移远小于上述 TCP 相对误差，不支持把26.5 mm/8.6°直接归因于这份保存的目标 frame 跳变。

按运行使用的 TRAIN 标准化区间重算，269×19个特征中仅1个关节特征略超[-1,1]，最大绝对归一化值为 **1.000253715**；实际未 clamp。这不是分布内或闭环可控的证明：多个维度各自在范围内，不代表联合状态被训练覆盖。

## 尚不能据此回答的成因

- 本次没有模型重推理，也没有比较相同输入下服务端预测与训练端预测。输入哈希/配置绑定是支持证据，不能替代该模型输出对照。
- 原始 command→physics 的标签映射继续依赖已批准 preparation/qualification 证据；本次核对了 frame 间50 ms与选定缓存块，没有重新扫描完整大 physics trace。`raw_physics_label_trace_revalidated=false` 已单列。
- 没有用最近邻距离或训练区间统计证明漂移原因。可证伪的下一步是把同一保存教师目标按原时间交给当前前缀/Ruckig执行器，并比较实际 command/measured 轨迹；以及固定同一 state19/receipt 比较直接 native 与服务响应。
- 需要区分提案误差、执行滞后及闭环累积偏离。本文仅排查输入契约，不把输入重建一致解释为提案动力学可行、接触成功或 M0 已通过。

## 来源与复现

完整字段检查、来源与证据 SHA、远端四个缓存的路径绑定/hash/大小、分组统计均保存在 [metrics.json](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/m0_act_input_audit/metrics.json)。原始 R4 的 outcome SHA 与 prepared `source_record` 一致；原始 frame/body/registration/manifest 又与原 qualification 中的 artifact SHA 一致。

唯一诊断脚本为 [analysis.py](/Users/williamqiu/Desktop/Harvard/WAM/results/ausc_policy/m0_act_input_audit/analysis.py)，每次复现只读已授权 `qwen-agent-l40s` 上的四个 pinned R4 小缓存，写入本目录这两份报告，不改源码/模型/场景。

```bash
/Users/williamqiu/opt/anaconda3/bin/python -B results/ausc_policy/m0_act_input_audit/analysis.py
```
