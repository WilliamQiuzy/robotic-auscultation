# M0 状态 ACT：512 步 TRAIN32 只读诊断

2026-09-29。诊断对象为 `m0-act-fit32-20260929`，代码提交 `e8d8ab1e84cd48529e42889c133672607f2a289c`，官方 LeRobot 提交 `eb7a54c6ff4983fc86075442bf4fa61503e7b5d1`。既有结果只读，以 NumPy 重算误差；没有加载模型权重、运行模型、训练、SSH、Isaac 或 GPU。文末另记录经授权实现的块相对对照及 CPU 验证；该新对照尚未运行。

**结论：这次有限拟合未达到基线比较要求，但已核证据没有指出明确的 ACT 适配器错误。** 模型已学习不同样本的绝对姿态和块内运动结构；512 步的主要残差是所有样本共享的关节偏置，且偏置方向随检查点变化。不能据此宣布数据错误、模型不具备拟合能力、M0 通过或闭环失败。M0 闭环尚未执行。

## 结果与指标口径

输入为 STATE7 + ENV12，不含图像、速度或 Hall；目标 ID 和阶段只在 ENV12 中出现，没有提示词。H=10、E=5、50 ms，标签为原始绝对关节弧度，无 VAE、无 dropout、无 bounded logit。TRAIN32 每步完整使用同一组 32 个窗口，只改变排列，512 步共 16,384 次窗口呈现。统计量由全部 8,685 个合格 TRAIN L2 窗口拟合，未使用 R5/L5。

下表单位均为 rad。“元素中位数”在 32×10×7 个绝对误差上取中位数；“样本 MAE 中位数”先对每个窗口的 10×7 元素求 MAE，再对 32 个 MAE 取中位数。两者均保留，不能互换。

| 方法/步数 | 整块 MAE | 元素中位数 | 样本 MAE 中位数 | h0 MAE |
|---|---:|---:|---:|---:|
| ACT 0 | 0.52770054 | 0.38243580 | 0.49127300 | 0.51943890 |
| ACT 128 | 0.02637675 | 0.02297994 | 0.02543152 | 0.02619520 |
| ACT 256 | 0.02630710 | 0.01908329 | 0.02658911 | 0.02616328 |
| ACT 512 | 0.03056228 | 0.01835795 | 0.02967418 | 0.03038162 |
| 原地保持 | 0.01279246 | 0.00472417 | 0.00950237 | 0.00399596 |
| 匀速外推 | 0.00253275 | 0.00151634 | 0.00212683 | 0.00195722 |

ACT 的元素中位数从 128 到 512 改善，但样本 MAE 中位数和整块 MAE 变差。512 的两种中位数都未超过两条基线。匀速外推使用实测 dq，ACT 按新规格没有 dq；这是要求保留的强诊断基线，并非完全相同输入下的策略对照。这里没有 TCP 位置误差、实测交接区容差或闭环成功率，不能把关节误差换算成尚未测量的交接成功结论。

证据：[各检查点结果](../../results/ausc_policy/m0_act_fit32/)、[完成记录](../../results/ausc_policy/m0_act_fit32/COMPLETE.json)。

## 管线核验

| 检查项 | 实际证据与判断 |
|---|---|
| 运行代码 | 诊断开始时本地 HEAD 与 `result.json.code_commit` 一致；适配器和基础配置相对该提交无差异。运行配置明确为 train32、512 步、batch 32、q dropout 0。后续块相对实现单列在文末，不归入本次绝对动作运行。 |
| 官方模型入口 | `make_native` 检查 vendor Git 提交、直接 ACT/config 源文件无改动、Python 环境和 ACTPolicy 实际导入路径；`native.json` 记录这些源码摘要。STATE7、ENV12、ACTION7 配置由官方 ACTPolicy 构造，不存在自行编写的 Transformer。 |
| 无图/无 VAE 协议 | 官方 ACT 支持 ENV token 分支；本配置同时提供 STATE 和 ENV。原生 CPU smoke 已产生 `[2,10,7]` 输出和 54 个有限梯度张量，记录 `images=false`、`vae=false`，未做 optimizer update。拟合使用相同源码摘要。 |
| 动作损失与推理 | 官方 `ACTPolicy.forward` 对动作块作 masked L1；本批 `action_is_pad` 全 false、无 VAE KL。`predict_action_chunk` 直接输出整块；没有使用 `select_action` 的 E 步队列，因此离线指标没有误把执行队列当 H=10 标签。 |
| 归一化 | 适配器在入模前各归一化一次，官方 normalization mapping 全为 IDENTITY，推理后反归一化一次。训练标签和评估参考来自同一 `row_from_episode`。未调用旧 Molmo processor，没有 clamp。 |
| train/eval 状态 | `evaluate` 调用 `eval()` 和 inference mode，结束后显式 `train()`；每次训练先 zero_grad、forward、backward、clip，再 optimizer.step。无 VAE和 dropout 的本次 full-batch 设置提供了直接数值交叉核验，见下一节。 |
| 输入坐标 | 旧 `task_observation.encode_task_state` 的末九维为“工具坐标中的目标位姿”，旋转前两列按列优先存储。`small_state` 使用 R 转置和 −Rᵀp 得到“目标坐标中的 TCP 位姿”，与新小状态定义一致；不是仅对平移取负。 |
| 时钟和标签 | 准备阶段对照原始 measured q、following q_command、时间和物理帧；动作源已经是 t+50 ms。`row_from_episode` 直接取 `actions[start:start+10]`，没有额外 shift，块覆盖 +50…+500 ms。 |
| L2 边界 | 准备记录报告 25,545 个标签配对核验、16 条原始 align→contact_cycle 静止交接证据；窗口过滤目标、FREE/APPROACH 阶段、连续采集帧、50 ms 时间和所有有效标签，不跨接触。此次本地诊断没有重新读取完整原始 trace，这部分证据来自哈希绑定的准备程序及其成功收据。 |
| 样本选择 | 以 seed+2=63 从每条 TRAIN episode 的 eligible starts 选两个起点，32 个 `(episode_id,start)` 均唯一、均在 prepared starts 内；本地重新生成的有序列表与 `probe_rows.json` 完全一致。四个检查点的 reference、measured_q、measured_dq 均逐元素一致。 |

源码：[适配器](../../sim/ausc_policy/act_baseline.py)、[原始相对位姿定义](../../sim/auscultation_pi05_v6/task_observation.py)、[官方 ACT 模型](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/modeling_act.py)、[官方 ACT 配置](https://github.com/huggingface/lerobot/blob/eb7a54c6ff4983fc86075442bf4fa61503e7b5d1/src/lerobot/policies/act/configuration_act.py)。收据：[native](../../results/ausc_policy/m0_act_fit32/native.json)、[CPU smoke](../../results/ausc_policy/m0_act_smoke/COMPLETE.json)、[准备记录](../../results/ausc_policy/m0_act_prepare/prepared.json)。

## 归一化、损失和更新的数值交叉核验

本地 `prepared.json` SHA-256 为 `f4d72efa2a1aec07c47e5f8c068f4406201204946fe109f7aea926f9836d889b`，与运行配置一致；`normalization.json` 为 `dbd5200d4713ecda6f1d292abb529d3c6a42f29b8165f672f5e46d8fef9557ab`，与 prepared binding 一致。标签经 scale→unscale 最大误差仅 `1.19e-7 rad`。

以保存的物理输出/参考重新计算 `mean(abs(scale(raw)-scale(reference)))`，与**下一训练步的更新前 loss**比较。两次计算的样本仅排列不同：

| 检查点 | 重算 normalized L1 | 下一步日志 loss | 结果 |
|---|---:|---:|---|
| 0 → step 1 | 0.6345071792602539 | 0.6345071792602539 | float32 完全一致 |
| 128 → step 129 | 0.03544946387410164 | 0.03544946387410164 | float32 完全一致 |
| 256 → step 257 | 0.034760408103466034 | 0.034760408103466034 | float32 完全一致 |
| 512 | 0.03446804732084274 | 无 step 513 | 不能作同权重比较 |

step 512 日志的 `0.0386596173` 是最后一次**更新前** loss，保存预测来自该更新之后；不能将两者差异误判成 train/eval 分歧。这些结果对“训练和评估使用不同标签、尺度或样本排列”提供了很强的反证，但不替代原始语义审计。

动作七维归一化宽度依次为 `[2.63822, 0.86145, 1.47683, 2.16571, 1.19634, 1.29651, 1.51495] rad`，没有动作维触发 span floor。状态的恒定 confidence 使用固定物理区间 [0,1]，旋转、目标和阶段使用固定区间，不把近常数 confidence 放大。统计文件明确 `actual_clamp=false`；quantile 外的值仍保留。

128→512 的 normalized L1 降低 2.77%，物理 MAE 却升高 15.87%。损失每维的权重为 `2/(high-low)`，物理 MAE 为等 rad 权重；误差从不同关节之间迁移可产生这种差异。512 的第 1、4 关节合计贡献物理绝对误差的 65.17%。这是透明的目标函数/报告单位区别，没有证据表明加载了错误统计。

512 行日志步号连续、loss 和梯度范数全部有限；每步记录的裁剪前梯度范数均大于阈值 10。最后 32 步 loss 为 0.017415–0.049919，均值 0.029885；梯度范数均值 28.181。全程最小更新前 loss 为 step 442 的 0.015892，该步未保存对应预测，不能把它当作已验证的物理最佳检查点。

代码包含真实 optimizer.step；输出从 step 0 到 128 的平均变化为 0.522865 rad，128→256 为 0.035113 rad，256→512 为 0.042386 rad，且损失最初大幅下降。这支持模型确实学习和更新，而非冻结模型/重复保存。此次未读取参数张量，未作逐参数 delta 证明。持续裁剪和后期波动是观测事实，尚不能据此唯一归因于学习率、某个权重矩阵或数据问题。

## 误差结构：主要是绝对基准偏置

令 `e[n,h,j]=prediction-reference`。对全部样本、horizon 求每个关节的均值 b[j]，用它分解输出误差。这只是在使用标签解释错误，**没有应用偏置校正，也不作为新模型成绩**。

| 关节 | 512 的 signed bias | MAE |
|---|---:|---:|
| 1 | −0.05462184 | 0.05462184 |
| 2 | −0.01620004 | 0.01620414 |
| 3 | −0.01693467 | 0.01697878 |
| 4 | +0.08480216 | 0.08480216 |
| 5 | −0.00930984 | 0.01303743 |
| 6 | +0.00276636 | 0.00807057 |
| 7 | −0.01911166 | 0.02022104 |

- 所有样本/步的第 1 关节误差为负，第 4 关节为正。共享 b[j] 解释 **90.76% 的误差 MSE**；`mean(abs(b))=0.02910665 rad`，仅为分解而去掉 b 后残差 MAE 为 0.00941649 rad。
- 进一步对每个样本、每个关节单独取块内平均偏置，解释 **99.53% 的误差 MSE**；去掉这些块常量后的残差 MAE 为 0.00178553 rad。这表明误差大多在块的绝对位置，而非块内形状。
- 将预测和参考分别减去自身 h0 后，整块 shape MAE 为 0.00298706 rad，h9 shape MAE 为 0.00528823 rad。此计算故意使用参考 h0，只用于诊断，不是可部署校正。
- 预测 h0 在 32 样本之间的逐关节相关系数为 0.99878–0.99993；32 个预测块没有完全重复。平均块端点变化幅度为预测 0.01745723 rad、参考 0.01754951 rad。不是所有样本输出同一块，也不是原地保持复制。
- 预测/参考的 h9−h0 变化向量 cosine 为 0.95611；参考端点变化绝对值大于 0.002 rad 的 137 个“样本×关节”中，方向一致率为 **97.81%**。模型已学到大部分块内运动方向，仍有形状误差。
- 各样本 MAE 为 0.025244–0.046852 rad，中位数 0.029674 rad；每个样本都受影响。L1 两个 episode 平均误差较高，约 0.041514 和 0.040554 rad，但并非少数坏行造成总体失败。
- 共享偏置随检查点明显漂移：例如第 7 关节 128/256/512 分别为 −0.051587、+0.063986、−0.019112 rad；第 4 关节为 −0.019574、−0.017489、+0.084802 rad。固定的反归一化常量错误难以单独解释这种变号，而更新中的输出偏置可与之相符。不能据此断言具体是 action head 的 bias 参数出错。

参考标签有一对完全相同的动作块：`R4_observed_armature02_v1:464` 与 `R4_train_B7300_approach_entry_v1_001:117`，所以为 32 个观测身份、31 个独特动作块。它不构成“同输入冲突标签”的证据；也没有给失败强行增加 32 个独立任务的含义。

## 仍缺什么，以及后续辨识顺序

现已核验 [32 个选定输入](../../results/ausc_policy/m0_act_fit_analysis/selected_inputs.npz)（17,400 bytes）与 [身份/阶段/时钟](../../results/ausc_policy/m0_act_fit_analysis/selected_rows.json)：

- state19、raw41、reference 全部有限；样本顺序、reference、measured q/dq 与原预测文件逐元素一致。从 raw41 和实际 target/phase 重新构造 state19 也逐元素一致。
- 归一化状态最大绝对值 1.291268；608 个元素仅 4 个在 [-1,1] 外，未截断，没有近常数维爆炸。实际 confidence 都为 0.9，归一化为 0.8。
- 22 个 FREE、10 个 APPROACH；目标数 R1/L1/R2/L2 各 4，R3/L3/L4 各 2、R4 为 10。没有 R5/L5；R4 较多来自既定每 episode 两个观测的抽样规则。
- 无完全重复输入。最近两个输入的归一化最大逐维差为 `8.20e-6`，正是上述两个相同标签的 R4 块，没有发现同输入冲突标签。

原始大 trace 的标签对齐由准备收据证明，本次未重复大数据扫描。参数级更新量、名义闭环、TCP 交接区容差仍没有本次新证据。

现有证据不支持把适配器更改称作修复已确认的错误。按已批准 [方案 §5、§9](../../docs/superpowers/plans/2026-09-29-ausc-diverse-data-policy-redesign.md)，**相同 32 个观测、初始化种子、官方 ACT、512 步预算和评估**的块相对关节对照能够针对这里观察到的绝对基准偏置：显式保留 measured q，再让模型学习未来运动，不必通过输出头重建大的当前绝对姿态。32 个标签的平均绝对幅度为 0.500432 rad，而相对块起点的运动平均幅度仅 0.01279246 rad。

该对照同时改变“固定起点锚定”和“动作统计尺度”，属于动作契约整体对照；即使改善，也不能唯一归因于其中一个因素。输出零残差天然等价于原地保持，所以仍须用原始绝对弧度指标证明超过 hold/CV，不能仅因绝对姿态偏置变小就宣布拟合或闭环通过。不延长预算、不改学习率或模型。

块相对标签必须是每个 k 的 `a_k − q_t`，整块共用同一个观测起点 q_t；不是逐相邻动作差分，也不是每步重新减 q。增量统计仅由原 TRAIN 合格窗口重算；推理恢复时加同一个 q_t 一次，保持 +50…+500 ms 标签语义。物理指标仍与本表比较，不能用减去真值偏置的诊断数字冒充成绩。

后续若仍失败，再沿输入、动作可辨识性、数据对齐和控制契约逐项核查；本次 512 步单独不能排除优化限制。M1 的目标相对 TCP 动作、IK 可行化及 staging 尚未因此完成。没有有限拟合和名义 Isaac 闭环证据前，仍保留 `m0_passed=false`，不启动 full VLA 训练。

## 经授权准备的块相对有限对照（未运行）

实现范围仅 [act_baseline.py](../../sim/ausc_policy/act_baseline.py)、[m0_act_relative_fit32.json](../../configs/ausc_policy/m0_act_relative_fit32.json)、[CPU 测试](../../tests/test_ausc_policy_act.py) 和本报告。基础 `m0_act.json` 未改动。

- 新表示仅为 `block_relative_joint_radians`，只允许 pinned prepared 来源的 train32 或 CPU native_smoke；拒绝相对动作的 prepare/2000-step train 入口。新配置仍 seed 61、同 TRAIN16、H10/E5、32 样本/512 updates、同模型和优化器。
- 原 prepared 仅提供哈希绑定的绝对动作缓存和合格窗口起点。相对分支遍历所有原 TRAIN 合格窗口重新计算相对动作统计，并要求重算状态统计与原 frozen 状态统计完全一致；不使用旧绝对动作范围冒充新范围，不改写源缓存/统计。
- `encode_actions`/`decode_actions` 使用独立的 acquired measured_q 锚点。输入 q dropout 不会改变标签锚点。训练按相对统计归一化，推理反归一化后加回同一个块起点 q 一次；原绝对分支仍是原数值路径。
- 输出保存实际 `normalization.json` 和 `action_contract.json`，明确动作表示、归一化文件及摘要、统计分母、原 prepared 只是绝对源缓存收据。relative `predictions.npz.raw` 仍为绝对弧度，与原 metrics 完全同口径；额外 `model_action_physical` 为加回锚点前的物理相对量。
- 新测试先观察失败，再实现通过。`python -B -m unittest discover -s tests -p 'test_ausc_policy*.py'`：最终共享工作区 **46/46 通过**（其中 ACT 17 项；其他任务期间新增了包内测试）。覆盖固定块起点、h0/h9、shape 错误、q dropout、全 TRAIN 相对统计重算、源统计不变、输出统计身份及旧边界拒绝测试。
- 对实际选定 32 个小输入，与原运行提交代码比较：绝对动作训练/推理 batch 逐元素不变；四个已有 checkpoint 经绝对解码逐元素不变。相对编码→解码的最大 float32 舍入误差 `4.66e-10 rad`（不是数学/位级完全无误差）。

现有统一入口仍为 `python -m sim.ausc_policy.run --config configs/ausc_policy/m0_act_relative_fit32.json --run-id <unique-owned-run-id>`；owner 再调用 `act_baseline --config PATH --output PATH`。由父任务检查后使用既有 native 环境、owner 和共享锁调度。相对分支尚未做真实全 TRAIN 统计重算、native smoke、512 拟合或闭环；本次只有纯 CPU fixture 和小数组回归，不将代码准备等同于实验完成。后续 runtime 必须读取本次输出 `action_contract.json` 绑定的归一化统计，不能误读 `prepared.json` 中的原绝对动作统计。
