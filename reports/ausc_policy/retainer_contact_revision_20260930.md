# 听诊头接触分区修正：C 接触，B 凹入约 1 mm

用户于 2026-09-30 明确：设备外圈 C 本来就应接触背部，中央膜片 B 比 C 凹入约 1 mm。此前把 C 当作禁止接触外壳、要求它离皮肤至少 0.3 mm，是建模假设错误。旧失败记录保留，但不能继续据此阻止正确的 C 接触。

## 新几何与规则

| 部件 | 实体与尺寸 | 患者接触规则 |
|---|---|---|
| A | 打印夹具及打印唇边；外形不变 | 禁止接触，原 3 mm 规划间隙不变 |
| B | 中央膜片，正面比 C 后退 1 mm | 允许接触；独立记录受力，不要求它单独承力 |
| C | 设备接触外圈，正面仍比 A 凸出 5 mm | 允许正面接触背部；独立记录受力，计入总力 |
| 后部外壳 | DeviceEdge 后部、锥壳、旋钮 | 不作为接触面，保留原 0.3 mm 患者间隙 |

1 mm 是用户提供的近似值，模型中以 1.000 mm 显式实现，不宣称精密量测。原模型的 21/23 mm 内外半径沿用；外圈拆成 32 个独立凸扇区，中央开口保留。膜片与后部壳体的碰撞分区也同时调整，不只是外观着色。

相机使用已批准的 7° 配置。患者、打印夹具、TCP（C 前沿，零压缩时距腕部 132 mm）、碰撞 contact/rest offset 均保留。质量和惯量仍是原 CAD/密度估算，不是新的称重结果。

七通道记录 B、C、打印唇边、打印接触边、其他滑架、固定架及禁止碰撞。连续合格接触按 B+C 合力计量：原 3.5–4.5 N、至少 3 秒；总接触力 6 N、Hall 5.5 N、压缩行程 6 mm 的门不变。打印部件受力仍须严格为零。

C 接触还核对原始 PhysX 冲量和实际滑架位姿，拒绝侧面/后侧受力。接触点位置容差 0.1 mm 是数值核对容差，不是修改几何、穿透许可或接触间隙。

## 已完成证据

- 几何源码：`44a8847b0667b53ce7188f6f907adf98bc0aafe1`。
- Linux 原生 Isaac 静态检查：`retainer-5mm-recess1mm-review-20260930-v2`，owner 完成并释放 GPU。
- 4 张原生渲染图、USD 和两份导出碰撞文件已同步 Mac，逐份 SHA-256 核对通过。
- 新场景 SHA-256：`d4ef5726096055c5cdbbb409825d853d610ec865a7bd32e7135c39ec7653a571`。
- 新几何版本：`stethoscope_retainer_contact_5mm_recess1mm_v2`。
- 接触、规划、采集及旧协议兼容相关 339 项 CPU 测试通过；这些是软件证据，不是物理接触合格。

![Linux Isaac 模型实图](../../results/ausc_policy/retainer_geometry_review_v2/output/review/after_oblique.png)

## 第一轮实际接触：接触分区正确，力验收未通过

`retainer-r3-contact-20260930-v2` 在 Linux Isaac 实际执行了 31,439 个物理步（130.996 秒），完成接近、接触、3 秒 HOLD、撤离及回到起点。其运行源码为 `9d678042fa02e86c09a376d6428de234d5b1fd03`。进程正常结束不等于物理验收通过；原结果明确保留 `success=false`、`physical_validation=false`。

| 原始记录复核 | 结果 |
|---|---:|
| C 接触力峰值 | 5.6153 N |
| B 膜片及所有禁止接触通道 | 全程 0 N |
| 3.5–4.5 N 连续力窗 | 1.9375 秒，仅在 CONTACT 阶段 |
| HOLD 内合格力样本 | 0 / 720 |
| HOLD 平均实际 C 力 / Hall 力代理 | 5.5238 / 3.9211 N |
| HOLD 的 C−Hall 平均差 | 1.6027 N |
| 压缩峰值 | 2.4395 mm |
| 撤离稳定段末尾七通道合力 / 压缩 | 0 N / 0 mm |

按原始冲量独立重算七通道，与保存力字段最大差为 `8.88e-16 N`。HOLD 最后 1 秒 C 力标准差仅 0.0079 N，因此主要表现为稳定偏差，不能描述成单纯的瞬态抖动。C 前缘位置审计通过；打印夹具零接触。

![实际受力记录](../../results/ausc_policy/retainer_r3_contact_analysis_v2/contact.png)

[完整原生录像（130.9 秒）](../../results/ausc_policy/retainer_r3_contact_v2/output/review/actual_fresh_expert.mp4) · [独立力审计](../../results/ausc_policy/retainer_r3_contact_analysis_v2/summary.json) · [预紧诊断](../../results/ausc_policy/retainer_r3_contact_analysis_v2/preload_diagnosis.json)

实际保存 USD 的滑块为 force drive，刚度 1000 N/m、目标位移 −1.5 mm，已包含约 1.5 N 预紧；运行代码又施加 −1.5 N，执行后读回也确认了该值。Hall 公式只算一次 1.5 N。下一轮只移除重复的内置预紧，保留名义 1.5 N、全部控制律、力带和 3 秒门；因果结论仍需同路线实测验证。

另修复了六个继承释放/交接检查中漏算 C 的代码，提交 `e161f56fb1a734daf09d794e85f6783d38edaabb`，原 0.2 N 释放门不变。本次 v2 录像使用修复前代码；事后七通道重算证明本次确实卸载，不等于运行时已测试新版释放门。

原始 RGB-D、ADC/Hall、关节和物理记录的采集收据完整；完整性审计仍与物理合格及训练资格分开。本轮不授予训练资格，无声学或学习模型验收结论。

## 十点新规划与后续

`retainer-tenpoint-plan-20260930-v2` 完成十点检查，8 点找到受约束几何路径；L1、L3 当前搜索未找到路径，不能据此宣称绝对不可达。L4、L5 找到了路径，但第二关节相对既定软件安全上限的最小余量为 0.2°，相对 URDF 机械上限为 3.2°；尚未完成实际动态验证。

随后在不改变限值的前提下，将 L1 搜索起点增加至12个、预接近距离搜索为30/25 mm；L3 另外使用与当前拟合文件逐字节绑定的3个IK初值，仍重新执行全部姿态、碰撞与路径检查。`retainer-l1-search-20260930-v3`、`retainer-l3-search-20260930-v3` 均找到新路径，十点几何路径已齐。几何路径成功不等于十点实际接触成功。

## 单份预紧重测：R3 接触、保持与撤离通过

`retainer-r3-single-preload-20260930-v3` 使用源码 `313cbba57a7873452ad178185fd2594c26c456f2`，于19:23:14 UTC正常结束并释放GPU。31,435个实际物理步，130.979秒；完整录像130.9秒。运行内与原始冲量独立复核均通过，原结果为 `success=true`、`physical_validation=true`。

| 指标 | 重复预紧 v2 | 单份预紧 v3 |
|---|---:|---:|
| HOLD实际接触力均值 | 5.5238 N | 3.9870 N |
| HOLD实际力范围 | 5.2458–5.6153 N | 3.7038–4.0749 N |
| 3秒保持中的合格采样 | 0/720 | 720/720 |
| 实际力与Hall代理均差 | 1.6027 N | 0.0657 N |
| B及全部禁止部件受力 | 0 N | 0 N |

修复仅将USD弹簧目标位置−1.5 mm归零，保留显式−1.5 N预紧。七通道释放门修复也已进入v3，原力、3秒保持、姿态、行程、相机与几何要求没有放宽。连续实际合格力窗3.75秒，HOLD完整3秒；压缩峰值2.4394 mm；撤离末端全部接触力与压缩归零。原始冲量重算误差小于1e−14 N。

完整原始采集审计核对31,435个物理步、13,098个ADC包、2,619帧RGB-D和所有逐步事件，来源哈希及Hall回放一致，无错误。事件文件原schema没有录制收据哈希，这个限制在审计中保留；审计报告记录其当前SHA并核对跨流一致性。

![保持阶段的实际前后对照](../../results/ausc_policy/retainer_preload_comparison_v3/comparison.png)

[v3完整实际录像](../../results/ausc_policy/retainer_r3_single_preload_v3/output/review/actual_fresh_expert.mp4) · [独立接触和释放核对](../../results/ausc_policy/retainer_r3_single_preload_analysis_v3/summary.json) · [多模态记录审计](../../results/ausc_policy/retainer_r3_raw_audit_v3.json) · [前后数值](../../results/ausc_policy/retainer_preload_comparison_v3/comparison.json)

这只是同一场景R3解析专家的物理成功，`training_accepted`及训练资格仍为false。L4/L5实际单份预紧测试已结束，结果见下；十点动态、交接、随机场景专家资格及新训练仍未完成。旧平面膜片几何的路径、示范和交接测量不再用于训练。


## 最新 L4/L5 动态批次：L4 接触超时，L5 未执行

`retainer-l4l5-single-preload-20260930-v3` 使用源码 `5f11b3f`，于2026-09-30 19:45:23 UTC结束并释放GPU。批次处理正常结束不代表物理通过：原始 `success=false`、`physical_validation=false`。L4运行17,959物理步、74.829秒后触发 `contact_timeout`，进入保持0秒；L5尚未执行。C峰值1.8197 N，B与全部禁止部件受力为零。仿真在故障处停止，没有完成撤离，不能记作释放成功。

运行收据给出L4预接近距离40 mm、阶段预算28.571秒。原预检只检查必要的最短时间27.047秒；同一收据中的两档推进、加载与稳定名义时间已达33.333秒，且尚不含加速度及跟踪延迟。这表明现有预检不足以筛除时间安排不合理的方案，是下一轮优先修复方向；尚未通过独立完整原始轨迹复核确定全部原因。不是已有证据证明关节限位触发了本次停止。

已将终态收据、控制器诊断及74.8秒实际录像复制回Mac并逐SHA核验；完整原始传感数据镜像与独立审计仍待完成。来源：`results/ausc_policy/retainer_l4l5_single_preload_v3/selected_mirror_verification.json`。这次失败记录不进入训练集。
