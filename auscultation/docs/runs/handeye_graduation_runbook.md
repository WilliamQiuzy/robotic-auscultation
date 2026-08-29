# 手眼标定毕业会跑单(R4 → validated)

目标:总不确定界 3.8cm → ≤2.5cm(实测门槛,过线即 L5/R5 回归、10 点全出;
毕业后预期 ~2.2cm,底部掩膜带缩至 <8% 画面)。

## 为什么(2026-08-29 实测链)

- 相机装在臂底座正上方 ~18cm(深度图证实:底缘中央 13cm 实体);肩部 link1/2
  离视锥仅 0.8cm,当前 ±3.8cm 界投影成底部掩膜带,盖住 L5/R5。
- 界的构成:3σ × workspace_1σ(6.1mm) + 临时档政策余量(6.5mm) + 网格几何项(~13mm)。
- 毕业后:3σ × ~3mm + 0 + 13mm ≈ 2.2cm ≤ 2.5cm 门槛(逐档扫描见
  2026-08-29 会话记录:2.5cm 全清,1.2cm 带几乎消失)。

## 前置条件(到场前确认)

1. 刚性非人体 3D 夹具在位(点触标定专用)。
2. 相机自标定后未移动(相机↔底座刚性连接;若动过一切作废)。
3. 相机延迟标定(camera latency)在有效期(preflight 会查)。
4. 重力补偿程序可用(标定时驱动臂回 home;工作区清空、E-stop 在手)。
5. 8801 服务可停(标定独占相机)。

## 会议流程(预计 1-2 小时)

1. `handeye_orbbec_preflight.py` — 相机/延迟/身份预检。
2. 起重力补偿(**先清工作区、扶住臂、E-stop 就位**)。
3. `handeye_calibrate.py` — 操作员牵臂点触:
   - 10 个空间位置 × 2 种腕姿态 = 20 次确认点击;
   - 3 个独立留出位置;
   - 每次点击落盘 RGB/深度/关节/FK/时序证据。
4. 质量门(自动):training RMS ≤ 3.0mm、单点 ≤ 5.0mm、留出 ≤ 5.0mm。
   - 通过 → validated 候选;
   - 未通过 → 可加测或按 R4 流程用户接受(不毕业,今天的带子不变)。
5. `activate_handeye.py`(validated 档路径)→ 新 handeye release。
6. **激活包换装修订**(工具的 handeye-swap 模式,毕业日前建好):
   - activation.json 的 handeye 外部工件条目(path+sha);
   - `occlusion_identity` 的 handeye_policy_sha256 / calibration_tier /
     capability_schema 交叉钉(!!与 back_mask 预算修订同款陷阱,必须同改);
   - 真实 loader 消费者验证。
7. unit 重装(新激活)→ Disable → 重启 → bring up。
8. **10 点验收**:坐下数 26 秒(8s 静默 + 12×1.5s 连击),网页应出全部 10 点。

## 失败回退

任何一步失败:激活包不换,现役 activation-416a3f1f(500ms/2M/R4)继续有效,
8 点工作流不受影响。

## 已知坑(2026-08 战役全集)

- 重启 8801 前必扫孤儿控制器:`pgrep -f 'controller/unilatera[l]'`(括号防自匹配)。
- 绑核哨兵(/tmp/pin_sentinel.log)机器重启后要重挂;新控制器出生 1 秒内须 SCHED_FIFO。
- Scarlett 声卡槽位被抢的复活配方见 2026-08-29 会话记录(unbind 摄像头音频→USB 重枚举)。
- 开录需听诊头有信号(敲/搓即过门)。
