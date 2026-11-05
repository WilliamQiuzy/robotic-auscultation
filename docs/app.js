'use strict';
const zh = {
  navTraining:'训练流程', heroTitle:'在仿真中，<br>学会温柔接触。',
  heroLead:'搭建一个听诊机器人训练环境，让 OpenArm 学习接近、贴合、保持和撤离背部的十个听诊点。',
  heroNote:'从合格的解析专家走向学习策略。实际动作、受力记录，以及失败，全部可查看。',
  watch:'观看仿真演示 <span>↓</span>',recipe:'查看训练方法 ↗',previewNote:'专家已完成接触任务；本页发布的 ACT 试训仍未通过闭环测试。两者都如实展示。',
  expertTab:'专家 · L5',actTab:'ACT · 验证 L3',trainTab:'ACT · 训练 R3',
  expertMetric:'专家物理资格',expertMetricNote:'固定的诊断回合',actMetric:'ACT 闭环成功',actMetricNote:'1 个 TRAIN + 1 个选模 VAL 场景',holdMetric:'接触保持要求',holdMetricNote:'法向力 3.5–4.5 N 连续保持',targetMetric:'任务听诊点',targetMetricNote:'L1–L5、R1–R5；本页未全部验证',
  trainingTitle:'学习接近。<br>实测接触。',trainingLead:'当前基线学习经碰撞检查的接近轨迹。本地控制器负责受力、保持和卸力。完整回合还必须回到 home。',
  p1:'搭建场景',p1d:'OpenArm、弹簧听诊组件、坐姿人体和已批准的 RGB-D 相机视角。',p2:'查询专家',p2d:'从当前状态规划，记录实际命令、实测运动与接触。',p3:'筛选训练窗口',p3d:'按阶段和动作来源筛选，按人体身份划分，再只用 TRAIN 拟合归一化。',p4:'训练官方 ACT',p4d:'预测 20 步，执行前 10 步，更新观测后再次预测。',p5:'实际闭环评估',p5d:'检查接近、接触、保持、撤离和返回，不以离线误差代替任务成功。',
  contractTitle:'先建立明确、可检查的控制基线。',contractText:'官方 LeRobot ACT，266 万参数，无 VAE。两帧 state17 构成 34 个输入特征；一个动作块始终使用同一个关节原点。',
  modalityNote:'当前 ACT 输入是状态与目标几何。RGB-D 和 Hall 有采集记录，Hall 用于解析接触控制器；这版 ACT 尚不是多模态 VLA，也没有学习模拟声学。',
  recipeNote:'训练数组和有许可证要求的场景资产需另行准备。提供可运行的参考训练入口；本页不分发原实验的完整资产包。',fullRecipe:'数组契约与复现边界 ↗',
  benchmarkTitle:'每个结论，都能检查。',benchmarkLead:'下表来自不同诊断批次，不是同一协议下的排行榜。本次发布尚未提供独立留出人体 benchmark 的结果。',
  closedCaption:'实际 Isaac 闭环证据 · 2026 年 10 月 28–29 日的保留实验',method:'方法',cohort:'评测范围',success:'成功数',meaning:'支持的结论',
  expertName:'解析专家',expertCohort:'固定 8 个资格回合',expertMeaning:'完成接近、接触、保持至少 3 秒、卸力与回 home',
  actCohort:'1 个 TRAIN + 1 个选模 VAL 场景',actMeaning:'两例均在接触前被姿态检查停止',heldoutName:'独立留出人体',heldoutCohort:'按身份隔离的独立协议',unreported:'尚无报告',heldoutMeaning:'目前不能宣称已证明泛化',
  astraCohort:'接口原型',notEvaluated:'尚未评测',astraMeaning:'具备 API 请求和校验代码；尚无 Isaac 成绩',
  fitTitle:'拟合变好了，动作仍然失败。',fitNote:'两轮均完成 5,000 次更新，按完整验证集选模。目前仅增强开启的模型有两次已发布的 Isaac 闭环测试。',
  gateTitle:'接触成功有明确的物理定义。',force:'保持期间物理法向力',hold:'连续有效保持',forbidden:'非允许部件受力',tilt:'工具轴与局部 23 mm 圆盘平面法向',return:'卸力、撤离、回到 home',required:'必须完成',
  gateNote:'允许 C 外圈承载接触；凹入的 B 膜片不要求受力。物理、停车、关节与碰撞门持续生效。',protocolLink:'完整评测协议、范围与测量定义 ↗',resultsLink:'下载机器可读结果 ↗',
  failureNote:'为什么公开失败？验证场景的首个动作块就有较大误差；训练场景则在偏离专家状态后逐渐失控。人体多样性与恢复示范分别针对这两种问题。',
  astraTitle:'Astra 做推理。<br>本地控制器执行。',astraLead:'第一步让 Astra 根据观测选择技能：下一个目标、继续控制或撤离恢复。原生执行器继续管理运动与接触力。',
  astraScope:'原型状态：请求构造与本地决策校验已测试。尚无实时 API 或 Isaac 执行结果。实际试验还需要 API 密钥和仿真技能执行桥。',
  astraQuestion:'推理能否提升完成率与恢复能力？',astraCompare:'在相同场景、目标、执行器与物理门下比较：解析专家、ACT + 接触控制器、Astra + 本地技能。同时报告调用次数、干预、墙钟耗时和 API 成本。',
  astraLatency:'云端推理不进入力控环。Astranomy 的操作实验显示推理延迟不可忽略；两次决策之间，接触控制器仍需在本地运行。',astraGuide:'接入设计与可运行的 dry run ↗',
  closingTitle:'检查场景。<br>复核证据。<br>改进策略。',openRepo:'打开 GitHub 项目 ↗',footerNote:'仿真研究预览 · 发布于 2026-11-05 · 证据来自 10 月 28–29 日'
};
const videos = {
  expert:{src:'assets/expert-l5.mp4',kind:'ANALYTIC EXPERT',result:'EXPERT QUALIFIED',tone:'pass',evidence:'data/expert_summary.json',en:'A full single-point expert episode: approach → stop → contact → 3 s hold → unload → home. Actual RobotCam recording; 104.05 s playback.',zh:'完整的单点解析专家回合：接近 → 停车 → 接触 → 保持 3 秒 → 卸力 → 回 home。真实 RobotCam 录像，播放时长 104.05 秒。'},
  act:{src:'assets/act-val-l3.mp4',kind:'ACT · SELECTION VALIDATION',result:'FAILED BEFORE CONTACT',tone:'fail',evidence:'data/act_summary.json',en:'Actual ACT weights on p001005 / L3 (selection validation). Stopped by the elbow-height gate after 34 queries and 334 actions. No contact; 17.7 s playback.',zh:'实际 ACT 权重在 p001005 / L3（选模验证人体）上的闭环。34 次查询、334 条动作后触发肘部高度门，未接触；播放时长 17.7 秒。'}
};
const originals = new Map([...document.querySelectorAll('[data-i18n]')].map(el=>[el,el.innerHTML]));
let language='en',selected='expert';
function caption(){document.querySelector('#video-caption').textContent=videos[selected][language];}
document.querySelector('#language').addEventListener('click',()=>{
  language=language==='en'?'zh':'en';document.documentElement.lang=language==='zh'?'zh-CN':'en';
  for(const [el,original] of originals)el.innerHTML=language==='zh'?(zh[el.dataset.i18n]||original):original;
  document.querySelector('#language').textContent=language==='zh'?'EN':'中文';caption();
});
for(const button of document.querySelectorAll('[data-video]'))button.addEventListener('click',()=>{
  selected=button.dataset.video;const data=videos[selected],video=document.querySelector('#main-video');
  video.pause();video.removeAttribute('poster');video.src=data.src;video.load();
  video.setAttribute('aria-label',data.kind+' actual Isaac Sim recording');
  document.querySelector('#video-kind').textContent=data.kind;
  const result=document.querySelector('#video-result');result.textContent=data.result;result.className=data.tone;
  document.querySelector('#video-evidence').href=data.evidence;
  for(const tab of document.querySelectorAll('[data-video]')){const active=tab===button;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));}
  caption();
});
for(const button of document.querySelectorAll('[data-copy]'))button.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);button.textContent=language==='zh'?'已复制':'Copied';}
  catch{button.textContent=language==='zh'?'请选中文本复制':'Select text to copy';}
});
