'use strict';
const zh = {
  navTraining:'训练流程', heroTitle:'在仿真中，<br>学会温柔接触。',
  heroLead:'搭建一个听诊机器人训练环境，让 OpenArm 学习接近、贴合、保持和撤离背部的十个听诊点。',
  heroNote:'从合格的解析专家走向学习策略。检查实际动作、受力测量，以及失败。',
  watch:'观看实际录像 <span aria-hidden="true">↗</span>',recipe:'查看训练方法 ↗',previewNote:'专家已完成接触任务；本页发布的 ACT 试训仍未通过闭环测试。两者都如实展示。',
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
  astraScope:'原型状态：请求构造与本地决策校验已测试。尚无实时 API 或 Isaac 执行结果。已公开的 Codex CLI 桥负责推理；有许可证要求的资产和原生仿真环境需要另行准备。',
  astraQuestion:'推理能否提升完成率与恢复能力？',astraCompare:'在相同场景、目标、执行器与物理门下比较：解析专家、ACT + 接触控制器、Astra + 本地技能。同时报告调用次数、干预、墙钟耗时和 API 成本。',
  astraLatency:'云端推理不进入力控环。Astranomy 的操作实验显示推理延迟不可忽略；两次决策之间，接触控制器仍需在本地运行。',astraGuide:'实际推理桥与接入契约 ↗',
  closingTitle:'检查场景。<br>复核证据。<br>改进策略。',openRepo:'打开 GitHub 项目 ↗',footerNote:'仿真研究预览 · 发布于 2026-11-05 · 证据来自 10 月 28–29 日'
};
Object.assign(zh, {
  navDemo:'演示',navBenchmark:'评测证据',navAstra:'Astra',navQuestions:'常见问题',
  previewLabel:'研究预览',recordedLabel:'实际仿真录像',evidenceLink:'证据 JSON ↗',
  metricsContext:'已发布证据 · 来自不同诊断批次',baselineLabel:'已发布 ACT 基线',codeLabel:'Linux · 使用合格数据',copyLabel:'复制',
  menuNote:'仿真研究预览。检查实际录像，以及录像背后的证据。',
  videoLoading:'正在加载实际录像…',videoRetry:'重新加载录像',
  taglineLabel:'研究原则',taglineText:'学习动作。<br>检验完整任务。',taglineNote:'更低的动作误差是有用信号。任务成功仍要求接触、有效保持、撤离，并回到 home。',
  tableHint:'在小屏幕上可横向滚动，检查表格中的每一列。 <span aria-hidden="true">↔</span>',
  fitLabel:'动作误差 ≠ 任务成功',gateLabel:'物理回合成功',astraCompareLabel:'值得开展的比较',
  faqLabel:'04 / 研究问题',faqTitle:'这个预览，<br>说明了什么？',faqLead:'范围和分数同样重要。以下回答描述已发布的证据，以及尚未完成的工作。',
  faqQ1:'这是临床或实体机器人结果吗？',faqA1:'本页报告 Isaac Sim 实验，不代表临床效果，也不代表已完成实体机器人验证。',
  faqQ2:'8/8 和 0/2 来自相同评测批次吗？',faqA2:'不是。专家通过八个固定资格回合。ACT 试训评测了一个 TRAIN 场景和一个选模 VAL 场景。这是不同诊断批次，不是排行榜。',
  faqQ3:'ACT 策略观察什么？',faqA3:'两帧 state17 构成 34 个输入特征，包括状态与目标几何。本版本没有向 ACT 输入 RGB、深度或 Hall 测量；Hall 由解析接触控制器使用。',
  faqQ4:'动作误差降低，为什么仍会失败？',faqA4:'离线预测误差不能衡量完整接触任务。两例已发布 ACT 闭环均在接触前被姿态检查停止。全部物理门仍必须满足。',
  faqQ5:'Astra 执行过仿真回合吗？',faqA5:'尚无结果报告。已发布原型构造技能请求，并在本地校验所提议的决策。请求不等于仿真器接受，也不等于物理成功。',
  faqQ6:'用本次发布可以复现什么？',faqA6:'仓库包含参考训练入口、数组契约、评测协议、证据摘要和两段实际录像。训练数组和有许可证要求的场景资产需要另行准备。',
  closingLabel:'开放研究，持续进行',checksumsLink:'资产校验和 ↗',siteNotesLink:'关于本预览 ↗'
});
const videos = {
  expert:{src:'assets/expert-l5.mp4',evidence:'data/expert_summary.json',tone:'pass',
    kind:{en:'Analytic expert',zh:'解析专家'},result:{en:'Expert qualified',zh:'专家资格通过'},
    en:'A full single point expert episode: approach → stop → contact → 3 s hold → unload → home. Actual RobotCam recording; 104.05 s playback.',
    zh:'完整的单点解析专家回合：接近 → 停车 → 接触 → 保持 3 秒 → 卸力 → 回 home。真实 RobotCam 录像，播放时长 104.05 秒。'},
  act:{src:'assets/act-val-l3.mp4',evidence:'data/act_summary.json',tone:'fail',
    kind:{en:'ACT · selection validation',zh:'ACT · 选模验证'},result:{en:'Failed before contact',zh:'接触前失败'},
    en:'Actual ACT weights on p001005 / L3 (selection validation). Stopped by the elbow height gate after 34 queries and 334 actions. No contact; 17.7 s playback.',
    zh:'实际 ACT 权重在 p001005 / L3（选模验证人体）上的闭环。34 次查询、334 条动作后触发肘部高度门，未接触；播放时长 17.7 秒。'}
};
const originals = new Map([...document.querySelectorAll('[data-i18n]')].map(el=>[el,el.innerHTML]));
const video = document.querySelector('#main-video');
const videoBox = document.querySelector('.video-box');
const stateMessage = document.querySelector('#video-state-message');
const retry = document.querySelector('#video-retry');
let language='en', selected='expert', videoState='loading', loadingTimer;
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let taglineObserver;

function updateVideoCopy(){
  const data=videos[selected];
  document.querySelector('#video-caption').textContent=data[language];
  document.querySelector('#video-kind').textContent=data.kind[language];
  const result=document.querySelector('#video-result');
  result.textContent=data.result[language]; result.className=data.tone;
  document.querySelector('#video-evidence').href=data.evidence;
  video.setAttribute('aria-label',`${data.kind[language]} ${language==='zh'?'实际 Isaac Sim 录像':'actual Isaac Sim recording'}`);
  const messages={
    loading:{en:'Loading the actual recording…',zh:'正在加载实际录像…'},
    buffering:{en:'Loading more of the recording…',zh:'正在加载后续录像…'},
    error:{en:'The recording could not load. Try again, or open the video file below.',zh:'录像未能加载。请重试，或打开下方视频文件。'}
  };
  if(messages[videoState]) stateMessage.textContent=messages[videoState][language];
  retry.textContent=language==='zh'?'重新加载录像':'Try loading again';
  document.querySelector('#video-file-link')?.remove();
  if(videoState==='error'){
    const link=document.createElement('a'); link.id='video-file-link'; link.href=data.src;
    link.className='caption-1'; link.textContent=language==='zh'?'打开实际视频文件 ↗':'Open the actual video file ↗';
    document.querySelector('#video-state').append(link);
  }
}
function setVideoState(state){
  videoState=state; videoBox.dataset.state=state==='buffering'?'loading':state;
  videoBox.setAttribute('aria-busy',String(state==='loading'||state==='buffering'));
  retry.hidden=state!=='error';
  if(state==='ready'||state==='error') clearTimeout(loadingTimer);
  updateVideoCopy();
}
function beginLoading(){
  clearTimeout(loadingTimer); setVideoState('loading');
  loadingTimer=setTimeout(()=>{if(video.readyState===0)setVideoState('error');},15000);
}
video.addEventListener('loadedmetadata',()=>setVideoState('ready'));
video.addEventListener('canplay',()=>setVideoState('ready'));
video.addEventListener('playing',()=>setVideoState('ready'));
video.addEventListener('waiting',()=>setVideoState('buffering'));
video.addEventListener('error',()=>setVideoState('error'));
video.addEventListener('stalled',()=>{if(video.readyState<2)setVideoState('buffering');});
retry.addEventListener('click',()=>{beginLoading();video.load();});
for(const button of document.querySelectorAll('[data-video]'))button.addEventListener('click',()=>{
  if(button.dataset.video===selected) return;
  selected=button.dataset.video;
  video.pause(); video.removeAttribute('poster');
  beginLoading(); video.src=videos[selected].src; video.load();
  for(const tab of document.querySelectorAll('[data-video]')){
    const active=tab===button;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));
  }
});

function prepareTagline(){
  taglineObserver?.disconnect();
  const tagline=document.querySelector('#tagline');
  const walker=document.createTreeWalker(tagline,NodeFilter.SHOW_TEXT);
  const nodes=[]; while(walker.nextNode()) nodes.push(walker.currentNode);
  let wordIndex=0;
  for(const node of nodes){
    const parts=node.textContent.split(/(\s+)/u);const fragment=document.createDocumentFragment();
    for(const part of parts){
      if(!part)continue;
      if(/^\s+$/u.test(part)){fragment.append(part);continue;}
      // Chinese phrases follow the same reading order without splitting characters.
      const span=document.createElement('span');span.className='tagline-word';span.textContent=part;span.dataset.order=wordIndex++;
      fragment.append(span);
    }
    node.replaceWith(fragment);
  }
  const words=[...tagline.querySelectorAll('.tagline-word')];
  if(motionPreference.matches||!('IntersectionObserver' in window)){
    words.forEach(word=>word.classList.add('is-active'));return;
  }
  taglineObserver=new IntersectionObserver(entries=>{
    entries.filter(entry=>entry.isIntersecting).sort((a,b)=>a.target.dataset.order-b.target.dataset.order).forEach((entry,index)=>{
      setTimeout(()=>entry.target.classList.add('is-active'),index*80);
      taglineObserver.unobserve(entry.target);
    });
  },{rootMargin:'0px 0px -22% 0px',threshold:1});
  words.forEach(word=>taglineObserver.observe(word));
}
const languageButton=document.querySelector('#language');
languageButton.addEventListener('click',()=>{
  language=language==='en'?'zh':'en';document.documentElement.lang=language==='zh'?'zh-CN':'en';
  for(const [el,original] of originals)el.innerHTML=language==='zh'?(zh[el.dataset.i18n]||original):original;
  languageButton.textContent=language==='zh'?'EN':'中文';
  languageButton.setAttribute('aria-label',language==='zh'?'Read this page in English':'Read this page in Chinese');
  document.querySelector('#menu-toggle').setAttribute('aria-label',language==='zh'?'打开导航':'Open navigation');
  document.querySelector('#menu-close').setAttribute('aria-label',language==='zh'?'关闭导航':'Close navigation');
  updateVideoCopy();prepareTagline();
});
for(const button of document.querySelectorAll('[data-copy]'))button.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);button.textContent=language==='zh'?'已复制':'Copied';}
  catch{button.textContent=language==='zh'?'请选中文本复制':'Select text to copy';}
});

const menu=document.querySelector('#mobile-menu');
const menuToggle=document.querySelector('#menu-toggle');
function closeMenu(){
  menu.close();menuToggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;
}
menuToggle.addEventListener('click',()=>{
  if(menu.open){closeMenu();return;}
  menuToggle.setAttribute('aria-expanded','true');document.body.dataset.menuOpen='';menu.showModal();
});
document.querySelector('#menu-close').addEventListener('click',closeMenu);
menu.addEventListener('keydown',event=>{
  if(event.key!=='Tab')return;
  const items=[...menu.querySelectorAll('a[href],button:not([disabled])')].filter(el=>!el.hidden);
  const first=items[0],last=items[items.length-1];
  if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
  else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
});
menu.addEventListener('cancel',()=>{menuToggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;});
menu.addEventListener('close',()=>{menuToggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;});
menu.querySelectorAll('nav a').forEach(link=>link.addEventListener('click',()=>closeMenu()));
window.matchMedia('(min-width: 801px)').addEventListener('change',event=>{if(event.matches&&menu.open)closeMenu();});

if('IntersectionObserver' in window){
  if(!motionPreference.matches) document.documentElement.classList.add('js-motion');
  const sectionObserver=new IntersectionObserver(entries=>{
    for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('is-visible');sectionObserver.unobserve(entry.target);}
  },{threshold:.08});
  document.querySelectorAll('.reveal').forEach(el=>sectionObserver.observe(el));
  const navObserver=new IntersectionObserver(entries=>{
    const visible=entries.filter(entry=>entry.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
    if(!visible.length)return;
    const href='#'+visible[0].target.id;
    document.querySelectorAll('.desktop-nav a,.mobile-menu nav a').forEach(link=>{
      if(link.getAttribute('href')===href)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
    });
  },{rootMargin:'-15% 0px -50% 0px',threshold:0});
  ['demo','training','benchmark','astra','questions'].forEach(id=>navObserver.observe(document.getElementById(id)));
}
motionPreference.addEventListener('change',()=>{
  document.documentElement.classList.toggle('js-motion',!motionPreference.matches);
  if(motionPreference.matches)document.querySelectorAll('.reveal').forEach(el=>el.classList.add('is-visible'));
  // Rebuild from the untranslated source, avoiding nested word spans.
  document.querySelector('#tagline').innerHTML=language==='zh'?zh.taglineText:originals.get(document.querySelector('#tagline'));
  prepareTagline();
});
prepareTagline();
document.documentElement.classList.add('js-video');
beginLoading();
if(video.error)setVideoState('error');
else if(video.readyState>=1)setVideoState('ready');
