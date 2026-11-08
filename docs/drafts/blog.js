'use strict';
const menu=document.getElementById('blog-menu');
const toggle=document.getElementById('blog-menu-toggle');
function closeMenu(){menu.close();toggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;}
toggle.addEventListener('click',()=>{toggle.setAttribute('aria-expanded','true');document.body.dataset.menuOpen='';menu.showModal();});
document.getElementById('blog-menu-close').addEventListener('click',closeMenu);
menu.addEventListener('close',()=>{toggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;});
menu.addEventListener('cancel',()=>{toggle.setAttribute('aria-expanded','false');delete document.body.dataset.menuOpen;});
menu.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',closeMenu));
menu.addEventListener('keydown',event=>{
 if(event.key!=='Tab')return;
 const items=[...menu.querySelectorAll('a[href],button:not([disabled])')];
 if(event.shiftKey&&document.activeElement===items[0]){event.preventDefault();items.at(-1).focus();}
 else if(!event.shiftKey&&document.activeElement===items.at(-1)){event.preventDefault();items[0].focus();}
});
window.matchMedia('(min-width:801px)').addEventListener('change',event=>{if(event.matches&&menu.open)closeMenu();});
document.documentElement.classList.add('js-video');
for(const video of document.querySelectorAll('.exp-recording video')){
 const box=video.closest('.video-box');
 if(!box)continue;
 const message=box.querySelector('#recording-message,[data-recording-message]');
 const retry=box.querySelector('#recording-retry,[data-recording-retry]');
 let loadingTimer,awaitingPlayback=!video.paused;
 function clearDeadline(){clearTimeout(loadingTimer);loadingTimer=undefined;}
 // Metadata is enough for an idle poster and native controls. Requested
 // playback needs future frames; do not force buffering for metadata preload.
 function available(){return video.readyState>=3||(!awaitingPlayback&&video.readyState>=1);}
 function state(value){
  box.dataset.state=value;box.setAttribute('aria-busy',String(value==='loading'));
  retry.hidden=value!=='error';
  message.textContent=value==='error'?'The recording could not load. Try again, or open the actual file below.':'Loading the actual recording…';
  if(value!=='loading')clearDeadline();
  // Repeated waiting/loadeddata events must not postpone the same stall.
  else if(loadingTimer===undefined)loadingTimer=setTimeout(()=>{
   loadingTimer=undefined;state(!video.error&&available()?'ready':'error');
  },15000);
 }
 function readiness(){state(video.error?'error':available()?'ready':'loading');}
 video.addEventListener('loadstart',()=>{awaitingPlayback=!video.paused;clearDeadline();readiness();});
 video.addEventListener('loadedmetadata',readiness);
 video.addEventListener('loadeddata',readiness);
 video.addEventListener('play',()=>{awaitingPlayback=true;readiness();});
 video.addEventListener('canplay',()=>state('ready'));
 video.addEventListener('playing',()=>{awaitingPlayback=true;state('ready');});
 video.addEventListener('waiting',()=>{awaitingPlayback=true;state('loading');});
 video.addEventListener('pause',()=>{awaitingPlayback=false;readiness();});
 video.addEventListener('ended',()=>{awaitingPlayback=false;readiness();});
 video.addEventListener('error',()=>state('error'));
 retry.addEventListener('click',()=>{awaitingPlayback=false;clearDeadline();state('loading');video.load();});
 readiness();
}
const reduced=window.matchMedia('(prefers-reduced-motion:reduce)');
const principle=document.querySelector('[data-word-reveal]');
const walker=document.createTreeWalker(principle,NodeFilter.SHOW_TEXT);const nodes=[];
while(walker.nextNode())nodes.push(walker.currentNode);
for(const node of nodes){
 const fragment=document.createDocumentFragment();
 for(const part of node.textContent.split(/(\s+)/u)){
  if(!part)continue;
  if(/^\s+$/u.test(part)){fragment.append(part);continue;}
  const span=document.createElement('span');span.className='word';span.textContent=part;fragment.append(span);
 }
 node.replaceWith(fragment);
}
if('IntersectionObserver' in window){
 if(!reduced.matches)document.documentElement.classList.add('js-motion');
 const sections=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('is-visible');sections.unobserve(entry.target);}},{threshold:.03});
 document.querySelectorAll('.reveal').forEach(el=>sections.observe(el));
 const words=new IntersectionObserver(entries=>{entries.filter(e=>e.isIntersecting).forEach((entry,index)=>{setTimeout(()=>entry.target.classList.add('is-active'),index*80);words.unobserve(entry.target);});},{rootMargin:'0px 0px -20% 0px',threshold:1});
 principle.querySelectorAll('.word').forEach(word=>{if(reduced.matches)word.classList.add('is-active');else words.observe(word);});
 const nav=new IntersectionObserver(entries=>{const entry=entries.find(e=>e.isIntersecting);if(!entry)return;const href='#'+entry.target.id;
  document.querySelectorAll('.desktop-nav a,.exp-toc a,.mobile-menu nav a').forEach(a=>{if(a.getAttribute('href')===href)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
 },{rootMargin:'-15% 0px -55% 0px',threshold:0});
 for(const id of ['comparison','recordings','offline-act','native-smokes','system-roles','hardware-gap','engineering','evidence','coverage','prefix','recovery','training','native-negative','student-states','iteration3','iteration4','v3-preparation','iteration10','evaluation']){const section=document.getElementById(id);if(section)nav.observe(section);}
}else principle.querySelectorAll('.word').forEach(word=>word.classList.add('is-active'));
reduced.addEventListener('change',event=>{if(event.matches){document.documentElement.classList.remove('js-motion');document.querySelectorAll('.reveal').forEach(el=>el.classList.add('is-visible'));principle.querySelectorAll('.word').forEach(word=>word.classList.add('is-active'));}});
