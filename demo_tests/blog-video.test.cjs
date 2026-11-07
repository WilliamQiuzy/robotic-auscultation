/* Production blog.js, real EventTarget events and a virtual clock.
   Minimal DOM surfaces isolate media readiness without a network or codec. */
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../docs/drafts/blog.js'),'utf8');
class Element extends EventTarget{
 constructor(){super();this.dataset={};this.attrs={};this.hidden=true;this.classList={add(){},remove(){}};}
 setAttribute(k,v){this.attrs[k]=v;}
 getAttribute(k){return this.attrs[k];}
 querySelectorAll(){return [];}
}
function fixture({readyState=0,paused=true}={}){
 let now=0,id=0;const tasks=new Map();
 const clock={pending:()=>tasks.size,advance(ms){const until=now+ms;for(let n=0;n<100;n++){const next=[...tasks.entries()].filter(([,t])=>t.at<=until).sort((a,b)=>a[1].at-b[1].at)[0];if(!next){now=until;return;}tasks.delete(next[0]);now=next[1].at;next[1].fn();}throw new Error('Unbounded timer loop');}};
 const box=new Element(),retry=new Element(),message=new Element();box.querySelector=s=>s.includes('message')?message:retry;
 const video=new Element();Object.assign(video,{readyState,paused,error:null,preload:'metadata',loads:0,plays:0});
 video.closest=()=>box;video.load=()=>{video.loads++;video.readyState=0;video.paused=true;video.error=null;video.dispatchEvent(new Event('loadstart'));};video.play=()=>{video.plays++;throw new Error('Loader must not force autoplay');};
 const bare=new Element();bare.closest=()=>null;
 const nodes=new Map();const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
 const principle=new Element();const document={body:{dataset:{}},documentElement:new Element(),getElementById:get,querySelector:()=>principle,querySelectorAll:s=>s==='.exp-recording video'?[video,bare]:[],createTreeWalker:()=>({nextNode:()=>false})};
 const window={matchMedia:()=>({matches:false,addEventListener(){}})};
 vm.runInNewContext(source,{document,window,NodeFilter:{SHOW_TEXT:4},setTimeout(fn,delay){const key=++id;tasks.set(key,{fn,at:now+delay});return key;},clearTimeout(key){tasks.delete(key);}},{filename:'docs/drafts/blog.js'});
 const emit=type=>video.dispatchEvent(new Event(type));
 const state=()=>({state:box.dataset.state,busy:box.getAttribute('aria-busy'),retryHidden:retry.hidden});
 return {video,retry,clock,emit,state};
}
const usable={state:'ready',busy:'false',retryHidden:true};
const loading={state:'loading',busy:'true',retryHidden:true};
const failed={state:'error',busy:'false',retryHidden:false};
test('paused metadata preload exposes poster/controls without forcing buffering or false timeout',()=>{
 const f=fixture();f.video.readyState=1;f.emit('loadedmetadata');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.clock.advance(30000);assert.deepEqual(f.state(),usable);assert.equal(f.video.preload,'metadata');assert.equal(f.video.loads,0);assert.equal(f.video.plays,0);
});
test('metadata already available before the script is usable without a later event',()=>{
 const f=fixture({readyState:1});assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);
});
test('a requested play that stalls at metadata gets a finite retry route',()=>{
 const f=fixture();f.video.readyState=1;f.video.paused=false;f.emit('play');f.emit('loadedmetadata');assert.deepEqual(f.state(),loading);f.clock.advance(15000);assert.deepEqual(f.state(),failed);assert.equal(f.clock.pending(),0);
});
test('post-canplay waiting rearms a deadline and reaches retry on a real buffering stall',()=>{
 const f=fixture();f.video.readyState=3;f.emit('canplay');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.video.readyState=1;f.video.paused=false;f.emit('waiting');assert.deepEqual(f.state(),loading);assert.equal(f.clock.pending(),1);f.clock.advance(15000);assert.deepEqual(f.state(),failed);
});
test('current-frame loadeddata does not pretend active playback has future data',()=>{
 const f=fixture();f.video.readyState=1;f.video.paused=false;f.emit('play');f.video.readyState=2;f.emit('loadeddata');assert.deepEqual(f.state(),loading);assert.equal(f.clock.pending(),1);f.video.readyState=3;f.emit('canplay');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.clock.advance(30000);assert.deepEqual(f.state(),usable);
});
test('pausing during buffering restores usable metadata controls and cancels the deadline',()=>{
 const f=fixture({readyState:3,paused:false});f.video.readyState=1;f.emit('waiting');f.clock.advance(5000);f.video.paused=true;f.emit('pause');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.clock.advance(30000);assert.deepEqual(f.state(),usable);
});
test('repeated waiting signals cannot postpone the same unresolved stall forever',()=>{
 const f=fixture({readyState:3});f.video.readyState=1;f.emit('waiting');f.clock.advance(10000);f.emit('waiting');assert.equal(f.clock.pending(),1);f.clock.advance(5000);assert.deepEqual(f.state(),failed);
});
test('Retry reloads once without autoplay; metadata/readiness clears the new deadline',()=>{
 const f=fixture();f.clock.advance(15000);assert.deepEqual(f.state(),failed);f.retry.dispatchEvent(new Event('click'));assert.deepEqual(f.state(),loading);assert.equal(f.clock.pending(),1);assert.equal(f.video.loads,1);assert.equal(f.video.plays,0);f.video.readyState=1;f.emit('loadedmetadata');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.clock.advance(30000);assert.deepEqual(f.state(),usable);
});
test('actual playing and media error events clear their pending deadlines',()=>{
 const f=fixture();f.video.readyState=2;f.video.paused=false;f.emit('play');f.emit('playing');assert.deepEqual(f.state(),usable);assert.equal(f.clock.pending(),0);f.video.readyState=1;f.emit('waiting');f.video.error={code:2};f.emit('error');assert.deepEqual(f.state(),failed);assert.equal(f.clock.pending(),0);
});
