/* Synthetic accounting fixtures only: no simulated/native benchmark outcomes. */
const test=require('node:test');
const assert=require('node:assert/strict');
const {summarize,verdict}=require('../docs/drafts/matched-results.js');
const template=require('../benchmarks/matched-eight-results-v1.template.json');
const fixture=()=>structuredClone(template);
const receipt={logical_name:'SYNTHETIC-TEST-ONLY',sha256:'a'.repeat(64),bytes:1,public_path:null};
function closed(arm,physical){Object.assign(arm,{execution_status:'completed',physical_success:physical,owner_terminal:receipt,native_audit:receipt,actual_native_episode:true,model_invoked:arm.method!=='expert',model_calls:arm.method!=='expert'?1:null});}
test('all 24 unknown outcomes stay pending, with eight planned slots per system',()=>{
 const r=summarize(fixture());assert.equal(r.reviewed,false);
 for(const s of Object.values(r.methods)){assert.equal(s.planned,8);assert.equal(s.pending,8);assert.equal(s.physicalUnknown,8);assert.equal(s.physicalRejected,0);assert.equal(s.actualNative,0);}
});
test('a missing frozen slot is rejected rather than reducing the denominator',()=>{const d=fixture();d.scenarios.pop();assert.throws(()=>summarize(d),/all eight/);});
test('a duplicate scenario cannot stand in for another planned slot',()=>{const d=fixture();d.scenarios[7]=structuredClone(d.scenarios[0]);assert.throws(()=>summarize(d),/exactly once/);});
test('unavailable infrastructure keeps a null physical verdict and is not a physical rejection',()=>{
 const d=fixture();Object.assign(d.scenarios[0].arms.act,{execution_status:'unavailable',failure_reason:'SYNTHETIC infrastructure counterexample'});
 const s=summarize(d).methods.act;assert.equal(s.planned,8);assert.equal(s.terminal,1);assert.equal(s.physicalUnknown,8);assert.equal(s.physicalRejected,0);assert.equal(s.categories.unavailable,1);
});
test('pending and infrastructure unknowns cannot be converted to false physical outcomes',()=>{
 for(const execution_status of ['pending','running','unavailable','not_launched','missing_receipt']){const d=fixture();Object.assign(d.scenarios[0].arms.act,{execution_status,physical_success:false});assert.throws(()=>summarize(d),/remain null/);}
});
test('a reviewed planned-slot completion count is valid even with infrastructure physical unknowns',()=>{
 const d=fixture();d.publication_status='reviewed_for_append';closed(d.scenarios[0].arms.act,true);closed(d.scenarios[1].arms.act,false);Object.assign(d.scenarios[2].arms.act,{execution_status:'missing_receipt',failure_reason:'SYNTHETIC missing audit'});
 const r=summarize(d),s=r.methods.act;assert.equal(r.reviewed,true);assert.equal(s.qualified,1);assert.equal(s.planned,8);assert.equal(s.physicalRejected,1);assert.equal(s.physicalUnknown,6);assert.equal(s.actualNative,2);assert.equal(s.modelInvoked,2);
});
test('audited physical rejections require the same terminal audit binding',()=>{const d=fixture();closed(d.scenarios[0].arms.act,false);d.scenarios[0].arms.act.native_audit=null;assert.throws(()=>summarize(d),/audit binding/);});
test('ACT or Astra qualification cannot imply an uninvoked model was tested',()=>{
 for(const m of ['act','astra']){const d=fixture();closed(d.scenarios[0].arms[m],true);d.scenarios[0].arms[m].model_invoked=false;assert.throws(()=>summarize(d),/contradicts/);}
});
test('UI labels keep observed negatives, missing audits and pending distinct',()=>{
 assert.equal(verdict({physical_success:false}),'Physical rejection');assert.equal(verdict({physical_success:null,execution_status:'pending'}),'Pending');assert.equal(verdict({physical_success:null,execution_status:'missing_receipt'}),'Missing audit · physical unknown');
});

const actualPartial=require('../docs/drafts/evidence/rebuilt-matched-20261106-v3.partial.json');
test('closed expert/ACT partial extract reports observed counts without completing Astra',()=>{
 const r=summarize(actualPartial);assert.equal(r.reviewed,false);assert.equal(r.methods.expert.qualified,8);assert.equal(r.methods.act.qualified,0);assert.equal(r.methods.act.physicalRejected,8);assert.equal(r.methods.act.modelInvoked,8);assert.equal(r.methods.expert.reportable,true);assert.equal(r.methods.act.reportable,true);assert.equal(r.methods.astra.reportable,false);assert.equal(r.methods.astra.physicalUnknown,8);assert.equal(r.methods.astra.pending,8);
});
test('a pending Astra arm cannot be declared a closed-arm result',()=>{const d=structuredClone(actualPartial);d.reported_methods.push('astra');assert.throws(()=>summarize(d),/unresolved slots/);});
test('a partial extract cannot claim whole-comparison completion or publication review',()=>{for(const change of [{whole_comparison_complete:true},{publication_status:'reviewed_for_append'}]){const d=structuredClone(actualPartial);Object.assign(d,change);assert.throws(()=>summarize(d),/partial extract/);}});

test('an actual Astra recording with a worker report does not become an independent qualified result',()=>{
 const scenario=actualPartial.scenarios.find(s=>s.scenario_id==='p001030-p000476_a001231_f0023-r3');assert.equal(scenario.arms.astra.recording.status,'available');assert.equal(scenario.arms.astra.physical_success,null);const r=summarize(actualPartial);assert.equal(r.methods.astra.reportable,false);assert.equal(r.methods.astra.physicalUnknown,8);assert.equal(r.methods.astra.pending,8);
});
