/* Presentation accounting only. Physical qualification remains the recorded native verdict. */
(function(root){
 'use strict';
 const methods=['expert','act','astra'];
 const subjects=['p001030','p000956'];
 const poses=['p000474_a001230_f0000','p000476_a001231_f0023'];
 const expected=new Set(subjects.flatMap(s=>poses.flatMap(p=>['r3','l3'].map(t=>`${s}-${p}-${t}`))));
 const statuses=new Set(['pending','running','completed','failed','aborted','unavailable','not_launched','missing_receipt']);
 const terminal=new Set(['completed','failed','aborted','unavailable','not_launched','missing_receipt']);
 function summarize(data){
  const partial=data.schema==='ausc-matched-closed-arm-extract-v1';
  if(!['ausc-matched-eight-results-v1','ausc-matched-closed-arm-extract-v1'].includes(data.schema)||data.expected_scenario_slots!==8||!Array.isArray(data.scenarios)||data.scenarios.length!==8)throw new Error('Expected all eight frozen scenario slots.');
  const seen=new Set();const summaries=Object.fromEntries(methods.map(m=>[m,{planned:8,pending:0,running:0,terminal:0,qualified:0,physicalRejected:0,physicalUnknown:0,actualNative:0,modelInvoked:0,categories:{}}]));
  for(const scenario of data.scenarios){
   if(!expected.has(scenario.scenario_id)||seen.has(scenario.scenario_id))throw new Error('Frozen scenarios must appear exactly once.');
   seen.add(scenario.scenario_id);
   for(const method of methods){
    const arm=scenario.arms?.[method];
    if(!arm||arm.method!==method||!statuses.has(arm.execution_status)||![null,true,false].includes(arm.physical_success))throw new Error('Incomplete or invalid method slot.');
    if(['pending','running','unavailable','not_launched','missing_receipt'].includes(arm.execution_status)&&arm.physical_success!==null)throw new Error('Unknown physical verdict must remain null.');
    if(arm.physical_success!==null&&(!arm.owner_terminal||!arm.native_audit))throw new Error('Physical verdict has no terminal audit binding.');
    if(arm.physical_success===true&&(arm.execution_status!=='completed'||arm.actual_native_episode!==true||(method!=='expert'&&arm.model_invoked!==true)))throw new Error('Qualified completion contradicts execution evidence.');
    const s=summaries[method];
    if(arm.execution_status==='pending')s.pending++;
    if(arm.execution_status==='running')s.running++;
    if(terminal.has(arm.execution_status))s.terminal++;
    if(arm.physical_success===true)s.qualified++;
    else if(arm.physical_success===false)s.physicalRejected++;
    else s.physicalUnknown++;
    if(arm.actual_native_episode===true)s.actualNative++;
    if(arm.model_invoked===true)s.modelInvoked++;
    s.categories[arm.execution_status]=(s.categories[arm.execution_status]||0)+1;
   }
  }
  const reviewed=data.publication_status==='reviewed_for_append';
  if(partial&&(reviewed||data.whole_comparison_complete!==false||!Array.isArray(data.reported_methods)||data.reported_methods.some(m=>!methods.includes(m))))throw new Error('A partial extract cannot claim whole-comparison completion.');
  for(const method of methods){
   const declared=partial&&data.reported_methods.includes(method);
   if(declared&&(!data.original_reports?.[method]||!data.original_qualifications?.[method]||summaries[method].terminal!==8||summaries[method].physicalUnknown!==0||data.scenarios.some(s=>!s.arms[method].qualification)))throw new Error('Closed-arm summary includes unresolved slots or missing qualification bindings.');
   summaries[method].reportable=reviewed||declared;
  }
  return {reviewed,methods:summaries};
 }
 function verdict(arm){
  if(arm.physical_success===true)return 'Qualified';
  if(arm.physical_success===false&&arm.physical_category)return {mesh_clearance:'Rejected · mesh clearance',posture_joint_envelope:'Rejected · posture envelope'}[arm.physical_category]||'Physical rejection';
  if(arm.physical_success===false)return 'Physical rejection';
  return {pending:'Pending',running:'Running · verdict pending',completed:'Closed · audit unavailable',failed:'Execution failed · physical unknown',aborted:'Aborted · physical unknown',unavailable:'Unavailable · physical unknown',not_launched:'Not launched',missing_receipt:'Missing audit · physical unknown'}[arm.execution_status];
 }
 function render(element,data){
  const result=summarize(data);
  for(const method of methods){
   const summary=result.methods[method];
   element.querySelector(`[data-summary="${method}"]`).textContent=summary.reportable?`${summary.qualified} / 8`:'Pending';
   for(const scenario of data.scenarios){
    const cell=element.querySelector(`[data-case="${scenario.scenario_id}"][data-method="${method}"]`);
    if(!cell)throw new Error('Missing scenario cell.');
    cell.textContent=verdict(scenario.arms[method]);
    cell.dataset.verdict=scenario.arms[method].physical_success===null?'unknown':String(scenario.arms[method].physical_success);
   }
  }
  const labels={expert:'Expert',act:'ACT',astra:'Astra + expert'};
  const reported=methods.filter(m=>result.methods[m].reportable);
  element.querySelector('#matched-status').textContent=result.reviewed?'End-to-end qualified completions per eight planned scenarios. Physical unknowns remain distinct from observed rejections.':reported.length?`Closed original qualifications: ${reported.map(m=>`${labels[m]} ${result.methods[m].qualified}/8`).join('; ')}. Astra’s eight slots remain pending. The whole comparison is incomplete.`:'Qualification pending. All 24 system slots remain pending; no matched aggregate score is available.';
  element.querySelector('#matched-coverage').textContent=reported.length?methods.map(m=>{const s=result.methods[m];if(!s.reportable)return `${labels[m]}: eight planned slots, all verdicts pending.`;return `${labels[m]}: ${s.actualNative}/8 actual native episodes; ${s.physicalRejected} audited physical rejections; ${s.physicalUnknown} unknown physical verdicts${m==='expert'?'':`; ${s.modelInvoked} model-invoked episodes`}.`;}).join(' '):'Eight planned slots per system. Pending is unknown, not an observed physical failure. Terminal infrastructure or missing-audit outcomes retain a null physical verdict.';
  return result;
 }
 const api={summarize,verdict,render};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 else root.MatchedResults=api;
 if(typeof document!=='undefined'){
  const element=document.querySelector('[data-results-source]');
  if(element)fetch(element.dataset.resultsSource).then(response=>{if(!response.ok)throw new Error('Result extract unavailable.');return response.json();}).then(data=>render(element,data)).catch(()=>{element.querySelector('#matched-status').textContent='The result extract could not be verified. The static table retains its last evidenced values; no new score is inferred.';});
 }
})(typeof window!=='undefined'?window:globalThis);
