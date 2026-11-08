"""Direct-file CPU bridge entrypoint; project imports come only from Source30.

This file truthfully belongs to the new producer source. It never impersonates
an archived module, executable, __file__ or spec. Pure selectors are also used
by the current metadata reader; main imports only the original project.
"""
from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

SOURCE_ROOT='/opt/wam-runtime/code/wam-act-recovery-20261106-v30'
SOURCE_COMMIT='8ad4cb7734cf390fe68e9ef6e6f4a57b36866113'
ARCHIVE=dict(root=SOURCE_ROOT,code_commit=SOURCE_COMMIT,file_count=1679,
    manifest=dict(path=SOURCE_ROOT+'/SOURCE_MANIFEST.json',sha256='01d63eb437cd477c58d1ab663c860062183e3c860ae50322be3342ba08f22f73'),
    archive=dict(path='/workspace/wam/source_archives/wam-act-recovery-20261106-v30/source.tar.gz',bytes=6174493,
        sha256='ecaee479e83474b170d2523329d796e16b0f747dbcdc8f272f23762e37394c12'))
SLOT_IDS=('r3_previous_12mm','r3_first_12mm','l3_first_12mm','l3_last_45s')
PAYLOAD_SCHEMA='m2_authenticated_source30_query_prefix_payload_v1'
RESULT_SCHEMA='m2_authenticated_source30_query_prefix_result_v1'
RUN_ID='act8-corridor-smoke-20261107-v1'
BASE='/opt/wam-runtime/fast-wam'
NATIVE_PYTHON='/opt/wam-runtime/envs/isaacsim60/bin/python'
THREAD_VARIABLES=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


def need(ok,message):
    if not ok:raise ValueError(message)


def select_rows(queries,*,gap_m,target):
    """Authenticated request observations; teacher safety never chooses cuts."""
    need(type(gap_m) in (int,float) and math.isfinite(gap_m) and gap_m>0,'Positive actual canonical point gap required')
    need(target in ('R3','L3'),'Unknown predeclared query selector target')
    need(isinstance(queries,list),'Explicit ordered eligible query list required')
    prior=-1;first=None
    for i,q in enumerate(queries):
        row=q.get('physics_row');t=q.get('time_s');h=q.get('history');times=q.get('history_times')
        need(type(row) is int and row>=12 and row%12==0 and row>prior,'Unique ordered actual complete12-tick queries required')
        need(type(t) in (int,float) and math.isfinite(t) and abs(t-row/240.)<=1e-8 and
            isinstance(times,list) and len(times)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in times) and
            abs(times[1]-t)<=1e-8 and abs(times[1]-times[0]-.05)<=1e-8,'Complete causal H2 history/clock required')
        need(isinstance(h,list) and len(h)==2 and all(isinstance(v,list) and len(v)==17 and
            all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in h) and
            q.get('context',[])[-1:] == ['APPROACH'],'Authenticated APPROACH state17 query required')
        error=math.sqrt(sum(x*x for x in (h[-1][7],h[-1][8],h[-1][9]+gap_m)))
        if first is None and error<=.012+1e-14:first=i
        prior=row
    reached=None if first is None else queries[first]['physics_row']
    if target=='R3':return [None if first in (None,0) else queries[first-1]['physics_row'],reached]
    early=[q['physics_row'] for q in queries if q['time_s']<=45.]
    return [reached,early[-1] if early else None]


def select_request_rows(binding,*,gap_m,target):
    """Project requests already accepted by original authenticate_source."""
    queries=[]
    for request in binding['requests']:
        context=request['context']
        if context['context'][-1]!='APPROACH':continue
        queries.append(dict(physics_row=context['physics_row'],time_s=context['observed_at_s'],
            context=context['context'],history=request['history'],history_times=request['history_times']))
    return select_rows(queries,gap_m=gap_m,target=target)


def query_at_slot(binding,slot,*,l2_started_at_s):
    """Validate this fixed cut with the unchanged original expert admission."""
    need(slot.get('status')=='selected' and slot.get('slot_id') in SLOT_IDS,
        'One fixed selected query slot required; missing/duplicate slots stay ineligible')
    from sim.ausc_policy import m2_student_oracle_proposals as oracle
    row=slot['physics_row']
    try:
        query,snapshot,commands=oracle.query_at(binding,row)
        need(l2_started_at_s<=query['time_s']<l2_started_at_s+120.,
            'Actual selected query outside original cumulative L2 budget')
        query['measured_pose_base']=deepcopy(binding['events'][row]['observed_before']['pose_base'])
        return query,snapshot,commands
    except (ValueError,KeyError,TypeError,RuntimeError) as error:
        raise ValueError('Fixed selected cut '+slot['slot_id']+' physics_row='+str(row)+' is ineligible: '+
            type(error).__name__+': '+str(error)[:1200]) from error


def slot_ledger(sources):
    slots=[]
    for target,names in (('R3',SLOT_IDS[:2]),('L3',SLOT_IDS[2:])):
        value=sources.get(target,{})
        for slot,row in zip(names,value.get('rows',[None,None])):
            source=value.get('source');identity=None if row is None else dict(config=source['config'],owner=source['owner'],physics_row=row)
            slots.append(dict(slot_id=slot,target_id=target,physics_row=row,identity=identity,
                status='missing' if row is None else 'selected',source=deepcopy(source)))
    for slot in slots:
        if slot['identity'] is not None and sum(s['identity']==slot['identity'] for s in slots)>1:slot['status']='duplicate'
    need(len(slots)==4,'Exactly FOUR frozen selector slots required')
    return slots


def freeze_slots(rt,selected,output):
    slots=slot_ledger(selected)
    ref=rt.write_new(Path(output)/'SLOT_LEDGER.json',dict(schema='m2_source30_query_recovery_slot_ledger_v1',
        fixed_denominator=4,slots=slots))
    return slots,ref


def verify_archive(descriptor=ARCHIVE):
    """Full archive/tree hash closure, before and after historical CPU work."""
    def bound(ref):
        p=Path(ref['path']);need(p.is_absolute() and all(not q.is_symlink() for q in (p,*p.parents)), 'Canonical nonsymlink archive path required')
        need(p.is_file() and ('bytes' not in ref or p.stat().st_size==ref['bytes']),'Original archive bytes changed')
        with p.open('rb') as stream:
            digest=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        need(digest.hexdigest()==ref['sha256'],'Original archive hash changed')
        return p
    root=Path(descriptor['root']);need(root.is_absolute() and root.resolve()==root,'Canonical original source root required')
    bound(descriptor['archive']);rows=json.loads(bound(descriptor['manifest']).read_text())
    need(isinstance(rows,list) and len(rows)==descriptor['file_count'],'Complete original source manifest required')
    need((root/'CODE_COMMIT').read_text().strip()==descriptor['code_commit'],'Original source commit differs')
    names=set()
    for row in rows:
        name=row.get('file');relative=Path(name) if isinstance(name,str) else None
        need(set(row)=={'file','bytes','sha256'} and relative is not None and not relative.is_absolute() and
            relative.parts and '..' not in relative.parts and name==relative.as_posix() and name not in names and
            name not in ('CODE_COMMIT','SOURCE_MANIFEST.json'),'Unique safe original manifest file required')
        p=root/relative;need(stat.S_ISREG(p.lstat().st_mode),'Regular original source file required')
        bound(dict(path=str(p),bytes=row['bytes'],sha256=row['sha256']));names.add(name)
    actual=set()
    for p in root.rglob('*'):
        need(not p.is_symlink(),'Original source cannot contain symlinks')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    need(actual==names|{'CODE_COMMIT','SOURCE_MANIFEST.json'},'Unknown or omitted original source file')


def source_metadata(rt,source):
    need(isinstance(source,dict) and set(source)=={'config','owner','child','native'},'Exact historical source references required')
    config=rt.read_bound(source['config']);dep=config.get('m2_given_contact_expanded_pilot_deployment',{})
    meta=dep.get('metadata',{})
    need(meta.get('split')=='train','Query recovery sources must be TRAIN before native payload access')
    rt.shared.require_membership('train',meta.get('subject_id'))
    need(meta.get('subject_id')=='p001120' and meta.get('target_id') in ('R3','L3'),'Exact original ACT8 TRAIN body/targets required')
    return config,meta


def _inventory(rt,config):
    dep=config.get('m2_given_contact_expanded_pilot_deployment',{});inventory=dep.get('source_inventory',{})
    need(not any(k in config for k in ('m2_given_contact_control','m2_given_contact_deployment',
        'm2_given_contact_evaluation','m2_given_contact_recovery','m2_act_deployment')),
        'Original learned source cannot mix teacher/formal/TEST routes')
    manifest=inventory.get('source_manifest',{})
    need(inventory.get('root')==ARCHIVE['root'] and inventory.get('code_commit')==SOURCE_COMMIT and
        all(manifest.get(k)==v for k,v in ARCHIVE['manifest'].items()),'Exact immutable Source30 namespace/manifest required')
    rows={r['file']:r for r in rt.read_bound(manifest)};files=inventory.get('files')
    need(isinstance(files,dict) and bool(files),'Actual original source inventory required')
    for name,ref in files.items():
        relative='sim/ausc_policy/'+name
        need(relative in rows and Path(ref['path'])==Path(ARCHIVE['root'])/relative and
            all(ref.get(k)==rows[relative][k] for k in ('bytes','sha256')),'Original runtime inventory path/hash differs')
        rt.bound_path(ref)


def _normal_owner(owner):
    stage=owner.get('stage',{})
    return (owner.get('completed') is True and owner.get('gpu_released') is True and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')))


def _closed_development(rt,gate):
    value=rt.read_bound(gate['needs_repair']);frozen=value.get('source30',{})
    need(Path(gate['needs_repair']['path']).name=='NEEDS_REPAIR.json' and
        value.get('schema')=='act8_known_native_development_v1' and value.get('complete') is False and
        value.get('qualified_dev_only') is False and value.get('source30_stable') is True and value.get('errors')==[] and
        value.get('normal_training_owner') is True and value.get('normal_smoke_owner') is True and
        value.get('native_owner_requested') is True and value.get('smoke_run')==RUN_ID and value.get('smoke_owner')==gate['owner'] and
        value.get('fixed_denominator')==value.get('executed_cases')==4 and type(value.get('physical_successes')) is int and
        0<=value['physical_successes']<4,'Actual ACT8 closed normal/source-stable FOUR-case NEEDS_REPAIR required; 4/4 uses fresh benchmark')
    need(set(frozen)=={'root','code_commit','files','manifest'} and frozen.get('root')==ARCHIVE['root'] and
        frozen.get('code_commit')==SOURCE_COMMIT and frozen.get('files')==ARCHIVE['file_count'] and
        all(frozen.get('manifest',{}).get(k)==v for k,v in ARCHIVE['manifest'].items()),
        'Exact actual ACT8 Source30 closure required')
    rt.bound_path(frozen['manifest'])
    model=value.get('model',{})
    need(model.get('checkpoint_schema')=='m2_act_auxiliary_act_pilot_v6' and model.get('complete')==value.get('training'),
        'Actual original ACT8 V6 selected training binding required')
    control=rt.read_bound(value['training_control']);training_owner=rt.read_bound(value['training_owner'])
    need(control.get('schema')=='act8_single_owned_training_execution_v1' and control.get('complete') is True and
        control.get('normal_owner') is True and control.get('completed_steps')==control.get('selected_step')==1000 and
        control.get('native_launched') is False and control.get('model_qualified') is False and control.get('source')==frozen and
        all(control.get(k)==value[k] for k in ('training','training_owner','model')) and
        Path(value['training_owner']['path'])==Path(BASE)/'runs/ausc_policy/act8-corridor-train-20261107-v1/result.json' and
        training_owner.get('run_id')=='act8-corridor-train-20261107-v1' and training_owner.get('code_commit')==SOURCE_COMMIT and
        _normal_owner(training_owner),'Original ACT8 training must already close normally; no new training')
    cases=value.get('outcomes')
    need(isinstance(cases,list) and len(cases)==4 and all(
        c.get('preflight_passed') is True and c.get('qualification_attempted') is True and c.get('executed') is True and
        c.get('worker_normal') is True and c.get('infrastructure_error') is False and c.get('native_files_present') is True and
        type(c.get('passed')) is bool and type(c.get('physical_gate_passed')) is bool and
        type(c.get('physics_steps')) is int and c['physics_steps']>0 and
        not any(k in c for k in ('preparation_error','audit_exception','execution_evidence_error')) for c in cases) and
        value['physical_successes']==sum(c['passed'] and c['physical_gate_passed'] for c in cases),
        'ALL FOUR actual audits/workers must execute normally without infrastructure errors')
    return value,cases


def require_gate(rt,request):
    """All four metadata/audit terminals first; no VAL trajectories are read."""
    need(request.get('profile')=='production_source30' and set(request)=={'schema','profile','gate','sources'} and
        request.get('schema')=='m2_source30_query_recovery_request_v1','Explicit production Source30 request required')
    sources=request['sources'];need(isinstance(sources,dict) and set(sources)=={'R3','L3'},'Both predeclared TRAIN sources required')
    for target,source in sources.items():
        config,meta=source_metadata(rt,source);need(meta['target_id']==target,'Source target differs');_inventory(rt,config)
    gate=request['gate'];need(isinstance(gate,dict) and set(gate)=={'owner','needs_repair','cases'},'Exact four-case completion gate required')
    negative,closed_cases=_closed_development(rt,gate)
    owner=rt.read_bound(gate['owner']);stage=owner.get('stage',{})
    need(Path(gate['owner']['path'])==Path(BASE)/'runs/ausc_policy'/RUN_ID/'result.json' and
        owner.get('run_id')==RUN_ID and owner.get('code_commit')==SOURCE_COMMIT and
        _normal_owner(owner),'Entire original ACT8 owner must close normally')
    rows=gate['cases'];need(isinstance(rows,list) and len(rows)==4,'ALL FOUR independent audits must close')
    expected={(s,t) for s in ('p001120','p000923') for t in ('R3','L3')};seen=set();headers=[];matched_cases=[]
    jobs=owner.get('config',{}).get('jobs',[])
    need(owner.get('config',{}).get('experiment')=='isaac_batch' and len(jobs)==4,'Entire four-worker ACT8 owner required')
    for row in rows:
        need(set(row)=={'source','audit','pose_id'} and row['pose_id']=='p000474_a001230_f0000','Exact original development pose required')
        source=row['source'];need(set(source)=={'config','owner','child','native'} and source['owner']==gate['owner'],'Same ACT8 owner required')
        config=rt.read_bound(source['config']);dep=config.get('m2_given_contact_expanded_pilot_deployment',{});meta=dep.get('metadata',{})
        _inventory(rt,config)
        key=(meta.get('subject_id'),meta.get('target_id'))
        need(key in expected and key not in seen and meta.get('split')==('train' if key[0]=='p001120' else 'validation'),
            'Exact TRAIN/selectionVAL metadata-only gate required');seen.add(key)
        if key[0]=='p001120':need(source==sources[key[1]],'Declared TRAIN source is outside completion gate')
        recorded=[c for c in closed_cases if (c.get('subject_id'),c.get('target_id'))==key]
        need(len(recorded)==1,'One exact recorded ACT8 outcome per source required');case=recorded[0]
        need(case.get('config')==source['config'] and case.get('worker_result')==source['child'] and
            case.get('native_complete')==source['native'] and case.get('qualification')==row['audit'] and
            case.get('pose_id')==row['pose_id'] and case.get('split')==meta['split'] and dep.get('model')==negative['model'] and
            case.get('source_episode_id')==dep.get('source_episode_id'),
            'Request refs/pose/model must exactly match recorded ACT8 outcome')
        child=rt.read_bound(source['child']);audit=rt.read_bound(row['audit'])
        matches=[j for j in jobs if j.get('config')==source['config']['path'] and j.get('sha256')==source['config']['sha256']]
        need(len(matches)==1 and matches[0]['id']==case.get('id'),'Exact ACT8 config/job/outcome binding required');job=matches[0]['id']
        root=Path(gate['owner']['path']).parent/'output'/job;identity=child.get('identity',{});group=stage.get('process_identity',{})
        need(Path(source['child']['path'])==root.parent/(job+'.result.json') and Path(source['native']['path'])==root/'COMPLETE.json' and
            child.get('id')==job and child.get('output')==str(root) and child.get('config_sha256')==source['config']['sha256'] and
            child.get('result_sha256')==source['native']['sha256'] and type(child.get('exit_code')) is int and child['exit_code']==0 and
            child.get('worker_complete') is True and type(child.get('physical_pass')) is bool and type(child.get('pid')) is int and
            type(group.get('pgid')) is int and isinstance(identity.get('start_ticks'),str) and child.get('pid')==identity.get('pid') and
            identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
            group.get('members',{}).get(str(child.get('pid')))==identity.get('start_ticks'),'All ACT8 workers must close normally under original owner')
        need(Path(row['audit']['path'])==Path(gate['needs_repair']['path']).parent/'qualifications'/(job+'.json') and
            audit.get('schema')=='m2_given_contact_diagnostic_native_qualification_v1' and type(audit.get('passed')) is bool and
            audit.get('configuration_sha256')==rt.digest(config) and audit.get('inputs',{}).get('configuration')==source['config'] and
            audit.get('inputs',{}).get('complete')==source['native'] and audit.get('inputs',{}).get('run')==case.get('native_result') and
            audit['passed']==case['passed'] and
            (audit.get('checks',{}).get('unchanged_physical_gate_passed') is True)==case['physical_gate_passed'] and
            (audit.get('error') is None if audit['passed'] else isinstance(audit.get('error'),str) and bool(audit['error'])),
            'Normally closed exact independent native audit required')
        headers.append((source,config,meta,child,audit));matched_cases.append(case)
    need(seen==expected,'Full fixed ACT8 denominator required')
    # COMPLETE JSON is metadata here. No native trace/capture or VAL CPU payload.
    for (source,config,meta,child,audit),case in zip(headers,matched_cases):
        run=rt.read_bound(source['native']);receipt=run.get('neutral_native_audit',{})
        native_result=rt.read_bound(case['native_result'])
        need(Path(case['native_result']['path'])==Path(source['native']['path']).with_name('fresh_expert_result.json') and
            rt.digest(native_result)==rt.digest(run) and case['physics_steps']==run.get('physics_steps') and
            case.get('reported_success')==run.get('success') and case.get('reported_physical_validation')==run.get('physical_validation'),
            'Recorded ACT8 execution/count/physical verdict metadata differs')
        need(run.get('physics_steps',0)>0 and run.get('errors')==[] and run.get('learned_policy') is True and
            run.get('expert_l2_fallback') is False and receipt.get('complete') is True and receipt.get('errors')==[] and
            receipt.get('metadata')==meta and receipt.get('source_inventory')==config['m2_given_contact_expanded_pilot_deployment'].get('source_inventory'),
            'All original workers must execute with stable source and errors[]')
        cpu=run.get('cpu_child_exit',{})
        need(cpu.get('normal_exit') is True and cpu.get('reaped') is True and type(cpu.get('exit_code')) is int and
            cpu['exit_code']==0 and cpu.get('signals')==[] and cpu.get('started') is True and
            run.get('learned_policy_receipt',{}).get('cpu_child_exit')==cpu,
            'All four original ACT8 CPU children must close normally')
        for path,sha in run.get('runtime_sources_sha256',{}).items():
            need(rt.reference(path)['sha256']==sha,'Actual Source30 runtime source changed')
        need(bool(run.get('runtime_sources_sha256')) and child['physical_pass']==bool(run.get('success') is True and run.get('physical_validation') is True),
            'Actual source physical verdict/worker differs')
    need(any(not h[3]['physical_pass'] for h in headers),'At least one actual physical negative required')
    return headers


def namespace_proof(root):
    result={}
    for name,module in sorted(sys.modules.items()):
        if name.split('.')[0] in ('sim','scripts'):
            file=getattr(module,'__file__',None);spec=getattr(module,'__spec__',None)
            origin=getattr(spec,'origin',None)
            paths=list(getattr(module,'__path__',[]))
            need((file is None or origin==file and Path(file).resolve().is_relative_to(root)) and
                (origin in (None,'namespace') or Path(origin).resolve().is_relative_to(root)) and
                all(Path(p).resolve().is_relative_to(root) for p in paths),'Project namespace escaped original immutable Source30')
            result[name]=dict(file=file,spec_origin=origin,package_paths=paths)
    need('sim.ausc_policy.m2_student_recovery_prefix' in result and 'sim.ausc_policy.m2_student_oracle_proposals' in result,
        'Original historical authenticators must actually run')
    return result


def require_output_directory(output,root):
    output=Path(output);root=Path(root)
    need(output.resolve()==output and not output.is_relative_to(root) and output.is_dir() and not any(output.iterdir()),
        'Empty new child output outside immutable Source30 required')


def main():
    root=Path(SOURCE_ROOT)
    need(sys.platform=='linux' and Path.cwd()==root and sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 and
        sys.executable==NATIVE_PYTHON and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        'AUSC_OWNED_RUN' not in os.environ and 'PYTHONPATH' not in os.environ and
        all(os.environ.get(k)=='1' for k in THREAD_VARIABLES),'Isolated original30 CPU-only environment required')
    verify_archive();sys.path.insert(0,str(root))
    from sim.ausc_policy import m2_given_contact_runtime as rt
    from sim.ausc_policy import m2_student_oracle_proposals as oracle
    from sim.ausc_policy import m2_student_recovery_prefix as student
    request_ref=rt.reference(sys.argv[1]);request=rt.read_bound(request_ref);require_gate(rt,request)
    output=Path(sys.argv[2]);require_output_directory(output,root)
    selected={};bindings={};details={}
    for target,source in request['sources'].items():
        binding=oracle.authenticate_source(source)  # exactly one original binding per source
        bindings[target]=binding;run=rt.read_bound(source['native']);meta=binding['metadata']
        reference=rt.pose.require_route(meta);plan_ref=reference.snapshot()['sources']['fresh_plan'];plan=rt.pose.bound_json(plan_ref)
        points=[p for p in run.get('points',[]) if p.get('target')==target];need(len(points)==1,'One actual source target point required')
        point=points[0];gap=point.get('phase_config',{}).get('precontact_gap_m')
        need(type(gap) in (int,float) and math.isfinite(gap) and gap>0 and point.get('checked_prepose_gap_m')==gap and
            plan.get('free_approach_start_gap_m')==gap,'Actual phase/checked/canonical gap must agree and be positive')
        starts=[h.get('start_time_s') for h in run.get('l2_handoffs',[]) if h.get('target')==target]
        # The ACT receipt owns this start clock; never infer it from an arbitrary query.
        need(len(starts)==1 and type(starts[0]) in (int,float) and math.isfinite(starts[0]) and starts[0]>=0,
            'Original source L2 start clock required')
        selected[target]=dict(source=source,rows=select_request_rows(binding,gap_m=gap,target=target))
        details[target]=dict(gap_m=gap,canonical_plan=plan_ref,l2_started_at_s=starts[0],
            cpu_trace=run['cpu_child_exit']['trace'],native_trace=run['neutral_native_audit']['trace'])
    # Freeze all FOUR row identities before any expert eligibility check. A
    # rejected selected cut fails this producer; it never chooses a replacement.
    ledger,ledger_ref=freeze_slots(rt,selected,output);prefixes={}
    for slot in ledger:
        if slot['status']!='selected':continue
        target=slot['target_id'];binding=bindings[target];detail=details[target];cut=slot['physics_row']
        query,snapshot,commands=query_at_slot(binding,slot,l2_started_at_s=detail['l2_started_at_s'])
        cells=student.extract_cells(binding['events'],binding['blocks'],cut_row=cut)
        start=cells['start_row'];initial=binding['events'][start]['observed_before'];meta=binding['metadata']
        initial_requests=[r for r in binding['requests'] if r['context']['physics_row']==start]
        need(len(initial_requests)==1,'Actual initial replay query context required')
        payload=dict(schema=PAYLOAD_SCHEMA,profile='production_source30',slot_id=slot['slot_id'],source=slot['source'],
            source_config=binding['source_config'],source_model=binding['model'],split='train',subject_id=meta['subject_id'],target_id=target,
            reference_sha256=meta['reference_sha256'],initial_measured_q=initial['q'],
            initial_context_phase=initial_requests[0]['context']['context'][-1],source_cut_query=query,
            source_cut_command_input=snapshot,source_cut_command_history=commands.tolist(),actual_gap_m=detail['gap_m'],
            canonical_plan=detail['canonical_plan'],l2_started_at_s=detail['l2_started_at_s'],l2_deadline_s=detail['l2_started_at_s']+120.,
            source_cut_remaining_budget_s=detail['l2_started_at_s']+120.-query['time_s'],
            source_cpu_trace=detail['cpu_trace'],source_native_trace=detail['native_trace'],
            source_episode_verdict_unchanged=True,learned_policy_executed=False,actual_student_command_prefix_replayed=True,**cells)
        prefixes[slot['slot_id']]=rt.write_new(output/(slot['slot_id']+'.payload.json'),payload)
    origins=namespace_proof(root);verify_archive();require_gate(rt,request)
    for source in request['sources'].values():
        for ref in source.values():rt.bound_path(ref)
    for target,detail in details.items():
        for key in ('cpu_trace','native_trace'):rt.bound_path(detail[key])
        for key in ('complete','checkpoint'):rt.bound_path(bindings[target]['model'][key])
    result=rt.write_new(output/'RESULT.json',dict(schema=RESULT_SCHEMA,complete=True,request=request_ref,source30=ARCHIVE,
        profile='production_source30',slots=ledger,slot_ledger=ledger_ref,prefixes=prefixes,namespace_origins=origins,
        child_entry=rt.reference(__file__),cwd=str(Path.cwd()),executable=sys.executable,
        completion_gate_verified_before=True,completion_gate_verified_after=True,
        source_episode_verdicts_unchanged=True,optimization_eligible=False))
    return dict(result=result,selected_slots=len(prefixes),fixed_slots=4)


if __name__=='__main__':
    try:
        # Historical helpers may log. Suppress their output rather than retain
        # unbounded histories, array representations or binary error payloads.
        with open(os.devnull,'w') as sink,redirect_stdout(sink),redirect_stderr(sink):summary=main()
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        print(type(error).__name__+': '+str(error)[:2000],file=sys.stderr,flush=True)
        raise SystemExit(1)
