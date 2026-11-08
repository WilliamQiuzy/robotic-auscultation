"""Synthetic local files only; production literal pins are tested independently."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_ausc_act_auxiliary_data import pose_inputs

ROWS = [5760, 5880, 14280, 14400, 28920, 29040]
SLOTS = ['near_entry.first','near_entry.next','before60.previous','before60.anchor','last.previous','last.anchor']


def module():
    name = 'sim.ausc_policy.m2_act_phase_data'
    assert importlib.util.find_spec(name) is not None, 'Separate V5 phase data module required'
    return importlib.import_module(name)


def fixture(tmp_path, valid_counts=None):
    m = module(); rt = m.rt
    (tmp_path/'pose').mkdir()
    reference, _, _, meta = pose_inputs(tmp_path/'pose')
    snap = reference.snapshot()
    patient_path = Path(snap['sources']['patient_receipt']['file'])
    patient = json.loads(patient_path.read_text()); patient.update(identity='p001120',proposal='p001120_p000474_a001230_f0000')
    patient_path.write_text(json.dumps(patient)); patient_ref = rt.reference(patient_path)
    snap['sources']['patient_receipt'] = dict(file=patient_ref['path'],sha256=patient_ref['sha256'])
    snap['sources']['task_manifest'] = deepcopy(snap['sources']['patient_receipt'])
    meta['subject_id'] = 'p001120'
    plan_ref = rt.reference(reference.snapshot()['sources']['fresh_plan']['file'])
    plan = rt.read_bound(plan_ref); plan['free_approach_start_gap_m'] = .025
    plan['patient_provenance'].update(identity='p001120',binding=snap['sources']['patient_receipt'])
    Path(plan_ref['path']).write_text(json.dumps(plan)); plan_ref = rt.reference(plan_ref['path'])
    snap['sources']['fresh_plan'] = dict(file=plan_ref['path'],sha256=plan_ref['sha256'])
    reference = rt.pose.GivenReference.from_snapshot(snap); meta = rt.pose.route_metadata(meta, reference)
    # Keep the fixture's real plan target; local descriptor explicitly declares it.
    subject, target = meta['subject_id'], meta['target_id']; pose = 'p000474_a001230_f0000'
    parent_sources = [dict(schema=m.v4.SOURCE_SCHEMA,kind='planned_ideal_tracking',metadata=deepcopy(meta),
        subject_id=subject,split='train',target_id=target,pose_id=pose,source_id='inherited-0',
        windows=[dict(input_index=i,phase='APPROACH') for i in range(2)],
        bindings=dict(source=plan_ref),**m.old.FLAGS)]
    parent_arrays = dict(source_histories=np.zeros((2,2,17),dtype='<f8'),source_times=np.array([[0,.05],[.05,.1]]),
        relative_setters=np.zeros((2,20,7)),setter_is_pad=np.tile(np.arange(20)>=3,(2,1)),source_indices=np.zeros(2,dtype='<i8'))
    parent_arrays['source_histories'][0,0,0] = -0.
    parent_arrays['relative_setters'][1,0,0] = -0.
    parent_arrays['source_histories'][:,-1,7:10] = [[.003,0,-.025],[0,0,0]]
    parent_dir = tmp_path/'parent'; parent_dir.mkdir()
    path = parent_dir/'AUXILIARY_ARRAYS.npz'; np.savez_compressed(path,**parent_arrays)
    parent_meta = dict(schema=m.v4.SCHEMA,sources=parent_sources,source_table_sha256=rt.digest(parent_sources),
        allowed_train_ids=[subject],optimization_admission=m.v4.ADMISSION,
        slot_ledger=[dict(slot_id='inherited-attempt',status='accepted')],request_refs=[plan_ref],
        provenance=dict(ordered_parent_proof={'previous':'inherited'},parent_archive_validation={'proof':'sealed'}),
        counts={'previous':'immutable'},**m.old.FLAGS)
    pm = rt.write_new(parent_dir/'METADATA.json',parent_meta)
    pc = rt.write_new(parent_dir/'COMPLETE.json',dict(schema=m.v4.SCHEMA,complete=True,metadata=pm,
        arrays=rt.reference(path),metadata_sha256=pm['sha256'],optimization_admission=m.v4.ADMISSION,**m.old.FLAGS))
    control = tmp_path/'probe'; control.mkdir()
    code = tmp_path/'source28'; code.mkdir()
    generator = code/'sim/ausc_policy/m2_student_oracle_proposals.py'; generator.parent.mkdir(parents=True)
    generator.write_text('original definition-only generator fixture\n')
    gm = rt.reference(generator)
    manifest = rt.write_new(code/'SOURCE_MANIFEST.json',[dict(file='sim/ausc_policy/m2_student_oracle_proposals.py',bytes=gm['bytes'],sha256=gm['sha256'])])
    (code/'CODE_COMMIT').write_text('d932993b13030da7873d04a4a9471df99bb836ec\n')
    closure = dict(root=str(code),code_commit='d932993b13030da7873d04a4a9471df99bb836ec',file_count=1,
        source_manifest=manifest,code_commit_file=rt.reference(code/'CODE_COMMIT'))
    operator_path = tmp_path/'producer.py'; operator_path.write_text('sealed producer fixture\n'); operator = rt.reference(operator_path)
    source = {k:rt.write_new(tmp_path/(k+'.json'),{'inherited_producer_proof':k}) for k in ('owner','child','native')}
    source['config'] = rt.write_new(tmp_path/'config.json',{'m2_given_contact_expanded_pilot_deployment':dict(metadata=meta)})
    model = dict(checkpoint=dict(path='/DO-NOT-OPEN/fixture-weights.bin',bytes=99,sha256='a'*64),step=18000)
    geometry = dict(point_native_report=source['native'],point_index=0,point_target_id=target,
        point_phase_precontact_gap_m=.025,point_checked_prepose_gap_m=.025,
        fresh_plan=snap['sources']['fresh_plan'],point_plan_provenance=dict(plan_file=plan_ref['path'],plan_sha256=plan_ref['sha256']),
        plan_free_approach_start_gap_m=.025,precontact_gap_m=.025,desired_skin_relative_position_m=[0.,0.,-.025],
        state17_position_semantics='Rref.T @ (TCP - skin_contact_point)',reference_normal_semantics='minus_reference_pose_Z_axis',
        selection_error_semantics='norm(state17[7:10] - [0,0,-actual_point_gap])')
    eligible = []
    # Six query origins are separated in eligible request order around the 60s window.
    for n,row in enumerate(ROWS):
        eligible.append(dict(physics_row=row,eligible_index=n,request_index=n,query_id=n+1,time_s=row/240.,
            skin_relative_position_norm_m=(.003**2+.025**2)**.5,precontact_relative_position_error_norm_m=.003))
    slots = [dict(e,slot_id=SLOTS[n],slot_index=n,pair_id=('near_entry','before60','last')[n//2],
        intended_role='potential_train' if n<4 else 'reserved_diagnostic',effective_role='potential_train' if n<4 else 'reserved_diagnostic',
        status='selected',pair_is_adjacent_eligible_requests=True,training_eligible=False) for n,e in enumerate(eligible)]
    role = rt.write_new(control/'SLOT_ROLE_LEDGER.json',dict(schema='act7_phase_slot_roles_v1',slots=slots,
        semantic_slot_denominator=6,potential_train_slot_denominator=4,reserved_diagnostic_slot_denominator=2,**m.PROBE_FLAGS))
    queries = []; outcomes = []; h2 = []; times = []; labels = []; masks = []
    oracle_dir = control/'oracle'; oracle_dir.mkdir()
    for n,row in enumerate(ROWS):
        h = np.zeros((2,17)); h[:,:7] = n/100.; h[:,7:10] = [.003,0,-.025]
        q = dict(history=h.tolist(),history_times=[row/240.-.05,row/240.],time_s=row/240.,physics_row=row,
            measured_q=h[-1,:7].tolist(),measured_dq=[0.]*7,metadata=meta,reference_sha256=meta['reference_sha256'],
            context=[meta['episode_id'],target,0,'APPROACH'],hall_N=0.,physical_contact_N=0.,compression_m=0.)
        snapshot = dict(fields=dict(current_position=[.02]*7,current_velocity=[.01]*7,current_acceleration=[.001]*7))
        qr = rt.write_new(control/'queries'/('query-'+str(row)+'.json'),dict(schema='act7_authenticated_measured_query_v1',
            source=source,physics_row=row,query=q,retained_interpolator_snapshot=snapshot,precontact_geometry=geometry,
            skin_relative_position_norm_m=eligible[n]['skin_relative_position_norm_m'],precontact_relative_position_error_norm_m=.003,
            pre_query_three_command_seam=[[.019]*7,[.0195]*7,[.02]*7],
            actual_original_student_absolute_h20=np.full((20,7),9.).tolist(),actual_student_output=True,
            positive_teacher_label=False,future_measured_observations=False,slot_ids=[SLOTS[n]],
            slot_roles=[{k:slots[n][k] for k in ('slot_id','intended_role','effective_role')}],reserved_diagnostic_only=n>=4,**m.PROBE_FLAGS))
        queries.append(qr)
        count = valid_counts[n] if valid_counts is not None else (16 if n==0 else 20)
        cells = np.full((count,7),.4+n/100.)
        proposal = dict(kind='recovery_proposal',physics_executed=False,training_eligible=False,query=q,query_sha256=rt.digest(q),
            preview=dict(cells=cells.tolist(),all_240hz_commands_verified=True,physics_executed=False,
                input_state=snapshot,input_sha256=rt.digest(snapshot)))
        pr = rt.write_new(oracle_dir/('query-'+str(row)+'.json'),proposal)
        outcome = dict(physics_row=row,proposal_accepted=True,valid_proposed_cells=count,computational_proposal=pr,
            actions_executed=False,physics_qualified=False,training_eligible=False)
        rt.write_new(oracle_dir/('outcome-'+str(row)+'.json'),outcome); outcomes.append(outcome)
        relative = np.zeros((20,7)); relative[:count] = cells-h[-1,:7]
        h2.append(h); times.append(q['history_times']); labels.append(relative); masks.append(np.arange(20)>=count)
    path = oracle_dir/'UNEXECUTED_PROPOSALS.npz'
    np.savez_compressed(path,source_observed_histories=np.asarray(h2),source_observation_times=np.asarray(times),
        source_physics_rows=np.array(ROWS,dtype=np.int64),proposed_relative_setters=np.asarray(labels),proposed_is_pad=np.asarray(masks))
    oa = rt.reference(path)
    plan_config = rt.write_new(tmp_path/'plan-config.json',{'same_original_geometry':True})
    rt.write_new(oracle_dir/'REQUEST.json',dict(source=source,plan_config=plan_config,physics_rows=ROWS,generator=gm,
        actions_executed=False,physics_qualified=False,training_eligible=False))
    oc = rt.write_new(oracle_dir/'COMPLETE.json',dict(schema='m2_student_cpu_oracle_diagnostic_v1',complete=True,outcomes=outcomes,
        attempted_queries=6,accepted_proposals=6,arrays=oa,actions_executed=False,physics_qualified=False,training_eligible=False,
        gpu_launched=False,source_subject_id=subject,source_split='train',source_episode_verdict_unchanged=True))
    frozen = rt.write_new(control/'FROZEN_REQUEST.json',dict(schema=m.PROBE_SCHEMA,operator=operator,source28=closure,
        source=source,source_metadata=meta,model=model,generator=gm,plan_config=plan_config,precontact_geometry=geometry,
        slots=slots,slot_role_ledger=role,diagnostic_queries=queries,physics_rows=ROWS,eligible_queries=eligible,
        eligible_request_ledger=[dict(e,eligible=True) for e in eligible],semantic_slot_denominator=6,distinct_query_denominator=6,
        input_refs=[plan_ref],**m.PROBE_FLAGS))
    final = rt.write_new(control/'FINAL_SLOT_LEDGER.json',dict(schema='act7_phase_slot_roles_v1',
        slots=[dict(s,proposal_attempted=True,proposal_accepted=True,proposal_outcome=o) for s,o in zip(slots,outcomes)],
        frozen_slot_role_ledger=role,frozen_request=frozen,semantic_slot_denominator=6,**m.PROBE_FLAGS))
    complete = rt.write_new(control/'COMPLETE.json',dict(schema=m.PROBE_SCHEMA,operator=operator,source28=closure,source28_end=closure,
        source=source,model=model,frozen_request=frozen,slot_role_ledger=role,final_slot_ledger=final,oracle_result=oc,
        source_native_physics_steps=29041,semantic_slot_denominator=6,potential_train_slot_denominator=4,
        reserved_diagnostic_slot_denominator=2,eligible_query_count=6,distinct_query_denominator=6,proposals_accepted=6,
        proposal_rejections=0,potential_train_distinct_accepted=4,reserved_distinct_accepted=2,**m.PROBE_FLAGS))
    seal = rt.write_new(control/'OUTPUT_MANIFEST.json',dict(schema=m.PROBE_SCHEMA,
        files=[rt.reference(p) for p in sorted(control.rglob('*')) if p.is_file()],**m.PROBE_FLAGS))
    inputs = m.LocalFixtureInputs(parent_complete=pc,parent_metadata=pm,parent_arrays=rt.reference(parent_dir/'AUXILIARY_ARRAYS.npz'),
        producer_complete=complete,producer_manifest=seal,producer_operator=operator,frozen_request=frozen,final_slot_ledger=final,
        oracle_complete=oc,oracle_arrays=oa,source28_manifest=manifest,parent_rows=2,parent_sources=1,parent_slots=1,
        source28_files=1,subject_id=subject,pose_id=pose,target_id=target)
    return m, SimpleNamespace(inputs=inputs,allowed={subject},control=control,parent_arrays=parent_arrays,parent_meta=parent_meta,
        source=source,meta=meta,queries=queries,root=tmp_path)


def export(m,f):return m.export_from_phase_probe(f.root/'output',allowed_train_ids=f.allowed,inputs=f.inputs)


def mutate(f,m,path,change):
    """Reseal fixture graph so negatives exercise semantic checks, not stale hashes."""
    path=Path(path); old=m.rt.reference(path); value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value));new=m.rt.reference(path)
    pending=[(old,new)]
    for before,after in pending:
        for p in sorted(f.control.rglob('*.json')):
            if p==Path(after['path']) or p.name=='OUTPUT_MANIFEST.json':continue
            obj=json.loads(p.read_text())
            def repl(v):
                if v==before:return deepcopy(after)
                if isinstance(v,dict):return {k:repl(x) for k,x in v.items()}
                if isinstance(v,list):return [repl(x) for x in v]
                return v
            updated=repl(obj)
            if updated!=obj:
                prior=m.rt.reference(p);p.write_text(json.dumps(updated));pending.append((prior,m.rt.reference(p)))
    seal=f.control/'OUTPUT_MANIFEST.json';seal.write_text(json.dumps(dict(schema=m.PROBE_SCHEMA,
        files=[m.rt.reference(p) for p in sorted(f.control.rglob('*')) if p.is_file() and p!=seal],**m.PROBE_FLAGS)))
    updates={}
    for key in ('producer_complete','producer_manifest','frozen_request','final_slot_ledger','oracle_complete','oracle_arrays'):
        ref=getattr(f.inputs,key);updates[key]=m.rt.reference(ref['path'])
    f.inputs=replace(f.inputs,**updates)


def test_production_descriptor_literals_are_independent_and_v4_immutable():
    m=module();p=m.PRODUCTION_INPUTS
    assert p.parent_complete==dict(path='/opt/wam-runtime/fast-wam/datasets/act6-terminal-full-chunks-20261107-v1/COMPLETE.json',bytes=1129,sha256='d5ec96f999498dd3133dc8984bc79b44ad2eb00be24f14bc1bebb269045bf7ce')
    assert p.parent_metadata['bytes']==13439079 and p.parent_metadata['sha256']=='133a78792793f071f0637de8673db1511b0174ef12209336635b0cfe320cb049'
    assert p.parent_arrays['bytes']==8338733 and p.parent_arrays['sha256']=='d1360dfcaab41741e7f17ce70c38f18506d4bccef883dc81e6971a7bb13075ba'
    assert p.producer_operator['bytes']==39293 and p.producer_operator['sha256']=='11054e4fc24a531a6cefe49202b29b286926c54078430601a254806338655e4e'
    assert p.producer_complete['sha256']=='146cefaeea1094b1fcd47b3dfd28bb78ca5f7e45475c3b3f299691c7b3e2a78e'
    assert p.producer_manifest['sha256']=='e07021c7b20aa69992c1c6665eadd34596811caf6c53e049fe84c9ab81c5e87a'
    assert p.frozen_request['sha256']=='e168d985005843a33d68fa61b480dc850e21f8f93852a14cef63417c17dcb355'
    assert p.final_slot_ledger['sha256']=='2c342776f4e5d19c02be33ca73a10b901efc126e74b5ea1133c0a8ab40dae4e4'
    assert p.oracle_complete['sha256']=='ebd92bae3816ff48437e43c6ad031e31019e84ba2644e0a2ddb576e145769a40'
    assert p.oracle_arrays['sha256']=='a5cca28b88d33021ccd53706a140eba38494f3226dc7bc7767e423c00b64a046'
    assert (p.parent_rows,p.parent_sources,p.parent_slots)==(35988,940,1115)
    assert p.source28_files==1673 and p.source28_manifest['sha256']=='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c'
    assert m.SCHEMA!=m.v4.SCHEMA and m.SOURCE_SCHEMA!=m.v4.SOURCE_SCHEMA


def test_four_training_origins_two_reserved_all_masks_parent_bytes_roundtrip(tmp_path,monkeypatch):
    m,f=fixture(tmp_path)
    monkeypatch.setattr(m.v4,'load_preparation',lambda *a,**k:pytest.fail('No historical replay'))
    monkeypatch.setattr(m.v4.frozen26,'load_parent',lambda *a,**k:pytest.fail('No historical factory'))
    ref=export(m,f);value=m.load_preparation(ref,f.allowed,inputs=f.inputs);s=value['samples']
    assert len(s['source_indices'])==6 and len(s['sources'])==5
    assert [x['frozen_physics_row'] for x in s['sources'][1:]]==ROWS[:4]
    assert [x['physics_row'] for x in value['phase_slot_ledger']]==ROWS
    for key,a in f.parent_arrays.items():assert s[key][:2].dtype==a.dtype and s[key][:2].tobytes()==a.tobytes()
    assert np.signbit(s['source_histories'][0,0,0]) and np.signbit(s['relative_setters'][1,0,0])
    assert np.count_nonzero(~s['setter_is_pad'][2])==16
    assert value['counts']['new_valid_computed_future_cells']==76
    assert value['counts']['new_measured_query_origins']==4 and value['counts']['reserved_diagnostic_sources']==2
    assert value['counts']['new_physical_demonstrations']==0
    assert value['samples']['optimization_admission']['computational_optimization_eligible'] is True
    assert value['provenance']['parent_admission']['mode']=='inherited_hash_bound_closed_snapshot'
    assert s['relative_setters'][2,0,0]==pytest.approx(.4) and s['relative_setters'][2,0,0]!=9.
    assert m.load_preparation(ref,f.allowed,inputs=f.inputs)['counts']==value['counts']
    with pytest.raises(ValueError,match='Fresh'):export(m,f)


@pytest.mark.parametrize('fault',['role','clock','history','action','reference','target','mask','proposal','shape','row_order','seam','gap','manifest'])
def test_semantic_or_binary_corruption_is_rejected(tmp_path,fault):
    m,f=fixture(tmp_path)
    if fault in ('mask','shape','row_order'):
        path=Path(f.inputs.oracle_arrays['path'])
        with np.load(path,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
        if fault=='mask':arrays['proposed_is_pad'][0,2]=True
        elif fault=='shape':arrays['source_observed_histories']=arrays['source_observed_histories'][:,:1]
        else:arrays['source_physics_rows']=arrays['source_physics_rows'][::-1]
        np.savez_compressed(path,**arrays)
        # Reseal array bindings by replacing in complete and then rest of graph.
        mutate(f,m,f.inputs.oracle_complete['path'],lambda v:v.update(arrays=m.rt.reference(path)))
    elif fault=='manifest':
        path=Path(f.inputs.producer_manifest['path']);value=json.loads(path.read_text());value['files'].pop();path.write_text(json.dumps(value));f.inputs=replace(f.inputs,producer_manifest=m.rt.reference(path))
    elif fault=='role':mutate(f,m,f.inputs.frozen_request['path'],lambda v:v['slots'][0].update(effective_role='reserved_diagnostic'))
    elif fault in ('clock','history','reference','target','seam','gap'):
        def change(v):
            if fault=='clock':v['query']['history_times'][0]-=.01
            elif fault=='history':v['query']['history'][-1][0]+=.1
            elif fault=='reference':v['query']['reference_sha256']='0'*64
            elif fault=='target':v['query']['metadata']['target_id']='L5'
            elif fault=='seam':v['retained_interpolator_snapshot']['fields']['current_velocity'][0]+=.1
            else:v['precontact_geometry']['precontact_gap_m']=.04
        mutate(f,m,f.queries[0]['path'],change)
    else:
        path=f.control/'oracle'/('query-'+str(ROWS[0])+'.json')
        mutate(f,m,path,lambda v:v['preview'].update(cells=[[9.]*7]*16) if fault=='action' else v.update(query_sha256='0'*64))
    with pytest.raises((ValueError,KeyError)):export(m,f)


def test_val_identity_rejected_before_query_or_binary_payload(tmp_path,monkeypatch):
    m,f=fixture(tmp_path)
    mutate(f,m,f.inputs.frozen_request['path'],lambda v:v['source_metadata'].update(split='validation'))
    original=m.rt.read_bound
    def guard(ref):
        assert '/queries/' not in ref['path'] and '/oracle/' not in ref['path'], 'Identity must precede payload reads'
        return original(ref)
    monkeypatch.setattr(m.rt,'read_bound',guard)
    with pytest.raises(ValueError,match='TRAIN'):export(m,f)


def test_pack_uses_external_native_stats_and_ignores_padding_poison(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs);s=v['samples']
    norm=dict(schema=m.rt.given.NORMALIZATION_SCHEMA,task_id=m.rt.pose.TASK_ID,action_semantics_id=m.rt.given.ACTION_SEMANTICS_ID,
        fitted_split='train',observation_contract=m.rt.pose.observation_contract(),state_mean=[0.]*17,state_std=[1.]*17,action_mean=[0.]*7,action_std=[1.]*7)
    before=deepcopy(norm)
    s['relative_setters'][2,16:]=np.nan
    batch=m.pack_batch(s,norm,[2])
    assert batch['action'].shape==(1,20,7) and batch['observation.environment_state'].shape==(1,27)
    assert np.isfinite(batch['action']).all() and norm==before
    assert batch['action_is_pad'][0].tolist()==[False]*16+[True]*4
    s['optimization_admission']=dict(s['optimization_admission'],computational_optimization_eligible=False)
    with pytest.raises(ValueError,match='admitted'):m.pack_batch(s,norm,[2])


def test_terminal_inventory_uses_bound_actual_gap_prepose_not_skin_or_40mm(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    missing=(next(x for x in m.scope.TRAIN16_IDS if x not in f.allowed),'pose-missing','L5')
    actual=(f.meta['subject_id'],f.inputs.pose_id,f.meta['target_id'])
    result=m.terminal_pool_inventory(v,allowed_train_ids=f.allowed,expected_strata=[actual,missing])
    assert result['eligible_row_indices']==[0]
    assert result['rows'][0]['precontact_error_norm_m']==pytest.approx(.003)
    assert result['rows'][0]['skin_relative_norm_m']>.025
    assert result['bucket_counts']['body'][f.meta['subject_id']]==1
    assert result['missing_strata']==[list(missing)] and not result['invalid_strata']
    assert result['diagnostic_threshold_m']==.006


def test_inventory_invalid_plan_is_visible_and_never_defaulted(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    p=Path(v['sources'][0]['bindings']['source']['path']);obj=json.loads(p.read_text());obj.pop('free_approach_start_gap_m');p.write_text(json.dumps(obj))
    result=m.terminal_pool_inventory(v,allowed_train_ids=f.allowed)
    assert result['eligible_row_indices']==[] and len(result['invalid_strata'])==1


def test_rejected_permitted_row_preserves_six_outcomes_but_never_admits_smaller_cohort(tmp_path):
    m,f=fixture(tmp_path)
    row=ROWS[0];outcome=dict(physics_row=row,proposal_accepted=False,rejection='ValueError: original oracle abstained',
        actions_executed=False,physics_qualified=False,training_eligible=False)
    mutate(f,m,f.control/'oracle'/('outcome-'+str(row)+'.json'),lambda v:(v.clear(),v.update(outcome)))
    path=Path(f.inputs.oracle_arrays['path'])
    with np.load(path,allow_pickle=False) as a:arrays={k:a[k][1:].copy() for k in a.files}
    np.savez_compressed(path,**arrays)
    def reject(v):v['outcomes'][0]=outcome;v.update(accepted_proposals=5,arrays=m.rt.reference(path))
    mutate(f,m,f.inputs.oracle_complete['path'],reject)
    mutate(f,m,f.inputs.final_slot_ledger['path'],lambda v:v['slots'][0].update(proposal_accepted=False,proposal_outcome=outcome,rejection=outcome['rejection']))
    mutate(f,m,f.inputs.producer_complete['path'],lambda v:v.update(proposals_accepted=5,proposal_rejections=1,potential_train_distinct_accepted=3))
    value=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    assert value['counts']['new_measured_query_origins']==3
    assert value['counts']['reserved_diagnostic_sources']==2 and len(value['phase_slot_ledger'])==6
    assert value['phase_slot_ledger'][0]['rejection']==outcome['rejection']
    assert value['samples']['optimization_admission']['computational_optimization_eligible'] is False
    assert value['counts']['optimization_admitted'] is False


def test_missing_and_duplicate_role_inventory_retains_fixed_denominators(tmp_path):
    m,f=fixture(tmp_path);header=m._probe_header(f.inputs,f.allowed)
    h=deepcopy(header);e=h['frozen']['eligible_queries'][:2]
    h['frozen']['eligible_queries']=e
    indices=[0,1,0,1,0,1]
    for n,i in enumerate(indices):
        s=h['frozen']['slots'][n];s.update(e[i]);s['effective_role']='duplicate_excluded' if n<4 else 'reserved_diagnostic'
        s['status']='duplicate_excluded' if n<4 else 'selected'
        if n<4:s['exclusion_reason']='reserved_diagnostic_role_wins'
    h['role']['slots']=deepcopy(h['frozen']['slots']);h['final']['slots']=deepcopy(h['frozen']['slots'])
    h['root'].update(eligible_query_count=2,distinct_query_denominator=2)
    h['frozen'].update(physics_rows=ROWS[:2],distinct_query_denominator=2)
    slots,rows=m._roles(h['frozen'],h['role'],h['final'],h['root'])
    assert rows==ROWS[:2] and len(slots)==6 and sum(s['effective_role']=='potential_train' for s in slots)==0
    for s in h['frozen']['slots']:
        for k in ('physics_row','eligible_index','request_index','query_id','time_s'):s[k]=None
        s.update(effective_role='missing',status='missing',pair_is_adjacent_eligible_requests=False)
    h['role']['slots']=deepcopy(h['frozen']['slots']);h['final']['slots']=deepcopy(h['frozen']['slots'])
    h['frozen'].update(eligible_queries=[],physics_rows=[],distinct_query_denominator=0)
    h['root'].update(eligible_query_count=0,distinct_query_denominator=0)
    slots,rows=m._roles(h['frozen'],h['role'],h['final'],h['root'])
    assert len(slots)==6 and rows==[] and all(s['status']=='missing' for s in slots)


def test_default_inventory_exposes_existing_strata_without_any_near_prepose_rows(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    v['samples']['source_histories'][:2,-1,7:10]=[0.,0.,0.]
    result=m.terminal_pool_inventory(v,allowed_train_ids=f.allowed)
    assert result['eligible_row_indices']==[]
    assert result['missing_strata']==[[f.meta['subject_id'],f.inputs.pose_id,f.meta['target_id']]]


def test_inventory_uses_30mm_gap_and_counts_diverse_existing_body_pose_target_sources(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    source=deepcopy(v['sources'][0]);source.update(subject_id='p000758',pose_id='other-pose',target_id='L2',source_id='other-existing')
    meta=source['metadata'];meta.update(subject_id='p000758',target_id='L2')
    snap=meta['given_reference'];snap['target_id']='L2'
    plan=m.rt.read_bound(source['bindings']['source']);plan.update(name='L2',free_approach_start_gap_m=.03)
    ref=m.rt.write_new(tmp_path/'other-plan.json',plan);source['bindings']['source']=ref
    snap['sources']['fresh_plan']=dict(file=ref['path'],sha256=ref['sha256'])
    meta=m.rt.pose.route_metadata(meta,m.rt.pose.GivenReference.from_snapshot(snap));source['metadata']=meta
    source['kind']='conditional_ideal_continuation';source['windows']=[dict(input_index=1,phase='APPROACH')]
    v['sources'].append(source);s=v['samples'];s['source_histories']=np.concatenate([s['source_histories'],np.zeros((1,2,17))])
    s['source_histories'][-1,-1,7:10]=[.004,0,-.03]
    s['source_indices']=np.concatenate([s['source_indices'],[len(v['sources'])-1]])
    result=m.terminal_pool_inventory(v,allowed_train_ids=f.allowed|{'p000758'})
    assert result['eligible_row_indices']==[0,6]
    assert result['bucket_counts']['body']=={f.meta['subject_id']:1,'p000758':1}
    assert result['bucket_counts']['pose']=={f.inputs.pose_id:1,'other-pose':1}
    assert result['bucket_counts']['target']=={f.meta['target_id']:1,'L2':1}
    assert result['rows'][-1]['actual_gap_m']==.03 and result['rows'][-1]['precontact_error_norm_m']==pytest.approx(.004)


@pytest.mark.parametrize('identity,split',[('p000923','validation'),('p000519','train'),('p000439','train')])
def test_unpermitted_sources_gate_before_all_queries_or_npz(tmp_path,monkeypatch,identity,split):
    m,f=fixture(tmp_path)
    mutate(f,m,f.inputs.frozen_request['path'],lambda v:v['source_metadata'].update(subject_id=identity,split=split))
    original=m.rt.read_bound
    def guard(ref):
        assert '/queries/' not in ref['path'] and '/oracle/' not in ref['path']
        return original(ref)
    monkeypatch.setattr(m.rt,'read_bound',guard)
    with pytest.raises(ValueError,match='TRAIN'):export(m,f)


def test_metadata_does_not_decode_any_npz_and_load_does_not_rewrite(tmp_path,monkeypatch):
    m,f=fixture(tmp_path);ref=export(m,f)
    before={str(p):p.read_bytes() for p in (f.root/'output').iterdir()}
    with monkeypatch.context() as patch:
        patch.setattr(m.np,'load',lambda *a,**k:pytest.fail('Metadata is binary hash-only'))
        complete,meta=m.metadata(ref,f.allowed,inputs=f.inputs)
    m.load_preparation(ref,f.allowed,inputs=f.inputs)
    assert before=={str(p):p.read_bytes() for p in (f.root/'output').iterdir()}
    assert complete['schema']==meta['schema']==m.SCHEMA


def test_modified_production_descriptor_cannot_silently_become_a_fixture():
    m=module()
    with pytest.raises(ValueError,match='production'):m._inputs(replace(m.PRODUCTION_INPUTS,parent_rows=2))
    with pytest.raises(ValueError,match='typed'):m._inputs({'profile':'production'})


def test_load_preserves_signed_zero_in_nonarray_parent_ledgers(tmp_path):
    m,f=fixture(tmp_path)
    # An exact closed parent may contain signed-zero descriptive values too.
    pm=deepcopy(f.parent_meta);pm['slot_ledger'][0]['numeric_note']=-0.
    path=Path(f.inputs.parent_metadata['path']);path.write_text(json.dumps(pm));pmref=m.rt.reference(path)
    pc=m.rt.read_bound(f.inputs.parent_complete);pc.update(metadata=pmref,metadata_sha256=pmref['sha256'])
    cp=Path(f.inputs.parent_complete['path']);cp.write_text(json.dumps(pc))
    f.inputs=replace(f.inputs,parent_metadata=pmref,parent_complete=m.rt.reference(cp))
    ref=export(m,f);complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata'])
    meta['slot_ledger'][0]['numeric_note']=0.
    mp=Path(complete['metadata']['path']);mp.write_text(json.dumps(meta));mr=m.rt.reference(mp)
    complete.update(metadata=mr,metadata_sha256=mr['sha256']);cp=Path(ref['path']);cp.write_text(json.dumps(complete));ref=m.rt.reference(cp)
    with pytest.raises(ValueError,match='prefix|ledger'):m.load_preparation(ref,f.allowed,inputs=f.inputs)


def test_no_fitting_training_or_process_dispatch_surface():
    import ast
    m=module();tree=ast.parse(Path(m.__file__).read_text())
    banned={'fit','train','load_model','load_state_dict','authenticate_queries','authenticate_sources','run_owner','diagnose','Popen','run','repr','load_parent','import_student_oracle'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            assert name not in banned


def test_parent_source_slot_request_prefix_has_canonical_byte_proof(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    assert v['provenance']['ordered_parent_metadata_proof']=={
        k:m.rt.digest(f.parent_meta[k]) for k in ('sources','slot_ledger','request_refs')}


@pytest.mark.parametrize('fault',['gap_missing','target','reference_hash','source_plan_path'])
def test_terminal_inventory_rejects_invalid_bound_gap_target_and_reference(tmp_path,fault):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs);source=v['sources'][0]
    plan=m.rt.read_bound(source['bindings']['source'])
    if fault=='gap_missing':plan.pop('free_approach_start_gap_m')
    elif fault=='target':plan['name']='L5'
    ref=m.rt.write_new(tmp_path/'bad-plan.json',plan)
    if fault in ('gap_missing','target'):
        source['bindings']['source']=ref;source['metadata']['given_reference']['sources']['fresh_plan']=dict(file=ref['path'],sha256=ref['sha256'])
        reference=m.rt.pose.GivenReference.from_snapshot(source['metadata']['given_reference'])
        source['metadata']=m.rt.pose.route_metadata(source['metadata'],reference)
    elif fault=='reference_hash':source['metadata']['reference_sha256']='0'*64
    else:source['bindings']['source']=ref
    result=m.terminal_pool_inventory(v,allowed_train_ids=f.allowed)
    assert result['eligible_row_indices']==[] and len(result['invalid_strata'])==1


def test_schema_and_validation_normalization_do_not_fall_through(tmp_path):
    m,f=fixture(tmp_path);v=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs);s=v['samples']
    norm=dict(schema=m.rt.given.NORMALIZATION_SCHEMA,task_id=m.rt.pose.TASK_ID,action_semantics_id=m.rt.given.ACTION_SEMANTICS_ID,
        fitted_split='validation',observation_contract=m.rt.pose.observation_contract(),state_mean=[0.]*17,state_std=[1.]*17,action_mean=[0.]*7,action_std=[1.]*7)
    with pytest.raises(ValueError,match='TRAIN'):m.pack_batch(s,norm,[2])
    norm['fitted_split']='train';s['schema']=m.v4.SCHEMA
    with pytest.raises(ValueError,match='V5'):m.pack_batch(s,norm,[2])


def test_actual_six_query_mask_pattern_preserves_58_training_cells_and_27_reserved_cells(tmp_path):
    pattern=[16,16,13,13,13,14]
    m,f=fixture(tmp_path,valid_counts=pattern)
    production_before=deepcopy(m.PRODUCTION_INPUTS)
    value=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs);samples=value['samples']
    assert value['counts']['new_measured_query_origins']==4
    assert value['counts']['new_valid_computed_future_cells']==58
    assert value['counts']['reserved_diagnostic_sources']==2
    assert [slot['proposal_outcome']['valid_proposed_cells'] for slot in value['phase_slot_ledger']]==pattern
    assert sum(slot['proposal_outcome']['valid_proposed_cells'] for slot in value['phase_slot_ledger'][4:])==27
    assert [source['frozen_physics_row'] for source in value['sources'][1:]]==ROWS[:4]
    assert len(samples['source_indices'])==f.inputs.parent_rows+4
    tail=slice(f.inputs.parent_rows,None)
    assert np.count_nonzero(~samples['setter_is_pad'][tail],axis=1).tolist()==pattern[:4]
    assert int(np.count_nonzero(~samples['setter_is_pad'][tail]))==sum(pattern[:4])==58
    assert np.all(samples['relative_setters'][tail][samples['setter_is_pad'][tail]]==0.)
    # Poison only masked future cells after admission; the batch packer must
    # preserve all valid teacher cells and never expose padding or reserved rows.
    raw=samples['relative_setters'][tail].copy()
    samples['relative_setters'][tail][samples['setter_is_pad'][tail]]=np.nan
    norm=dict(schema=m.rt.given.NORMALIZATION_SCHEMA,task_id=m.rt.pose.TASK_ID,
        action_semantics_id=m.rt.given.ACTION_SEMANTICS_ID,fitted_split='train',
        observation_contract=m.rt.pose.observation_contract(),state_mean=[0.]*17,
        state_std=[1.]*17,action_mean=[0.]*7,action_std=[1.]*7)
    batch=m.pack_batch(samples,norm,list(range(f.inputs.parent_rows,f.inputs.parent_rows+4)))
    assert np.isfinite(batch['action']).all()
    assert np.count_nonzero(~batch['action_is_pad'],axis=1).tolist()==pattern[:4]
    assert np.all(batch['action'][batch['action_is_pad']]==0.)
    np.testing.assert_array_equal(batch['action'][~batch['action_is_pad']],raw[~batch['action_is_pad']].astype(np.float32))
    assert m.PRODUCTION_INPUTS==production_before
