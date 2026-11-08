"""Auxiliary supervision stays separate from native physical demonstrations."""
from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest

from test_ausc_given_contact_capture import new_inputs

POSE='p000474_a001230_f0000'
OTHER_POSE='p000476_a001231_f0023'


def pose_inputs(tmp_path,pose=POSE):
    from sim.ausc_policy import m2_given_contact_runtime as rt
    from test_ausc_given_contact_pose import bound
    ref,rows,audit,meta=new_inputs(tmp_path)
    value=ref.snapshot();patient=rt.pose.bound_json(value['sources']['patient_receipt'])
    patient['proposal']=meta['subject_id']+'_'+pose
    patient_ref=bound(tmp_path,'posed-patient.json',patient)
    plan=rt.pose.bound_json(value['sources']['fresh_plan']);plan['patient_provenance']['binding']=patient_ref
    value['sources'].update(patient_receipt=patient_ref,task_manifest=patient_ref,
        fresh_plan=bound(tmp_path,'posed-plan.json',plan))
    ref=rt.pose.GivenReference.from_snapshot(value);meta=rt.pose.route_metadata(meta,ref)
    return ref,rows,audit,meta


def module():
    name='sim.ausc_policy.m2_act_auxiliary_data'
    assert importlib.util.find_spec(name) is not None, 'Separate auxiliary data module required'
    return importlib.import_module(name)


def fixture(tmp_path, n=3):
    m=module()
    from sim.ausc_policy import m2_given_contact_runtime as rt
    ref,_,_,meta=pose_inputs(tmp_path)
    binding=rt.reference(ref.snapshot()['sources']['fresh_plan']['file'])
    h=np.arange(n*34,dtype=float).reshape(n,2,17)/1000
    times=np.column_stack((np.arange(n)*.05,np.arange(n)*.05+.05))
    cells=np.full((n+1,7),.7)
    a=np.zeros((n,20,7))
    for i in range(n):a[i,:2]=cells[i:i+2]-h[i,-1,:7]
    source=m.make_source('planned_ideal_tracking',meta,POSE,
        dict(source=binding,geometry=binding,generator=rt.reference(m.__file__)),
        [dict(input_index=i,phase='APPROACH') for i in range(n)])
    attach_schedule(tmp_path/'raw-schedule',source,cells,h[0,-1,:7])
    arrays=dict(source_histories=h,source_times=times,relative_setters=a,
        setter_is_pad=np.tile(np.arange(20)>=2,(n,1)),source_indices=np.zeros(n,dtype=np.int64))
    return m.bind_samples(arrays,[source],{meta['subject_id']})


def attach_schedule(output,source,cells,q):
    m=module()
    from test_ausc_given_contact_handoff import FixtureTrajectory
    initial=m.rt.handoff.input_snapshot(FixtureTrajectory(q,dt=m.rt.handoff.DT,**m.rt.handoff.PLANNING_LIMITS))
    preview=dict(input_state=initial,input_sha256=m.rt.digest(initial),cells=np.asarray(cells).tolist(),
        all_240hz_commands_verified=True,physics_executed=False)
    bound=m.write_planned_schedule(output,preview,source['metadata'],source['pose_id'],{source['subject_id']})
    header=m.rt.read_bound(bound)
    source['bindings']['raw_schedule']=bound
    source['preview']={**{k:v for k,v in preview.items() if k!='cells'},
        **{k:header[k] for k in ('raw_setter_cells','raw_setter_sha256')}}


@pytest.mark.parametrize('split',['validation','test'])
def test_split_rejected_before_any_numeric_array_access(tmp_path,split):
    m=module();s=fixture(tmp_path);sources=deepcopy(s['sources']);sources[0]['split']=split
    class Unreadable(dict):
        def __getitem__(self,key):
            if key in m.ARRAY_KEYS:pytest.fail('Numeric data read before TRAIN eligibility')
            return super().__getitem__(key)
    with pytest.raises(ValueError,match='TRAIN'):
        m.validate_samples(Unreadable(s),sources,s['allowed_train_ids'])


@pytest.mark.parametrize('fault',['source_index','clock','target','geometry','padding','mask','binding'])
def test_arrays_and_source_bindings_cannot_drift(tmp_path,fault):
    m=module();s=fixture(tmp_path)
    if fault=='source_index':s['source_indices'][0]=1
    elif fault=='clock':s['source_times'][0,0]+=.001
    elif fault=='target':s['sources'][0]['target_id']='R3'
    elif fault=='geometry':s['sources'][0]['metadata']['geometry_id']='old'
    elif fault=='padding':s['relative_setters'][0,2]=1.
    elif fault=='mask':s['setter_is_pad'][0,0]=True
    else:s['sources'][0]['bindings']['source']['sha256']='0'*64
    with pytest.raises(ValueError):m.validate_samples(s,s['sources'],s['allowed_train_ids'])


def test_pack_matches_native_numeric_math_and_native_reader_refuses_route(tmp_path):
    m=module();s=fixture(tmp_path)
    from sim.ausc_policy import m2_given_contact_act as native
    from sim.ausc_policy import m2_act as shared
    native_samples=dict(schema=native.MASKED_DATASET_SCHEMA,task_id=native.TASK_ID,
        observation_contract=native.observation_contract(),action_semantics_id=native.ACTION_SEMANTICS_ID,
        states=s['source_histories'][:,1],histories=s['source_histories'],actions=s['relative_setters'],
        action_is_pad=s['setter_is_pad'],label_protocol=shared.MASKED_LABEL_PROTOCOL)
    stats=native.fit_normalization(native_samples,split='train')
    got=m.pack_batch(s,stats,np.array([2,0]));want=native.pack_batch(native_samples,stats,[2,0],training=False)
    for key in want:np.testing.assert_array_equal(got[key],want[key])
    with pytest.raises(ValueError):native.require_samples(s)
    stats['fitted_split']='validation'
    with pytest.raises(ValueError):m.pack_batch(s,stats,[0])


def test_export_load_hashes_and_retains_rejected_slots_before_npz(tmp_path,monkeypatch):
    m=module();s=fixture(tmp_path)
    ledger=[dict(slot_id='accepted',subject_id='p000439',pose_id=POSE,target_id='L1',
        kind='planned_ideal_tracking',status='accepted',source_id=s['sources'][0]['source_id']),
        dict(slot_id='rejected',subject_id='p000439',pose_id=POSE,target_id='R3',
        kind='planned_ideal_tracking',status='rejected',rejection='original collision blocked')]
    ledger[0]['slot_id']='planned:p000439:'+POSE+':L1'
    ledger[1]['slot_id']='planned:p000439:'+POSE+':R3'
    from sim.ausc_policy import m2_given_contact_runtime as rt
    request=rt.write_new(tmp_path/'scene_manifest.json',dict(schema='m2_rebuild_scene_manifest_v1',purpose='capture',
        targets=['L1','R3'],subject_ids=['p000439'],requested_scenes=1,requested_target_slots=2,
        scenes=[dict(subject_id='p000439',split='train',cohort='train',pose_id=POSE,
            plans=dict(L1={'plan':s['sources'][0]['bindings']['source']},R3={'status':'unavailable','reason':'blocked'}))]))
    with pytest.raises(ValueError,match='slot'):
        m.export_preparation(tmp_path/'shrunk',s,ledger[:1],s['allowed_train_ids'],request_refs=[request])
    ref=m.export_preparation(tmp_path/'export',s,ledger,s['allowed_train_ids'],request_refs=[request])
    got=m.load_preparation(ref,s['allowed_train_ids'])
    assert got['slot_ledger']==ledger and got['counts']['planned_ideal_tracking']['samples']==3
    for key in m.ARRAY_KEYS:np.testing.assert_array_equal(got['samples'][key],s[key])
    complete=json.loads(open(ref['path']).read());metadata_path=complete['metadata']['path']
    data=json.loads(open(metadata_path).read());data['sources'][0]['split']='validation'
    open(metadata_path,'w').write(json.dumps(data))
    from sim.ausc_policy import m2_given_contact_runtime as rt
    complete['metadata']=rt.reference(metadata_path)
    new=rt.write_new(tmp_path/'tampered.json',complete)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('VAL arrays must stay unopened'))
    with pytest.raises(ValueError,match='TRAIN'):m.load_preparation(new,s['allowed_train_ids'])


def test_sampler_weights_available_body_pose_target_groups_not_window_counts(tmp_path):
    m=module();s=fixture(tmp_path,n=40)
    other=tmp_path/'other';other.mkdir();ref,_,_,meta=pose_inputs(other,OTHER_POSE)
    source=deepcopy(s['sources'][0]);source['pose_id']=OTHER_POSE;source['metadata']=meta
    source['windows']=[dict(input_index=0,phase='APPROACH')]
    binding=m.rt.reference(ref.snapshot()['sources']['fresh_plan']['file'])
    source['bindings'].update(source=binding,geometry=binding)
    attach_schedule(other/'raw-schedule',source,np.full((2,7),.7),s['source_histories'][0,-1,:7])
    arrays={k:v.copy() for k,v in s.items() if k in m.ARRAY_KEYS}
    arrays={k:np.concatenate([v,v[:1]],axis=0) for k,v in arrays.items()}
    arrays['source_indices'][-1]=1
    s=m.bind_samples(arrays,[s['sources'][0],source],s['allowed_train_ids'])
    ids=m.sample_indices(np.random.default_rng(1),s,s['sources'],2000)
    assert .45<np.mean(ids==40)<.55


def oracle_fixture(tmp_path):
    m=module();from sim.ausc_policy import m2_given_contact_runtime as rt
    ref,_,_,meta=pose_inputs(tmp_path)
    archive=tmp_path/'frozen-source';(archive/'sim/ausc_policy').mkdir(parents=True)
    generator=archive/'sim/ausc_policy/m2_student_oracle_proposals.py';generator.write_text('# immutable source20 fixture')
    genref=rt.reference(generator)
    manifest=rt.write_new(archive/'SOURCE_MANIFEST.json',[dict(file='sim/ausc_policy/m2_student_oracle_proposals.py',
        bytes=genref['bytes'],sha256=genref['sha256'])])
    original=rt.write_new(tmp_path/'student-config.json',dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=meta)))
    source=dict(config=original,owner=rt.write_new(tmp_path/'owner.json',{}),child=rt.write_new(tmp_path/'child.json',{}),
        native=rt.write_new(tmp_path/'native.json',dict(success=False)))
    plan_config=rt.write_new(tmp_path/'plan-config.json',dict(metadata=meta))
    out=tmp_path/'diagnostic';out.mkdir();rows=[24,36,48]
    request=rt.write_new(out/'REQUEST.json',dict(source=source,plan_config=plan_config,physics_rows=rows,generator=genref,
        actions_executed=False,physics_qualified=False,training_eligible=False))
    histories=[];times=[];relative=[];masks=[];outcomes=[]
    for row in rows:
        accepted=row!=36
        outcome=dict(physics_row=row,proposal_accepted=accepted,actions_executed=False,physics_qualified=False,training_eligible=False)
        if accepted:
            q=np.full(7,row/1000.)
            h=np.array([rt.shared.contract.encode_state(q-.001,ref.pose_base,1.,metadata=meta),
                rt.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)])
            t=np.array([row/240.-.05,row/240.])
            query=dict(metadata=meta,measured_q=q.tolist(),measured_dq=[0.]*7,history=h.tolist(),history_times=t.tolist(),
                time_s=float(t[-1]),physics_row=row,context=[meta['episode_id'],'L1',0,'APPROACH'],reference_sha256=ref.sha256)
            proposal=dict(kind='recovery_proposal',query=query,query_sha256=rt.digest(query),physics_executed=False,training_eligible=False,
                preview=dict(cells=[[.7]*7,[.8]*7],all_240hz_commands_verified=True,physics_executed=False))
            outcome.update(computational_proposal=rt.write_new(out/f'query-{row}.json',proposal),valid_proposed_cells=2)
            a=np.zeros((20,7));a[:2]=np.array([[.7]*7,[.8]*7])-q
            histories.append(h);times.append(t);relative.append(a);masks.append(np.arange(20)>=2)
        else:outcome['rejection']='original numerical rejection'
        rt.write_new(out/f'outcome-{row}.json',outcome);outcomes.append(outcome)
    arrays=out/'UNEXECUTED_PROPOSALS.npz'
    with arrays.open('wb') as stream:np.savez_compressed(stream,source_observed_histories=np.asarray(histories),
        source_observation_times=np.asarray(times),source_physics_rows=np.array([24,48],dtype=np.int64),
        proposed_relative_setters=np.asarray(relative),proposed_is_pad=np.asarray(masks))
    perjob=rt.write_new(out/'COMPLETE.json',dict(schema='m2_student_cpu_oracle_diagnostic_v1',complete=True,
        outcomes=outcomes,attempted_queries=3,accepted_proposals=2,arrays=rt.reference(arrays),source_subject_id=meta['subject_id'],
        source_split='train',source_episode_verdict_unchanged=True,gpu_launched=False,
        actions_executed=False,physics_qualified=False,training_eligible=False))
    frozen=rt.write_new(tmp_path/'FROZEN_REQUEST.json',dict(cases=[dict(job='genuine-source',source=source,plan_config=plan_config,physics_rows=rows)],
        fixed_query_denominator=3,source_manifest=manifest,actions_executed=False,physics_qualified=False,training_eligible=False))
    complete=rt.write_new(tmp_path/'COMPLETE.json',dict(results=[dict(job='genuine-source',result=perjob)],fixed_query_denominator=3,
        actions_executed=False,physics_qualified=False,training_eligible=False,gpu_launched=False))
    return complete,frozen,meta,perjob,request


def test_frozen_oracle_import_preserves_genuine_h2_but_only_first_setter(tmp_path):
    m=module();complete,frozen,meta,perjob,_=oracle_fixture(tmp_path)
    got=m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
    s=got['samples'];assert len(s['source_histories'])==2 and len(got['slot_ledger'])==3
    assert [r['status'] for r in got['slot_ledger']]==['accepted','rejected','accepted']
    np.testing.assert_allclose(s['source_histories'][:,-1,:7],[[.024]*7,[.048]*7])
    np.testing.assert_allclose(s['relative_setters'][:,0],[[.676]*7,[.652]*7])
    assert s['setter_is_pad'].tolist()==[[False]+[True]*19]*2
    assert np.all(s['relative_setters'][:,1:]==0)
    assert s['sources'][0]['source_episode_verdict_unchanged'] is True
    assert s['sources'][0]['observation_origin']=='actual_authenticated_TRAIN_student_query'
    assert 'histories' not in s and 'actions' not in s


@pytest.mark.parametrize('fault',['split','numeric','query','generator'])
def test_frozen_oracle_rejects_rebinding_before_data_admission(tmp_path,monkeypatch,fault):
    m=module();complete,frozen,meta,perjob,request=oracle_fixture(tmp_path)
    from sim.ausc_policy import m2_given_contact_runtime as rt
    per=rt.read_bound(perjob);req=rt.read_bound(request)
    if fault=='split':
        cfg=rt.read_bound(req['source']['config']);cfg['m2_given_contact_expanded_pilot_deployment']['metadata']['split']='validation'
        # Rehash every declared parent, so split gating, not stale byte hashes, is exercised.
        changed=rt.write_new(tmp_path/'validation-config.json',cfg);req['source']['config']=changed
        fr=rt.read_bound(frozen);fr['cases'][0]['source']['config']=changed
        frozen=rt.write_new(tmp_path/'new-frozen.json',fr)
        Path=__import__('pathlib').Path;Path(request['path']).write_text(json.dumps(req))
        monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Non-TRAIN proposal arrays must stay unopened'))
    elif fault=='numeric':
        path=rt.bound_path(per['arrays'])
        with np.load(path) as archive:arrays={k:archive[k].copy() for k in archive.files}
        arrays['proposed_relative_setters'][0,0,0]+=.01
        with path.open('wb') as stream:np.savez(stream,**arrays)
        per['arrays']=rt.reference(path)
    elif fault=='query':
        p=rt.read_bound(per['outcomes'][0]['computational_proposal']);p['query']['history'][1][0]+=.001
        per['outcomes'][0]['computational_proposal']=rt.write_new(tmp_path/'changed-query.json',p)
    else:
        req['generator']=rt.write_new(tmp_path/'other-generator.json',{})
        __import__('pathlib').Path(request['path']).write_text(json.dumps(req))
    if fault in ('numeric','query'):
        # Saved outcomes and completion must agree; numeric/query binding still refuses.
        if fault=='query':__import__('pathlib').Path(perjob['path']).parent.joinpath('outcome-24.json').write_text(json.dumps(per['outcomes'][0]))
        newper=rt.write_new(tmp_path/'new-perjob.json',per)
        # Keep directory convention; importer derives REQUEST/outcomes from original completion directory.
        __import__('pathlib').Path(perjob['path']).write_text(json.dumps(per));newper=rt.reference(perjob['path'])
        root=rt.read_bound(complete);root['results'][0]['result']=newper
        complete=rt.write_new(tmp_path/'new-complete.json',root)
    with pytest.raises(ValueError):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


def test_declared_pose_must_match_bound_installed_patient(tmp_path):
    m=module();s=fixture(tmp_path)
    source=deepcopy(s['sources'][0]);source['pose_id']=OTHER_POSE
    with pytest.raises(ValueError,match='pose'):
        m.bind_samples({k:s[k] for k in m.ARRAY_KEYS},[source],s['allowed_train_ids'])


def test_new_binding_cannot_change_window_clock_schedule(tmp_path):
    m=module();s=fixture(tmp_path)
    arrays={k:s[k].copy() for k in m.ARRAY_KEYS};arrays['source_times']+=.01
    with pytest.raises(ValueError,match='clock'):
        m.bind_samples(arrays,s['sources'],s['allowed_train_ids'])


def test_request_expansion_preserves_source_request_order(tmp_path):
    m=module();_,frozen,meta,_,_=oracle_fixture(tmp_path)
    from sim.ausc_policy import m2_given_contact_runtime as rt
    manifest=rt.write_new(tmp_path/'scene-request.json',dict(schema='m2_rebuild_scene_manifest_v1',purpose='capture',
        targets=['L1'],subject_ids=[meta['subject_id']],requested_scenes=1,requested_target_slots=1,
        scenes=[dict(subject_id=meta['subject_id'],split='train',cohort='train',pose_id=POSE,
            plans={'L1':{'status':'unavailable','reason':'retained'}})]))
    slots=m.requested_slots([frozen,manifest],{meta['subject_id']})
    assert [s['kind'] for s in slots]==['student_oracle_unexecuted']*3+['planned_ideal_tracking']


@pytest.mark.parametrize('fault',['swap_sources','proposal_reference','frozen_request_reference'])
def test_same_rollout_oracle_queries_cannot_be_reassigned_to_other_slots(tmp_path,fault):
    m=module();complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    imported=m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
    samples=imported['samples'];ledger=deepcopy(imported['slot_ledger'])
    # Both proposals are valid for the same actual rollout/body/pose/target.
    valid=m.export_preparation(tmp_path/'valid',samples,ledger,{meta['subject_id']},request_refs=[frozen])
    assert len(m.load_preparation(valid,{meta['subject_id']})['slot_ledger'])==3
    if fault=='swap_sources':
        ledger[0]['source_id'],ledger[2]['source_id']=ledger[2]['source_id'],ledger[0]['source_id']
    else:
        from sim.ausc_policy import m2_given_contact_runtime as rt
        sources=deepcopy(samples['sources'])
        if fault=='proposal_reference':sources[0]['bindings']['proposal']=sources[1]['bindings']['proposal']
        else:sources[0]['bindings']['frozen_request']=rt.write_new(tmp_path/'other-frozen-request.json',rt.read_bound(frozen))
        samples=m.bind_samples({k:samples[k] for k in m.ARRAY_KEYS},sources,{meta['subject_id']})
        ledger[0]['source_id']=samples['sources'][0]['source_id']
    with pytest.raises(ValueError,match='frozen|proposal|query'):
        m.export_preparation(tmp_path/'reassigned',samples,ledger,{meta['subject_id']},request_refs=[frozen])
