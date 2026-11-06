"""Finite formal bridge IPC, without official model execution or native physics."""
from copy import deepcopy
import importlib
import json
import os
import threading
import time

import numpy as np
import pytest

from test_ausc_given_contact_pose import reference,metadata


def module():
    name='sim.ausc_policy.m2_given_contact_deployment'
    assert importlib.util.find_spec(name) is not None, 'Formal given CPU bridge missing'
    return importlib.import_module(name)


def request_fixture(tmp_path):
    m=module();ref=reference(tmp_path);meta=dict(metadata(ref),handoff_policy_id=m.runtime.handoff.POLICY_ID,
        episode_id='fixed-one',subject_id='p000439',split='train',target_id='L1')
    attempt=dict(episode_id='fixed-one',attempt_id='fixed-one',subject_id='p000439',split='train',target_id='L1',metadata=meta)
    permit=dict(protocol=m.protocol(),task_contract=m.runtime.task_contract(),admission=dict(sha256='a'*64),
        model=dict(checkpoint=dict(sha256='b'*64)),attempts=[attempt])
    dep=dict(episode_id='fixed-one',metadata=meta,permit=dict(sha256='c'*64))
    state=m.runtime.shared.contract.encode_state(np.zeros(7),ref.pose_base,1,metadata=meta)
    context=dict(observed_at_s=.05,physics_row=12,source_id='given-contact-'+ref.sha256,
        context=['fixed-one','L1',0,'FREE'])
    request=m.make_request(dep,permit,1,np.zeros(7),ref.pose_base,1,metadata=meta,
        history=np.stack([state,state]),history_times=[0,.05],context=context)
    return m,ref,dep,permit,request


def test_ipc_request_binds_full_reference_history_clock_and_current_origin(tmp_path):
    m,ref,dep,permit,request=request_fixture(tmp_path)
    state=m.validate_request(request,dep,permit,previous=None)
    assert state['request_id']==1 and state['stamp']==.05
    response=m.response_for(request,np.zeros((20,7)))
    block=m.decode_response(response,request)
    assert block.values.shape==(20,7) and block.metadata['checkpoint_policy']==m.runtime.given.CHECKPOINT_SCHEMA
    assert np.array_equal(block.observed_q,np.zeros(7))


@pytest.mark.parametrize('fault',['old_schema','wrong_permit','reference','full_roll','revision','q','nan','history','history_time','stamp','context','label','duplicate','regression'])
def test_request_refuses_stale_wrong_reference_or_labels_before_inference(tmp_path,fault):
    m,ref,dep,permit,request=request_fixture(tmp_path);previous=None
    if fault=='old_schema':request['schema']='m2_act_pilot_protocol'
    elif fault=='wrong_permit':request['binding']['permit_sha256']='d'*64
    elif fault=='reference':request['binding']['reference_sha256']='d'*64
    elif fault=='full_roll':request['estimated_target_base'][0][0]+=.01
    elif fault=='revision':request['binding']['reference_revision']=1
    elif fault=='q':request['measured_q'][0]=.1
    elif fault=='nan':request['history'][0][0]=float('nan')
    elif fault=='history':request['history'].pop()
    elif fault=='history_time':request['history_times'][0]=.01
    elif fault=='stamp':request['context']['observed_at_s']=.06
    elif fault=='context':request['context']['context'][1]='R1'
    elif fault=='label':request['action']=[[0]*7]*20
    elif fault=='duplicate':previous=m.validate_request(request,dep,permit,previous=None)
    else:previous=dict(request_id=0,stamp=0,physics_row=0,phase='APPROACH')
    with pytest.raises(ValueError):m.validate_request(request,dep,permit,previous=previous)


@pytest.mark.parametrize('fault',['sequence','origin','shape','nonfinite','echo','reference','metadata','extra'])
def test_response_refuses_reordered_truncated_or_changed_same_time_bindings(tmp_path,fault):
    m,_,_,_,request=request_fixture(tmp_path);response=m.response_for(request,np.zeros((20,7)))
    if fault=='sequence':response['request_id']=2
    elif fault=='origin':response['origin'][0]=.1
    elif fault=='shape':response['relative'].pop()
    elif fault=='nonfinite':response['relative'][0][0]=float('inf')
    elif fault=='echo':response['request_sha256']='e'*64
    elif fault=='reference':response['binding']['reference_revision']=1
    elif fault=='metadata':response['metadata']['given_reference']['reference_pose_base'][0][0]+=.01
    else:response['error']='fallback'
    with pytest.raises(ValueError):m.decode_response(response,request)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n',b'[]\n',b'{bad}\n'])
def test_wire_json_has_no_duplicate_keys_nonfinite_values_or_nondictionaries(raw):
    with pytest.raises(ValueError):module().wire_decode(raw)


def test_transport_one_deadline_covers_write_and_read_and_refuses_duplicate_frame():
    m=module();read_fd,write_fd=os.pipe();response_read,response_write=os.pipe()
    reader=os.fdopen(response_read,'rb',buffering=0);writer=os.fdopen(write_fd,'wb',buffering=0)
    transport=m.FramedPipe(reader,writer)
    def peer():
        os.read(read_fd,32768);os.write(response_write,b'{"ok":true}\n{"ok":true}\n');os.close(response_write);os.close(read_fd)
    thread=threading.Thread(target=peer);thread.start()
    try:
        with pytest.raises(ValueError,match='unsolicited|duplicate'):transport.exchange(dict(test=True),timeout=.5)
    finally:reader.close();writer.close();thread.join()


@pytest.mark.parametrize('mode',['deadline','eof','oversize','truncated'])
def test_transport_errors_are_bounded_without_fallback(mode):
    m=module();a,b=os.pipe();reader=os.fdopen(a,'rb',buffering=0)
    try:
        if mode=='eof':os.close(b);b=None
        elif mode=='oversize':
            def flood():
                try:os.write(b,b'x'*131073)
                except BrokenPipeError:pass
            thread=threading.Thread(target=flood);thread.start()
        elif mode=='truncated':os.write(b,b'{"partial":');os.close(b);b=None
        with pytest.raises(ValueError):m.FramedPipe(reader,None).read(timeout=.05)
    finally:
        reader.close()
        if b is not None:os.close(b)
        if mode=='oversize':thread.join(timeout=1)


def permit_document(tmp_path):
    m=module();splits=json.loads(m.runtime.shared.contract.IDENTITY_SPLIT_PATH.read_text())['splits']
    attempts=[]
    for i,(split,target) in enumerate(zip(('train','train','validation','validation'),('L1','L2','R1','R2'))):
        root=tmp_path/str(i);root.mkdir();ref=reference(root);snapshot=ref.snapshot();snapshot['target_id']=target
        ref=m.runtime.pose.GivenReference.from_snapshot(snapshot)
        meta=dict(metadata(ref),episode_id='episode-'+str(i),subject_id=splits[split][i%2],split=split,target_id=target,
            handoff_policy_id=m.runtime.handoff.POLICY_ID)
        attempts.append(dict(episode_id=meta['episode_id'],attempt_id='job-'+str(i),job_id='job-'+str(i),
            owner_run_id='fixed-smoke',output='/workspace/wam/runs/ausc_policy/fixed-smoke/output/job-'+str(i),
            subject_id=meta['subject_id'],split=split,target_id=target,metadata=meta,seed=2026102800+i,
            runtime_config_sha256='e'*64,pose_template=('p000474_a001230_f0000','p000476_a001231_f0023')[i%2],
            pose_binding=dict(patient_receipt=ref.snapshot()['sources']['patient_receipt'],preparation_config=dict(file='/prep',sha256='f'*64))))
    permit=dict(schema=m.PERMIT_SCHEMA,plan_schema=m.SMOKE_PLAN_SCHEMA,protocol=m.protocol(),task_contract=m.runtime.task_contract(),
        production_ready=True,simulation_only=True,scope='trainval_smoke_only',test_payloads_opened=False,
        physical_acceptance=False,real_hardware_validated=False,closed_loop_evaluated=False,
        admission=dict(path='/admission.json',sha256='a'*64),model=dict(checkpoint=dict(path='/best.pt',sha256='b'*64),
            complete=dict(path='/COMPLETE.json',sha256='c'*64)),processing={},source_inventory={},
        allocation_mapping=dict(path='/mapping.json',sha256=m.runtime.dataset.ALLOCATION_MAPPING_SHA256),attempts=attempts)
    return m,permit


def test_finite_simulation_permit_has_exact_four_2_train_2_val_two_poses_and_distinct_targets(tmp_path):
    m,permit=permit_document(tmp_path);m._validate_permit(permit)
    assert len(permit['attempts'])==4 and permit['physical_acceptance'] is False


@pytest.mark.parametrize('split',['train','validation'])
def test_smoke_permit_requires_two_distinct_identities_in_each_split(tmp_path,split):
    m,p=permit_document(tmp_path);rows=[r for r in p['attempts'] if r['split']==split]
    rows[1]['subject_id']=rows[0]['subject_id'];rows[1]['metadata']['subject_id']=rows[0]['subject_id']
    with pytest.raises(ValueError,match='identit'):m._validate_permit(p)


@pytest.mark.parametrize('role',['task_manifest','fresh_plan','patient_receipt'])
def test_frozen_mapping_authentication_precedes_every_snapshot_source_read(tmp_path,monkeypatch,role):
    m,p=permit_document(tmp_path);row=p['attempts'][0]
    sentinel=tmp_path/'synthetic-untrusted-body.npz';sentinel.write_bytes(b'not JSON; never a real heldout asset')
    bound=m.runtime.reference(sentinel);snapshot=row['metadata']['given_reference']
    snapshot['sources'][role]=dict(file=bound['path'],sha256=bound['sha256'])
    ref=m.runtime.pose.GivenReference.from_snapshot(snapshot)
    row['metadata']=m.runtime.pose.route_metadata(row['metadata'],ref)
    if role=='patient_receipt':row['pose_binding']['patient_receipt']=snapshot['sources'][role]
    m._validate_permit(p)  # A valid TRAIN header cannot authenticate a leaf path.
    reads=[];original=m.runtime.pose.bound_json
    monkeypatch.setattr(m.runtime.pose,'bound_json',lambda ref:(reads.append(ref['file']) or original(ref)))
    with pytest.raises(ValueError,match='frozen allocation|Original frozen'):m._pose_binding(row,dict(scenes=[]))
    assert reads==[]


def frozen_reference_fixture(tmp_path):
    from test_ausc_given_contact_dataset import metadata_row
    m=module();raw,meta,*_=metadata_row(tmp_path)
    ref=m.runtime.pose.require_route(meta);snapshot=ref.snapshot()
    plan=m.runtime.pose.bound_json(snapshot['sources']['fresh_plan'])
    plan['selected_fit']=dict(first_contact=dict(contact_plane_normal_world=[0.,0.,-1.],
        plane_tilt_limit_deg=3.,contact_plane_radius_m=.023,tilt_relaxed=False))
    saved=m.runtime.write_new(tmp_path/'checked-plan.json',plan)
    snapshot['sources']['fresh_plan']=dict(file=saved['path'],sha256=saved['sha256'])
    ref=m.runtime.pose.GivenReference.from_snapshot(snapshot);meta=m.runtime.pose.route_metadata(meta,ref)
    prep=m.runtime.write_new(tmp_path/'original-preparation.json',dict(cpu_fixture_only=True))
    prep=dict(file=prep['path'],bytes=prep['bytes'],sha256=prep['sha256'])
    row=dict(subject_id=meta['subject_id'],split=meta['split'],target_id=meta['target_id'],metadata=meta,
        pose_template='p000474_a001230_f0000',pose_binding=dict(patient_receipt=snapshot['sources']['patient_receipt'],preparation_config=prep))
    scene=dict(subject_id=row['subject_id'],split=row['split'],template=row['pose_template'],proposal_id='original-proposal',
        patient_receipt=snapshot['sources']['patient_receipt'],preparation_config=prep)
    allocation=dict(subject_id=row['subject_id'],target=row['target_id'],proposal_id=scene['proposal_id'],
        scene_ref=scene['proposal_id'],fresh_plan=dict(saved, file=saved['path']))
    allocation['fresh_plan'].pop('path')
    return m,row,dict(scenes=[scene],ordered_allocations={row['split']:[allocation]})


def test_original_frozen_role_bindings_then_reconstruct_the_complete_reference(tmp_path):
    m,row,mapping=frozen_reference_fixture(tmp_path)
    assert m._reference_roles(row,mapping).sha256==row['metadata']['reference_sha256']
    assert m._pose_binding(row,mapping).sha256==row['metadata']['reference_sha256']


@pytest.mark.parametrize('fault',['subject','target','pose','proposal','fresh_plan','patient','preparation','task_alias'])
def test_original_mapping_rejects_changed_role_or_scene_before_any_leaf(tmp_path,monkeypatch,fault):
    m,row,mapping=frozen_reference_fixture(tmp_path);scene=mapping['scenes'][0];allocation=mapping['ordered_allocations']['train'][0]
    if fault=='subject':allocation['subject_id']='another-identity'
    elif fault=='target':allocation['target']='R1'
    elif fault=='pose':scene['template']='p000476_a001231_f0023'
    elif fault=='proposal':allocation['scene_ref']='different-proposal'
    elif fault=='fresh_plan':allocation['fresh_plan']['sha256']='0'*64
    elif fault=='patient':scene['patient_receipt']['sha256']='0'*64
    elif fault=='preparation':row['pose_binding']['preparation_config']=dict(file='/wrong/original-preparation.json',sha256='0'*64)
    else:
        snapshot=deepcopy(row['metadata']['given_reference']);snapshot['sources']['task_manifest']['file']='/same-content-unsafe-alias.npz'
        ref=m.runtime.pose.GivenReference.from_snapshot(snapshot);row['metadata']=m.runtime.pose.route_metadata(row['metadata'],ref)
    monkeypatch.setattr(m.runtime.pose,'bound_json',lambda ref:pytest.fail('Unauthenticated role reached a leaf read'))
    with pytest.raises(ValueError,match='Original frozen'):m._pose_binding(row,mapping)


def test_public_permit_authenticates_all_four_source_role_maps_before_first_leaf(tmp_path,monkeypatch):
    m,p=permit_document(tmp_path);ref=m.runtime.write_new(tmp_path/'permit.json',p)
    monkeypatch.setattr(m,'_allocation_mapping',lambda *args:({},p['allocation_mapping']))
    monkeypatch.setattr(m,'_admission_metadata',lambda ref:dict(admission=dict(model=p['model']),config=dict(processing=p['processing'])))
    monkeypatch.setattr(m,'_sources',lambda binding:p['source_inventory'])
    seen=[]
    def roles(row,mapping):
        seen.append(row['episode_id'])
        if len(seen)==4:raise ValueError('last source role rejected')
    monkeypatch.setattr(m,'_reference_roles',roles)
    monkeypatch.setattr(m,'_pose_binding',lambda *a:pytest.fail('First reference read preceded last metadata role'))
    with pytest.raises(ValueError,match='last source role'):m.verify_permit(ref)
    assert seen==[row['episode_id'] for row in p['attempts']]


@pytest.mark.parametrize('fault',['alias','wrong_bytes'])
def test_mapping_role_refuses_same_content_alias_or_declared_size_before_source_leaf(tmp_path,monkeypatch,fault):
    m=module();original=m.runtime.write_new(tmp_path/'original-mapping.json',dict(scenes=[]))
    monkeypatch.setattr(m.runtime.dataset,'ALLOCATION_MAPPING_SHA256',original['sha256'])
    plan=m.runtime.write_new(tmp_path/'plan.json',dict(allocation_mapping=dict(file=original['path'],
        bytes=original['bytes'],sha256=original['sha256'])))
    candidate=deepcopy(original)
    if fault=='alias':candidate=m.runtime.write_new(tmp_path/'same-content-alias.json',dict(scenes=[]))
    else:candidate['bytes']+=1
    assert candidate['sha256']==original['sha256']
    with pytest.raises(ValueError,match='mapping'):
        m._allocation_mapping(candidate,dict(plan=plan))


def test_mapping_role_normalizes_file_and_path_against_authenticated_original_plan(tmp_path,monkeypatch):
    m=module();original=m.runtime.write_new(tmp_path/'original-mapping.json',dict(scenes=[]))
    monkeypatch.setattr(m.runtime.dataset,'ALLOCATION_MAPPING_SHA256',original['sha256'])
    plan=m.runtime.write_new(tmp_path/'plan.json',dict(allocation_mapping=dict(file=original['path'],
        bytes=original['bytes'],sha256=original['sha256'])))
    value,canonical=m._allocation_mapping(dict(path=original['path'],sha256=original['sha256']),dict(plan=plan))
    assert value==dict(scenes=[]) and canonical==original


@pytest.mark.parametrize('entrypoint',['verify','freeze'])
def test_every_public_permit_entrypoint_refuses_same_hash_mapping_alias_before_reference_leaves(tmp_path,monkeypatch,entrypoint):
    m,p=permit_document(tmp_path)
    original=m.runtime.write_new(tmp_path/'original-mapping.json',dict(scenes=[]))
    alias=m.runtime.write_new(tmp_path/'alias-mapping.json',dict(scenes=[]))
    assert alias['sha256']==original['sha256'] and alias['path']!=original['path']
    monkeypatch.setattr(m.runtime.dataset,'ALLOCATION_MAPPING_SHA256',original['sha256'])
    plan=m.runtime.write_new(tmp_path/'plan.json',dict(allocation_mapping=original))
    p['allocation_mapping']=alias;p['processing']=dict(plan=plan)
    monkeypatch.setattr(m,'_admission_metadata',lambda ref:dict(admission=dict(model=p['model']),config=dict(processing=p['processing'])))
    monkeypatch.setattr(m,'_sources',lambda binding:p['source_inventory'])
    monkeypatch.setattr(m,'_reference_roles',lambda *a:pytest.fail('Mapping alias reached reference role authentication'))
    monkeypatch.setattr(m,'_pose_binding',lambda *a:pytest.fail('Mapping alias reached a reference leaf'))
    with pytest.raises(ValueError,match='mapping'):
        if entrypoint=='verify':m.verify_permit(m.runtime.write_new(tmp_path/'alias-permit.json',p))
        else:m.freeze_smoke_permit(p['admission'],alias,p['attempts'],tmp_path/'must-not-create.json')
    assert not (tmp_path/'must-not-create.json').exists()


@pytest.mark.parametrize('fault',['old','pilot','false_ready','fixture','test','quarantine','missing','duplicate','targets','pose','seed','owner_path','teacher_fallback','no_model'])
def test_permit_cannot_unlock_old_pilot_fixture_test_or_outcome_selected_smokes(tmp_path,fault):
    m,p=permit_document(tmp_path)
    if fault in ('old','pilot'):p['schema']='m2_act_'+fault
    elif fault=='false_ready':p['production_ready']=False
    elif fault=='fixture':p['cpu_fixture_only']=True
    elif fault in ('test','quarantine'):p['attempts'][0]['split']=fault
    elif fault=='missing':p['attempts'].pop()
    elif fault=='duplicate':p['attempts'][1]=deepcopy(p['attempts'][0])
    elif fault=='targets':p['attempts'][1]['target_id']=p['attempts'][0]['target_id']
    elif fault=='pose':p['attempts'][1]['pose_template']=p['attempts'][0]['pose_template']
    elif fault=='seed':p['attempts'][0]['seed']=True
    elif fault=='owner_path':p['attempts'][0]['output']='/tmp/replacement'
    elif fault=='teacher_fallback':p['protocol']['expert_l2_fallback']=True
    else:p['model'].pop('complete')
    with pytest.raises(ValueError):m._validate_permit(p)


def test_original_owner_registers_only_explicit_finite_fixture_and_train_routes():
    from sim.ausc_policy import run
    m=module()
    assert run.MODULES['m2_given_contact_numeric_fixture']=='sim.ausc_policy.m2_given_contact_runtime'
    assert run.MODULES['m2_given_contact_train']=='sim.ausc_policy.m2_given_contact_train'
    assert 'm2_given_contact_deployment' not in run.MODULES


def test_formal_preflight_runs_before_scene_store_patient_or_simulation_payload(tmp_path,monkeypatch):
    from sim.ausc_policy import geometry_qualification as geometry
    m=module();config=tmp_path/'config.json';config.write_text(json.dumps(dict(m2_given_contact_deployment={})))
    monkeypatch.setattr(geometry,'require_current',lambda _:None)
    monkeypatch.setattr(geometry,'preparation_backend',lambda *a:pytest.fail('Payload/backend reached before formal preflight'))
    monkeypatch.setattr(m,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('formal metadata rejected')) ,raising=False)
    monkeypatch.setattr(m.sys,'argv',['geometry','--config',str(config),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='formal metadata rejected'):geometry.main()
    assert not (tmp_path/'out').exists()


def test_formal_factory_preserves_refusal_for_missing_old_or_unqualified_permit():
    from sim.ausc_policy import m2_given_contact_act as given
    with pytest.raises(RuntimeError,match='bridge|qualification|permit'):
        given.make_controller(deployment=dict(schema=given.DEPLOYMENT_SCHEMA))


@pytest.mark.parametrize('fault',['schema','metrics','selected','cpu_count','cpu_fixture','source','stats'])
def test_admission_metadata_refuses_unqualified_or_changed_model_bindings(tmp_path,monkeypatch,fault):
    m=module()
    # The caller must not get as far as native payload/model construction.
    admission=dict(schema=m.runtime.given.ADMISSION_SCHEMA,formal_run_schema=m.runtime.FORMAL_RUN_SCHEMA,
        task_contract=m.runtime.task_contract(),passed=True,test_payloads_opened=False,test_sealed=True,
        physical_acceptance=False,production_ready=False,production_bridge_pending=True,
        comparison=dict(passed=True,cpu_byte_identical=True),model={},inputs={},processing={},cpu_results=[])
    if fault=='schema':admission['schema']='m2_act_pilot_admission'
    elif fault=='metrics':admission['comparison']['passed']=False
    elif fault=='cpu_fixture':admission['cpu_fixture_only']=True
    elif fault=='cpu_count':admission['cpu_results']=[{}]
    elif fault=='selected':admission['model']={'checkpoint':{}}
    elif fault=='source':admission['task_contract']['horizon']=10
    else:admission['normalization']={}
    ref=m.runtime.write_new(tmp_path/'admission.json',admission)
    with pytest.raises(ValueError):m._admission_metadata(ref)


def bridge_fixture(tmp_path,monkeypatch):
    m,p=permit_document(tmp_path);row=p['attempts'][0]
    dep=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
        permit=dict(path='/permit.json',sha256='c'*64),episode_id=row['episode_id'],metadata=row['metadata'])
    binary=m.runtime.reference(__import__('pathlib').Path(m.sys.executable).resolve())
    software_ref=m.runtime.write_new(tmp_path/'software.json',dict(python=binary))
    admission=dict(source_inventory=software_ref);admission_ref=m.runtime.write_new(tmp_path/'admission.json',admission)
    p['admission']=admission_ref
    monkeypatch.setattr(m,'verify_permit',lambda ref:p)
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    parent=dict(pid=os.getpid(),pgid=50,sid=50,start_ticks='12345',state='R')
    child=dict(pid=700,pgid=50,sid=50,start_ticks='12346',state='S')
    from sim.ausc_policy import process_guard as guard
    monkeypatch.setattr(guard,'process_identity',lambda pid:parent if pid==os.getpid() else child)
    monkeypatch.setattr(m,'_official_python',lambda software:m.sys.executable,raising=False)
    execution=dict(device='cpu',parameter_devices=['cpu'],parameter_dtypes=['torch.float32'],default_dtype='torch.float32',
        intraop_threads=2,interop_threads=1,deterministic_algorithms=True,matmul_precision='highest',matmul_tf32=False,
        cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8',torch_version='CPU fixture',
        cuda_version=None,device_name='cpu')
    ready=dict(ready=True,schema=m.runtime.given.PROTOCOL_SCHEMA,protocol_sha256=m.runtime.digest(m.protocol()),
        permit=dep['permit'],model=p['model'],episode_id=row['episode_id'],runtime_config_sha256=row['runtime_config_sha256'],
        metadata_sha256=m.runtime.digest(row['metadata']),source_inventory=p['source_inventory'],software=software_ref,
        python=binary,prefix=m.runtime.dataset.OFFICIAL_ACT_PREFIX,identity=child,parent_identity=parent,execution=execution)
    ready=deepcopy(ready)
    class Pipe:
        def close(self):pass
    class Child:
        pid=700;returncode=None
        stdin=Pipe();stdout=Pipe();terminated=0;killed=0;waits=0
        def poll(self):return self.returncode
        def wait(self,timeout):
            self.waits+=1
            if self.returncode is None:raise m.subprocess.TimeoutExpired('mock',timeout)
            return self.returncode
        def terminate(self):self.terminated+=1;self.returncode=-15
        def kill(self):self.killed+=1;self.returncode=-9
    process=Child();calls=[]
    monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kwargs:(calls.append((command,kwargs)) or process))
    monkeypatch.setattr(m.FramedPipe,'read',lambda self,**k:deepcopy(ready))
    return m,dep,p,ready,process,calls


def test_cpu_bridge_uses_original_session_official_cpu_environment_and_reaps_only_owned_child(tmp_path,monkeypatch):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch)
    bridge=m.CPUBridge(dep,output=tmp_path/'native-job')
    command,options=calls[0]
    assert command[1:5]==['-u','-B','-m',m.MODULE] and options['start_new_session'] is False
    assert options['env']['CUDA_VISIBLE_DEVICES']=='' and options['env']['PYTHONDONTWRITEBYTECODE']=='1'
    assert options['env']['OMP_NUM_THREADS']=='2' and options['env']['PYTHONNOUSERSITE']=='1'
    result=bridge.close();assert process.terminated==1 and process.killed==0 and process.poll()==-15
    assert result['reaped'] is True and result['identity']==ready['identity']
    assert bridge.close()==result


@pytest.mark.parametrize('sampled,reported',[('R','S'),('S','R'),('R','D')])
def test_live_ready_identity_allows_scheduler_state_change_without_changing_stable_identity(tmp_path,monkeypatch,sampled,reported):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch)
    from sim.ausc_policy import process_guard as guard
    original=guard.process_identity
    def identity(pid):
        value=deepcopy(original(pid));value['state']=sampled;return value
    monkeypatch.setattr(guard,'process_identity',identity)
    ready['identity']['state']=reported;ready['parent_identity']['state']=reported
    bridge=m.CPUBridge(dep,output=tmp_path/'native-job')
    assert bridge.runtime['identity']['state']==reported and bridge.identity['state']==sampled
    bridge.close()


@pytest.mark.parametrize('role',['identity','parent_identity'])
@pytest.mark.parametrize('state',['Z','X','x','',None])
def test_ready_identity_refuses_exited_or_missing_process_state_and_reaps_child(tmp_path,monkeypatch,role,state):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch);ready[role]['state']=state
    with pytest.raises(ValueError):m.CPUBridge(dep,output=tmp_path/'native-job')
    assert process.poll()==-15 and process.terminated==1


@pytest.mark.parametrize('field',['pid','start_ticks','pgid','sid','ppid','argv','cwd'])
def test_live_identity_keeps_every_stable_field_binding(field):
    m=module();left=dict(pid=700,pgid=50,sid=50,start_ticks='12346',state='R',ppid=600,
        argv=['/official/python','-B','-m',m.MODULE],cwd='/frozen/processing')
    right=deepcopy(left);right['state']='S'
    assert m._same_live_process(left,right)
    if field in ('pid','pgid','sid','ppid'):right[field]+=1
    elif field=='start_ticks':right[field]='12347'
    elif field=='argv':right[field]=['/other/python']
    else:right[field]='/other/root'
    assert not m._same_live_process(left,right)


def test_child_that_exits_after_ready_cannot_pass_live_handshake(tmp_path,monkeypatch):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch)
    def ready_then_exit(*args,**kwargs):process.returncode=0;return deepcopy(ready)
    monkeypatch.setattr(m.FramedPipe,'read',ready_then_exit)
    with pytest.raises(ValueError):m.CPUBridge(dep,output=tmp_path/'native-job')
    assert process.poll()==0 and process.waits==1


@pytest.mark.parametrize('fault',['prefix','module','checkpoint','software','identity','session','threads','reference','startup'])
def test_bad_child_ready_or_startup_exception_always_reaps_direct_child(tmp_path,monkeypatch,fault):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch)
    if fault=='prefix':ready['prefix']='/other/python'
    elif fault=='module':ready['source_inventory']={'wrong':'source'}
    elif fault=='checkpoint':ready['model']['checkpoint']['sha256']='0'*64
    elif fault=='software':ready['software']['sha256']='0'*64
    elif fault=='identity':ready['identity']['start_ticks']='another'
    elif fault=='session':ready['identity']['sid']=51
    elif fault=='threads':ready['execution']['intraop_threads']=3
    elif fault=='reference':ready['metadata_sha256']='0'*64
    else:monkeypatch.setattr(m.FramedPipe,'read',lambda *a,**k:(_ for _ in ()).throw(ValueError('startup EOF')))
    with pytest.raises(ValueError):m.CPUBridge(dep,output=tmp_path/'native-job')
    assert process.poll()==-15 and process.terminated==1


def test_formal_controller_exception_before_run_closes_the_created_child(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_controller as controller
    m,p=permit_document(tmp_path);row=p['attempts'][0];closed=[]
    dep=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
        permit=dict(path='/permit.json',sha256='a'*64),episode_id=row['episode_id'],metadata=row['metadata'])
    provider=m.runtime.pose.GivenPoseProvider(m.runtime.pose.require_route(row['metadata']),row['episode_id'],row['metadata'])
    class Bridge:
        runtime={}
        def __init__(self,*a,**k):pass
        def close(self):closed.append(True)
    monkeypatch.setattr(m,'verify_permit',lambda ref:p)
    monkeypatch.setattr(m,'CPUBridge',Bridge)
    monkeypatch.setattr(controller.GivenContactACT,'__init__',lambda *a,**k:(_ for _ in ()).throw(ValueError('native setup failed')))
    from types import SimpleNamespace
    row['output']=str(tmp_path/'native-job');env=SimpleNamespace(rows=[],state=lambda:dict(time_s=0.,q=np.zeros(7)))
    with pytest.raises(ValueError,match='native setup failed'):
        m.make_controller(env,deployment=dep,estimate_provider=provider,output=tmp_path/'native-job')
    assert closed==[True]


def test_formal_query_sets_authoritative_context_and_clears_it_after_base_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from sim.ausc_policy import m2_given_contact_controller as controller
    m,_,_,_,request=request_fixture(tmp_path)
    instance=m.FormalGivenACT.__new__(m.FormalGivenACT)
    instance.env=SimpleNamespace(rows=[{}]*12);instance.predictor=SimpleNamespace(context=None)
    observation=dict(observed_at_s=.05,context=['fixed-one','L1',0,'FREE'],source_id=request['context']['source_id'])
    def base(self,value):
        assert self.predictor.context==request['context'];raise ValueError('inference stopped')
    monkeypatch.setattr(controller.GivenContactACT,'_query',base)
    with pytest.raises(ValueError,match='inference stopped'):instance._query(observation)
    assert instance.predictor.context is None


def test_cpu_child_permit_never_loads_planner_waypoints_or_runtime_patient_payload(tmp_path,monkeypatch):
    m,p=permit_document(tmp_path);ref=m.runtime.write_new(tmp_path/'permit.json',p)
    monkeypatch.setattr(m,'_admission_metadata',lambda ref:dict(admission=dict(model=p['model']),config=dict(processing=p['processing'])))
    monkeypatch.setattr(m,'_sources',lambda binding:p['source_inventory'])
    monkeypatch.setattr(m,'_pose_binding',lambda *a:pytest.fail('CPU child opened privileged expert plan'))
    assert m._child_permit(ref)==p


def test_formal_native_setup_failure_retains_learned_identity_and_denominator(tmp_path,monkeypatch):
    from sim.ausc_policy import fresh_expert
    m=module();dep=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,metadata=dict(target_id='L1'))
    monkeypatch.setattr(m,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('unqualified source')))
    monkeypatch.setattr(fresh_expert,'_run_probe',lambda *a,**k:pytest.fail('Unqualified route reached native payload'))
    result=fresh_expert.run_probe(None,None,tmp_path,dict(m2_given_contact_deployment=dep),{})
    assert result['success'] is False and result['learned_policy'] is True and result['expert_demonstration'] is False
    assert result['l2_handoff_attempt_count']==1 and result['l2_handoff_success_count']==0
    assert result['task_id']==m.runtime.pose.TASK_ID and result['training_eligible'] is False


def scene_metadata_fixture(tmp_path):
    from sim.ausc_policy import scene_store as store
    m,p=permit_document(tmp_path);row=p['attempts'][0]
    prepared=tmp_path/'prepared-assets';prepared.mkdir()
    patient=dict(identity=row['subject_id'],split=row['split'],asset_sha256='a'*64,non_patient_scene_unchanged=True)
    receipt=m.runtime.write_new(prepared/'patient_receipt.json',patient)
    snapshot=row['metadata']['given_reference'];snapshot['sources']['patient_receipt']={k:receipt[k] for k in ('sha256',)}
    snapshot['sources']['patient_receipt']['file']=receipt['path']
    ref=m.runtime.pose.GivenReference.from_snapshot(snapshot)
    row['metadata']=m.runtime.pose.route_metadata(row['metadata'],ref)
    row['metadata'].update(patient_asset_sha256='a'*64,installed_patient_receipt_sha256=receipt['sha256'])
    row['pose_binding']['patient_receipt']=ref.snapshot()['sources']['patient_receipt']
    asset=dict(directory=str(tmp_path/'candidate'),file='train.npz',expected_split='train')
    prep=dict(patient_asset=asset,prepare_only=True,geometry_id=m.runtime.geometry.GEOMETRY_ID,
        geometry_source='/frozen/geometry.usd',geometry_source_sha256=m.runtime.geometry.GEOMETRY_SOURCE_SHA256,posture_limits={},freshly_generated=True)
    prep_ref=m.runtime.write_new(tmp_path/'prep.json',prep)
    complete=m.runtime.write_new(prepared/'COMPLETE.json',dict(completed=True,patient=patient,effective_scene_sha256='1'*64,
        geometry_artifacts=dict(collision_geometry_sha256='2'*64,collision_arrays_sha256='3'*64)))
    sources={name:dict(file=str(prepared/filename),bytes=1,sha256='1'*64 if name=='scene' else '2'*64 if name=='collision_geometry' else '3'*64)
        for name,filename in store.FILES.items()}
    sources['preparation_config']=dict(file=prep_ref['path'],bytes=prep_ref['bytes'],sha256=prep_ref['sha256'])
    sources['prepared_complete']=dict(file=complete['path'],bytes=complete['bytes'],sha256=complete['sha256'])
    sources['patient_receipt']=dict(file=receipt['path'],bytes=receipt['bytes'],sha256=receipt['sha256'])
    sources['sdk_mdl_manifest']=dict(file='/fixed/sdk-manifest.json',bytes=1,sha256='4'*64)
    for i in range(244):sources['sdk_mdl_'+str(i).zfill(4)]=dict(file='/fixed/sdk/'+str(i),bytes=1,sha256='5'*64)
    row['pose_binding']['preparation_config']=dict(file=prep_ref['path'],bytes=prep_ref['bytes'],sha256=prep_ref['sha256'])
    scene=dict(subject_id=row['subject_id'],split=row['split'],patient_receipt=sources['patient_receipt'],
        preparation_config=row['pose_binding']['preparation_config'],prepared_complete=sources['prepared_complete'],template=row['pose_template'])
    binding=dict(schema=store.SDK_SCHEMA,dependency_policy=store.SDK_DEPENDENCY_POLICY,store_root='/workspace/wam/scene_store',batch_id='smoke',sources=sources)
    config={k:v for k,v in prep.items() if k!='prepare_only'}
    config.update(compact_scene_storage=binding,fresh_expert=dict(collision_profile='full'),isaac_python='/native/python')
    recipe=dict(collector_conditions=dict(fresh=deepcopy(config['fresh_expert']),isaac_python=config['isaac_python'],
        scene_dependencies=dict(schema=store.SDK_SCHEMA,dependency_policy=store.SDK_DEPENDENCY_POLICY,
            sources={k:deepcopy(v) for k,v in sources.items() if k.startswith('sdk_mdl')})))
    return m,row,config,dict(scenes=[scene]),recipe


def test_metadata_preflight_authenticates_scene_patient_sdk_and_gates_without_opening_arrays(tmp_path):
    m,row,config,mapping,recipe=scene_metadata_fixture(tmp_path)
    m._scene_metadata(config,row,mapping,recipe)
    assert not __import__('pathlib').Path(config['compact_scene_storage']['sources']['scene']['file']).exists()


@pytest.mark.parametrize('fault',['patient_asset','patient_split','scene_path','scene_sha','arrays_path','complete_ref','sdk','gates','posture'])
def test_metadata_first_scene_binding_rejects_hidden_test_or_changed_native_inputs_before_payload(tmp_path,fault):
    m,row,config,mapping,recipe=scene_metadata_fixture(tmp_path);sources=config['compact_scene_storage']['sources']
    if fault=='patient_asset':config['patient_asset']['file']='TEST.npz'
    elif fault=='patient_split':config['patient_asset']['expected_split']='test'
    elif fault=='scene_path':sources['scene']['file']='/heldout/test.usdc'
    elif fault=='scene_sha':sources['scene']['sha256']='9'*64
    elif fault=='arrays_path':sources['collision_arrays']['file']='/heldout/test.npz'
    elif fault=='complete_ref':sources['prepared_complete']['sha256']='9'*64
    elif fault=='sdk':sources['sdk_mdl_0000']['file']='/heldout/test.npz'
    elif fault=='gates':config['fresh_expert']['collision_profile']='fast'
    else:config['posture_limits']={'relaxed':True}
    with pytest.raises(ValueError):m._scene_metadata(config,row,mapping,recipe)


def admitted_metadata_fixture(tmp_path,monkeypatch):
    from test_ausc_given_contact_train import config_fixture
    from test_ausc_given_contact_admission import comparison_fixture
    a=tmp_path/'prepared';a.mkdir();train,r,config,prepared,_=config_fixture(a,monkeypatch)
    b=tmp_path/'model';b.mkdir();admission,r,_,complete,arrays,workers=comparison_fixture(b)
    m=module();config_ref=r.write_new(tmp_path/'formal-config.json',config)
    software=r.write_new(tmp_path/'official-software.json',dict(schema=r.RUNTIME_SCHEMA,prefix=r.dataset.OFFICIAL_ACT_PREFIX,
        vendor=dict(root=r.VENDOR_ROOT,upstream_commit=r.UPSTREAM_COMMIT),python=r.reference(__import__('pathlib').Path(m.sys.executable).resolve())))
    stats=r.write_new(tmp_path/'stats.json',prepared['normalization']);inventory=r.write_new(tmp_path/'refs.json',prepared['reference_inventory'])
    rebound=r.write_new(tmp_path/'input-rebound.json',dict(passed=True))
    inputs=dict(config=config_ref,dataset=config['dataset'],software=software,normalization=stats,reference_inventory=inventory)
    execution=deepcopy(workers[0]['execution']);execution.update(device='cuda',parameter_devices=['cuda'],device_name='metadata-only CPU fixture',cuda_version='not executed')
    complete.update(schema=r.FORMAL_RUN_SCHEMA,completed_steps=20000,complete=True,source_audit_passed=True,
        cpu_fixture_only=False,test_payloads_opened=False,task_contract=r.task_contract(),inputs=inputs,best=complete['evaluations'][0]['checkpoint'],
        last=complete['evaluations'][-1]['checkpoint'],processing=config['processing'],execution=execution,input_rebinding=rebound)
    model=dict(checkpoint=complete['best'],complete=r.write_new(b/'COMPLETE.json',complete))
    for worker in workers:worker.update(model=model,inputs=inputs,test_payloads_opened=False)
    comparison=admission.compare_cpu_results(model,complete,arrays,workers)
    value=dict(schema=r.given.ADMISSION_SCHEMA,formal_run_schema=r.FORMAL_RUN_SCHEMA,task_contract=r.task_contract(),
        passed=True,test_payloads_opened=False,test_sealed=True,physical_acceptance=False,production_ready=False,production_bridge_pending=True,
        comparison=comparison,model=model,inputs=inputs,processing=config['processing'],normalization=stats,reference_inventory=inventory,
        source_inventory=software,cpu_results=[r.write_new(tmp_path/('cpu'+str(i)+'.json'),w) for i,w in enumerate(workers)],input_rebinding=rebound)
    monkeypatch.setattr(m,'_training_owner',lambda *a:dict(passed=True,scope='CPU metadata unit fixture'))
    return m,value


def test_native_admission_verification_rebinds_immutable_evidence_without_dataset_decode_or_model_inference(tmp_path,monkeypatch):
    m,value=admitted_metadata_fixture(tmp_path,monkeypatch)
    for name in ('verify_final_seal','prepare_dataset'):
        monkeypatch.setattr(m.runtime.dataset,name,lambda *a,**k:pytest.fail('Bridge startup re-decoded full population'))
    monkeypatch.setattr(m.runtime,'infer',lambda *a,**k:pytest.fail('Bridge startup reran forty-snapshot admission'))
    result=m._admission_metadata(m.runtime.write_new(tmp_path/'admission.json',value))
    assert result['admission']==value and result['owner']['passed']


@pytest.mark.parametrize('fault',['stats_ref','model','source_ref','gate','argmin','cpu_identity','cpu_count','source_changed'])
def test_bound_admission_metadata_negative_uses_otherwise_complete_evidence(tmp_path,monkeypatch,fault):
    m,value=admitted_metadata_fixture(tmp_path,monkeypatch);r=m.runtime
    if fault=='stats_ref':value['normalization']=r.write_new(tmp_path/'other-stats.json',{})
    elif fault=='model':value['model']['checkpoint']=r.write_new(tmp_path/'other-weights.json',{})
    elif fault=='source_ref':value['source_inventory']=r.write_new(tmp_path/'other-source.json',{})
    elif fault=='gate':value['comparison']['train_gate']['passed']=False
    elif fault in ('argmin','cpu_identity'):
        index=0 if fault=='argmin' else 1;worker=r.read_bound(value['cpu_results'][index])
        if fault=='argmin':worker['evaluations'].pop()
        else:worker['identity']=r.read_bound(value['cpu_results'][0])['identity']
        value['cpu_results'][index]=r.write_new(tmp_path/'changed-worker.json',worker)
    elif fault=='cpu_count':value['cpu_results'].pop()
    else:__import__('pathlib').Path(value['source_inventory']['path']).write_text('changed')
    with pytest.raises(ValueError):m._admission_metadata(r.write_new(tmp_path/'admission.json',value))


def test_formal_scene_store_preflight_precedes_any_binding_or_payload_reader(monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();config=dict(m2_given_contact_deployment={},fresh_expert={})
    monkeypatch.setattr(m,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('sealed metadata rejected')))
    monkeypatch.setattr(scene_store,'_validate_binding',lambda *a:pytest.fail('Scene payload reached before formal preflight'))
    with pytest.raises(ValueError,match='sealed metadata rejected'):scene_store.validate_config(config)


def test_scene_store_authoring_strips_only_a_preflight_authenticated_formal_route(monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();calls=[];config=dict(m2_given_contact_deployment={},fresh_expert={})
    monkeypatch.setattr(m,'preflight',lambda c,**k:calls.append(deepcopy(c)))
    assert scene_store._authoring(config)==dict(fresh_expert={}) and calls==[config]
    monkeypatch.setattr(m,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('forged formal flag')))
    with pytest.raises(ValueError,match='forged formal flag'):scene_store._authoring(config)


def test_only_formal_scene_route_can_omit_expert_capture_without_faking_teacher_flags(monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();config=dict(m2_given_contact_deployment={},fresh_expert={},compact_scene_storage=dict(sources={}))
    prep=dict(fresh_expert={},prepare_only=True);calls=[]
    monkeypatch.setattr(m,'preflight',lambda c,**k:calls.append(c))
    monkeypatch.setattr(scene_store,'_validate_binding',lambda b:(prep,{},{}))
    binding,_=scene_store.validate_config(config);assert binding==config['compact_scene_storage'] and calls
    old=deepcopy(config);old.pop('m2_given_contact_deployment')
    with pytest.raises(ValueError,match='compact expert'):scene_store.validate_config(old)


def test_ready_module_inventory_uses_canonical_module_name_under_python_dash_m(monkeypatch):
    m=module();monkeypatch.setattr(m,'__name__','__main__')
    inventory=m.module_inventory();assert m.MODULE in inventory and '__main__' not in inventory


@pytest.mark.parametrize('failure',[False,True])
def test_real_bridge_call_retains_full_h20_same_time_trace_and_failure_closes_child(tmp_path,monkeypatch,failure):
    m,dep,p,ready,process,calls=bridge_fixture(tmp_path,monkeypatch)
    bridge=m.CPUBridge(dep,output=tmp_path/'native-job');ref=m.runtime.pose.require_route(dep['metadata'])
    state=m.runtime.shared.contract.encode_state(np.zeros(7),ref.pose_base,1,metadata=dep['metadata'])
    bridge.context=dict(observed_at_s=.05,physics_row=12,source_id='given-contact-'+ref.sha256,
        context=[dep['episode_id'],dep['metadata']['target_id'],0,'FREE'])
    def exchange(request,**kwargs):
        assert kwargs==dict(timeout=10.)
        if failure:raise ValueError('owned peer EOF')
        return m.response_for(request,np.zeros((20,7)))
    monkeypatch.setattr(bridge.pipe,'exchange',exchange)
    try:
        if failure:
            with pytest.raises(ValueError,match='owned peer EOF'):
                bridge(np.zeros(7),ref.pose_base,1,metadata=dep['metadata'],history=np.stack([state,state]),history_times=[0,.05])
            assert process.poll()==-15
        else:
            block=bridge(np.zeros(7),ref.pose_base,1,metadata=dep['metadata'],history=np.stack([state,state]),history_times=[0,.05])
            assert block.values.shape==(20,7) and process.poll() is None
    finally:bridge.close()
    records=[json.loads(line) for line in (tmp_path/'native-job/given_act_ipc.jsonl').read_text().splitlines()]
    assert records[-1]['accepted'] is not failure and records[-1]['request']['context']==bridge.context
    assert records[-1]['simulated_time_pause_s']==0. and records[-1]['wall_latency_s']>=0


@pytest.mark.parametrize('after_step_error',[False,True])
def test_formal_tick_records_actual_poststep_pose_and_live_raw_setter_without_replacing_native_chain(tmp_path,monkeypatch,after_step_error):
    from types import SimpleNamespace
    from sim.ausc_policy import m2_given_contact_controller as controller
    m,ref,dep,p,request=request_fixture(tmp_path)
    instance=m.FormalGivenACT.__new__(m.FormalGivenACT);state=dict(time_s=1/240.,q=np.ones(7),
        tcp_base=np.array([.1,.2,.3]),rotation_world=np.eye(3))
    instance.env=SimpleNamespace(rows=[],state=lambda:state,base=np.eye(4));instance.reference=ref
    instance.control_owner='ACT_L2';instance._current_intent_audit=dict(setter='actual native setter',live_target_q=[.1]*7,live_target_after_step_q=[.1]*7)
    calls=[]
    def original(self,command,subphase,**kwargs):
        calls.append((command,subphase,kwargs));self.env.rows.append(dict(q_command=np.asarray(command).tolist(),time_s=1/240.))
        if after_step_error:raise ValueError('original physical gate failed')
        return state
    monkeypatch.setattr(controller.GivenContactACT,'_tick',original)
    if after_step_error:
        with pytest.raises(ValueError,match='original physical gate failed'):instance._tick(np.zeros(7),'given_l2_motion')
    else:assert instance._tick(np.zeros(7),'given_l2_motion') is state
    assert len(calls)==1
    audit=instance.env.rows[-1]['formal_act_attribution']
    assert audit['physics_row']==1 and audit['time_s']==state['time_s']
    assert audit['observed_tcp_pose_base'][0][3]==.1 and audit['raw_intent']==instance._current_intent_audit
    assert audit['actual_control_owner']=='ACT_L2' and audit['reference_sha256']==ref.sha256


@pytest.mark.parametrize('fault',['none','factory','run','trace','fingerprint'])
def test_formal_native_route_preserves_new_given_identity_sources_reference_and_cleanup(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    from pathlib import Path
    from test_ausc_policy_m2_deployment_integration import setup_runtime
    from sim.ausc_policy import fresh_expert
    m=module();config,events,created=setup_runtime(tmp_path,monkeypatch,fault)
    ref_root=tmp_path/'reference';ref_root.mkdir();ref=reference(ref_root)
    meta=dict(metadata(ref),handoff_policy_id=m.runtime.handoff.POLICY_ID,episode_id='formal-smoke',subject_id='p000439',split='train',target_id='L1')
    dep=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
        permit=m.runtime.write_new(tmp_path/'permit.json',{}),episode_id='formal-smoke',metadata=meta)
    weight=m.runtime.reference(tmp_path/'checkpoint.pt')
    evidence={k:m.runtime.write_new(tmp_path/(k+'.json'),{}) for k in ('normalization','reference_inventory','source_inventory','processing_freeze','complete')}
    admission=m.runtime.write_new(tmp_path/'admission.json',evidence)
    p=dict(admission=admission,model=dict(checkpoint=weight,complete=evidence['complete']),source_inventory=m.module_inventory(),
        processing=dict(processing_freeze=evidence['processing_freeze']))
    config.pop('m2_act_deployment');config.pop('m2_target_estimate');config['m2_given_contact_deployment']=dep
    monkeypatch.setattr(m,'preflight',lambda *a,**k:dict(deployment=dep,permit=p,attempt=dict(seed=2026102801)))
    runtime_meta=dict(geometry_id=m.runtime.geometry.GEOMETRY_ID,geometry_source_sha256=m.runtime.geometry.GEOMETRY_SOURCE_SHA256,
        freshly_generated=True,geometry_artifacts={},world_from_left_base=np.eye(4).tolist(),plans=[],
        patient_provenance=dict(identity=meta['subject_id'],split=meta['split'],receipt_sha256=ref.snapshot()['sources']['patient_receipt']['sha256']))
    monkeypatch.setattr(fresh_expert,'read_plans',lambda *a:([(SimpleNamespace(name='L1'),{})],deepcopy(runtime_meta)))
    monkeypatch.setattr(fresh_expert,'audit_physics',lambda outcome,rows,**kwargs:dict(outcome,physical_validation=bool(outcome.get('success'))))
    class Controller:
        def __init__(self,env,provider):
            self.env=env;self.estimate_provider=provider;self.l2_handoffs=[];self.l2_actions=[];self.l2_queries=[];self.control_rows=[]
            self.policy_receipt=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA)
            provider.bound=dict(bound_at_s=0.,physics_row=0,reference_sha256=ref.sha256)
        def run(self,points):
            self.env.rows.append(dict(target='L1',q_command=[0.]*7));self.env.value['time_s']=1/240.
            if fault=='run':raise ValueError('formal controller fixture failure')
            if fault=='fingerprint':Path(weight['path']).write_bytes(b'changed')
            self.l2_handoffs.append(dict(target='L1',l2_handoff_success=True,l3_completed=True))
            return dict(success=True,points=[dict(target='L1',success=True)])
        def close_bridge(self):events.append('cpu_reap');return dict(normal_exit=True,reaped=True,exit_code=0)
    def factory(env,kin,checker,**kwargs):
        events.append('formal_factory');assert kwargs['deployment']==dep and not hasattr(env,'raw_capture')
        if fault=='factory':raise ValueError('formal setup fixture failure')
        value=Controller(env,kwargs['estimate_provider']);created.append(value);return value
    monkeypatch.setattr(m,'make_controller',factory)
    # This fixture isolates factory/cleanup and has no native tick API. The
    # neutral observer's real tick/exception/source behavior has its own tests.
    from sim.ausc_policy import m2_given_contact_evaluation
    monkeypatch.setattr(m2_given_contact_evaluation,'attach_native_audit',lambda *a,**k:events.append('neutral_hook'))
    result=fresh_expert.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert result['success'] is (fault=='none') and result['learned_policy'] and not result['expert_demonstration']
    assert result['observation_contract']==m.runtime.pose.observation_contract() and result['task_id']==m.runtime.pose.TASK_ID
    assert result['executed_controller_class']==m.MODULE+'.FormalGivenACT' and result['l2_handoff_attempt_count']==1
    assert events.index('video_close')<events.index('stop') and result['raw_capture_complete'] is False
    if created:
        snapshot=json.loads((tmp_path/'fresh_expert_inputs/given_reference_before_motion.json').read_text())
        assert snapshot['availability_binding']['physics_row']==0
        assert events.index('cpu_reap')<events.index('video_close')
    assert any(Path(path).name=='m2_given_contact_deployment.py' for path in result['runtime_sources_sha256'])


def test_factory_refuses_any_physics_advance_during_cpu_child_startup(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m,p=permit_document(tmp_path);row=p['attempts'][0];row['output']=str(tmp_path/'native-job')
    dep=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
        permit=dict(path='/permit.json',sha256='a'*64),episode_id=row['episode_id'],metadata=row['metadata'])
    provider=m.runtime.pose.GivenPoseProvider(m.runtime.pose.require_route(row['metadata']),row['episode_id'],row['metadata'])
    value=dict(time_s=0.,q=np.zeros(7));env=SimpleNamespace(rows=[],state=lambda:deepcopy(value));closed=[]
    class Bridge:
        runtime={}
        def __init__(self,*a,**k):value['time_s']=1/240.
        def close(self):closed.append(True)
    monkeypatch.setattr(m,'verify_permit',lambda ref:p);monkeypatch.setattr(m,'CPUBridge',Bridge)
    with pytest.raises(ValueError,match='startup|Physics'):m.make_controller(env,deployment=dep,estimate_provider=provider,output=tmp_path/'native-job')
    assert closed==[True]


@pytest.fixture(scope='module')
def current_deployment_population(tmp_path_factory):
    """Actual pure producer/adapter lineage, with original geometry JSON headers."""
    from pathlib import Path
    from test_ausc_given_contact_lossless_mapping import current_population
    patch=pytest.MonkeyPatch();root=tmp_path_factory.mktemp('current-deployment');m=module();write=m.runtime.write_new
    def original_geometry(path,value):
        if Path(path).name=='COMPLETE.json' and isinstance(value,dict) and {'completed','patient','effective_scene_sha256'}<=set(value):
            value=deepcopy(value);value.setdefault('geometry_artifacts',dict(collision_geometry_sha256='d'*64,collision_arrays_sha256='d'*64))
        return write(path,value)
    try:
        with patch.context() as local:
            local.setattr(m.runtime,'write_new',original_geometry)
            _,rows,original,mapping_ref,processing,report=current_population(root,patch)
        mapping,_=m._allocation_mapping(mapping_ref,processing)
        plan=m.runtime.read_bound(processing['plan']);recipe=m.runtime.read_bound(plan['source_recipe'])
        (root/'permit').mkdir()
        _,permit=permit_document(root/'permit');permit.update(processing=processing,allocation_mapping=mapping_ref)
        configs=[];attempts=[]
        for n,index in enumerate((0,11,202,213)):
            row=deepcopy(rows[index]);entry=report['plans'][index];config=m.runtime.read_bound(entry['runtime'])
            scene=next(s for s in mapping['scenes'] if s['subject_id']==row['subject_id'] and s['split']==row['split'])
            prep=m.runtime.read_bound(dict(path=scene['preparation_config']['file'],sha256=scene['preparation_config']['sha256']))
            patient=m.runtime.read_bound(dict(path=scene['patient_receipt']['file'],sha256=scene['patient_receipt']['sha256']))
            config.pop('m2_given_contact_control');config['fresh_expert'].pop('raw_multimodal_capture')
            for key in ('patient_asset','geometry_id','geometry_source','geometry_source_sha256','posture_limits','freshly_generated'):
                if key in prep:config[key]=deepcopy(prep[key])
            row['metadata'].update(patient_asset_sha256=patient['asset_sha256'],installed_patient_receipt_sha256=scene['patient_receipt']['sha256'])
            row.update(episode_id=row['metadata']['episode_id'],attempt_id='current-'+str(n),job_id='current-'+str(n),owner_run_id='current-act-smoke',
                output='/workspace/wam/runs/ausc_policy/current-act-smoke/output/current-'+str(n),seed=config['seed'],runtime_config_sha256=m.runtime_config_sha256(config))
            config['m2_given_contact_deployment']=dict(schema=m.runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
                permit=dict(path=str(root/'permit.json'),sha256='c'*64),episode_id=row['episode_id'],metadata=deepcopy(row['metadata']))
            configs.append(config);attempts.append(row)
        permit['attempts']=attempts;m._validate_permit(permit)
        yield dict(module=m,root=root,mapping=mapping,original=original,plan=plan,recipe=recipe,configs=configs,rows=attempts,permit=permit,report=report)
    finally:patch.undo()


@pytest.mark.parametrize('index',range(4))
def test_current_policy_public_preflight_accepts_real_producer_historical_files(current_deployment_population,monkeypatch,index):
    context=current_deployment_population;m=context['module']
    from pathlib import Path
    from sim.ausc_policy import scene_store
    read=Path.read_bytes;open_file=Path.open;opened=[]
    def metadata_only(path):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', str(path)
        opened.append(str(path));return read(path)
    def forbidden(*args,**kwargs):raise AssertionError('Payload/authoring validator called during deployment metadata preflight')
    def open_metadata(path,*args,**kwargs):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', str(path)
        return open_file(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',metadata_only)
    monkeypatch.setattr(Path,'open',open_metadata)
    for name in ('read_source','source_ref','_binding_metadata','_validate_binding','_layered_metadata','_layered_binding'):
        if hasattr(scene_store,name):monkeypatch.setattr(scene_store,name,forbidden)
    monkeypatch.setattr(m,'verify_permit',lambda ref:deepcopy(context['permit']))
    result=m.preflight(deepcopy(context['configs'][index]))
    assert result['attempt']==context['rows'][index] and opened
    config=context['configs'][index];sources=config['compact_scene_storage']['sources']
    scene=next(s for s in context['mapping']['scenes'] if s['subject_id']==result['attempt']['subject_id'] and s['split']==result['attempt']['split'])
    historical=Path(scene['current_raw']['historical_prepared_complete']['file']).parent
    assert historical!=Path(sources['prepared_complete']['file']).parent
    assert all(sources[role]['file']==str(historical/name) for role,name in {**scene_store.FILES,**scene_store.OPTIONAL}.items() if role in sources)


def current_metadata_call(context,*,config=None,mapping=None,recipe=None,plan=None):
    return context['module']._scene_metadata(config if config is not None else deepcopy(context['configs'][0]),
        context['rows'][0],mapping if mapping is not None else deepcopy(context['mapping']),
        recipe if recipe is not None else deepcopy(context['recipe']),plan=plan if plan is not None else deepcopy(context['plan']))


def current_leaf_reads(monkeypatch,plan):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    read=bridge._CurrentMetadata.read;leaves=[]
    top={bridge._reference(plan[role])['path'] for role in ('allocation_mapping','derivation','source_recipe')}
    def track(reader,ref):
        path=bridge._reference(ref)['path']
        if path not in top:leaves.append(path)
        return read(reader,ref)
    monkeypatch.setattr(bridge._CurrentMetadata,'read',track)
    return leaves


@pytest.mark.parametrize('fault',['old_policy','unknown_policy','missing_policy','schema','optin','mixed_recipe','recipe_unknown',
    'missing_plan','unregistered','historical_parent','historical_complete','foreign_sidecar','missing_sidecar','optional_parent',
    'foreign_patient','source_unknown','late_identity','late_plan','late_scene_source','late_attempt'])
def test_current_deployment_rejects_mixed_policy_and_all_late_headers_before_leaf(current_deployment_population,monkeypatch,fault):
    from pathlib import Path
    from sim.ausc_policy import scene_store
    context=current_deployment_population;config=deepcopy(context['configs'][0]);mapping=deepcopy(context['mapping'])
    recipe=deepcopy(context['recipe']);plan=deepcopy(context['plan']);binding=config['compact_scene_storage'];sources=binding['sources']
    if fault=='old_policy':binding['dependency_policy']=scene_store.SDK_LAYER_DEPENDENCY_POLICY
    elif fault=='unknown_policy':binding['dependency_policy']='unknown-current-policy'
    elif fault=='missing_policy':binding.pop('dependency_policy')
    elif fault=='schema':binding['schema']=scene_store.SDK_SCHEMA
    elif fault=='optin':recipe['current_raw_equivalence']=1
    elif fault=='mixed_recipe':recipe['collector_conditions']['scene_dependencies']['dependency_policy']=scene_store.SDK_LAYER_DEPENDENCY_POLICY
    elif fault=='recipe_unknown':recipe['unapproved']=False
    elif fault=='missing_plan':plan.pop('current_raw_equivalence')
    elif fault=='unregistered':sources['preparation_config']['file']='/unregistered/prepare.json'
    elif fault=='historical_parent':
        old=sources['collision_arrays'];old['file']=str(Path(sources['prepared_complete']['file']).parent/'geometry/collision_geometry.npz')
        plan['accounting_scope']['extra_input_refs'].append(dict(category='source',reference=dict(path=old['file'],sha256=old['sha256'],bytes=old['bytes'])))
    elif fault=='historical_complete':sources['prepared_complete']=deepcopy(mapping['scenes'][0]['current_raw']['historical_prepared_complete'])
    elif fault=='foreign_sidecar':sources['layered_runtime_equivalence']=deepcopy(mapping['scenes'][-1]['current_raw']['layered_runtime_equivalence'])
    elif fault=='missing_sidecar':sources.pop('layered_runtime_equivalence')
    elif fault=='optional_parent':
        sources['stool']=dict(file=str(Path(sources['prepared_complete']['file']).parent/scene_store.OPTIONAL['stool']),sha256='e'*64,bytes=1)
        plan['accounting_scope']['extra_input_refs'].append(dict(category='source',reference=dict(path=sources['stool']['file'],sha256='e'*64,bytes=1)))
    elif fault=='foreign_patient':sources['patient_receipt']=deepcopy(context['configs'][1]['compact_scene_storage']['sources']['patient_receipt'])
    elif fault=='source_unknown':sources['foreign']=deepcopy(sources['patient_receipt'])
    elif fault=='late_identity':mapping['ordered_allocations']['validation'][-1]['subject_id']=mapping['ordered_allocations']['validation'][0]['subject_id']
    elif fault=='late_plan':mapping['ordered_allocations']['validation'][-1]['fresh_plan']=deepcopy(mapping['ordered_allocations']['validation'][0]['fresh_plan'])
    elif fault=='late_scene_source':mapping['scenes'][-1]['current_raw']['historical_prepared_complete']=deepcopy(mapping['scenes'][0]['current_raw']['historical_prepared_complete'])
    else:plan['attempts'][-1]['source_freeze']=deepcopy(plan['attempts'][0]['source_freeze'])
    leaves=current_leaf_reads(monkeypatch,plan)
    with pytest.raises(ValueError):current_metadata_call(context,config=config,mapping=mapping,recipe=recipe,plan=plan)
    assert leaves==[], 'Bad current RAW runtime/header reached leaf JSON: '+str(len(leaves))


@pytest.mark.parametrize('fault',['alias','unknown','unregistered'])
def test_current_deployment_rejects_last_prepared_json_header_before_leaf(current_deployment_population,monkeypatch,fault):
    from pathlib import Path
    from test_ausc_given_contact_lossless_mapping import rebind_current_json
    context=current_deployment_population;plan=deepcopy(context['plan']);path=Path(plan['derivation']['path']);saved=path.read_bytes()
    report=json.loads(saved);prepared=report['layered_authoring']['prepared']
    assert prepared[-1] is not report['scenes'][-1]['derived']['derivation']
    if fault=='alias':prepared[-1]['patient_receipt']=deepcopy(prepared[0]['patient_receipt'])
    elif fault=='unknown':prepared[-1]['unknown']=False
    else:prepared[-1]['patient_receipt']=dict(file='/unregistered/last-patient.json',sha256='a'*64)
    try:
        rebind_current_json(plan,plan['derivation'],report);leaves=current_leaf_reads(monkeypatch,plan)
        with pytest.raises(ValueError):current_metadata_call(context,plan=plan)
        assert leaves==[]
    finally:path.write_bytes(saved)


def replace_current_reference(value,path,ref):
    if isinstance(value,dict):
        if value.get('file',value.get('path'))==str(path) and 'sha256' in value:
            value['sha256']=ref['sha256']
            if 'bytes' in value:value['bytes']=ref['bytes']
        for child in value.values():replace_current_reference(child,path,ref)
    elif isinstance(value,list):
        for child in value:replace_current_reference(child,path,ref)


@pytest.mark.parametrize('fault',['complete_extra','complete_patient','sidecar_index','sidecar_unknown','sidecar_historical','sidecar_policy','duplicate','nonfinite'])
def test_current_deployment_rejects_rehashed_current_complete_and_sidecar_proofs(current_deployment_population,monkeypatch,fault):
    from pathlib import Path
    from test_ausc_given_contact_lossless_mapping import rebind_current_json
    context=current_deployment_population;plan=deepcopy(context['plan']);mapping=deepcopy(context['mapping']);config=deepcopy(context['configs'][0])
    reportpath=Path(plan['derivation']['path']);reportdata=reportpath.read_bytes();report=json.loads(reportdata)
    record=report['current_raw_prepared'][0];role='prepared_complete' if fault.startswith('complete') else 'layered_runtime_equivalence'
    path=Path(record[role]['file']);saved=path.read_bytes();value=json.loads(saved)
    if fault=='complete_extra':value['unapproved']=False
    elif fault=='complete_patient':value['patient']['identity']='p999999'
    elif fault=='sidecar_index':value['scene_index']=False
    elif fault=='sidecar_unknown':value['unknown']=False
    elif fault=='sidecar_historical':value['historical_prepared_complete']=deepcopy(report['current_raw_prepared'][-1]['historical_prepared_complete'])
    elif fault=='sidecar_policy':value['dependency_policy']='hash_bound_static_usd_and_isaacsim60_mdl_v3'
    elif fault=='nonfinite':value['scene_index']=float('nan')
    try:
        rebind_current_json(plan,record[role],value)
        if fault=='duplicate':
            path.write_text(json.dumps(value)[:-1]+',"scene_index":0}')
        ref=context['module'].runtime.reference(path)
        for item in (plan,report,mapping,config):replace_current_reference(item,path,ref)
        rebind_current_json(plan,plan['derivation'],report)
        with pytest.raises(ValueError):current_metadata_call(context,plan=plan,mapping=mapping,config=config)
    finally:path.write_bytes(saved);reportpath.write_bytes(reportdata)
