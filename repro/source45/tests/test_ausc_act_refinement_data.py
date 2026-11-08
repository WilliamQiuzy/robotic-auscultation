"""Invented CPU evidence only; no Native or historical payload access."""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest
from test_ausc_given_contact_capture import new_inputs
from test_ausc_given_contact_recovery import ledger


def module():
    name='sim.ausc_policy.m2_act_refinement_data'
    assert importlib.util.find_spec(name) is not None, 'Typed refinement data API missing'
    return importlib.import_module(name)


def test_query32_bridge_requires_exact_parent_before_any_payload(monkeypatch):
    m=module();assert callable(getattr(m,'export_query32',None)), 'Separate original41 typed query consumer missing'
    calls=[];monkeypatch.setattr(m.rt,'read_bound',lambda r:calls.append(r))
    with pytest.raises(ValueError):m.export_query32({},'unused')
    assert not calls


def query41_fixture(m,tmp_path,monkeypatch):
    """Invented original41 sources/58-window graph, no real archived reader."""
    from pathlib import Path
    root=tmp_path/'query-original41';root.mkdir();(root/'CODE_COMMIT').write_text('synthetic-query41\n')
    names=dict(bank='m2_measured_recovery_data.py',capture_decoder='m2_given_contact_dataset.py',
        role_filter='m2_given_contact_recovery_dataset.py',native_physical='m2_stop_residual_trajectory.py',
        query_prefix='m2_query_recovery_prefix.py',teacher_reader='m2_teacher_stop_reader.py')
    bindings={};rows=[]
    for key,name in names.items():
        file=root/'sim/ausc_policy'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# invented '+name+'\n')
        ref=m.rt.reference(file);bindings[key]=ref;rows.append(dict(file='sim/ausc_policy/'+name,bytes=ref['bytes'],sha256=ref['sha256']))
    manifest=m.rt.write_new(root/'SOURCE_MANIFEST.json',rows)
    source=dict(root=str(root),code_commit='synthetic-query41',manifest_files=len(rows),source_manifest=manifest,code_commit_file=m.rt.reference(root/'CODE_COMMIT'))
    monkeypatch.setattr(m.warm,'SOURCE41',source)
    from test_ausc_act_training_v7 import v7_cache
    p,_=v7_cache(tmp_path,monkeypatch);q=p['measured'];meta=deepcopy(q['metadata']);sample=q['samples']
    arrays={k:np.concatenate([sample[k][:1]]*58) for k in m.ARRAY_KEYS};windows=[]
    for i in range(58):
        w=deepcopy(sample['derived_index']['windows'][0]);w['inclusive_native_ticks']=[23+12*i,35+12*i]
        windows.append(w)
    index=[];cursor=0
    for i,s in enumerate(meta['sources']):
        n=16 if i==3 else 14;s.update(windows=n,split_sample_range=[cursor,cursor+n])
        index.extend(dict(slot_id=s['slot_id'],sample_index=j,window=windows[j]) for j in range(cursor,cursor+n));cursor+=n
        meta['attempt_ledger'][i].update(windows=n,episode=deepcopy(s['declaration']),receipt=deepcopy(s['receipt']))
    path=tmp_path/'query-original-arrays.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    meta.update(schema=m.query.SCHEMA_V2,profile=m.query.PROFILE_V2,data_code=bindings,windows=58,
        evidence_profile=m.query.teacher.EVIDENCE_PROFILE,recorded_source=deepcopy(m.query.teacher.SOURCE32),
        normalization=m.NORMALIZATION,arrays=m.rt.reference(path),array_proofs={k:m.query.array_proof(v) for k,v in arrays.items()},
        index=m.rt.write_new(tmp_path/'query-original-index.json',dict(samples=index,order_sha256=m.rt.digest(index))))
    gate=m.query.admission(meta['attempt_ledger'],profile=m.query.PROFILE_V2);meta['optimization_admission']=gate
    complete=dict(schema=m.query.SCHEMA_V2,profile=m.query.PROFILE_V2,complete=True,windows=58,
        arrays=meta['arrays'],normalization=m.NORMALIZATION,optimization_admission=gate,metadata=m.rt.write_new(tmp_path/'query-original-meta.json',meta))
    parent=m.rt.write_new(tmp_path/'query-original-complete.json',complete);monkeypatch.setattr(m,'SOURCE41_QUERY32',parent)
    return source,parent,meta,arrays


def query_transport_fixture(m,tmp_path,source,parent,meta,arrays):
    root=tmp_path;root.mkdir(parents=True,exist_ok=False)
    with (root/'TRAIN.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    value=dict(schema=m.QUERY_TRANSPORT_SCHEMA,complete=True,parent_bank=parent,reader_source=source,
        original_metadata=m.rt.read_bound(parent)['metadata'],original_index=meta['index'],original_code=meta['data_code'],
        arrays=m.rt.reference(root/'TRAIN.npz'),array_proofs=meta['array_proofs'],windows=58,
        processing_environment=dict(executable=m.recovery_python(),prefix='/opt/wam-runtime/envs/isaacsim60',
            isolated=True,bytecode_disabled=True,cuda_visible_devices='',no_owner=True,no_torch_world=True,linux=True))
    return m.rt.write_new(root/'COMPLETE.json',value)


def test_query_consumer_preserves_original_metadata_arrays_masks_index_and_code(tmp_path,monkeypatch):
    m=module();assert hasattr(m,'QUERY_SCHEMA'), 'Query consumer schema missing'
    source,parent,meta,arrays=query41_fixture(m,tmp_path,monkeypatch)
    transport=query_transport_fixture(m,tmp_path/'transport',source,parent,meta,arrays)
    exit=m.rt.write_new(tmp_path/'exit.json',dict(schema=m.QUERY_EXIT_SCHEMA,normal_exit=True,exit_code=0,
        archive_authenticated_before=True,archive_authenticated_after=True,reader_source=source,
        parent_bank=parent,command_flags=['-I','-B'],sdk_configuration=m.SDK_CONFIGURATION,python=m.recovery_python()))
    monkeypatch.setattr(m,'_original41_query_transport',lambda p,o:(transport,exit))
    ref=m.export_query32(parent,tmp_path/'consumer');complete,new=m.metadata(ref);loaded,again=m.load_arrays(ref)
    assert new['schema']==m.QUERY_SCHEMA and new['original_metadata']==m.rt.read_bound(parent)['metadata']
    assert new['data_code']==meta['data_code'] and new['sources']==meta['sources'] and new['index']==meta['index']
    assert new['source_manifest']==meta['source_manifest'] and new['recorded_source']==meta['recorded_source']
    assert new['reader_source']==source and new['code_binding']==m.code_binding() and again==new
    for k in m.ARRAY_KEYS:np.testing.assert_array_equal(loaded[k],arrays[k])
    from pathlib import Path
    Path(transport['path']).parent.joinpath('TRAIN.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError):m.load_arrays(ref)


@pytest.mark.parametrize('fault',['reader','module_ref','normalization','index','unknown','alias','mask'])
def test_query_bridge_rejects_archive_transport_and_semantic_corruption(tmp_path,monkeypatch,fault):
    m=module();assert hasattr(m,'QUERY_SCHEMA'), 'Query consumer schema missing'
    source,parent,meta,arrays=query41_fixture(m,tmp_path,monkeypatch)
    transport=query_transport_fixture(m,tmp_path/'transport',source,parent,meta,arrays)
    from pathlib import Path
    value=m.rt.read_bound(transport)
    if fault=='reader':source['code_commit']='wrong'
    elif fault=='module_ref':meta['data_code']['bank']['path']='/alias/m2_measured_recovery_data.py'
    elif fault=='normalization':meta['normalization']=dict(meta['normalization'],sha256='0'*64)
    elif fault=='index':
        value['original_index']=dict(meta['index'],sha256='0'*64)
    elif fault=='unknown':Path(transport['path']).parent.joinpath('unknown').write_text('bad')
    elif fault=='alias':
        path=Path(value['arrays']['path']);path.rename(path.with_suffix('.real'));path.symlink_to(path.with_suffix('.real'))
    else:
        arrays['action_is_pad'][0,0]=True
        path=Path(value['arrays']['path']);path.unlink()
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    if fault in ('module_ref','normalization'):
        c=m.rt.read_bound(parent);c['metadata']=m.rt.write_new(tmp_path/'changed-meta.json',meta)
        parent=m.rt.write_new(tmp_path/'changed-complete.json',c);monkeypatch.setattr(m,'SOURCE41_QUERY32',parent)
    if fault=='index':transport=m.rt.write_new(tmp_path/'bad-transport.json',value)
    with pytest.raises((ValueError,OSError)):m._read_original41_query_transport(transport,parent)


@pytest.mark.parametrize('fault',['none','child_failure','changed_after'])
def test_query_child_is_data_only_isolated_and_authenticates_after_failure(tmp_path,monkeypatch,fault):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    m=module();assert hasattr(m,'QUERY_SCHEMA'), 'Query consumer schema missing'
    source,parent,meta,arrays=query41_fixture(m,tmp_path,monkeypatch);checks=[]
    def auth():
        checks.append(True)
        if fault=='changed_after' and len(checks)>1:raise ValueError('archive changed')
        return deepcopy(source)
    monkeypatch.setattr(m,'_authenticate_original41_query',auth)
    monkeypatch.setattr(m.sys,'platform','linux');read=m.rt.read_bound
    monkeypatch.setattr(m.rt,'read_bound',lambda r:dict(ld_library_path='synthetic-sdk') if r==m.SDK_CONFIGURATION else read(r))
    def run(command,**kwargs):
        payload=json.loads(kwargs['input']);env=kwargs['env']
        assert command[:3]==[m.recovery_python(),'-I','-B'] and env['CUDA_VISIBLE_DEVICES']=='' and env['OMNI_KIT_ACCEPT_EULA']=='YES'
        assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
        assert 'AUSC_OWNED_RUN' not in env and 'PYTHONPATH' not in env
        query_transport_fixture(m,Path(payload['output']),source,parent,meta,arrays)
        return SimpleNamespace(returncode=3 if fault=='child_failure' else 0)
    monkeypatch.setattr(m.subprocess,'run',run);output=tmp_path/'child'
    if fault=='none':
        ref,exit=m._original41_query_transport(parent,output)
        assert m.rt.read_bound(exit)['normal_exit'] and m.rt.read_bound(ref)['windows']==58
    else:
        with pytest.raises(ValueError):m._original41_query_transport(parent,output)
        assert not (output/'payloads').exists() and not m.rt.shared._json(output/'reader-exit.json')['normal_exit']
    assert len(checks)>=2


def test_legacy_source32_header_still_rejects_changed_current_controller(tmp_path,monkeypatch):
    m=module();teacher=m.query.teacher;source=teacher.SOURCE32
    commit=tmp_path/'legacy-CODE_COMMIT';commit.write_text(source['code_commit']+'\n')
    freeze=dict(schema=m.data.SOURCE_FREEZE_SCHEMA,code_commit=source['code_commit'],
        code_commit_file=source['code_commit_file'],source_manifest=source['source_manifest'])
    monkeypatch.setattr(teacher.rt,'bound_path',lambda r:commit)
    monkeypatch.setattr(teacher.rt,'read_bound',lambda r:[])
    # Isolate the unchanged current-module comparison; archive validation itself
    # is separately tested with complete invented inventories.
    monkeypatch.setattr(teacher,'_validate_source32_inventory',lambda *a:None)
    with pytest.raises(ValueError,match='incompatible'):
        teacher.validate_source32_header(freeze,evidence_profile=teacher.EVIDENCE_PROFILE,recorded_source=source)


def test_query_metadata_does_not_load_npz_and_consumer_load_checks_all7_bytes(tmp_path,monkeypatch):
    m=module();source,parent,meta,arrays=query41_fixture(m,tmp_path,monkeypatch)
    transport=query_transport_fixture(m,tmp_path/'transport',source,parent,meta,arrays)
    exit=m.rt.write_new(tmp_path/'exit.json',dict(schema=m.QUERY_EXIT_SCHEMA,normal_exit=True,exit_code=0,
        archive_authenticated_before=True,archive_authenticated_after=True,reader_source=source,
        parent_bank=parent,command_flags=['-I','-B'],sdk_configuration=m.SDK_CONFIGURATION,python=m.recovery_python()))
    monkeypatch.setattr(m,'_original41_query_transport',lambda p,o:(transport,exit))
    original=np.load
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('Query metadata opened numerical arrays'))
    ref=m.export_query32(parent,tmp_path/'consumer');m.metadata(ref)
    monkeypatch.setattr(np,'load',original)
    loaded,_=m.load_arrays(ref)
    assert len(loaded['states'])==58


def test_query_transport_rejects_noncanonical_dotdot_path_alias(tmp_path,monkeypatch):
    m=module();source,parent,meta,arrays=query41_fixture(m,tmp_path,monkeypatch)
    transport=query_transport_fixture(m,tmp_path/'transport',source,parent,meta,arrays)
    value=m.rt.read_bound(transport);value['arrays']['path']=str(tmp_path/'transport')+'/../transport/TRAIN.npz'
    from pathlib import Path
    path=Path(transport['path']);path.unlink();m.rt.write_new(path,value)
    alias=dict(m.rt.reference(path),path=str(tmp_path/'transport')+'/../transport/COMPLETE.json')
    with pytest.raises(ValueError):m._read_original41_query_transport(alias,parent)


def proof(audit,boundary):
    entries=[]
    for event in audit:
        raw=event.get('raw_intent')
        if raw and raw['native_tick_in_cell']==0:
            setter=raw['setter'];start=setter['physics_row_start'];end=min(start+12,len(audit))
            entries.append(dict(intent_id=setter['intent_id'],setter_sha256=setter['setter_sha256'],
                physics_row_start=start,physics_row_end_exclusive=end,completed=end-start==12,
                kind=('learner_prefix' if setter['control_owner']=='ACT_L2' else 'excluded') if start<boundary else 'recovery_teacher'))
    return dict(schema='m2_refinement_completed_teacher_roles_v1',boundary_row=boundary,events=entries)


def mixed(audit,boundary):
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    from types import SimpleNamespace
    t=SimpleNamespace(input=SimpleNamespace(target_position=[0.]*7))
    t.set_target=lambda q:setattr(t.input,'target_position',q.tolist())
    d=IntentDispatcher(t,reference_sha256=audit[0]['raw_intent']['setter']['reference_sha256'])
    for i,event in enumerate(audit):
        owner='ACT_L2' if i<boundary else 'TEACHER_L2'
        if i%12==0:
            old=event['raw_intent']['setter']
            d.issue(old['target_q'],time_s=i/240.,physics_row=i,context=old['context'],owner=owner,role='given_l2_motion')
        raw=d.snapshot(time_s=i/240.,physics_row=i);raw['live_target_after_step_q']=raw['live_target_q']
        event['raw_intent']=raw;event['actual_control_owner']=owner
        for side in ('controller_before','controller_after'):event[side]['actual_control_owner']=owner


def test_teacher_windows_keep_mixed_h2_real_short_future_and_verbatim_targets(tmp_path):
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=7);mixed(audit,36)
    result=m.teacher_windows(records,audit,meta,boundary_row=36,role_proof=proof(audit,36))
    assert result['derived_index']['windows'][0]['inclusive_native_ticks'][0]==35
    assert result['derived_index']['windows'][0]['history_actual_owners']==['ACT_L2','ACT_L2']
    assert result['derived_index']['windows'][1]['history_actual_owners']==['ACT_L2','TEACHER_L2']
    assert (~result['action_is_pad']).sum(1).tolist()==[4,3,2,1]
    np.testing.assert_array_equal(result['histories'][0],np.asarray([r['state17'] for r in records[1:3]]))
    target=np.asarray(audit[36]['raw_intent']['setter']['target_q'])
    np.testing.assert_array_equal(result['actions'][0,0],target-result['origins'][0])
    assert np.all(result['actions'][result['action_is_pad']]==0)


@pytest.mark.parametrize('fault',['role','unfinished','preview','owner','physical','source','endpoint','reference'])
def test_corrupt_or_noncompleted_teacher_never_supplies_labels(tmp_path,fault):
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=5);p=proof(audit,24)
    if fault=='role':p['events'][2]['kind']='perturbation'
    elif fault=='unfinished':p['events'][2]['completed']=False
    elif fault=='preview':p['events'][2]['kind']='unexecuted_preview'
    elif fault=='owner':audit[24]['actual_control_owner']='ACT_L2'
    elif fault=='physical':audit[30]['physical_violations']=['force']
    elif fault=='source':records[0]['target_estimate_source_id']='invented'
    elif fault=='endpoint':records[1]['q_command'][0]+=.1
    elif fault=='reference':audit[24]['raw_intent']['setter']['reference_sha256']='0'*64
    try:r=m.teacher_windows(records,audit,meta,boundary_row=24,role_proof=p)
    except ValueError:return
    assert all(w['inclusive_native_ticks'][0]>=35 for w in r['derived_index']['windows'])


def test_parent_subsequence_checks_all_seven_arrays_masks_semantics_order(tmp_path):
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=8)
    enriched=m.teacher_windows(records,audit,meta,boundary_row=24,role_proof=proof(audit,24))
    parent=deepcopy(enriched)
    for k in m.ARRAY_KEYS:parent[k]=parent[k][[0,2]].copy()
    parent['derived_index']['windows']=[deepcopy(enriched['derived_index']['windows'][i]) for i in (0,2)]
    assert m.parent_subsequence(parent,enriched)==[0,2]
    for key in m.ARRAY_KEYS:
        bad=deepcopy(parent)
        if key=='action_is_pad':bad[key][0,0]=True
        else:bad[key].flat[0]+=.001
        with pytest.raises(ValueError):m.parent_subsequence(bad,enriched)
    parent['derived_index']['windows'].reverse()
    with pytest.raises(ValueError):m.parent_subsequence(parent,enriched)


def test_enrichment_pins_parent_and_qualification_before_reading_manifest(monkeypatch):
    m=module();calls=[];monkeypatch.setattr(m.rt,'read_bound',lambda ref:calls.append(ref))
    with pytest.raises(ValueError):m.export_enriched_dart({},dict(m.SOURCE37_BANK,bytes=1),m.SOURCE37_QUALIFICATION,'unused')
    assert not calls


def test_archived_full_reader_is_a_distinct_finite_transport_api():
    m=module();assert callable(m._original40_transport)
    assert callable(m._authenticate_original40)
    assert callable(m._read_original40_transport)


def test_archived_transport_rejects_hash_tamper_and_omitted_qualified_slot(tmp_path,monkeypatch):
    m=module()
    assert callable(m._read_original40_transport)
    payload=m.rt.write_new(tmp_path/'capture.json',dict(records=[],audit=[],metadata={},sidecar={}))
    slots=[r['slot_id'] for r in m.dart.planned_requests()][:17]
    bundle=dict(schema='m2_act_refinement_original40_transport_v1',complete=True,manifest={},parent_bank=m.SOURCE37_BANK,
        qualification=m.SOURCE37_QUALIFICATION,physical_reader={},fixed_denominator=20,qualified_slots=slots,
        sources=[dict(slot_id=slot,payload=payload) for slot in slots])
    ref=m.rt.write_new(tmp_path/'complete.json',bundle)
    monkeypatch.setattr(m,'_authenticate_original40',lambda q:bundle['physical_reader'])
    with pytest.raises(ValueError):m._read_original40_transport(ref,{},slots[:-1])
    Path=__import__('pathlib').Path;Path(payload['path']).write_text('{}')
    with pytest.raises(ValueError):m._read_original40_transport(ref,{},slots)


def synthetic_archive(m,tmp_path,monkeypatch):
    root=tmp_path/'archive';root.mkdir();(root/'CODE_COMMIT').write_text('synthetic-original40\n')
    file=root/'reader.py';file.write_text('# invented CPU reader fixture\n')
    inventory=[dict(file='reader.py',bytes=file.stat().st_size,sha256=m.rt.shared.sha256(file))]
    manifest=m.rt.write_new(root/'SOURCE_MANIFEST.json',inventory)
    descriptor=dict(root=str(root),code_commit='synthetic-original40',files=1,manifest=manifest)
    monkeypatch.setattr(m,'SOURCE40',descriptor)
    return dict(reader_source=descriptor),root,file


@pytest.mark.parametrize('fault',['none','changed','omitted','unknown','symlink','hardlink','commit'])
def test_original_archive_authenticates_full_inventory_and_rejects_aliases(tmp_path,monkeypatch,fault):
    m=module();q,root,file=synthetic_archive(m,tmp_path,monkeypatch)
    if fault=='changed':file.write_text('# changed source\n')
    elif fault=='omitted':file.unlink()
    elif fault=='unknown':(root/'extra.py').write_text('# undeclared\n')
    elif fault=='symlink':file.rename(root/'real.py');file.symlink_to(root/'real.py')
    elif fault=='hardlink':__import__('os').link(file,tmp_path/'hardlink.py')
    elif fault=='commit':(root/'CODE_COMMIT').write_text('different\n')
    if fault=='none':assert m._authenticate_original40(q)==q['reader_source']
    else:
        with pytest.raises((ValueError,OSError)):m._authenticate_original40(q)


def transport_fixture(m,tmp_path,monkeypatch,root,manifest,slots):
    """Invented hashed transport, never an actual original40/Native result."""
    import gzip,json
    root.mkdir(parents=True,exist_ok=True);sources=[]
    for slot in slots:
        path=root/(slot+'.json.gz')
        with gzip.open(path,'xt') as stream:json.dump(dict(invented=True),stream)
        sources.append(dict(slot_id=slot,payload=m.rt.reference(path)))
    actual=tmp_path/'cpu-binary-fixture';actual.write_text('synthetic environment bytes')
    original_bound=m.rt.bound_path
    prefix='/opt/wam-runtime/envs/isaacsim60'
    def fake_ref(name):return dict(m.rt.reference(actual),path=prefix+'/'+name)
    fake_paths={prefix+'/'+name for name in ('bin/python','lib/numpy.py','lib/scipy.py','lib/ruckig.so')}
    def bound(ref):
        if ref.get('path') in fake_paths:
            assert ref['sha256']==m.rt.shared.sha256(actual) and ref['bytes']==actual.stat().st_size
            return actual
        return original_bound(ref)
    monkeypatch.setattr(m.rt,'bound_path',bound)
    fingerprint=dict(prefix=prefix,executable=prefix+'/bin/python',python=fake_ref('bin/python'),
        python_version='3.12.15',numpy_version='2.3.1',numpy=fake_ref('lib/numpy.py'),scipy_version='synthetic',
        scipy=fake_ref('lib/scipy.py'),ruckig_version='0.15.3',ruckig=fake_ref('lib/ruckig.so'),
        numeric_engine_sha256=m.dart.teacher.NUMERIC_ENGINE_SHA256,isolated=True,bytecode_disabled=True,
        cuda_visible_devices='',ld_library_path='synthetic-sdk-libraries',no_owner=True,no_torch_world=True)
    bundle=dict(schema=m.TRANSPORT_SCHEMA,complete=True,manifest=manifest,parent_bank=m.SOURCE37_BANK,
        qualification=m.SOURCE37_QUALIFICATION,physical_reader=m.SOURCE40,fixed_denominator=20,qualified_slots=slots,
        sources=sources,windows=274,full_original_physical_kernel=True,processing_environment=fingerprint)
    return m.rt.write_new(root/'COMPLETE.json',bundle)


def test_all17_hashed_transport_positive_and_each_payload_tamper_rejects(tmp_path,monkeypatch):
    m=module();slots=[r['slot_id'] for r in m.dart.planned_requests()][:17]
    ref=transport_fixture(m,tmp_path,monkeypatch,tmp_path/'transport',{},slots)
    value=m._read_original40_transport(ref,{},slots);assert len(value['sources'])==17
    from pathlib import Path
    Path(value['sources'][-1]['payload']['path']).write_bytes(b'changed last capture')
    with pytest.raises(ValueError):m._read_original40_transport(ref,{},slots)


@pytest.mark.parametrize('fault',['none','child_failure','archive_changed_after'])
def test_original_reader_child_is_finite_normal_and_always_reauthenticates_and_cleans(tmp_path,monkeypatch,fault):
    import json
    from types import SimpleNamespace
    m=module();slots=[r['slot_id'] for r in m.dart.planned_requests()][:17];checks=[];launches=[]
    manifest=dict(path='/synthetic/manifest.json',bytes=1,sha256='a'*64)
    monkeypatch.setattr(m,'_qualification',lambda *a:dict(reader_source=m.SOURCE40,manifest=manifest))
    def authenticate(q):
        checks.append(True)
        if fault=='archive_changed_after' and len(checks)==2:raise ValueError('Original40 source changed after child')
        return deepcopy(m.SOURCE40)
    monkeypatch.setattr(m,'_authenticate_original40',authenticate)
    read=m.rt.read_bound
    def read_bound(ref):
        if ref==m.SDK_CONFIGURATION:return dict(ld_library_path='synthetic-sdk-libraries')
        if ref==m.SOURCE37_BANK:return dict(metadata={'synthetic_parent':True})
        if ref=={'synthetic_parent':True}:return dict(attempt_ledger=[dict(request=dict(slot_id=s),status='qualified') for s in slots])
        return read(ref)
    monkeypatch.setattr(m.rt,'read_bound',read_bound);monkeypatch.setattr(m.sys,'platform','linux')
    def run(command,**kwargs):
        launches.append(command);payload=json.loads(kwargs['input']);env=kwargs['env']
        assert command[:3]==[m.recovery_python(),'-I','-B'] and len(payload['qualified_slots'])==17
        assert env['LD_LIBRARY_PATH']=='synthetic-sdk-libraries' and env['CUDA_VISIBLE_DEVICES']=='' and 'AUSC_OWNED_RUN' not in env
        assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
        from pathlib import Path
        transport_fixture(m,tmp_path,monkeypatch,Path(payload['output']),manifest,slots)
        return SimpleNamespace(returncode=2 if fault=='child_failure' else 0)
    monkeypatch.setattr(m.subprocess,'run',run)
    output=tmp_path/'read'
    if fault=='none':
        ref,exit=m._original40_transport(manifest,m.SOURCE37_BANK,m.SOURCE37_QUALIFICATION,output)
        assert m.rt.read_bound(exit)['normal_exit'] and len(m.rt.read_bound(ref)['sources'])==17
    else:
        with pytest.raises(ValueError):m._original40_transport(manifest,m.SOURCE37_BANK,m.SOURCE37_QUALIFICATION,output)
        assert not (output/'payloads').exists() and m.rt.shared._json(output/'reader-exit.json')['normal_exit'] is False
    assert len(checks)==2 and len(launches)==1


def historical_parent_fixture(m,tmp_path,monkeypatch):
    """Invented bank graph tests authentication; never executes original40."""
    from pathlib import Path
    root=tmp_path/'original40';root.mkdir();(root/'CODE_COMMIT').write_text('invented-reader\n')
    names=dict(bank='m2_measured_dart_data.py',capture_decoder='m2_given_contact_dataset.py',
        role_filter='m2_given_contact_recovery_dataset.py',teacher_reader='m2_teacher_stop_reader.py',
        sample_validator='m2_measured_recovery_data.py',collector='m2_given_contact_recovery.py')
    bindings={};rows=[]
    for key,name in names.items():
        file=root/'sim/ausc_policy'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# synthetic '+name+'\n')
        ref=m.rt.reference(file);bindings[key]=ref;rows.append(dict(file='sim/ausc_policy/'+name,bytes=ref['bytes'],sha256=ref['sha256']))
    inventory=m.rt.write_new(root/'SOURCE_MANIFEST.json',rows)
    source=dict(root=str(root),code_commit='invented-reader',files=6,manifest=inventory);monkeypatch.setattr(m,'SOURCE40',source)
    _,records,audit,meta=new_inputs(tmp_path,n=22)
    sample=m.teacher_windows(records,audit,meta,boundary_row=24,role_proof=proof(audit,24))
    arrays={k:np.repeat(sample[k][:1],274,axis=0) for k in m.ARRAY_KEYS}
    arrays['observation_times']=.1+np.arange(274)*.05
    windows=[]
    for i in range(274):
        w=deepcopy(sample['derived_index']['windows'][0]);delta=12*i
        for key in ('inclusive_native_ticks','history_native_ticks','raw_intent_native_ticks'):w[key]=[v+delta for v in w[key]]
        for key in ('observation_indices','action_record_indices'):w[key]=[v+i for v in w[key]]
        windows.append(w)
    sample['derived_index']['windows']=windows
    archive=tmp_path/'parent.npz'
    with archive.open('xb') as stream:np.savez(stream,**arrays)
    arrays_ref=m.rt.reference(archive);array_proofs={k:m.dart.measured.array_proof(v) for k,v in arrays.items()}
    attempts=[];ledger=[];sources=[];index=[];slots=[];cursor=0;count_qualified=0
    for i,request in enumerate(m.dart.planned_requests()):
        available=i not in (4,5,6);slots.append(dict(request=request,status='prepared' if available else 'cpu_unavailable',reason='',evidence={}))
        episode=dict(episode_id='synthetic-'+str(i),**{k:request[k] for k in ('subject_id','split','target_id')}) if available else None
        attempt=dict(request=request,episode=episode,nominal_source_config=None,outcome='collected' if available else 'cpu_unavailable',
            reason='synthetic',evidence={});attempts.append(attempt)
        entry=dict(attempt,status='qualified' if available else 'cpu_unavailable',windows=0,infrastructure_contamination=False)
        if available:
            n=18 if count_qualified==16 else 16;count_qualified+=1
            receipt=dict(native_reset_identity=dict(pid=100+i,pgid=100+i,sid=100+i,start_ticks=str(100+i)))
            entry.update(windows=n,receipt=receipt,native_reset_identity=receipt['native_reset_identity'])
            sources.append(dict(slot_id=request['slot_id'],declaration=episode,receipt=receipt,windows=n,split_sample_range=[cursor,cursor+n]))
            index.extend(dict(slot_id=request['slot_id'],sample_index=j,window=sample['derived_index']['windows'][j]) for j in range(cursor,cursor+n));cursor+=n
        ledger.append(entry)
    precollection=dict(schema=m.dart.CPU_PLAN_SCHEMA,frozen_utc='2026-11-07T00:00:00Z',frozen_before_first_native=True,slots=slots)
    # CPU-unavailable entries require a reason in the frozen plan.
    for slot in slots:
        if slot['status']=='cpu_unavailable':slot['reason']='synthetic unavailable'
    source_manifest=m.rt.write_new(tmp_path/'source-manifest.json',dict(attempts=attempts))
    monkeypatch.setattr(m.dart,'_headers',lambda *a:(attempts,{},precollection))
    gate=m.dart.admission(ledger,precollection=precollection);assert gate['pilot_optimization_eligible']
    index_ref=m.rt.write_new(tmp_path/'index.json',dict(samples=index,order_sha256=m.rt.digest(index)))
    metadata=dict(schema=m.dart.SCHEMA,profile=m.dart.PROFILE_SOURCE37,source_manifest=source_manifest,windows=274,
        normalization=m.NORMALIZATION,normalization_refitted=False,base_dataset=m.dart.BASE29,base_counts=m.dart.BASE_COUNTS,
        fixed_denominator=20,array_proofs=array_proofs,arrays=arrays_ref,optimization_admission=gate,data_code=bindings,
        precollection_plan=precollection,attempt_ledger=ledger,sources=sources,index=index_ref)
    metadata_ref=m.rt.write_new(tmp_path/'metadata.json',metadata)
    bank=m.rt.write_new(tmp_path/'bank.json',dict(schema=m.dart.SCHEMA,complete=True,profile=m.dart.PROFILE_SOURCE37,metadata=metadata_ref,
        windows=274,normalization=m.NORMALIZATION,arrays=arrays_ref,optimization_admission=gate))
    qualification=m.rt.write_new(tmp_path/'qualification.json',dict(schema='act10_source40_independent_dart_pilot_v1',complete=True,
        bank_qualified=True,bank=bank,windows=274,planned=20,fixed_denominator=20,prepared=17,seven_arrays_masks_loaded=True,
        optimization_admission=gate,manifest=source_manifest,array_proofs=array_proofs,reader_source=source))
    monkeypatch.setattr(m,'SOURCE37_BANK',bank);monkeypatch.setattr(m,'SOURCE37_QUALIFICATION',qualification)
    return bank,qualification,archive,Path(index_ref['path']),Path(bindings['capture_decoder']['path'])


@pytest.mark.parametrize('fault',['none','reader','array','index'])
def test_pinned_original_parent_positive_and_reader_array_index_negatives(tmp_path,monkeypatch,fault):
    m=module();bank,q,arrays,index,reader=historical_parent_fixture(m,tmp_path,monkeypatch)
    if fault=='reader':reader.write_text('# changed archived reader')
    elif fault=='array':arrays.write_bytes(b'changed raw-array fixture')
    elif fault=='index':index.write_text('{}')
    if fault=='none':
        actual,meta=m._historical_parent(bank,q);assert len(actual['states'])==274 and len(meta['sources'])==17
    else:
        with pytest.raises(ValueError):m._historical_parent(bank,q)
