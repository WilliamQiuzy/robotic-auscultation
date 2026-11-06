"""Software probe ownership/admission tests. No Kit or GPU is executed."""
from copy import deepcopy
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


NAME='sim.ausc_policy.layered_runtime_equivalence_probe'


def module():
    assert importlib.util.find_spec(NAME) is not None, 'Owned software equivalence probe is missing'
    return importlib.import_module(NAME)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode())
    return dict(file=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path,monkeypatch):
    m=module();from sim.ausc_policy import native_usd_preflight as pre,scene_store as store
    source=dict(root=str(Path(m.__file__).parents[2]),code_commit='a'*40,source_manifest=dict(file='/frozen/SOURCE_MANIFEST.json',bytes=2,sha256='a'*64))
    (tmp_path/'base/locks').mkdir(parents=True)
    frozen_source=deepcopy(source)
    monkeypatch.setattr(pre,'source_snapshot',lambda:deepcopy(frozen_source))
    calls=[]
    def metadata(binding):
        calls.append(binding['sources']['patient_receipt']['file'])
        refs=binding['sources']
        return tuple(pre.decode(pre.read_file(refs[k]['file'])[1]) for k in ('preparation_config','prepared_complete','patient_receipt'))
    monkeypatch.setattr(store,'_binding_metadata',metadata)
    config=dict(schema='m2_layered_runtime_equivalence_probe_config_v1',experiment='layered_runtime_equivalence_probe',
        base=str(tmp_path/'base'),runtime_project=source['root'],isaac_python=pre.NATIVE_PYTHON,
        ld_library_path='/opt/wam-runtime/sysroot/usr/lib/x86_64-linux-gnu',time_limit_seconds=1800,
        child_timeout_seconds=1740,source=source,cases=[])
    for case_id,subject_id,split,site in m.CASES:
        root=tmp_path/case_id
        patient=dict(identity=subject_id,split=split,non_patient_scene_unchanged=True)
        sources={k:write(root/name,value) for k,name,value in (
            ('preparation_config','prepare.json',dict(prepare_only=True)),
            ('prepared_complete','COMPLETE.json',dict(completed=True,patient=patient)),
            ('patient_receipt','patient_receipt.json',patient),
            ('scene','scene.usdc',b'USD body must not be read before all eight headers'),
            ('scene_base','base.usdc',b'BASE body'),
            ('scene_authoring_receipt','authoring.json',dict(schema='historical')),
            ('layered_runtime_equivalence','equivalence.json',dict(schema='current-raw-sidecar')))}
        binding=dict(schema='m2_compact_scene_storage_v3',dependency_policy='hash_bound_static_usd_current_raw_and_sdk_v4',
            store_root=str(tmp_path/'cas'),batch_id='fixture',sources=sources)
        identity=dict(episode_id=case_id,subject_id=subject_id,split=split,target_id=site,target_source='given_contact_pose')
        runtime=dict(compact_scene_storage=binding,m2_given_contact_control=dict(schema='m2_given_contact_pose_control_v1',
            episode_id=case_id,metadata=identity),fresh_expert=dict(raw_multimodal_capture=dict(
            enabled=True,profile='compact',episode_id=case_id,compact=dict(metadata=identity))))
        config['cases'].append(dict(case_id=case_id,subject_id=subject_id,split=split,site=site,
            runtime_config=write(root/'runtime.json',runtime)))
    path=tmp_path/'probe.json';write(path,config)
    return m,pre,store,config,path,calls


def rewrite(path,config):write(path,config)


def change_runtime(config,index,mutate):
    ref=config['cases'][index]['runtime_config'];path=Path(ref['file']);value=json.loads(path.read_bytes());mutate(value)
    config['cases'][index]['runtime_config']=write(path,value)


def test_original_dispatch_is_one_literal_only():
    m=module();from sim.ausc_policy.run import MODULES
    assert MODULES['layered_runtime_equivalence_probe']==NAME and m.EXPERIMENT=='layered_runtime_equivalence_probe'


@pytest.mark.parametrize('fault',['unknown','bool_time','float_time','wrong_schema','missing_case','extra_case','reorder',
    'duplicate_case','test','wrong_id','wrong_site','bad_ref','ref_extra','old_policy','missing_sidecar','learned',
    'sidecar_role','last_metadata','false_zero','config_source','runtime_project'])
def test_whole_population_refusal_precedes_payload_child_and_app(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    if fault=='unknown':c['allow_success']=True
    elif fault=='bool_time':c['time_limit_seconds']=True
    elif fault=='float_time':c['child_timeout_seconds']=1740.0
    elif fault=='wrong_schema':c['schema']='old'
    elif fault=='missing_case':c['cases'].pop()
    elif fault=='extra_case':c['cases'].append(deepcopy(c['cases'][0]))
    elif fault=='reorder':c['cases'].reverse()
    elif fault=='duplicate_case':c['cases'][-1]=deepcopy(c['cases'][0])
    elif fault=='test':c['cases'][-1]['split']='test'
    elif fault=='wrong_id':c['cases'][-1]['subject_id']='p001005'
    elif fault=='wrong_site':c['cases'][-1]['site']='L5'
    elif fault=='bad_ref':c['cases'][-1]['runtime_config']['bytes']=False
    elif fault=='ref_extra':c['cases'][-1]['runtime_config']['success']=True
    elif fault=='old_policy':change_runtime(c,-1,lambda v:v['compact_scene_storage'].update(dependency_policy='hash_bound_static_usd_and_isaacsim60_mdl_v3'))
    elif fault=='missing_sidecar':change_runtime(c,-1,lambda v:v['compact_scene_storage']['sources'].pop('layered_runtime_equivalence'))
    elif fault=='learned':change_runtime(c,-1,lambda v:v.update(m2_act_deployment={}))
    elif fault=='sidecar_role':change_runtime(c,-1,lambda v:v['compact_scene_storage']['sources'].update(layered_runtime_equivalence=v['compact_scene_storage']['sources']['scene']))
    elif fault=='last_metadata':
        def mutate(v):
            ref=v['compact_scene_storage']['sources']['patient_receipt'];v['compact_scene_storage']['sources']['patient_receipt']=write(Path(ref['file']),dict(identity='p001191',split='test',non_patient_scene_unchanged=True))
        change_runtime(c,-1,mutate)
    elif fault=='false_zero':change_runtime(c,-1,lambda v:v['fresh_expert']['raw_multimodal_capture'].update(enabled=1))
    elif fault=='config_source':c['source']['code_commit']='b'*40
    else:c['runtime_project']=str(tmp_path)
    rewrite(path,c)
    real=pre.read_file
    def read(p):
        assert Path(p).suffix!='.usdc','scene payload read before whole header admission'
        return real(p)
    monkeypatch.setattr(pre,'read_file',read)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('child before population admission'))
    with pytest.raises(ValueError):m.load_inputs(path)


@pytest.mark.parametrize('fault',['duplicate','nonfinite','symlink','hardlink','relative','dot_alias'])
def test_raw_config_json_and_canonical_single_file_paths(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    if fault=='duplicate':path.write_text('{"schema":1,"schema":2}')
    elif fault=='nonfinite':path.write_text('{"value":NaN}')
    elif fault=='symlink':alias=tmp_path/'alias';alias.symlink_to(path);path=alias
    elif fault=='hardlink':os.link(path,tmp_path/'alias')
    elif fault=='relative':path=Path('probe.json')
    else:path=str(path.parent)+'/./probe.json'
    with pytest.raises(ValueError):m.load_inputs(path)


def test_admission_reads_all_headers_then_full_preflights_without_usd_import(tmp_path,monkeypatch):
    import builtins
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    # Other tests may already have imported USD in the shared pytest process.
    # Check this metadata call's imports directly, even if pxr is cached.
    prior={name:module for name,module in sys.modules.items() if name=='pxr' or name.startswith('pxr.')}
    original_import=builtins.__import__
    def metadata_import(name,*args,**kwargs):
        assert name!='pxr' and not name.startswith('pxr.'),'USD import during metadata admission'
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',metadata_import)
    value=m.load_inputs(path)
    assert len(seen)==8 and len(value['cases'])==8
    assert {name:module for name,module in sys.modules.items() if name=='pxr' or name.startswith('pxr.')}==prior
    assert value['config_ref']==pre.file_ref(path)


@pytest.mark.parametrize('fault',['config','source','runtime','metadata','payload'])
def test_post_checks_detect_mutations(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);value=m.load_inputs(path)
    if fault=='config':path.write_bytes(path.read_bytes()+b' ')
    elif fault=='source':monkeypatch.setattr(pre,'source_snapshot',lambda:dict(c['source'],code_commit='b'*40))
    elif fault=='runtime':Path(c['cases'][0]['runtime_config']['file']).write_bytes(b'{}')
    else:
        runtime=value['cases'][0]['runtime'];name='patient_receipt' if fault=='metadata' else 'scene'
        Path(runtime['compact_scene_storage']['sources'][name]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError):m.rebind(value,full=True)


def fake_native(m,pre,store,monkeypatch,events,*,failure=None,close_failure=False):
    monkeypatch.setattr(pre,'clean_parent',lambda:events.append('clean') or dict(pxr_modules=[],standalone_libraries=[]))
    identity=dict(pid=300,ppid=200,pgid=100,sid=100,start_ticks='1')
    monkeypatch.setattr(pre,'native_identity',lambda:identity)
    def validate(runtime,path):
        events.append('preflight')
        return dict(passed=True,full_original_validator=True,parent_USD_clean=True,child_reaped=True,
            child_exit_code=0,physics_executed=False,kit_started=False,training_eligible=False,
            config_ref=pre.file_ref(path),parent_identity=identity,source=pre.source_snapshot())
    monkeypatch.setattr(pre,'validate_before_kit',validate)
    class Stage:
        def GetUsedLayers(self):return [SimpleNamespace(anonymous=True,identifier='anon:fixture')]
        def GetRootLayer(self):return self.GetUsedLayers()[0]
    def prepared(binding):
        assert events.count('preflight')==8
        events.append('stage')
        if failure is not None and events.count('stage')==3:raise failure
        return Stage()
    monkeypatch.setattr(store,'prepared_runtime_stage',prepared)
    class App:
        def __init__(self,settings):
            assert events.count('preflight')==8 and events[-1]=='clean'
            assert settings['fast_shutdown'] is False
            events.append('app')
        def update(self):pytest.fail('software probe advanced app')
        def close(self):
            events.append('close')
            if close_failure:raise RuntimeError('close failed')
    monkeypatch.setattr(m,'simulation_app_class',lambda:App)
    monkeypatch.setattr(m,'registry_diagnostics',lambda:dict(version=[26,8],modules={},registry={}))
    monkeypatch.setattr(store,'runtime_stage_leases',lambda:[])
    return identity


def test_native_one_app_eight_fresh_private_stages_then_close_and_software_only(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    fake_native(m,pre,store,monkeypatch,events)
    result=m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert events.count('app')==events.count('close')==1 and events.count('stage')==8
    assert result['completed'] is True and len(result['cases'])==8
    assert all(result[key] is False for key in ('physics_executed','physical_validation','model_qualification','training_eligible'))
    assert (tmp_path/'native/COMPLETE.json').is_file()


@pytest.mark.parametrize('cleanup_failure',[False,True])
def test_native_preserves_original_oracle_exception_and_partial_progress_on_close_error(tmp_path,monkeypatch,cleanup_failure):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[];error=ValueError('original RAW refusal')
    fake_native(m,pre,store,monkeypatch,events,failure=error,close_failure=cleanup_failure)
    with pytest.raises(ValueError,match='original RAW refusal') as caught:
        m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert caught.value is error and events.count('close')==1
    report=json.loads((tmp_path/'native/FAILURE.json').read_bytes())
    assert len(report['cases'])==2 and report['failed_case']=='given-p000465-r3'
    assert report['error']['message']=='original RAW refusal' and report['completed'] is False
    assert bool(report['cleanup_error'])==cleanup_failure
    assert not (tmp_path/'native/COMPLETE.json').exists()


def test_native_rejects_forged_preflight_before_app(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    fake_native(m,pre,store,monkeypatch,events)
    monkeypatch.setattr(pre,'validate_before_kit',lambda *a:dict(passed=True,child_reaped=False))
    with pytest.raises(ValueError):m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert 'app' not in events


def test_native_anonymous_layers_and_post_close_rebinding_are_mandatory(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    fake_native(m,pre,store,monkeypatch,events)
    monkeypatch.setattr(store,'prepared_runtime_stage',lambda *_:SimpleNamespace(GetRootLayer=lambda:SimpleNamespace(anonymous=False)))
    with pytest.raises(ValueError):m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert events.count('close')==1


@pytest.mark.parametrize('mode',['timeout','nonzero','success'])
def test_single_native_transport_reaps_same_group_and_records_identity(tmp_path,monkeypatch,mode):
    m=module();pidfile=tmp_path/'pid'
    monkeypatch.setattr(m,'process_identity',lambda pid:dict(pid=pid,pgid=os.getpgid(pid),sid=os.getsid(pid),start_ticks='local-cpu-test'))
    script="import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
    script+=('time.sleep(20)' if mode=='timeout' else 'sys.exit(7)' if mode=='nonzero' else 'sys.exit(0)')
    command=[sys.executable,'-B','-c',script,str(pidfile)]
    status=m.run_native_child(command,dict(os.environ),tmp_path,timeout_s=.5)
    assert status['reaped'] is True and status['identity']['pid']==int(pidfile.read_text())
    assert status['identity']['pgid']==os.getpgrp() and status['identity']['sid']==os.getsid(0)
    assert status['timed_out']==(mode=='timeout') and status['command']==command
    with pytest.raises(ProcessLookupError):os.kill(status['pid'],0)


def test_wrapper_two_live_idle_checks_inside_original_lock_and_one_native(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    output=tmp_path/'base/runs/ausc_policy/probe/output';output.parent.mkdir(parents=True)
    parent=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),start_ticks='1')
    monkeypatch.setattr(m,'require_owner',lambda *a,**k:parent)
    monkeypatch.setattr(m,'remaining_descendants',lambda *_:[])
    import fcntl
    flock=fcntl.flock
    def lock(fd,op):events.append('lock');return flock(fd,op)
    monkeypatch.setattr(fcntl,'flock',lock)
    monkeypatch.setattr(m,'gpu_idle',lambda:events.append('idle') or dict(pids=[]))
    def child(command,environment,folder,**kw):
        events.append('child');assert kw['timeout_s']<=1740 and environment['CUDA_VISIBLE_DEVICES']=='0'
        status=dict(pid=123,identity=dict(pid=123,pgid=parent['pgid'],sid=parent['sid'],start_ticks='22'),
            returncode=0,reaped=True,timed_out=False,command=command,error=None)
        (folder/'native').mkdir();write(folder/'native/COMPLETE.json',good_receipt(m,m.load_inputs(path),status,parent))
        return status
    monkeypatch.setattr(m,'run_native_child',child)
    m.run_owned(path,output)
    assert events[:4]==['lock','idle','idle','child'] and events.count('child')==1
    assert (output/'COMPLETE.json').is_file()


def good_receipt(m,inputs,status,parent):
    from sim.ausc_policy.native_lifecycle import shared_app_settings
    identity=dict(status['identity'],ppid=parent['pid'])
    return dict(schema=m.RESULT_SCHEMA,completed=True,config_ref=inputs['config_ref'],source=inputs['source'],identity=identity,
        cases=[dict(case_id=c['case_id'],runtime_config=c['runtime_config'],returned=True,private_anonymous=True,layer_identifiers=['anon:fixture']) for c in inputs['cases']],
        cpu_preflights=[dict(passed=True,full_original_validator=True,parent_USD_clean=True,child_reaped=True,child_exit_code=0,
            physics_executed=False,kit_started=False,training_eligible=False,config_ref=c['runtime_config'],
            source=inputs['source'],parent_identity=identity) for c in inputs['cases']],
        failed_case=None,registry={},normal_close=dict(app_close_returned=True,fast_shutdown=False,software_only=True,
            stage_cache_leases=[],settings=shared_app_settings()),error=None,cleanup_error=None,**m.FALSE_FLAGS)


@pytest.mark.parametrize('occupied_check',[1,2])
def test_occupied_gpu_never_launches_native(tmp_path,monkeypatch,occupied_check):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);output=tmp_path/'out';checks=[]
    monkeypatch.setattr(m,'require_owner',lambda *a,**k:{})
    def idle():
        checks.append(1)
        if len(checks)==occupied_check:raise RuntimeError('GPU occupied')
        return dict(pids=[])
    monkeypatch.setattr(m,'gpu_idle',idle)
    monkeypatch.setattr(m,'run_native_child',lambda *a,**k:pytest.fail('occupied GPU launched native'))
    with pytest.raises(RuntimeError,match='GPU occupied'):m.run_owned(path,output)


def test_source_has_no_physics_attachment_cache_publication_or_global_environment_edits():
    m=module();tree=ast.parse(inspect.getsource(m))
    prohibited={'World','SimulationContext','IsaacContactEnv','step','update','attach_stage','attach_stage_with_callback',
        'Export','ExportToString','prepare_job','runtime_stage','acquire_runtime_stage'}
    calls=[node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
        for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not prohibited.intersection(calls)
    assert not any(isinstance(node,ast.Assign) and any(isinstance(target,ast.Subscript) and
        isinstance(target.value,ast.Attribute) and target.value.attr=='environ' for target in node.targets) for node in ast.walk(tree))


def owner_fixture(m,c,path,tmp_path,monkeypatch,*,native=False):
    output=tmp_path/'base/runs/ausc_policy/probe/output';output.parent.mkdir(parents=True)
    from sim.ausc_policy import process_guard
    write(output.parent/'run.json',dict(config=c,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        code_commit=c['source']['code_commit'],completed=False,resource_guard=dict(path=process_guard.__file__,
        sha256=hashlib.sha256(Path(process_guard.__file__).read_bytes()).hexdigest()),
        command=[m.OWNER_PYTHON,'-u','-B','-m',NAME,'--config',str(path),'--output',str(output)]))
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setattr(m.sys,'executable',m.pre.NATIVE_PYTHON if native else m.OWNER_PYTHON)
    monkeypatch.setenv('AUSC_OWNED_RUN','1')
    identity=dict(pid=os.getpid(),pgid=os.getpid(),sid=os.getpid(),start_ticks='123')
    monkeypatch.setattr(m,'process_identity',lambda pid:dict(identity,pid=pid))
    return output,identity


@pytest.mark.parametrize('fault',[None,'env','python','bytecode','optimized','owner_command','owner_commit','owner_config',
    'guard','not_leader','output_alias'])
def test_original_owner_authentication_has_no_alternate_caller_allowance(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);inputs=m.load_inputs(path)
    output,identity=owner_fixture(m,c,path,tmp_path,monkeypatch)
    if fault=='env':monkeypatch.delenv('AUSC_OWNED_RUN')
    elif fault=='python':monkeypatch.setattr(m.sys,'executable','/other/python')
    elif fault=='bytecode':monkeypatch.setattr(m.sys,'dont_write_bytecode',False)
    elif fault=='optimized':monkeypatch.setattr(m.sys,'flags',SimpleNamespace(optimize=1))
    elif fault=='not_leader':monkeypatch.setattr(m,'process_identity',lambda pid:dict(identity,pgid=99))
    elif fault=='output_alias':output=str(output.parent)+'/./output'
    elif fault:
        record=output.parent/'run.json';value=json.loads(record.read_bytes())
        if fault=='owner_command':value['command'][4]='other.module'
        elif fault=='owner_commit':value['code_commit']='b'*40
        elif fault=='owner_config':value['config']['time_limit_seconds']=True
        else:value['resource_guard']['sha256']='b'*64
        write(record,value)
    if fault is None:assert m.require_owner(inputs,output)==identity
    else:
        with pytest.raises(ValueError):m.require_owner(inputs,output)


@pytest.mark.parametrize('alias',['symlink','hardlink','directory'])
def test_original_lock_rejects_aliases_before_idle_or_child(tmp_path,monkeypatch,alias):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);lock=tmp_path/'base/locks/isaac_gpu.lock'
    other=tmp_path/'other';other.touch()
    if alias=='symlink':lock.symlink_to(other)
    elif alias=='hardlink':os.link(other,lock)
    else:lock.mkdir()
    monkeypatch.setattr(m,'require_owner',lambda *a,**k:{})
    monkeypatch.setattr(m,'gpu_idle',lambda:pytest.fail('invalid lock reached idle check'))
    with pytest.raises((ValueError,OSError)):m.run_owned(path,tmp_path/'output')


def test_runtime_module_must_come_from_same_frozen_source(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    (tmp_path/'other.py').write_text('# wrong root\n')
    monkeypatch.setattr(m,'__file__',str(tmp_path/'other.py'))
    with pytest.raises(ValueError):m.load_inputs(path)


@pytest.mark.parametrize('fault',['missing_original','original_changed','extra_original_file','historical_changed'])
def test_explicit_historical_and_original_receipt_closure_is_bound_before_app(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    original=write(tmp_path/'old/COMPLETE.json',dict(completed=True));body=write(tmp_path/'old/body.npz',b'original body')
    historical=write(tmp_path/'historical/COMPLETE.json',dict(completed=True))
    def mutate(runtime):
        sources=runtime['compact_scene_storage']['sources']
        sources['scene_authoring_receipt']=write(Path(sources['scene_authoring_receipt']['file']),dict(
            original_selected=[dict(prepared_complete={k:original[k] for k in ('file','sha256')})],input_bindings=[[original,body]]))
        sources['layered_runtime_equivalence']=write(Path(sources['layered_runtime_equivalence']['file']),dict(
            historical_prepared_complete=historical,historical_prepared_sources=dict(scene=body)))
    change_runtime(c,0,mutate);rewrite(path,c);inputs=m.load_inputs(path)
    if fault=='missing_original':Path(body['file']).unlink()
    elif fault=='original_changed':Path(body['file']).write_bytes(b'new original')
    elif fault=='extra_original_file':write(tmp_path/'old/extra.npz',b'extra')
    else:Path(historical['file']).write_bytes(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)):m.rebind(inputs,full=True)


def test_header_asset_manifest_rows_are_not_mistaken_for_new_absolute_payload_roles(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    def mutate(runtime):
        ref=runtime['compact_scene_storage']['sources']['preparation_config']
        runtime['compact_scene_storage']['sources']['preparation_config']=write(Path(ref['file']),dict(
            prepare_only=True,asset_metadata=dict(file='body.npz',sha256='a'*64)))
    change_runtime(c,0,mutate);rewrite(path,c)
    assert len(m.load_inputs(path)['cases'])==8


def test_only_approved_sdk_payload_roles_keep_existing_regular_hardlinks(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);from sim.ausc_policy import sdk_mdl
    sdk=tmp_path/'sdk';sdk.mkdir();monkeypatch.setattr(sdk_mdl,'APPROVED_ROOT',sdk)
    payload=write(sdk/'approved.mdl',b'approved SDK');os.link(payload['file'],tmp_path/'existing-sdk-cache-link')
    def mutate(runtime):runtime['compact_scene_storage']['sources']['sdk_mdl_0000']=payload
    change_runtime(c,0,mutate);rewrite(path,c);inputs=m.load_inputs(path)
    assert m.rebind(inputs,full=True) is True


def test_one_returned_stage_is_released_before_next_prepared_stage(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[];fake_native(m,pre,store,monkeypatch,events)
    import weakref
    previous=[]
    class Stage:
        def GetRootLayer(self):return SimpleNamespace(anonymous=True)
        def GetUsedLayers(self):return [SimpleNamespace(anonymous=True,identifier='anon:test')]
    def prepared(binding):
        assert all(ref() is None for ref in previous),'prior returned USD stage still live'
        stage=Stage();previous.append(weakref.ref(stage));return stage
    monkeypatch.setattr(store,'prepared_runtime_stage',prepared)
    m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert len(previous)==8 and all(ref() is None for ref in previous)


@pytest.mark.parametrize('at',['final_receipt','failure_receipt'])
def test_receipt_write_failure_preserves_error_and_never_leaves_complete(tmp_path,monkeypatch,at):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[];original=ValueError('original current RAW refusal')
    fake_native(m,pre,store,monkeypatch,events,failure=original if at=='failure_receipt' else None)
    real=m.write_new
    def write_failure(path,value):
        real(path,value)
        if Path(path).name in ('COMPLETE.json','FAILURE.json','.COMPLETE.json.pending'):raise OSError('receipt fsync failed')
    monkeypatch.setattr(m,'write_new',write_failure)
    with pytest.raises((ValueError,OSError)) as caught:m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    if at=='failure_receipt':assert caught.value is original
    assert not (tmp_path/'native/COMPLETE.json').exists()


def test_post_close_rebind_has_no_metadata_validator_or_usd_calls(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[];fake_native(m,pre,store,monkeypatch,events)
    class App:
        def __init__(self,settings):pass
        def close(self):
            monkeypatch.setattr(store,'_binding_metadata',lambda *_:pytest.fail('post-close metadata validator'))
            monkeypatch.setattr(store,'prepared_runtime_stage',lambda *_:pytest.fail('post-close USD stage'))
            monkeypatch.setattr(store,'runtime_stage_leases',lambda:pytest.fail('post-close USD-dependent lease validator'))
    monkeypatch.setattr(m,'simulation_app_class',lambda:App)
    assert m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)['completed'] is True


def test_busy_original_lock_never_reaches_idle_or_native(tmp_path,monkeypatch):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);import fcntl
    monkeypatch.setattr(m,'require_owner',lambda *a,**k:{})
    monkeypatch.setattr(m,'gpu_idle',lambda:pytest.fail('busy lock reached idle'))
    with (tmp_path/'base/locks/isaac_gpu.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.run_owned(path,tmp_path/'out')


@pytest.mark.parametrize('fault',['success_flag','parent_group','parent_start','source','deadline','command'])
def test_native_entry_authenticates_original_request_without_allowance_flags(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch)
    output,identity=owner_fixture(m,c,path,tmp_path,monkeypatch,native=True);output.mkdir()
    parent=dict(pid=os.getppid(),pgid=identity['pgid'],sid=identity['sid'],start_ticks='12')
    command=[pre.NATIVE_PYTHON,'-u','-B','-m',NAME,'--native-worker','--config',str(path),'--output',str(output/'native')]
    request=dict(schema=m.REQUEST_SCHEMA,parent=parent,command=command,config_ref=pre.file_ref(path),source=c['source'],deadline_monotonic=time.monotonic()+20.)
    if fault=='success_flag':request['allow_success']=True
    elif fault=='parent_group':request['parent']['pgid']+=1
    elif fault=='parent_start':request['parent']['start_ticks']='13'
    elif fault=='source':request['source']=dict(c['source'],code_commit='b'*40)
    elif fault=='deadline':request['deadline_monotonic']=time.monotonic()+2000.
    else:request['command'][4]='other.module'
    write(output/'native_worker_request.json',request)
    monkeypatch.setattr(pre,'clean_parent',lambda:{})
    monkeypatch.setattr(pre,'native_identity',lambda:dict(identity,ppid=parent['pid']))
    monkeypatch.setattr(m,'process_identity',lambda pid:identity if pid==identity['pid'] else dict(parent,start_ticks='12',pgid=identity['pgid']))
    monkeypatch.setattr(m,'exercise',lambda *a:pytest.fail('unauthorized native request entered Kit'))
    with pytest.raises(ValueError):m.native_worker(path,output/'native')


@pytest.mark.parametrize('fault',[None,'unknown_success','pid','parent_group','config','close','shutdown','lease',
    'proof_source','unreaped_cpu','short_cases','case_ref','nonanonymous','false_zero','error'])
def test_wrapper_rejects_forged_or_incomplete_native_success(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);inputs=m.load_inputs(path)
    parent=dict(pid=100,pgid=100,sid=100,start_ticks='1')
    status=dict(identity=dict(pid=200,pgid=100,sid=100,start_ticks='2'))
    value=good_receipt(m,inputs,status,parent)
    if fault=='unknown_success':value['allow_success']=True
    elif fault=='pid':value['identity']['pid']=201
    elif fault=='parent_group':value['identity']['pgid']=10
    elif fault=='config':value['config_ref']=dict(value['config_ref'],bytes=False)
    elif fault=='close':value['normal_close']['app_close_returned']=False
    elif fault=='shutdown':value['normal_close']['fast_shutdown']=True
    elif fault=='lease':value['normal_close']['stage_cache_leases']=[dict(stage_cache_id=1)]
    elif fault=='proof_source':value['cpu_preflights'][-1]['source']={}
    elif fault=='unreaped_cpu':value['cpu_preflights'][-1]['child_reaped']=False
    elif fault=='short_cases':value['cases'].pop()
    elif fault=='case_ref':value['cases'][-1]['runtime_config']={}
    elif fault=='nonanonymous':value['cases'][-1]['layer_identifiers']=['/disk/file.usdc']
    elif fault=='false_zero':value['physics_executed']=0
    elif fault=='error':value['error']=dict(type='ValueError',message='RAW refused')
    if fault is None:assert m.validate_native_receipt(value,inputs,status,parent) is value
    else:
        with pytest.raises(ValueError):m.validate_native_receipt(value,inputs,status,parent)


def test_failed_anonymous_layer_check_releases_layers_before_close(tmp_path,monkeypatch):
    import weakref
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    fake_native(m,pre,store,monkeypatch,events);observed={}
    class Layer:
        anonymous=False
    class Stage:
        def __init__(self,layer):self.layer=layer
        def GetRootLayer(self):return SimpleNamespace(anonymous=True)
        def GetUsedLayers(self):return [self.layer]
    def prepared(binding):
        layer=Layer();observed['ref']=weakref.ref(layer)
        return Stage(layer)
    class App:
        def __init__(self,settings):pass
        def close(self):observed['layer_alive_at_close']=observed['ref']() is not None
    monkeypatch.setattr(store,'prepared_runtime_stage',prepared)
    monkeypatch.setattr(m,'simulation_app_class',lambda:App)
    with pytest.raises(ValueError,match='Every returned layer'):
        m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+30.)
    assert observed['layer_alive_at_close'] is False


@pytest.mark.parametrize('code,expected',[(0,1),(None,1),(False,1),(7,7)])
@pytest.mark.parametrize('route',['native_worker','run_owned'])
def test_entry_cannot_exit_zero_after_systemexit_failure(code,expected,route):
    arguments=['probe']+(['--native-worker'] if route=='native_worker' else [])+['--config','/ignored','--output','/ignored']
    script=("import sys;from sim.ausc_policy import layered_runtime_equivalence_probe as m;"
        "m."+route+"=lambda *a:(_ for _ in ()).throw(SystemExit("+repr(code)+"));"
        "sys.argv="+repr(arguments)+";m.main()")
    result=subprocess.run([sys.executable,'-B','-c',script],env=dict(os.environ),capture_output=True,text=True,timeout=10)
    assert result.returncode==expected


def test_argument_help_keeps_its_successful_exit_before_dispatch():
    result=subprocess.run([sys.executable,'-B','-m',NAME,'--help'],env=dict(os.environ),
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and '--native-worker' in result.stdout


@pytest.mark.parametrize('fault',['lease','close_source_drift','close_config_drift','SystemExit','release_failure','journal_failure'])
def test_normal_software_close_does_not_turn_errors_into_complete(tmp_path,monkeypatch,fault):
    m,pre,store,c,path,seen=fixture(tmp_path,monkeypatch);events=[]
    fake_native(m,pre,store,monkeypatch,events)
    if fault=='lease':monkeypatch.setattr(store,'runtime_stage_leases',lambda:[dict(stage_cache_id=1)])
    elif fault in ('close_source_drift','close_config_drift'):
        class App:
            def __init__(self,settings):events.append('app')
            def close(self):
                events.append('close')
                if fault=='close_source_drift':monkeypatch.setattr(pre,'source_snapshot',lambda:dict(c['source'],code_commit='b'*40))
                else:path.write_bytes(path.read_bytes()+b' ')
        monkeypatch.setattr(m,'simulation_app_class',lambda:App)
    elif fault=='SystemExit':monkeypatch.setattr(store,'prepared_runtime_stage',lambda *_:(_ for _ in ()).throw(SystemExit(0)))
    elif fault=='release_failure':monkeypatch.setattr(m.gc,'collect',lambda:(_ for _ in ()).throw(RuntimeError('release failed')))
    else:
        original=m.os.fsync
        def fsync(fd):
            if events.count('stage')>=1:raise OSError('journal fsync failed')
            return original(fd)
        monkeypatch.setattr(m.os,'fsync',fsync)
    with pytest.raises(BaseException):m.exercise(m.load_inputs(path),tmp_path/'native',time.monotonic()+20)
    assert not (tmp_path/'native/COMPLETE.json').exists()
    if events.count('app'):assert events.count('close')==1
