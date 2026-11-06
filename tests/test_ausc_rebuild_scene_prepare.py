from copy import deepcopy
from pathlib import Path
import json

import pytest


def module():
    from sim.ausc_policy import rebuild_scene_prepare
    return rebuild_scene_prepare


def template():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root/'configs/ausc_policy/fresh_expert_retainer_tenpoint_feedback_v2.json').read_text())


def test_preparation_uses_explicit_parking_solver_and_preserves_limits():
    m = module()
    original = template()
    before = deepcopy(original)
    spec = {'file': 'new-observed.npz', 'reconstruction_cohort': 'train'}
    configured = m.preparation_config(original, spec, base='/new/base', asset_root='/new/assets',
        native_python='/new/python', ld_library_path='/new/libs', approved_geometry='/new/approved.usdc')
    assert original == before
    assert configured['physics'] == {'position_iterations': 64, 'velocity_iterations': 1, 'armature_kg_m2': .02}
    assert configured['posture_limits'] == original['posture_limits']
    assert configured['fresh_expert']['passive_preload_N'] == 1.5
    assert configured['fresh_expert']['passive_preload_model'] == {'mode': 'single_explicit_v1'}
    assert configured['fresh_expert']['force_feedback'] == original['fresh_expert']['force_feedback']
    assert configured['fresh_expert']['plans'] == []
    assert 'raw_multimodal_capture' not in configured['fresh_expert']
    assert configured['prepare_only'] is True and configured['patient_asset'] == spec


@pytest.mark.parametrize('targets', [['R3'], ['R3','R3'], ['R3','L3','XX'], ['L1','R1','L2','R2','L3']])
def test_requested_targets_are_bounded_and_explicit(targets):
    with pytest.raises(ValueError): module().validate_targets(targets)


def test_changed_tool_or_preload_template_is_rejected():
    m = module()
    for field, value in [('geometry_source_sha256', '0'*64), ('geometry_id', 'old')]:
        value_template = template(); value_template[field] = value
        with pytest.raises(ValueError): m.preparation_config(value_template, {}, base='b', asset_root='a',
            native_python='p', ld_library_path='l', approved_geometry='g')
    value_template = template(); value_template['fresh_expert']['passive_preload_N'] = 1.
    with pytest.raises(ValueError): m.preparation_config(value_template, {}, base='b', asset_root='a',
        native_python='p', ld_library_path='l', approved_geometry='g')


def test_plan_failure_keeps_every_requested_slot_and_raw_failure(tmp_path):
    m = module()
    path = tmp_path/'R3';path.mkdir()
    (path/'COMPLETE.json').write_text(json.dumps({'completed': True,
        'targets': [{'name':'R3','geometry_path_found':False,'fit_candidates':0}],
        'physics_executed':False,'training_eligible':False}))
    slot, attempt = m.plan_slot('R3', path, {'returncode':0,'timed_out':False})
    assert slot['status'] == 'unavailable' and slot['reason']
    assert attempt['target'] == 'R3' and attempt['complete']['path'].endswith('COMPLETE.json')
    assert attempt['geometry_path_found'] is False


def test_successful_plan_is_never_accepted_after_failed_process(tmp_path):
    m = module()
    path = tmp_path/'L3';path.mkdir()
    (path/'L3_fresh_plan.json').write_text(json.dumps({'name':'L3','success':True}))
    slot, attempt = m.plan_slot('L3', path, {'returncode':9,'timed_out':False})
    assert slot['status'] == 'unavailable'
    assert attempt['retained_plan']['path'].endswith('L3_fresh_plan.json')


def test_command_runner_keeps_cpu_environment_and_finite_timeout(tmp_path, monkeypatch):
    m=module();calls=[]
    class Done: returncode=0
    def run(command, **kwargs):
        calls.append((command, kwargs));return Done()
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('AUSC_OWNED_RUN','1')
    result=m.run_cpu(['python','-B','-m','a.module'],tmp_path/'run.log',12.,ld_library_path='/new/libs')
    assert result['returncode']==0
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']==''
    assert calls[0][1]['env']['AUSC_OWNED_RUN']=='1'
    assert calls[0][1]['timeout']==12.
    assert calls[0][1]['env']['LD_LIBRARY_PATH']=='/new/libs'
    assert calls[0][1].get('start_new_session',False) is False
    monkeypatch.delenv('AUSC_OWNED_RUN')
    with pytest.raises(ValueError,match='owner'):m.run_cpu(['python'],tmp_path/'unowned.log',12.)
    assert len(calls)==1 and not (tmp_path/'unowned.log').exists()


@pytest.mark.parametrize('parallelism',[1,3])
def test_driver_retains_failed_scene_and_target_denominators(tmp_path, monkeypatch,parallelism):
    m=module()
    source=tmp_path/'source.json';source.write_text('{}')
    template_path=tmp_path/'template.json';template_path.write_text(json.dumps(template()))
    geometry=tmp_path/'approved.usdc';geometry.write_bytes(b'fixture approved geometry')
    real_ref=m.sources.reference;real_verify=m.sources.verify_ref
    def source_ref(path):
        result=real_ref(path)
        if Path(path)==geometry:result['sha256']=m.GEOMETRY_SOURCE_SHA256
        return result
    monkeypatch.setattr(m.sources,'reference',source_ref)
    monkeypatch.setattr(m.sources,'verify_ref',lambda ref: None if Path(ref['file'])==geometry else real_verify(ref))
    bundle_ref=real_ref(source)
    spec={'reconstruction_source':bundle_ref,'reconstruction_cohort':'train'}
    scenes=[dict(subject_id='p001120',split='train',cohort='train',pose_id=pose,
                 patient_spec=spec,numeric_candidate_screen_passed=good)
            for pose,good in [('p000474_a001230_f0000',True),('p000476_a001231_f0023',False)]]
    monkeypatch.setattr(m,'load_scenes',lambda *args,**kwargs:(scenes,bundle_ref,[bundle_ref]))
    monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('AUSC_OWNED_RUN','1')
    planner_calls=[]
    def build(**kwargs):
        planner_calls.append(kwargs)
        return {'config':{}}
    monkeypatch.setattr(m.builder,'build_geometry_plan',build)
    def run(command,logfile,timeout,**kwargs):
        out=Path(command[command.index('--output')+1]);out.mkdir()
        if '--cpu-prepare-only' in command:
            patient=dict(reconstruction_source=bundle_ref,reconstruction_cohort='train',original_depth_plane_bytes_recovered=False)
            data={'COMPLETE.json':dict(completed=True,source_sha256=m.GEOMETRY_SOURCE_SHA256,
                    physics_executed=False,training_eligible=False,patient=patient),
                'patient_receipt.json':patient,'physics_profile.json':{'profile':m.PHYSICS},
                'approved_camera_restoration.json':dict(camera_sha256=m.APPROVED_CAMERA_SHA256,restored_user_approved_pose=True),
                'passive_preload_saved_usd_readback.json':dict(mode='single_explicit_v1',nominal_preload_N=1.5,
                    explicit_effort_N=-1.5,drive={'target_position_m':0.})}
            for name,value in data.items():(out/name).write_text(json.dumps(value))
        else:
            target='R3' if out.name.endswith('R3') else 'L3'
            (out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,physics_executed=False,
                targets=[dict(name=target,geometry_path_found=target=='R3')])))
            if target=='R3':(out/'R3_fresh_plan.json').write_text(json.dumps(dict(name='R3',success=True)))
        return {'returncode':0,'timed_out':False}
    monkeypatch.setattr(m,'run_cpu',run)
    output=tmp_path/'new-output'
    result=m.prepare_and_plan(complete_ref=real_ref(source),template_ref=real_ref(template_path),
        approved_geometry=geometry,base=tmp_path,asset_root=tmp_path,native_python='native-python',
        planner_python='planner-python',ld_library_path='/libs',output=output,targets=['R3','L3'],cohorts=['train'],parallelism=parallelism)
    manifest=json.loads((output/'scene_manifest.json').read_text())
    assert result['completed'] is True
    assert manifest['requested_scenes']==2 and manifest['requested_target_slots']==4
    assert manifest['available_plans']==1 and manifest['unavailable_target_slots']==3
    assert set(manifest['scenes'][0]['plans'])==set(manifest['scenes'][1]['plans'])=={'R3','L3'}
    assert manifest['scenes'][1]['preparation_error']
    assert manifest['scenes'][1]['prepared_complete'] is None
    assert all(call['collision_profile']=='full' and call['plane_tilt_limit_deg']==3. and
               call['planning_plane_margin_deg']==.25 for call in planner_calls)
    prepared=next(output.glob('*/prepared'))
    (prepared/'physics_profile.json').write_text(json.dumps({'profile':dict(m.PHYSICS,position_iterations=32)}))
    with pytest.raises(ValueError,match='solver'):
        m.verify_preparation(prepared,spec)


def test_original_owner_registers_driver_with_hidden_cuda(tmp_path, monkeypatch):
    from sim.ausc_policy import run, process_guard
    config=tmp_path/'owner.json'
    config.write_text(json.dumps(dict(experiment='rebuild_scene_prepare',base=str(tmp_path),time_limit_seconds=60)))
    calls=[]
    def stage(name,command,args,metadata,env):
        calls.append((name,command,env))
        args.output.mkdir();(args.output/'COMPLETE.json').write_text('{}')
        return {'passed':True}
    monkeypatch.setattr(process_guard,'run_stage',stage)
    monkeypatch.setattr(process_guard,'wait_gpu_release',lambda:True)
    monkeypatch.setattr(process_guard,'STAGE_LIMIT_S',process_guard.STAGE_LIMIT_S)
    monkeypatch.setattr(process_guard,'GUARD',dict(process_guard.GUARD))
    monkeypatch.setattr(run.sys,'platform','linux')
    monkeypatch.setattr(run.sys,'argv',['run','--config',str(config),'--run-id','cpu-rebuild-test'])
    assert run.main()==0
    assert calls[0][1][calls[0][1].index('-m')+1]=='sim.ausc_policy.rebuild_scene_prepare'
    assert calls[0][2]['CUDA_VISIBLE_DEVICES']=='' and calls[0][2]['AUSC_OWNED_RUN']=='1'


def test_completion_cannot_silently_shrink_bound_source_cohort(tmp_path):
    m=module()
    cohort={'name':'train','split':'train','identities':['p001120','p000758']}
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'schema':m.sources.BUNDLE_SCHEMA,'cohorts':[cohort]}))
    entry={'cohort':dict(cohort,identities=['p001120'])}
    complete=tmp_path/'complete.json'
    complete.write_text(json.dumps(dict(schema='m2_observed_reconstruction_complete_v1',completed=True,
        training_eligible=False,reconstruction_source=m.sources.reference(bundle),cohorts=[entry])))
    with pytest.raises(ValueError,match='shrank'):
        m.load_scenes(m.sources.reference(complete),['train'])


def test_owner_config_dispatches_bound_sources_to_cpu_driver(tmp_path,monkeypatch):
    m=module();geometry=tmp_path/'geometry.usdc';geometry.write_bytes(b'fixture geometry')
    request=dict(schema=m.REQUEST_SCHEMA,experiment='rebuild_scene_prepare',base='/base',
        reconstruction_complete={'file':'/complete.json','sha256':'1'*64},
        template={'file':'/template.json','sha256':'2'*64},approved_geometry=m.sources.reference(geometry),
        asset_root='/assets',native_python='/native-python',planner_python='/planner-python',
        ld_library_path='/libs',targets=['R3','L3'],cohorts=['train'],per_gap_seconds=90.)
    calls=[]
    monkeypatch.setattr(m,'prepare_and_plan',lambda **kwargs:calls.append(kwargs) or {'completed':True})
    assert m.run_request(request,tmp_path/'output')['completed'] is True
    assert calls[0]['complete_ref']==request['reconstruction_complete']
    assert calls[0]['targets']==['R3','L3'] and calls[0]['per_gap_seconds']==90.
    assert calls[0]['native_python']=='/native-python'


def test_frozen_scope_requires_explicit_purpose_and_excludes_capture_cohorts(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='frozen'):
        m.load_scenes({},['frozen_evaluation'])
    for cohorts in (['train'],['selection_validation'],['frozen_evaluation','train']):
        with pytest.raises(ValueError,match='frozen'):
            m.load_scenes({},cohorts,purpose='frozen_evaluation')


def reconstruction_fixture(tmp_path,monkeypatch,*,overlap=False):
    m=module();roles=[dict(name='selection_validation',split='validation',identities=['p000923']),
        dict(name='frozen_evaluation',split='validation',identities=['p000923'] if overlap else ['p001030','p000956'])]
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps(dict(schema=m.sources.BUNDLE_SCHEMA,cohorts=roles)))
    bundle_ref=m.sources.reference(bundle);entries=[]
    monkeypatch.setattr(m.sources,'validate_reconstruction',lambda *args:None)
    for cohort in roles:
        directory=tmp_path/cohort['name'];directory.mkdir();rows=[];specs=[]
        for subject in cohort['identities']:
            for identity,sequence,frame,_ in m.sources.SOURCES:
                name=subject+'_'+sequence+f'_f{frame:04d}.npz'
                rows.append(dict(file=name,identity=subject,numeric_candidate_screen_passed=True,
                    source_pose=dict(sequence_path='/observed/'+sequence+'/pose.npz',frame_index=frame)))
                specs.append(dict(directory=str(directory),file=name,reconstruction_cohort=cohort['name'],
                    expected_split=cohort['split'],reconstruction_source=bundle_ref))
        manifest=directory/'asset_manifest.json';manifest.write_text(json.dumps(rows))
        for spec in specs:spec['manifest_sha256']=m.sources.reference(manifest)['sha256']
        summary=directory/'summary.json';summary.write_text('{}')
        specs_path=directory/'specs.json';specs_path.write_text(json.dumps(specs))
        entries.append(dict(cohort=cohort,summary=m.sources.reference(summary),patient_specs=m.sources.reference(specs_path)))
    complete=tmp_path/'complete.json';complete.write_text(json.dumps(dict(schema='m2_observed_reconstruction_complete_v1',
        completed=True,training_eligible=False,reconstruction_source=bundle_ref,cohorts=entries)))
    return m.sources.reference(complete)


def test_explicit_frozen_load_retains_both_bodies_and_poses_and_rejects_overlap(tmp_path,monkeypatch):
    m=module();complete=reconstruction_fixture(tmp_path,monkeypatch)
    scenes,_,_=m.load_scenes(complete,['frozen_evaluation'],purpose='frozen_evaluation')
    assert len(scenes)==4 and {s['subject_id'] for s in scenes}=={'p001030','p000956'}
    assert all(s['split']=='validation' and s['cohort']=='frozen_evaluation' for s in scenes)
    other=tmp_path/'overlap';other.mkdir();bad=reconstruction_fixture(other,monkeypatch,overlap=True)
    with pytest.raises(ValueError,match='overlap'):
        m.load_scenes(bad,['frozen_evaluation'],purpose='frozen_evaluation')


@pytest.mark.parametrize('parallelism',[0,4,True,1.5])
def test_parallelism_is_bounded_before_source_or_child_access(tmp_path,monkeypatch,parallelism):
    m=module();monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    with pytest.raises(ValueError,match='parallelism'):
        m.prepare_and_plan(complete_ref={},template_ref={},approved_geometry=tmp_path/'absent',base=tmp_path,
            asset_root=tmp_path,native_python='native',planner_python='planner',ld_library_path='/libs',
            output=tmp_path/'out',targets=['R3','L3'],cohorts=['frozen_evaluation'],purpose='frozen_evaluation',parallelism=parallelism)
    assert not (tmp_path/'out').exists()


def test_parallel_scene_workers_overlap_but_manifest_and_attempt_order_are_stable(tmp_path,monkeypatch):
    import threading
    m=module();assert hasattr(m,'prepare_scene'),'Per-scene helper required for bounded concurrency'
    source=tmp_path/'source.json';source.write_text('{}');template_path=tmp_path/'template.json';template_path.write_text(json.dumps(template()))
    geometry=tmp_path/'approved.usdc';geometry.write_bytes(b'approved fixture');real_ref=m.sources.reference;real_verify=m.sources.verify_ref
    def source_ref(path):
        value=real_ref(path)
        if Path(path)==geometry:value['sha256']=m.GEOMETRY_SOURCE_SHA256
        return value
    monkeypatch.setattr(m.sources,'reference',source_ref)
    monkeypatch.setattr(m.sources,'verify_ref',lambda ref:None if Path(ref['file'])==geometry else real_verify(ref))
    bound=real_ref(source);scenes=[dict(subject_id='subject'+str(i),pose_id='pose',cohort='frozen_evaluation',split='validation') for i in range(3)]
    monkeypatch.setattr(m,'load_scenes',lambda *a,**k:(scenes,bound,[bound]))
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    barrier=threading.Barrier(3);release_first=threading.Event();lock=threading.Lock();active=0;maximum=0;finished=[]
    def worker(source,**kwargs):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        barrier.wait(timeout=5.)
        if source['subject_id']=='subject0':
            assert release_first.wait(timeout=5.)
            with lock:active-=1;finished.append(source['subject_id'])
        else:
            with lock:active-=1;finished.append(source['subject_id'])
            release_first.set()
        row=dict(source,preparation_error='retained fixture failure',plans={t:dict(status='unavailable',reason='fixture failure') for t in kwargs['targets']})
        return row,[dict(scene_id=source['subject_id'],stage='prepare_unavailable',reason='fixture failure')]
    monkeypatch.setattr(m,'prepare_scene',worker)
    output=tmp_path/'out';result=m.prepare_and_plan(complete_ref=bound,template_ref=real_ref(template_path),approved_geometry=geometry,
        base=tmp_path,asset_root=tmp_path,native_python='native',planner_python='planner',ld_library_path='/libs',
        output=output,targets=['R3','L3'],cohorts=['frozen_evaluation'],purpose='frozen_evaluation',parallelism=3)
    manifest=json.loads((output/'scene_manifest.json').read_text())
    assert maximum==3 and finished[0]!='subject0'
    assert [r['subject_id'] for r in manifest['scenes']]==[r['subject_id'] for r in scenes]
    assert [a['scene_id'] for a in manifest['attempts']]==[r['subject_id'] for r in scenes]
    assert manifest['purpose']==result['purpose']=='frozen_evaluation' and manifest['parallelism']==3
    assert manifest['requested_target_slots']==manifest['unavailable_target_slots']==6
    assert manifest['training_eligible'] is False and manifest['physics_executed'] is False


@pytest.mark.parametrize('purpose,cohorts,parallelism',[('capture',['train','selection_validation'],1),('frozen_evaluation',['frozen_evaluation'],3)])
def test_owner_request_defaults_and_explicit_frozen_parallelism_forward_without_scope_changes(tmp_path,monkeypatch,purpose,cohorts,parallelism):
    m=module();geometry=tmp_path/'approved.usdc';geometry.write_bytes(b'fixture')
    request=dict(schema=m.REQUEST_SCHEMA,experiment='rebuild_scene_prepare',base='/base',
        reconstruction_complete={},template={},approved_geometry=m.sources.reference(geometry),
        asset_root='/assets',native_python='/native',planner_python='/planner',ld_library_path='/libs',targets=['R3','L3'])
    if purpose=='frozen_evaluation':request.update(purpose=purpose,parallelism=parallelism)
    calls=[];monkeypatch.setattr(m,'prepare_and_plan',lambda **kwargs:calls.append(kwargs) or {})
    m.run_request(request,tmp_path/'out')
    assert calls[0]['purpose']==purpose and calls[0]['cohorts']==cohorts and calls[0]['parallelism']==parallelism
