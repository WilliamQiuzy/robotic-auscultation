"""CPU diagnostic fixtures only; these do not qualify any native PhysX runtime."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.ausc_policy import retainer_contact as rc
from sim.ausc_policy.geometry_epoch import GEOMETRY_SOURCE_SHA256, qualify_contact
from sim.ausc_policy.retainer_geometry import ring_sector_meshes

DT=1/240
BODY='/World/Stethoscope/Carriage'
PATIENT='/World/Patient/Body'


def ref(path):
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def contract_fixture(tmp_path):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return ref(p)
    binary=tmp_path/'diagnostic_physx.so';binary.write_bytes(b'CPU diagnostic binary, not native qualification')
    geometry=Path(rc.__file__).with_name('retainer_geometry.py')
    run='diagnostic-oracle';review=run+'/output/review/'
    evidence={key:save(review+name,dict(cpu_diagnostic_only=True)) for key,name in
              [('cases','cases.json'),('native','native_contacts.jsonl'),('residuals','residuals.json')]}
    complete=dict(schema='native_convex_triangle_contact_oracle_v1',completed=True,
        execution_completed=True,evidence_complete=True,evidence_status='complete',
        cpu_diagnostic_only=True,semantics_rule_qualified=False,
        native_binding_file=str(binary),native_binding_sha256=ref(binary)['sha256'],
        retainer_geometry_source_sha256=ref(geometry)['sha256'],
        artifacts={Path(r['file']).name:dict(bytes=Path(r['file']).stat().st_size,sha256=r['sha256'])
                   for r in evidence.values()})
    evidence['complete']=save(review+'COMPLETE.json',complete)
    evidence['owner_exit']=save(run+'/output/native_worker_exit.json',dict(cpu_diagnostic_only=True,
        worker='sim.ausc_policy.contact_semantics_oracle',pid=123,returncode=0,
        timed_out=False,clean_exit=True,complete_file_present=True))
    evidence['owner_terminal']=save(run+'/result.json',dict(cpu_diagnostic_only=True,run_id=run,
        completed=True,gpu_released=True,ended_utc='2026-10-26T00:00:00Z',
        config=dict(experiment='contact_semantics_oracle'),
        stage=dict(passed=True,process_group_drained=True,exit_code=0)))
    receipt=dict(schema='qualified_native_retainer_witness_v1',qualified=True,
        cpu_diagnostic_only=True,
        scope=dict(retainer_pair_index=0,retainer_shape='convexHull',patient_shape='staticTriangleMesh'),
        rule=dict(witness='p+s*n',normal_direction='shape1_to_shape0',
                  pose_frames=['pre_step_current','post_step_current'],pair1_qualified=False),
        native_binding=ref(binary),geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        retainer_geometry_source_sha256=ref(geometry)['sha256'],
        execution=dict(terminal=True,exit_code=0,complete=True),evidence=evidence)
    binding=save('qualification.json',receipt)
    return dict(schema='retainer_front_audit_v3',qualification=binding),receipt


def module():
    spec=importlib.util.find_spec('sim.ausc_policy.retainer_witness')
    assert spec is not None, 'Missing independently qualified native witness route'
    return importlib.import_module('sim.ausc_policy.retainer_witness')


def tick(world=None, *, row=0):
    world=np.eye(4) if world is None else world
    def frame(label):
        return dict(sample=label,source='omni.physx.IPhysx.get_rigidbody_transformation',
                    path=BODY,available=True,world_from_carriage=world.tolist(),
                    contact_generation_pose_claimed=False)
    return dict(schema='native_contact_tick_v1',physics_row=row,interval_start_s=row*DT,
                physics_dt_s=DT,pre_step=frame('pre_step_current'),
                post_step=frame('post_step_current'),observer_error=None)


def point(index=0,world=None,normal=None,witness=None,force=4.,separation=.00035):
    world=np.eye(4) if world is None else world
    vertices,_=ring_sector_meshes()[index]
    witness=vertices[:4].mean(axis=0) if witness is None else np.asarray(witness)
    normal=np.array([0.,0.,1.]) if normal is None else np.asarray(normal)
    n=world[:3,:3]@normal
    w=world[:3,:3]@witness+world[:3,3]
    return dict(colliders=[rc.ROOT+f'DeviceContactRing/sector_{index:02d}',PATIENT],
                position_world=(w-separation*n).tolist(),normal_world=n.tolist(),
                separation_m=separation,impulse_Ns=(n*force*DT).tolist(),
                face_index0=0,face_index1=0,native_fields_missing=[],native_field_errors={})


def audit(points,evidence,force,contract):
    return module().audit_retainer_witness(points,evidence,force,contract,
        dt=DT,physics_row=evidence['physics_row'],time_s=evidence['interval_start_s']+DT)


def test_receipt_binding_requires_qualified_terminal_artifacts_and_actual_binary(tmp_path):
    contract,receipt=contract_fixture(tmp_path);m=module()
    assert m.load_contract(contract)['qualified'] is True
    m.verify_native_binding(contract,receipt['native_binding']['file'])
    Path(receipt['native_binding']['file']).write_bytes(b'different API')
    with pytest.raises(ValueError,match='binary'):m.verify_native_binding(contract,receipt['native_binding']['file'])


@pytest.mark.parametrize('change',['unqualified','pair1','scope','rule','exit','geometry','artifact','refhash'])
def test_invalid_receipt_never_qualifies(tmp_path,change):
    contract,receipt=contract_fixture(tmp_path)
    if change=='unqualified':receipt['qualified']=False
    elif change=='pair1':receipt['scope']['retainer_pair_index']=1
    elif change=='scope':receipt['scope']['patient_shape']='convexHull'
    elif change=='rule':receipt['rule']['witness']='p-s*n'
    elif change=='exit':receipt['execution']['exit_code']=1
    elif change=='geometry':receipt['retainer_geometry_source_sha256']='f'*64
    elif change=='artifact':Path(receipt['evidence']['native']['file']).write_text('mutated')
    else:contract['qualification']['sha256']='f'*64
    if change not in ('artifact','refhash'):
        path=Path(contract['qualification']['file']);path.write_text(json.dumps(receipt));contract['qualification']=ref(path)
    with pytest.raises(ValueError):module().load_contract(contract)


@pytest.mark.parametrize('index',range(32))
@pytest.mark.parametrize('transformed',[False,True])
def test_all_sectors_front_witness_accepts_nonzero_raw_offsets_without_raw_edits(tmp_path,index,transformed):
    contract,_=contract_fixture(tmp_path);world=np.eye(4)
    if transformed:
        world[:3,:3]=Rotation.from_euler('xyz',[.3,-.8,1.2]).as_matrix();world[:3,3]=[.12,-.35,.74]
    p=point(index,world);before=deepcopy(p)
    report=audit([p],tick(world),4.,contract)
    assert report['passed'] and report['off_front_force_N']==0.
    assert report['points'][0]['retainer_pair_index']==0
    assert report['points'][0]['frames']['pre_step']['front_unique_max']
    assert len(report['points'][0]['frames']['post_step']['normal_scores'])==6
    assert p==before
    assert not rc.audit_retainer_front([p],world,4.)['passed']


@pytest.mark.parametrize('index',range(32))
@pytest.mark.parametrize('transformed',[False,True])
@pytest.mark.parametrize('feature',['side','back','inner','angular','front_edge_side','tie','outside_sector'])
def test_side_back_edge_and_wrong_sector_cannot_acquire_front_permission(tmp_path,feature,index,transformed):
    contract,_=contract_fixture(tmp_path);v,_=ring_sector_meshes()[0];w=v[:4].mean(axis=0);n=np.array([0.,0.,1.])
    radial=np.array([np.cos(np.pi/32),np.sin(np.pi/32),0.])
    if feature=='side':w=v[[1,2,5,6]].mean(axis=0);n=-radial
    elif feature=='back':w=v[4:].mean(axis=0);n=-n
    elif feature=='inner':w=v[[0,3,4,7]].mean(axis=0);n=radial
    elif feature=='angular':w=v[[0,1,4,5]].mean(axis=0);n=np.array([0.,1.,0.])
    elif feature=='front_edge_side':w=v[[1,2]].mean(axis=0);n=-radial
    elif feature=='tie':w=v[[1,2]].mean(axis=0);n=(np.array([0.,0.,1.])-radial)/np.sqrt(2)
    else:w=ring_sector_meshes()[10][0][:4].mean(axis=0)
    angle=2*np.pi*index/32;rotation=Rotation.from_euler('z',angle).as_matrix()
    world=np.eye(4)
    if transformed:
        world[:3,:3]=Rotation.from_euler('xyz',[.3,-.8,1.2]).as_matrix();world[:3,3]=[.12,-.35,.74]
    report=audit([point(index,world,normal=rotation@n,witness=rotation@w)],tick(world),4.,contract)
    assert not report['passed'] and report['off_front_force_N']==pytest.approx(4.)
    assert report['points'][0]['reasons']


@pytest.mark.parametrize('change',['pair1','normal_missing','normal_nonunit','nan_separation','missing_post','source','path','time','row','matrix','observer','unavailable','frame_label','post_disagrees'])
def test_missing_or_unqualified_native_semantics_fail_closed(tmp_path,change):
    contract,_=contract_fixture(tmp_path);p=point();ev=tick()
    if change=='pair1':p['colliders'].reverse()
    elif change=='normal_missing':p.pop('normal_world')
    elif change=='normal_nonunit':p['normal_world']=[0.,0.,2.]
    elif change=='nan_separation':p['separation_m']=float('nan')
    elif change=='missing_post':ev.pop('post_step')
    elif change=='source':ev['pre_step']['source']='commanded_FK'
    elif change=='path':ev['post_step']['path']='/World/Other'
    elif change=='time':ev['physics_dt_s']=2*DT
    elif change=='row':ev['physics_row']=1.5
    elif change=='matrix':ev['pre_step']['world_from_carriage'][0][0]=2.
    elif change=='observer':ev['observer_error']='native callback failed'
    elif change=='unavailable':ev['post_step']['available']=False
    elif change=='frame_label':ev['pre_step']['sample']='callback_current'
    else:ev['post_step']['world_from_carriage'][2][3]=.001
    report=audit([p],ev,4.,contract)
    assert not report['passed'] and report['off_front_force_N']==pytest.approx(4.)


def test_positive_ledger_has_no_new_force_epsilon_and_zero_impulse_remains_observable(tmp_path):
    contract,_=contract_fixture(tmp_path)
    bad=point(normal=[0.,0.,-1.],force=1e-20)
    result=audit([point(force=0.),bad],tick(),1e-20,contract)
    assert result['off_front_force_N']>0 and not result['passed']
    assert len(result['points'])==2 and result['positive_points_qualified']==0
    assert not audit([],tick(),4.,contract)['force_matches_ledger']
    assert not audit([point()],tick(),4.+1e-5,contract)['force_matches_ledger']


def test_gate_enforces_expected_v3_contract_even_zero_force_rows_and_retains_archival_v2(tmp_path):
    from test_ausc_policy_retainer_contact import rows
    contract,_=contract_fixture(tmp_path)
    data=rows();front=audit([point()],tick(),4.,contract)
    for row in data:row.update(native_retainer_audit=deepcopy(contract),retainer_front_audit=deepcopy(front))
    assert qualify_contact(data,retainer_witness_contract=contract)['passed']
    assert qualify_contact(rows())['passed']
    data[10].pop('native_retainer_audit')
    assert not qualify_contact(data,retainer_witness_contract=contract)['passed']
    assert not qualify_contact(data)['passed']
    empty=rows(membrane=4.,retainer=0.)
    assert not qualify_contact(empty,retainer_witness_contract=contract)['passed']


def test_builder_selector_scene_store_and_provider_are_explicit_and_bound(tmp_path,monkeypatch):
    from test_ausc_policy_m2_pilot_config import fixture as prepared_fixture
    from sim.ausc_policy import m2_pilot_config,scene_store
    refs,plan,_=prepared_fixture(tmp_path,monkeypatch)
    qualified_dir=tmp_path/'oracle';qualified_dir.mkdir()
    contract,receipt=contract_fixture(qualified_dir)
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):
        (tmp_path/name).write_text('{}')
    args=dict(**refs,fresh_plan=plan,episode_id='witness',noise_seed=1)
    old=m2_pilot_config.build_compact_expert(**args)
    result=m2_pilot_config.build_compact_expert(**args,retainer_witness_contract=contract,
        scene_store=dict(root=str(tmp_path/'cas'),batch_id='cpu-diagnostic'))
    cfg=result['config'];fresh=cfg['fresh_expert']
    assert fresh['native_retainer_audit']==contract
    assert fresh['raw_multimodal_capture']['compact']['metadata']['native_retainer_audit']==contract
    assert result['receipt']['native_retainer_audit']==contract
    assert module().settings(fresh)==contract
    provider=module().NativeRetainerAudit(module().settings(fresh),native_binding_file=receipt['native_binding']['file'])
    assert provider.audit([point()],tick(),4.,dt=DT,physics_row=0,time_s=DT)['passed']
    scene_store.validate_config(cfg)
    scene=scene_store.prepare_job(cfg,tmp_path/'job')
    assert 'native_retainer_qualification' in scene['binding']['sources']
    assert scene['binding']['sources']['native_retainer_qualification']['sha256']==contract['qualification']['sha256']
    assert 'native_retainer_audit' not in old['config']['fresh_expert']
    cfg['fresh_expert']['physics_change']=True
    with pytest.raises(ValueError,match='authoring'):scene_store.validate_config(cfg)


def test_missing_runtime_selector_does_not_downgrade_bound_v3_capture(tmp_path):
    contract,_=contract_fixture(tmp_path)
    fresh=dict(raw_multimodal_capture=dict(compact=dict(metadata=dict(native_retainer_audit=contract))))
    with pytest.raises(ValueError,match='binding'):module().settings(fresh)
    fresh['native_retainer_audit']=contract
    assert module().settings(fresh)==contract
    fresh['raw_multimodal_capture']['compact']['metadata']['native_retainer_audit']['schema']='unknown'
    with pytest.raises(ValueError):module().settings(fresh)


def test_cpu_diagnostic_receipt_cannot_initialize_native_provider(tmp_path):
    contract,_=contract_fixture(tmp_path)
    with pytest.raises(ValueError,match='CPU diagnostic'):module().NativeRetainerAudit(contract)


def test_v3_native_step_uses_tick_frames_without_single_rigid_prim_or_fk_fallback(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    from sim.ausc_policy import fresh_expert as fresh
    contract,receipt=contract_fixture(tmp_path)
    env=fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    env._native_retainer_auditor=module().NativeRetainerAudit(contract,native_binding_file=receipt['native_binding']['file'])
    env.native_retainer_audit=contract
    env.rows=[];env.t=0.;env.slider=7;env.passive_preload_N=1.5
    env.robot=SimpleNamespace(apply_action=lambda _:None);env.video_recorder=None
    env.contact_points=[point()];env.last_contacts={'retainer':4.}
    env.bodies=[]
    def sample(self,label):
        frame=tick()['pre_step' if label=='pre_step_current' else 'post_step'];frame['sample']=label
        return frame
    monkeypatch.setattr(fresh.FreshContactEnv,'_sample_native_carriage',sample)
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',SimpleNamespace(ArticulationAction=SimpleNamespace))
    def step(self,*args):
        self.t=DT
        self.rows.append(dict(time_s=DT,phase='CONTACT',target='L1',subphase='contact_cycle'))
        return dict(axis_world=[0.,0.,-1.])
    monkeypatch.setattr(fresh.IsaacContactEnv,'step',step)
    env.step(np.zeros(7),'CONTACT','L1')
    row=env.rows[-1]
    assert row['retainer_front_audit']['passed']
    assert row['native_retainer_audit']==contract
    assert row['native_retainer_tick']['pre_step']['source']==module().POSE_SOURCE
    assert row['native_retainer_tick']['post_step']['source']==module().POSE_SOURCE
    assert env._retainer_first_contact_payload


def test_qualification_gate_reauthenticates_the_expected_receipt(tmp_path):
    from test_ausc_policy_retainer_contact import rows
    contract,_=contract_fixture(tmp_path);data=rows();front=audit([point()],tick(),4.,contract)
    for row in data:row.update(native_retainer_audit=deepcopy(contract),retainer_front_audit=deepcopy(front))
    Path(contract['qualification']['file']).write_text('{"qualified":false}')
    assert not qualify_contact(data,retainer_witness_contract=contract)['passed']


def test_lost_v3_provider_is_rejected_before_any_effort_or_position_command(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from sim.ausc_policy import fresh_expert as fresh
    contract,_=contract_fixture(tmp_path);env=fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    import sys
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',SimpleNamespace(ArticulationAction=SimpleNamespace))
    env.native_retainer_audit=contract;env._native_retainer_auditor=None
    env.slider=7;env.passive_preload_N=1.5
    env.robot=SimpleNamespace(apply_action=lambda _:pytest.fail('No command after loss of configured v3 provider'))
    with pytest.raises(ValueError,match='auditor'):env.step(np.zeros(7),'CONTACT','L1')


@pytest.mark.parametrize('front_weight,expected',[(.8,True),(.6,False),(1/np.sqrt(2),False)])
def test_front_must_be_the_unique_largest_support_score_at_an_edge(tmp_path,front_weight,expected):
    contract,_=contract_fixture(tmp_path);v,_=ring_sector_meshes()[0]
    side=np.array([np.cos(np.pi/32),np.sin(np.pi/32),0.])
    direction=front_weight*np.array([0.,0.,-1.])+np.sqrt(1-front_weight**2)*side
    p=point(normal=-direction,witness=v[[1,2]].mean(axis=0))
    report=audit([p],tick(),4.,contract)
    assert report['passed'] is expected


def test_precision_guard_rejects_unresolved_near_tie_without_normalization(tmp_path):
    contract,_=contract_fixture(tmp_path);v,_=ring_sector_meshes()[0]
    side=np.array([np.cos(np.pi/32),np.sin(np.pi/32),0.])
    a=1+module().NORMAL_SCORE_GUARD/4
    direction=(a*np.array([0.,0.,-1.])+side)/np.sqrt(a*a+1)
    p=point(normal=-direction,witness=v[[1,2]].mean(axis=0))
    assert not audit([p],tick(),4.,contract)['passed']


@pytest.mark.parametrize('separation',[-.00035,0.,.00035])
def test_signed_separation_kept_and_witness_is_not_midpoint(tmp_path,separation):
    contract,_=contract_fixture(tmp_path);p=point(separation=separation)
    assert audit([p],tick(),4.,contract)['passed']


def test_optional_independent_analysis_is_authenticated(tmp_path):
    contract,receipt=contract_fixture(tmp_path)
    path=tmp_path/'feature_analysis.json';path.write_text('{"cpu_diagnostic_only":true}')
    receipt['independent_analysis']=ref(path)
    qualification=Path(contract['qualification']['file']);qualification.write_text(json.dumps(receipt))
    contract['qualification']=ref(qualification)
    module().load_contract(contract)
    path.write_text('mutated independent audit')
    with pytest.raises(ValueError):module().load_contract(contract)


def test_pure_feature_wrapper_retains_nonproduction_body_coordinates():
    v,_=ring_sector_meshes()[0]
    feature=getattr(module(),'sector_feature_audit',None)
    assert callable(feature), 'Missing pure diagnostic geometry API'
    result=feature(np.eye(4),v[:4].mean(axis=0),[0.,0.,1.],0)
    assert result['passed'] and len(result['normal_scores'])==6


def test_zero_impulses_have_observable_records_without_expensive_feature_receipts(tmp_path):
    contract,_=contract_fixture(tmp_path)
    result=audit([point(force=0.) for _ in range(100)],tick(),0.,contract)
    assert result['zero_impulse_points']==100 and result['positive_points_qualified']==0
    assert all(p['frames']=={} and p['on_front'] is None for p in result['points'])


@pytest.mark.parametrize('change',['bool_pair_index','bool_exit','bool_pair1'])
def test_receipt_scope_and_execution_have_strict_json_types(tmp_path,change):
    contract,receipt=contract_fixture(tmp_path)
    if change=='bool_pair_index':receipt['scope']['retainer_pair_index']=False
    elif change=='bool_exit':receipt['execution']['exit_code']=False
    else:receipt['rule']['pair1_qualified']=0
    path=Path(contract['qualification']['file']);path.write_text(json.dumps(receipt));contract['qualification']=ref(path)
    with pytest.raises(ValueError):module().load_contract(contract)


def test_v3_failure_snapshot_preserves_explicitly_missing_native_world(tmp_path):
    from types import SimpleNamespace
    from sim.ausc_policy.retainer_failure import latch_retainer_failure
    contract,_=contract_fixture(tmp_path);ev=tick();ev['post_step'].update(available=False,world_from_carriage=None)
    p=point();report=audit([p],ev,4.,contract)
    env=SimpleNamespace(rows=[dict(time_s=DT,phase='CONTACT',target='L1')],contact_points=[p],
        retainer_front_audit=report,last_contacts={'retainer':4.},contact_tick_evidence=ev)
    latch_retainer_failure(env,None)
    saved=json.loads(env._retainer_failure_payload)
    assert saved['world_from_carriage'] is None
    assert saved['contact_tick_evidence']==ev and saved['points'][0]['raw']==p
    assert 'world_from_carriage' in saved['missing_fields']


def test_compact_capture_binds_native_contract_and_keeps_bounded_tick_summary(tmp_path):
    import gzip
    from test_ausc_policy_compact_expert_capture import compact_binding,tick as capture_tick
    from test_ausc_policy_expert_capture import Env,Backend,GEOMETRY
    from sim.ausc_policy import expert_capture
    contract,_=contract_fixture(tmp_path)
    value=compact_binding();value['compact']['metadata']['native_retainer_audit']=deepcopy(contract)
    geometry=dict(deepcopy(GEOMETRY),native_retainer_audit=deepcopy(contract))
    capture=expert_capture.create_expert_capture(tmp_path/'capture',binding=value,geometry_metadata=geometry)
    env=Env();capture.start(env,backend_factory=Backend);advance=env.advance
    def with_witness():
        advance()
        env.rows[-1].update(native_retainer_audit=deepcopy(contract),
            retainer_front_audit=audit([point()],tick(),4.,contract))
    env.advance=with_witness
    capture_tick(env,capture);capture.close()
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:row=json.loads(next(stream))
    assert row['native_retainer_audit']==contract
    assert row['retainer_front_audit']['native_retainer_audit']==contract
    assert row['retainer_front_audit']['schema']=='retainer_front_audit_v3'
    assert 'points' not in row['retainer_front_audit']
    manifest=json.loads((capture.output/'manifest.json').read_text())
    assert manifest['binding']['compact']['metadata']['native_retainer_audit']==contract
    with pytest.raises(ValueError,match='retainer audit'):
        expert_capture.create_expert_capture(tmp_path/'downgraded',binding=value,geometry_metadata=GEOMETRY)


def test_shared_scene_charges_and_authenticates_independent_analysis(tmp_path,monkeypatch):
    from test_ausc_policy_m2_pilot_config import fixture as prepared_fixture
    from sim.ausc_policy import m2_pilot_config,scene_store
    refs,plan,_=prepared_fixture(tmp_path,monkeypatch)
    qdir=tmp_path/'oracle';qdir.mkdir();contract,receipt=contract_fixture(qdir)
    analysis=qdir/'analysis.json';analysis.write_text('{"cpu_diagnostic_only":true}')
    receipt['independent_analysis']=ref(analysis)
    q=Path(contract['qualification']['file']);q.write_text(json.dumps(receipt));contract['qualification']=ref(q)
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):
        (tmp_path/name).write_text('{}')
    result=m2_pilot_config.build_compact_expert(**refs,fresh_plan=plan,episode_id='analysis',noise_seed=1,
        retainer_witness_contract=contract,scene_store=dict(root=str(tmp_path/'cas'),batch_id='diagnostic'))
    cfg=result['config'];source=cfg['compact_scene_storage']['sources']['native_retainer_independent_analysis']
    assert source==dict(ref(analysis),bytes=analysis.stat().st_size)
    scene_store.validate_config(cfg)
    cfg['compact_scene_storage']['sources'].pop('native_retainer_independent_analysis')
    with pytest.raises(ValueError,match='analysis'):scene_store.validate_config(cfg)


@pytest.mark.parametrize('change',['owner_failed','not_drained','gpu_held','wrapper_exit',
    'worker_failed','worker_not_clean','worker_timeout','worker_incomplete','worker_wrong',
    'complete_failed','evidence_partial','native_binding','geometry_binding',
    'native_hash','native_bytes','native_path','owner_path','worker_path'])
def test_receipt_rejects_misbound_or_failed_native_oracle_chain(tmp_path,change):
    contract,receipt=contract_fixture(tmp_path)
    evidence=receipt['evidence']
    key=('owner_terminal' if change in ('owner_failed','not_drained','gpu_held','wrapper_exit','owner_path') else
         'owner_exit' if change.startswith('worker_') else 'complete')
    path=Path(evidence[key]['file']);value=json.loads(path.read_text())
    if change=='owner_failed':value['completed']=False
    elif change=='not_drained':value['stage']['process_group_drained']=False
    elif change=='gpu_held':value['gpu_released']=False
    elif change=='wrapper_exit':value['stage']['exit_code']=1
    elif change=='worker_failed':value['returncode']=1
    elif change=='worker_not_clean':value['clean_exit']=False
    elif change=='worker_timeout':value['timed_out']=True
    elif change=='worker_incomplete':value['complete_file_present']=False
    elif change=='worker_wrong':value['worker']='other'
    elif change=='complete_failed':value['execution_completed']=False
    elif change=='evidence_partial':value['evidence_complete']=False
    elif change=='native_binding':value['native_binding_sha256']='e'*64
    elif change=='geometry_binding':value['retainer_geometry_source_sha256']='e'*64
    elif change=='native_hash':value['artifacts']['native_contacts.jsonl']['sha256']='e'*64
    elif change=='native_bytes':value['artifacts']['native_contacts.jsonl']['bytes']+=1
    elif change=='native_path':
        source=Path(evidence['native']['file']);other=tmp_path/'other-native.jsonl';other.write_bytes(source.read_bytes())
        evidence['native']=ref(other)
    elif change in ('owner_path','worker_path'):
        other=tmp_path/'other.json';other.write_bytes(path.read_bytes());path=other
    path.write_text(json.dumps(value));evidence[key]=ref(path)
    qualification=Path(contract['qualification']['file']);qualification.write_text(json.dumps(receipt))
    contract['qualification']=ref(qualification)
    with pytest.raises(ValueError):module().load_contract(contract)


@pytest.mark.parametrize('change',['receipt','owner','stage','worker','complete','artifacts'])
def test_malformed_qualification_documents_raise_explicit_value_error(tmp_path,change):
    contract,receipt=contract_fixture(tmp_path)
    if change=='receipt':receipt=[]
    else:
        key={'owner':'owner_terminal','stage':'owner_terminal','worker':'owner_exit',
             'complete':'complete','artifacts':'complete'}[change]
        path=Path(receipt['evidence'][key]['file']);value=json.loads(path.read_text())
        if change=='stage':value['stage']=[]
        elif change=='artifacts':value['artifacts']=[]
        else:value=[]
        path.write_text(json.dumps(value));receipt['evidence'][key]=ref(path)
    q=Path(contract['qualification']['file']);q.write_text(json.dumps(receipt));contract['qualification']=ref(q)
    with pytest.raises(ValueError):module().load_contract(contract)
