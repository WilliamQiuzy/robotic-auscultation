"""Pure configuration fixtures; no scene, physics or data qualification."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from sim.ausc_policy import geometry_epoch as epoch,m2_contract as contract

ROOT=Path(__file__).resolve().parents[1]


def module():
    from sim.ausc_policy import m2_pilot_config
    return m2_pilot_config


def test_bound_metadata_parses_exact_verified_bytes(tmp_path,monkeypatch):
    m=module();p=tmp_path/'meta.json';p.write_text('{"authenticated":true}')
    ref=m.reference(p);read=Path.read_bytes
    def replace(path):
        data=read(path)
        if path==p:path.write_text('{"authenticated":false}')
        return data
    monkeypatch.setattr(Path,'read_bytes',replace)
    assert m.bound_json(ref)=={'authenticated':True}


def fixture(tmp_path,monkeypatch,split='train',identity='p000439'):
    m=module();body=tmp_path/'body.npz';body.write_bytes(b'CPU fixture body')
    original=tmp_path/'original.usdc';original.write_bytes(b'CPU fixture original')
    real_hash=m.hash_file
    monkeypatch.setattr(m,'hash_file',lambda p:epoch.GEOMETRY_SOURCE_SHA256 if Path(p)==original else real_hash(p))
    def save(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return dict(file=str(path),sha256=real_hash(path))
    manifest=[dict(file=body.name,identity=identity,split=split,sha256=real_hash(body),numeric_candidate_screen_passed=True,mesh_scale=1.)]
    manifest_ref=save('asset_manifest.json',manifest)
    summary_ref=save('summary.json',dict(schema='m2_observed_seated_retarget_v1',identity_split_sha256=contract.IDENTITY_SPLIT_SHA256))
    target=dict(name='L1',position_world_m=[.1,.2,1.],normal_world=[0,0,1],
        contact_plane_normal_world=[0,0,1],contact_plane_radius_m=.023)
    patient=dict(identity=identity,split=split,asset_sha256=real_hash(body),identity_split_sha256=contract.IDENTITY_SPLIT_SHA256,
        targets=[target],non_patient_scene_unchanged=True,layout_translation_world_m=[.15,0,.06])
    patient_ref=save('patient_receipt.json',patient)
    geometry=tmp_path/'geometry';geometry.mkdir();(geometry/'collision_geometry.json').write_text('{}')
    (geometry/'collision_geometry.npz').write_bytes(b'CPU collision fixture')
    artifacts=dict(collision_geometry_sha256=real_hash(geometry/'collision_geometry.json'),collision_arrays_sha256=real_hash(geometry/'collision_geometry.npz'))
    scene=tmp_path/'candidate_physics.usdc';scene.write_bytes(b'CPU installed scene fixture')
    base=np.eye(4).tolist()
    complete=save('COMPLETE.json',dict(completed=True,geometry_id=epoch.GEOMETRY_ID,source_sha256=epoch.GEOMETRY_SOURCE_SHA256,
        effective_scene_sha256=real_hash(scene),patient=patient,geometry_artifacts=artifacts,world_from_left_base=base,
        physics_executed=False,physical_validation=False,training_eligible=False))
    current=json.loads((ROOT/'configs/ausc_policy/fresh_expert_retainer_tenpoint_near_v4.json').read_text())
    current.update(geometry_source=str(original),prepare_only=True,raw_target_provenance={'wrong':'nominal'},
        patient_asset=dict(geometry_id=epoch.GEOMETRY_ID,geometry_source_sha256=epoch.GEOMETRY_SOURCE_SHA256,freshly_generated=True,
        directory=str(tmp_path),file=body.name,manifest_sha256=manifest_ref['sha256'],summary_sha256=summary_ref['sha256'],
        identity_split=str(contract.IDENTITY_SPLIT_PATH),expected_split=split,layout=dict(xy_m=[.15,0],height_m=.06)))
    prep=save('prep.json',current)
    plan=dict(name='L1',success=True,geometry_id=epoch.GEOMETRY_ID,geometry_source_sha256=epoch.GEOMETRY_SOURCE_SHA256,
        freshly_generated=True,geometry_artifacts=artifacts,world_from_left_base=base,posture_limits=current['posture_limits'],
        loaded_compression_m=.0025,free_approach_start_gap_m=.025,position_world=target['position_world_m'],normal_world=target['normal_world'],
        patient_provenance=dict(identity=identity,asset_sha256=real_hash(body),binding=patient_ref,identity_split_sha256=contract.IDENTITY_SPLIT_SHA256),
        selected_fit=dict(first_contact=dict(contact_plane_normal_world=[0,0,1],contact_plane_radius_m=.023,plane_tilt_limit_deg=3.,tilt_relaxed=False)))
    return dict(preparation_config=prep,prepared_complete=complete,patient_receipt=patient_ref),save('L1_fresh_plan.json',plan),save


def test_scene_store_opt_in_is_versioned_and_does_not_change_legacy_config(tmp_path,monkeypatch):
    refs,plan,_=fixture(tmp_path,monkeypatch);m=module()
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):
        (tmp_path/name).write_text('{}')
    args=dict(**refs,fresh_plan=plan,episode_id='a',noise_seed=1)
    old=m.build_compact_expert(**args)
    new=m.build_compact_expert(**args,scene_store=dict(root=str(tmp_path/'cas'),batch_id='native-batch1'))
    binding=new['config'].pop('compact_scene_storage')
    assert new['config']==old['config'] and binding['schema']=='m2_compact_scene_storage_v1'
    assert binding['sources']['scene']['sha256']==old['receipt']['effective_scene']['sha256']
    assert new['receipt']['compact_scene_storage']==binding


def test_scene_store_sdk_manifest_argument_selects_only_explicit_v2(tmp_path,monkeypatch):
    from test_ausc_policy_sdk_mdl import sdk_fixture
    refs,plan,_=fixture(tmp_path,monkeypatch);m=module();sdk,_,_=sdk_fixture(tmp_path,monkeypatch)
    manifest=sdk.prepare_manifest(tmp_path/'sdk-manifest.json')
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):(tmp_path/name).write_text('{}')
    args=dict(**refs,fresh_plan=plan,episode_id='a',noise_seed=1)
    legacy=m.build_compact_expert(**args,scene_store=dict(root=str(tmp_path/'cas'),batch_id='batch'))
    modern=m.build_compact_expert(**args,scene_store=dict(root=str(tmp_path/'cas'),batch_id='batch',sdk_mdl_manifest=manifest))
    old_binding=legacy['config'].pop('compact_scene_storage');binding=modern['config'].pop('compact_scene_storage')
    assert old_binding['schema']=='m2_compact_scene_storage_v1' and binding['schema']=='m2_compact_scene_storage_v2'
    assert binding['sources']['sdk_mdl_manifest']['sha256']==manifest['sha256']
    assert modern['config']==legacy['config']
    with pytest.raises(ValueError):
        m.build_compact_expert(**args,scene_store=dict(root=str(tmp_path/'cas'),batch_id='batch',sdk_mdl_manifest=None))


def test_geometry_builder_binds_actual_new_scene_and_preserves_posture(tmp_path,monkeypatch):
    refs,_,_=fixture(tmp_path,monkeypatch);result=module().build_geometry_plan(**refs,target='L1')
    cfg=result['config'];prep=json.loads(Path(refs['preparation_config']['file']).read_text())
    assert cfg['plane_tilt_limit_deg']==3. and cfg['loaded_compression_m']==.0025
    assert cfg['fresh_pre_gap_search_m']==[.025,.03] and cfg['posture_limits']==prep['posture_limits']
    assert cfg['collision_directory']==str(tmp_path/'geometry') and cfg['patient_receipt']==refs['patient_receipt']
    assert cfg['targets'][0]['contact_plane_radius_m']==.023 and not result['receipt']['training_eligible']


def test_geometry_builder_optional_margin_is_explicit_and_planning_only(tmp_path,monkeypatch):
    refs,_,_=fixture(tmp_path,monkeypatch);m=module()
    old=m.build_geometry_plan(**refs,target='L1')
    assert 'planning_plane_margin_deg' not in old['config'] and 'planning_plane_policy' not in old['receipt']
    new=m.build_geometry_plan(**refs,target='L1',planning_plane_margin_deg=.25)
    assert new['config']==dict(old['config'],planning_plane_margin_deg=.25)
    policy=new['receipt'].pop('planning_plane_policy')
    assert new['receipt']==old['receipt']
    assert policy['margin_deg']==.25 and policy['candidate_tilt_limit_deg']==2.75
    assert policy['runtime_tilt_limit_deg']==3. and policy['physical_tracking_bound_certified'] is False


@pytest.mark.parametrize('value',[True,0.,-.1,1.5001,float('nan'),float('inf'),'0.25'])
def test_geometry_builder_rejects_invalid_planning_margin(tmp_path,monkeypatch,value):
    refs,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='planning.*margin'):
        module().build_geometry_plan(**refs,target='L1',planning_plane_margin_deg=value)


def test_geometry_builder_does_not_apply_strict_margin_to_relaxed_protocol(tmp_path,monkeypatch):
    refs,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='planning.*margin'):
        module().build_geometry_plan(**refs,target='L1',planning_plane_margin_deg=.25,
            plane_tilt_limit_deg=10.,fallback_reason='Explicit fallback')


def test_compact_builder_keeps_margin_plan_bound_without_tightening_physics(tmp_path,monkeypatch):
    refs,plan,save=fixture(tmp_path,monkeypatch);m=module()
    proposal=m.build_geometry_plan(**refs,target='L1',planning_plane_margin_deg=.25)
    document=json.loads(Path(plan['file']).read_text())
    document['selected_fit']['first_contact'].update(
        plane_tilt_deg=2.75,planning_plane_policy=proposal['receipt']['planning_plane_policy'])
    new_plan=save('L1_margin_plan.json',document)
    result=m.build_compact_expert(**refs,fresh_plan=new_plan,episode_id='margin-L1',noise_seed=17)
    assert result['config']['fresh_expert']['plans']==[new_plan]
    prep=json.loads(Path(refs['preparation_config']['file']).read_text())
    assert result['config']['physics']==prep['physics']
    for key,value in prep['fresh_expert'].items():
        if key not in ('plans','raw_multimodal_capture','collision_profile'):
            assert result['config']['fresh_expert'][key]==value
    assert m.bound_json(new_plan)['selected_fit']['first_contact']['plane_tilt_limit_deg']==3.


def test_ten_degree_fallback_is_explicit(tmp_path,monkeypatch):
    refs,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='fallback'):module().build_geometry_plan(**refs,target='L1',plane_tilt_limit_deg=10.)
    result=module().build_geometry_plan(**refs,target='L1',plane_tilt_limit_deg=10.,fallback_reason='Recorded three degree bounded fit found no candidate')
    assert result['receipt']['tilt_relaxed'] is True


def test_builder_rejects_seed_budget_unsupported_by_planner(tmp_path,monkeypatch):
    refs,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='bounded planner work'):
        module().build_geometry_plan(**refs,target='L1',seeds_per_candidate=13)


def test_compact_config_noise_visible_and_deterministic_without_normal_noise(tmp_path,monkeypatch):
    refs,plan,_=fixture(tmp_path,monkeypatch);m=module()
    args=dict(**refs,fresh_plan=plan,episode_id='pilot-L1',noise_seed=17)
    a=m.build_compact_expert(**args);b=m.build_compact_expert(**args)
    assert a==b
    cfg=a['config'];settings=cfg['fresh_expert']['raw_multimodal_capture']['compact'];noise=a['receipt']['target_estimate_noise']
    assert settings['max_frames']==5000 and settings['limit_bytes']==100_000_000
    assert settings['metadata']['source_kind']=='isaac_live_physics' and settings['metadata']['replay'] is False
    assert settings['metadata']['target_source']=='gt_plus_measured_noise'
    assert np.linalg.norm(noise['position_bias_base_m'])==pytest.approx(.004)
    assert noise['normal_noise_applied'] is False
    np.testing.assert_allclose(np.asarray(settings['estimated_target_base'])[:3,3],np.array([.1,.2,1.])+noise['position_bias_base_m'])
    assert not cfg.get('prepare_only') and cfg['fresh_expert']['plans']==[plan]
    assert cfg['raw_target_provenance']['patient_receipt']==refs['patient_receipt']
    assert cfg['physics']==json.loads(Path(refs['preparation_config']['file']).read_text())['physics']
    from sim.ausc_policy.expert_capture import capture_settings
    assert capture_settings(cfg['fresh_expert'])['profile']=='compact'


@pytest.mark.parametrize('split',['train','validation'])
def test_real_compact_builder_feeds_v2_provider_without_capture_metadata_injection(tmp_path,monkeypatch,split):
    """Exercise the real builder/config bytes/provider path; no capture or physics."""
    from types import SimpleNamespace
    from sim.ausc_policy import expert_capture,m2_estimate_provider
    def forbidden_capture(*args,**kwargs):
        pytest.fail('Provider metadata must come from the frozen builder, before capture exists')
    monkeypatch.setattr(expert_capture.CompactExpertCapture,'__init__',forbidden_capture)
    identity=json.loads(contract.IDENTITY_SPLIT_PATH.read_text())['splits'][split][0]
    refs,plan,save=fixture(tmp_path,monkeypatch,split,identity)
    result=module().build_compact_expert(**refs,fresh_plan=plan,episode_id='source-L1',noise_seed=17)
    compact=result['config']['fresh_expert']['raw_multimodal_capture']['compact']
    original=deepcopy(compact['metadata'])
    config=dict(schema=m2_estimate_provider.SCHEMA,
        compact_config=save('built_compact_config.json',result['config']),
        noise_receipt=save('built_compact_receipt.json',result['receipt']),revision=0)
    patient=dict(identity=identity,split=split,asset_sha256=original['patient_asset_sha256'],
                 receipt_sha256=refs['patient_receipt']['sha256'])
    provider=m2_estimate_provider.FrozenEstimateProvider(config,
        metadata=dict(original,episode_id='deployment-L1'),episode_id='deployment-L1',target_id='L1',patient=patient)
    state=dict(q=np.zeros(7),time_s=.05)
    controller=SimpleNamespace(episode_id='deployment-L1',target=SimpleNamespace(name='L1'),
        env=SimpleNamespace(state=lambda:deepcopy(state)))
    acquired=provider(controller,state)
    assert acquired['episode_id']=='deployment-L1' and acquired['observed_at_s']==.05
    np.testing.assert_array_equal(acquired['estimated_target_base'],compact['estimated_target_base'])
    assert compact['metadata']==original
    assert original['observation_contract']==contract.observation_contract()
    assert result['receipt']['observation_contract']==contract.observation_contract()
    assert result['receipt']['schema']=='m2_pilot_compact_expert_config_v2'
    assert provider.snapshot()['source_metadata']['observation_contract']==contract.observation_contract()


@pytest.mark.parametrize('split,identity',[('test','p000438'),('train','p000452'),('validation','p000439')])
def test_rejects_test_quarantine_or_membership_before_payload_hashing(tmp_path,monkeypatch,split,identity):
    refs,_,_=fixture(tmp_path,monkeypatch,split,identity);m=module();real=m.hash_file;opened=[]
    def audit(path):
        opened.append(str(path));return real(path)
    monkeypatch.setattr(m,'hash_file',audit)
    with pytest.raises(ValueError,match='identity|split|quarantin'):m.build_geometry_plan(**refs,target='L1')
    assert not any(Path(p).name in ('body.npz','candidate_physics.usdc','collision_geometry.npz') for p in opened)


def test_changed_scene_and_false_plan_rejected(tmp_path,monkeypatch):
    refs,plan,save=fixture(tmp_path,monkeypatch);m=module()
    obj=json.loads(Path(plan['file']).read_text());obj['success']=False
    with pytest.raises(ValueError,match='successful'):m.build_compact_expert(**refs,fresh_plan=save('bad.json',obj),episode_id='a',noise_seed=1)
    (tmp_path/'candidate_physics.usdc').write_bytes(b'changed')
    with pytest.raises(ValueError,match='scene'):m.build_geometry_plan(**refs,target='L1')


def test_failed_target_missing_plane_and_stale_patient_binding_rejected(tmp_path,monkeypatch):
    refs,plan,save=fixture(tmp_path,monkeypatch);m=module()
    with pytest.raises(ValueError,match='target'):m.build_geometry_plan(**refs,target='R5')
    obj=json.loads(Path(plan['file']).read_text());obj['patient_provenance']['binding']['sha256']='a'*64
    with pytest.raises(ValueError,match='patient'):m.build_compact_expert(**refs,fresh_plan=save('badplan.json',obj),episode_id='a',noise_seed=1)


def test_explicit_cache_profile_and_frame_bound(tmp_path,monkeypatch):
    refs,plan,_=fixture(tmp_path,monkeypatch);m=module()
    result=m.build_compact_expert(**refs,fresh_plan=plan,episode_id='a',noise_seed=1,collision_profile='certified_temporal_v1',max_frames=6000)
    assert result['config']['fresh_expert']['collision_profile']=='certified_temporal_v1'
    with pytest.raises(ValueError):m.build_compact_expert(**refs,fresh_plan=plan,episode_id='a',noise_seed=1,max_frames=6001)


def test_validation_split_is_explicit_and_does_not_change_force_settings(tmp_path,monkeypatch):
    subject=json.loads(contract.IDENTITY_SPLIT_PATH.read_text())['splits']['validation'][0]
    refs,plan,_=fixture(tmp_path,monkeypatch,'validation',subject)
    cfg=module().build_compact_expert(**refs,fresh_plan=plan,episode_id='validation-L1',noise_seed=4)['config']
    previous=json.loads(Path(refs['preparation_config']['file']).read_text())
    for key,value in previous['fresh_expert'].items():
        if key not in ('plans','raw_multimodal_capture','collision_profile'):assert cfg['fresh_expert'][key]==value
    assert cfg['fresh_expert']['raw_multimodal_capture']['compact']['metadata']['split']=='validation'


def test_noise_changes_translation_only_in_real_base_frame():
    m=module();target=dict(position_world_m=[1.,2.,3.],contact_plane_normal_world=[0,1.,0])
    base=np.eye(4);base[:3,3]=[1.,1.,1.]
    a,ar=m.target_estimate(target,base,1);b,br=m.target_estimate(target,base,2)
    assert not np.array_equal(a[:3,3],b[:3,3])
    np.testing.assert_array_equal(a[:3,:3],b[:3,:3])
    np.testing.assert_allclose(a[:3,3]-ar['position_bias_base_m'],[0.,1.,2.])
    np.testing.assert_allclose(a[:3,2],[0,-1.,0])


def test_plan_collision_binding_and_layout_mismatch_fail_closed(tmp_path,monkeypatch):
    refs,plan,save=fixture(tmp_path,monkeypatch);m=module()
    obj=json.loads(Path(plan['file']).read_text());obj['geometry_artifacts']['collision_arrays_sha256']='a'*64
    with pytest.raises(ValueError,match='geometry'):m.build_compact_expert(**refs,fresh_plan=save('badgeometry.json',obj),episode_id='a',noise_seed=1)
    prep=json.loads(Path(refs['preparation_config']['file']).read_text());prep['patient_asset']['layout']['height_m']=0.
    refs['preparation_config']=save('different-layout.json',prep)
    with pytest.raises(ValueError,match='layout'):m.build_geometry_plan(**refs,target='L1')
