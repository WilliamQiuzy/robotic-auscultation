"""Real USD, pure CPU builder/CAS integration; no simulator acceptance."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_ausc_policy_m2_pilot_config import fixture


def given_config(tmp_path, monkeypatch, sdk=False):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd
    from sim.ausc_policy import m2_pilot_config as config
    refs, plan, save = fixture(tmp_path, monkeypatch)
    root = Path(refs['prepared_complete']['file']).parent
    stage = Usd.Stage.CreateInMemory(); world = stage.DefinePrim('/World')
    world.CreateAttribute('mdl', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    stage.GetRootLayer().Export(str(root/'candidate_physics.usdc'))
    complete = json.loads(Path(refs['prepared_complete']['file']).read_text())
    complete['effective_scene_sha256'] = config.hash_file(root/'candidate_physics.usdc')
    refs['prepared_complete'] = save('COMPLETE.json', complete)
    row = json.loads(Path(plan['file']).read_text())
    row.update(command_rotation_world=np.diag([1., -1., -1.]).tolist(), command_normal_world=[0., 0., 1.])
    plan = save('L1_fresh_plan.json', row)
    for name in ('mass_estimates.json', 'physics_profile.json', 'approved_camera_restoration.json'):
        (root/name).write_text('{}')
    store = dict(root=str(tmp_path/'cas'), batch_id='given-control-test')
    if sdk:
        from test_ausc_policy_sdk_mdl import sdk_fixture
        mdl, _, _ = sdk_fixture(tmp_path, monkeypatch)
        store['sdk_mdl_manifest'] = mdl.prepare_manifest(tmp_path/'sdk-manifest.json')
    result = config.build_given_contact_expert(**refs, fresh_plan=plan, episode_id='given-cas-test', scene_store=store)
    return result['config'], root


@pytest.mark.parametrize('sdk', [False, True])
def test_real_given_builder_validates_both_cas_versions_before_authoring_comparison(tmp_path, monkeypatch, sdk):
    from sim.ausc_policy import scene_store
    config, root = given_config(tmp_path, monkeypatch, sdk)
    before = (root/'candidate_physics.usdc').read_bytes()
    binding, complete = scene_store.validate_config(config)
    assert binding['schema'] == (scene_store.SDK_SCHEMA if sdk else scene_store.SCHEMA)
    assert complete['effective_scene_sha256'] == scene_store.sha(root/'candidate_physics.usdc')
    assert (root/'candidate_physics.usdc').read_bytes() == before
    assert 'm2_given_contact_control' not in scene_store._authoring(config)


@pytest.mark.parametrize('fault', ['schema', 'extra', 'episode', 'metadata_episode', 'capture_episode',
    'capture_metadata', 'reference_hash', 'reference_source', 'plan', 'patient', 'given_capture_without_route',
    'old_capture', 'unknown_top_route', 'physics'])
def test_only_exact_bound_given_teacher_intent_is_excluded(tmp_path, monkeypatch, fault):
    from sim.ausc_policy import scene_store
    config, _ = given_config(tmp_path, monkeypatch)
    route = config['m2_given_contact_control']; capture = config['fresh_expert']['raw_multimodal_capture']
    if fault == 'schema': route['schema'] = 'unknown'
    elif fault == 'extra': route['future_field'] = True
    elif fault == 'episode': route['episode_id'] = 'another'
    elif fault == 'metadata_episode': route['metadata']['episode_id'] = 'another'
    elif fault == 'capture_episode': capture['episode_id'] = 'another'
    elif fault == 'capture_metadata': capture['compact']['metadata']['reference_revision'] = 1
    elif fault == 'reference_hash': route['metadata']['reference_sha256'] = '0'*64
    elif fault == 'reference_source':
        path = Path(route['metadata']['given_reference']['sources']['fresh_plan']['file']); path.write_text(path.read_text()+' ')
    elif fault == 'plan': config['fresh_expert']['plans'][0]['sha256'] = '0'*64
    elif fault == 'patient':
        config['compact_scene_storage']['sources']['patient_receipt']['sha256'] = '0'*64
    elif fault == 'given_capture_without_route': config.pop('m2_given_contact_control')
    elif fault == 'old_capture': capture['compact']['metadata']['target_source'] = 'gt_plus_measured_noise'
    elif fault == 'unknown_top_route': config['m2_unknown_control'] = config.pop('m2_given_contact_control')
    else: config['fresh_expert']['passive_preload_N'] = 2.
    with pytest.raises(ValueError): scene_store.validate_config(config)


def test_unknown_runtime_route_still_fails_with_legacy_compact_capture(tmp_path):
    from test_ausc_policy_scene_store import bound
    from sim.ausc_policy import scene_store
    _, config = bound(tmp_path)
    config['m2_unknown_control'] = dict(schema='unknown')
    with pytest.raises(ValueError, match='authoring'): scene_store.validate_config(config)
