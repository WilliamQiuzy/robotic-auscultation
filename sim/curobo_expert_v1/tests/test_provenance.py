import hashlib
import json
from pathlib import Path
import pytest

from sim.curobo_expert_v1.provenance import REQUIRED_SOURCES, bind_manifest, verify_candidate_sources


def setup_episode(tmp_path, policy='revised_shortcut'):
    files={};objects=tmp_path/'input_snapshot/objects';objects.mkdir(parents=True)
    for name in REQUIRED_SOURCES:
        content=('test source '+name).encode();digest=hashlib.sha256(content).hexdigest()
        (objects/digest).write_bytes(content)
        files['source/new_expert/'+name]={'sha256':digest}
    snapshot=tmp_path/'input_snapshot/manifest.json'
    snapshot.write_text(json.dumps({'files':files}))
    outcome={'candidate_policy':policy,'candidate_method':'sim.curobo_expert_v1.controller.ExpertController',
             'input_snapshot':'input_snapshot/manifest.json',
             'input_snapshot_manifest_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
             'planner_calls':[], 'success':False}
    (tmp_path/'outcome.json').write_text(json.dumps(outcome))
    bind_manifest(tmp_path,policy)
    return json.loads((tmp_path/'outcome.json').read_text())


def test_captured_sources_validate(tmp_path):
    outcome=setup_episode(tmp_path)
    assert verify_candidate_sources(tmp_path,outcome)['status']=='passed'


def test_changed_source_object_rejected(tmp_path):
    outcome=setup_episode(tmp_path)
    manifest=json.loads((tmp_path/'candidate_sources_manifest.json').read_text())
    (tmp_path/manifest['files']['controller.py']['path']).write_text('changed')
    assert verify_candidate_sources(tmp_path,outcome)['status']=='failed'


def test_different_policy_rejected(tmp_path):
    outcome=setup_episode(tmp_path);outcome['candidate_policy']='curobo'
    assert verify_candidate_sources(tmp_path,outcome)['status']=='failed'


def test_generated_route_tamper_rejected(tmp_path):
    setup_episode(tmp_path)
    d=tmp_path/'planner_requests/001';d.mkdir(parents=True)
    (d/'result.json').write_text('{"success":false}')
    bind_manifest(tmp_path,'revised_shortcut')
    outcome=json.loads((tmp_path/'outcome.json').read_text())
    (d/'result.json').write_text('changed')
    assert verify_candidate_sources(tmp_path,outcome)['status']=='failed'


def test_curobo_cannot_pass_without_captured_world_and_launcher(tmp_path):
    outcome=setup_episode(tmp_path,'curobo')
    outcome.update(success=True,points=[{} for _ in range(10)],
        planner_calls=[{'directory':'planner_requests/001','proposal_accepted':False,
                        'motion_complete':False} for _ in range(9)])
    report=verify_candidate_sources(tmp_path,outcome)
    assert report['status']=='failed'
    assert any('bridge' in reason for reason in report['reasons'])


@pytest.mark.parametrize('field',['files','generated_files'])
def test_invalid_container_fails_without_crashing(tmp_path,field):
    outcome=setup_episode(tmp_path)
    path=tmp_path/'candidate_sources_manifest.json'
    manifest=json.loads(path.read_text());manifest[field]=list(manifest[field])
    path.write_text(json.dumps(manifest))
    outcome['candidate_sources_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_candidate_sources(tmp_path,outcome)['status']=='failed'


def test_invalid_calls_container_fails_without_crashing(tmp_path):
    outcome=setup_episode(tmp_path);outcome['planner_calls']={}
    assert verify_candidate_sources(tmp_path,outcome)['status']=='failed'
