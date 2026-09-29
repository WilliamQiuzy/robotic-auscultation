"""Task conditioning cannot silently alter the acquired sensor/action arrays."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim.auscultation_pi05_v6 import derive_task_dataset as module
from sim.auscultation_pi05_v6.task_observation import observation_binding


def fixture(tmp_path,monkeypatch):
    raw=np.arange(64,dtype=np.float32).reshape(2,32)/100
    state=np.concatenate((raw,np.ones((2,9),np.float32)),axis=1)
    parent=tmp_path/'parent.npz';np.savez_compressed(parent,state=raw,actions=np.zeros((2,7)),rgb=np.zeros((2,2,2,3),np.uint8))
    output=tmp_path/'derived.npz';members=module.replace_state(parent,output,state)
    evidence=tmp_path/'evidence';evidence.mkdir()
    digest=hashlib.sha256(b'{}').hexdigest()
    snapshot=dict(files={key:{'sha256':digest} for key in (*module.DEPENDENCIES,'R4/plan')})
    for name in module.INPUTS:
        path=evidence/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    (evidence/'task_frame_registration.json').write_text('{"R4":{}}')
    (evidence/'input_snapshot/manifest.json').write_text(json.dumps(snapshot))
    (evidence/'input_snapshot/objects').mkdir()
    (evidence/'input_snapshot/objects'/digest).write_text('{}')
    inputs={str(p.relative_to(evidence)):module.sha256(p) for p in evidence.rglob('*') if p.is_file()}
    audit=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,
        audit_source_sha256=module.sha256(Path(module.__file__).with_name('audit_observed_supervisor.py')),
        artifact_sha256=inputs)
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps(audit))
    (evidence/'observed_supervisor_audit.json').write_bytes(audit_path.read_bytes())
    binding=dict(path='audit.json',sha256=module.sha256(audit_path),artifact_sha256=audit['artifact_sha256'])
    pairs=[dict(frame_index=2),dict(frame_index=3)]
    record=dict(path='derived.npz',sha256=module.sha256(output))
    derivation=dict(schema=module.SCHEMA,observation_contract=observation_binding(),source_sha256=module.sha256(module.__file__),
        arrays_sha256=record['sha256'],parent_episode=dict(path='parent.npz',sha256=module.sha256(parent)),
        evidence_root='evidence',input_sha256=dict(audit['artifact_sha256'],**{'observed_supervisor_audit.json':binding['sha256']}),
        qualified_supervisor_audit_sha256=binding['sha256'],
        label_pairs_sha256=hashlib.sha256(module.json_bytes(pairs)).hexdigest(),array_members=members)
    report=dict(checks=dict(task_observation_verified=True),evidence=dict(task_observation_derivation=derivation,label_pairs=pairs,
        joint_action_derivation=dict(phase_derivation=dict(independent_supervisor_audit=binding))))
    monkeypatch.setattr(module,'derive_states',lambda *args:state.copy())
    return record,report,{'state':state.copy()}


def test_only_state_member_changes_and_original_state_prefix_is_preserved(tmp_path,monkeypatch):
    record,report,arrays=fixture(tmp_path,monkeypatch)
    assert module.verify_task_derivation(tmp_path,record,report,arrays)
    with zipfile.ZipFile(tmp_path/'parent.npz') as a,zipfile.ZipFile(tmp_path/'derived.npz') as b:
        assert a.read('actions.npy')==b.read('actions.npy')
        assert a.read('rgb.npy')==b.read('rgb.npy')


@pytest.mark.parametrize('change',['raw_sensor','task_pose','input_file','label_pairs','observation_contract','action_member','substituted_audit','omitted_binding','omitted_object'])
def test_missing_or_altered_goal_provenance_is_rejected(tmp_path,monkeypatch,change):
    record,report,arrays=fixture(tmp_path,monkeypatch)
    if change=='raw_sensor':arrays['state'][0,20]+=.01
    elif change=='task_pose':arrays['state'][0,36]+=.01
    elif change=='input_file':(tmp_path/'evidence/outcome.json').write_text('{"changed":true}')
    elif change=='label_pairs':report['evidence']['label_pairs'][0]['frame_index']=12
    elif change=='observation_contract':report['evidence']['task_observation_derivation']['observation_contract']['version']='old'
    elif change=='substituted_audit':
        # An attacker can update the derivative's own hash, but not the bound parent audit.
        path=tmp_path/'evidence/observed_supervisor_audit.json'
        changed=json.loads(path.read_text());changed['artifact_sha256']['outcome.json']='0'*64
        path.write_text(json.dumps(changed))
        report['evidence']['task_observation_derivation']['input_sha256']['observed_supervisor_audit.json']=module.sha256(path)
    elif change=='omitted_binding':del report['evidence']['task_observation_derivation']['input_sha256']['outcome.json']
    elif change=='omitted_object':
        values=report['evidence']['task_observation_derivation']['input_sha256']
        del values[next(k for k in values if k.startswith('input_snapshot/objects/'))]
    else:
        with np.load(tmp_path/'derived.npz') as source:values={name:source[name] for name in source.files}
        values['actions'][0,0]=.1;np.savez_compressed(tmp_path/'derived.npz',**values)
    with pytest.raises(ValueError):module.verify_task_derivation(tmp_path,record,report,arrays)
