"""Failure evidence only: native-shaped CPU inputs, no physical qualification."""
from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy.retainer_contact import audit_retainer_front,ROOT
from sim.ausc_policy import fresh_expert as fresh


def module():return importlib.import_module('sim.ausc_policy.retainer_failure')


def native_failure(*,bad_force=.25,ledger=None):
    world=np.eye(4);world[:3,3]=[.1,.2,1.]
    world[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    points=[]
    for index,(local,force) in enumerate([([.022,0.,-.047],3.),([.023,0.,-.046],0.),([.023,0.,-.046],bad_force)]):
        points.append(dict(colliders=[ROOT+f'DeviceContactRing/sector_{index:02d}','/World/Patient/SMPLSurface'],
            position_world=(world[:3,:3]@local+world[:3,3]).tolist(),
            impulse_Ns=[0.,0.,force/240],separation_m=-.00003))
    env=SimpleNamespace(rows=[dict(time_s=1/240,phase='CONTACT',target='R3',subphase='contact_cycle')],
        contact_points=points,last_contacts={'retainer':3.+bad_force if ledger is None else ledger})
    env.retainer_front_audit=audit_retainer_front(points,world,env.last_contacts['retainer'])
    return env,world


def test_latches_first_native_failure_with_exact_raw_points_and_zero_counts():
    m=module();env,world=native_failure();before=deepcopy(env.__dict__)
    m.latch_retainer_failure(env,world)
    payload=env._retainer_failure_payload;doc=json.loads(payload)
    assert len(payload)<=m.MAX_BYTES==131072
    assert doc['physics_row']==0 and doc['time_s']==1/240 and doc['phase']=='CONTACT' and doc['target']=='R3'
    np.testing.assert_array_equal(doc['world_from_carriage'],world)
    assert doc['counts']==dict(contact_points=3,retainer_patient_points=3,positive_impulse_points=2,
        zero_impulse_points=1,positive_impulse_off_front_points=1,zero_impulse_off_front_points=1)
    assert doc['retainer_front_audit']['off_front_force_N']==pytest.approx(.25)
    assert not doc['truncated'] and not doc['missing_fields'] and doc['points_omitted']==0
    for index,point in enumerate(doc['points']):
        assert point['raw']==env.contact_points[index] and point['sector']==index
        assert point['audit']==env.retainer_front_audit['points'][index]
    assert doc['physical_gates_unchanged'] and not doc['training_eligible']
    assert env.contact_points==before['contact_points'] and env.retainer_front_audit==before['retainer_front_audit']
    env.contact_points[0]['position_world'][0]=999.;env.rows.append(dict(time_s=2/240))
    m.latch_retainer_failure(env,world)
    assert env._retainer_failure_payload==payload


def test_zero_impulse_off_front_candidates_do_not_trigger_new_failure_or_file(tmp_path):
    m=module();env,world=native_failure(bad_force=0.)
    assert env.retainer_front_audit['passed']
    m.latch_retainer_failure(env,world)
    assert not hasattr(env,'_retainer_failure_payload')
    assert m.save_retainer_failure(env,tmp_path) is None and not list(tmp_path.iterdir())


def test_unreconciled_ledger_without_raw_points_has_explicit_missing_evidence(tmp_path):
    m=module();env,world=native_failure();env.contact_points=[]
    env.retainer_front_audit=audit_retainer_front([],world,3.)
    m.latch_retainer_failure(env,world)
    doc=json.loads(env._retainer_failure_payload)
    assert not doc['retainer_front_audit']['force_matches_ledger']
    assert doc['counts']['retainer_patient_points']==0 and doc['points']==[]
    assert 'positive_ledger_without_retainer_points' in doc['missing_fields']


def test_bounded_snapshot_truncates_whole_points_without_losing_total_counts(tmp_path):
    m=module();env,world=native_failure();env.contact_points=env.contact_points*500
    env.last_contacts['retainer']*=500
    env.retainer_front_audit=audit_retainer_front(env.contact_points,world,env.last_contacts['retainer'])
    m.latch_retainer_failure(env,world)
    doc=json.loads(env._retainer_failure_payload)
    assert len(env._retainer_failure_payload)<=131072 and doc['truncated']
    assert doc['counts']['retainer_patient_points']==1500
    assert doc['points_omitted']==1500-len(doc['points'])>0
    receipt=m.save_retainer_failure(env,tmp_path)
    path=tmp_path/'retainer_failure.json';data=path.read_bytes()
    assert receipt['saved'] and receipt['sha256']==hashlib.sha256(data).hexdigest()
    assert receipt['bytes']==len(data)<=131072 and receipt['truncated']


def test_finalizer_preserves_sha_and_cleanup_errors_without_overwrite(tmp_path):
    m=module();env,world=native_failure();m.latch_retainer_failure(env,world)
    calls=[];env.save=lambda _:calls.append('trace');env.world=SimpleNamespace(stop=lambda:calls.append('stop'))
    _,errors=fresh.finalize_probe(env,None,None,tmp_path)
    assert not errors and calls==['trace','stop']
    assert env.retainer_failure_evidence['saved'] and env.retainer_failure_evidence['sha256']
    data=(tmp_path/'retainer_failure.json').read_bytes()
    _,errors=fresh.finalize_probe(env,None,None,tmp_path)
    assert any(row['stage']=='retainer_failure_save' for row in errors)
    assert (tmp_path/'retainer_failure.json').read_bytes()==data and calls[-1]=='stop'


def test_compact_failure_evidence_stays_inside_existing_auxiliary_and_total_budget(tmp_path):
    from test_ausc_policy_compact_expert_capture import make,tick
    m=module();runtime,capture=make(tmp_path)
    for _ in range(12):tick(runtime,capture)
    capture.close('retainer failure')
    env,world=native_failure();m.latch_retainer_failure(env,world)
    receipt=m.save_retainer_failure(env,tmp_path,capture=capture)
    assert receipt['budget_accounting']=='compact_auxiliary_included_in_episode_total'
    assert receipt==capture.receipts['retainer_failure_evidence']
    assert (capture.output/'retainer_failure.json').exists()
    result=capture.result(12,execution_completed=False)
    assert result['total_bytes']==sum(p.stat().st_size for p in capture.output.rglob('*') if p.is_file())
    assert result['total_bytes']<=capture.limit_bytes and not result['compact_capture_complete']
    assert result['receipts']['retainer_failure_evidence']['sha256']==receipt['sha256']


def test_compact_insufficient_auxiliary_budget_reports_missing_without_writing(tmp_path):
    from test_ausc_policy_compact_expert_capture import make
    m=module();_,capture=make(tmp_path);capture.close('retainer failure')
    capture.auxiliary_limit=capture.AUXILIARY_RESERVE_BYTES
    env,world=native_failure();m.latch_retainer_failure(env,world)
    with pytest.raises(ValueError,match='budget'):
        m.save_retainer_failure(env,tmp_path,capture=capture)
    assert not (capture.output/'retainer_failure.json').exists()
    assert env.retainer_failure_evidence['missing'] and not env.retainer_failure_evidence['saved']


def test_compact_remaining_auxiliary_capacity_truncates_evidence_before_write(tmp_path):
    from test_ausc_policy_compact_expert_capture import make
    m=module();_,capture=make(tmp_path);capture.close('retainer failure')
    used=sum(path.stat().st_size for path in capture.output.iterdir() if path.is_file())
    capture.auxiliary_limit=used+capture.AUXILIARY_RESERVE_BYTES+8192
    env,world=native_failure();env.contact_points*=100;env.last_contacts['retainer']*=100
    env.retainer_front_audit=audit_retainer_front(env.contact_points,world,env.last_contacts['retainer'])
    m.latch_retainer_failure(env,world)
    assert len(env._retainer_failure_payload)>8192
    receipt=m.save_retainer_failure(env,tmp_path,capture=capture)
    document=json.loads((capture.output/'retainer_failure.json').read_bytes())
    assert receipt['limit_bytes']==8192 and receipt['bytes']<=8192 and receipt['truncated']
    assert document['counts']['retainer_patient_points']==300
    assert document['points_omitted']==300-len(document['points'])
    total_aux=sum(path.stat().st_size for path in capture.output.iterdir() if path.is_file())
    assert total_aux+capture.AUXILIARY_RESERVE_BYTES<=capture.auxiliary_limit


def test_actual_step_latches_before_unchanged_retainer_failure_gate(monkeypatch):
    import sys
    m=module();fixture,world=native_failure();env=fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    env.__dict__.update(fixture.__dict__);env.rows=[];env.slider=7;env.passive_preload_N=1.5
    env.robot=SimpleNamespace(apply_action=lambda _:None);env.video_recorder=None
    from scipy.spatial.transform import Rotation
    q=Rotation.from_matrix(world[:3,:3]).as_quat()
    env.bodies=[('/World/Stethoscope/Carriage',SimpleNamespace(get_world_pose=lambda:(world[:3,3],q[[3,0,1,2]])))]
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',SimpleNamespace(ArticulationAction=SimpleNamespace))
    def step(self,*args):
        self.rows.append(dict(time_s=1/240,phase='CONTACT',target='R3'))
        return dict(axis_world=[0.,0.,-1.])
    monkeypatch.setattr(fresh.IsaacContactEnv,'step',step)
    state=env.step(np.zeros(7),'CONTACT','R3')
    actual=world.copy();actual[:3,:3]=Rotation.from_quat(q).as_matrix()
    np.testing.assert_array_equal(json.loads(env._retainer_failure_payload)['world_from_carriage'],actual)
    controller=fresh.FreshExpert.__new__(fresh.FreshExpert);controller.env=env
    monkeypatch.setattr(fresh.SkillExpert,'_violations',lambda *args:([],3.25))
    assert controller._violations(state)==(['device retainer side/back or unreconciled contact'],3.25)


def test_snapshot_serialization_failure_is_visible_without_replacing_original_gate(tmp_path):
    m=module();env,world=native_failure();env.contact_points[0]['separation_m']=float('nan')
    m.latch_retainer_failure(env,world)
    doc=json.loads(env._retainer_failure_payload)
    assert doc['missing_fields']==['snapshot_failed'] and doc['snapshot_error']
    assert env.retainer_front_audit['passed'] is False
    receipt=m.save_retainer_failure(env,tmp_path)
    assert receipt['saved'] and receipt['missing_fields']==['snapshot_failed']


def test_disk_failure_is_reported_and_world_cleanup_still_runs(tmp_path,monkeypatch):
    from pathlib import Path
    m=module();env,world=native_failure();m.latch_retainer_failure(env,world)
    events=[];env.save=lambda _:None;env.world=SimpleNamespace(stop=lambda:events.append('stop'))
    original=Path.open
    def fail(path,*args,**kwargs):
        if path.name=='retainer_failure.json':raise OSError('injected evidence disk failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail)
    _,errors=fresh.finalize_probe(env,None,None,tmp_path)
    assert events==['stop'] and any(row['stage']=='retainer_failure_save' for row in errors)
    assert env.retainer_failure_evidence['missing'] and env.retainer_failure_evidence['error']


@pytest.mark.parametrize('outer_failure',[False,True])
def test_runtime_result_binds_failure_artifact_even_after_unexpected_error(tmp_path,monkeypatch,outer_failure):
    from test_ausc_policy_m2_deployment_integration import setup_runtime
    from sim.ausc_policy import m2_act_controller as adapter
    m=module();config,events,_=setup_runtime(tmp_path,monkeypatch,'run')
    original=adapter.make_controller
    def factory(env,*args,**kwargs):
        controller=original(env,*args,**kwargs)
        native,world=native_failure();m.latch_retainer_failure(native,world)
        env._retainer_failure_payload=native._retainer_failure_payload
        env.retainer_failure_evidence=native.retainer_failure_evidence
        return controller
    monkeypatch.setattr(adapter,'make_controller',factory)
    if outer_failure:monkeypatch.setattr(fresh,'audit_physics',lambda *args:{})
    result=fresh.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert not result['success'] and events[-1]=='stop'
    receipt=result['retainer_failure_evidence']
    assert receipt['saved'] and receipt['sha256']==hashlib.sha256((tmp_path/'retainer_failure.json').read_bytes()).hexdigest()
    saved=json.loads((tmp_path/'fresh_expert_result.json').read_text())
    assert saved['retainer_failure_evidence']==receipt
    assert any(path.endswith('/retainer_failure.py') for path in result['runtime_sources_sha256'])
