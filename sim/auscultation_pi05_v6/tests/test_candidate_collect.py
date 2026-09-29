import copy
import pytest
from sim.auscultation_pi05_v6 import candidate_collect as candidate


def manifests():
    source={f'source/{i}':dict(sha256=f'{i:064x}') for i in range(143)}
    original=dict(points=['R5'],files={**source,**{f'R5/{key}':dict(sha256='a'*64) for key in ('plan','audit','targets','runtime','collision_metadata','collision_arrays','kinematics_urdf')}})
    updated=copy.deepcopy(original);updated['files']['R5/plan']['sha256']=candidate.PLAN_SHA;updated['files']['R5/audit']['sha256']='b'*64
    return original,updated


def test_only_plan_and_audit_may_change():
    original,updated=manifests();candidate.verify_manifest(original,updated)
    for key in ('source/0','R5/targets','R5/collision_arrays'):
        bad=copy.deepcopy(updated);bad['files'][key]['sha256']='c'*64
        with pytest.raises(ValueError):candidate.verify_manifest(original,bad)
    bad=copy.deepcopy(updated);bad['files']['source/new']={'sha256':'d'*64}
    with pytest.raises(ValueError):candidate.verify_manifest(original,bad)


def test_command_uses_exact_new_output_and_last_candidate_root():
    args=candidate.collection_command()
    assert args[-2:]==['--source-project',candidate.CANDIDATE_SOURCE]
    assert args[args.index('--output')+1].endswith('/R5_expert_observed_redundancy30_v1')
    assert args[args.index('--speed-scale')+1]=='.7'


def test_default_only_prepares(monkeypatch):
    monkeypatch.setattr(candidate.sys,'argv',['candidate_collect'])
    monkeypatch.setattr(candidate,'prepare',lambda:dict(prepared=True))
    monkeypatch.setattr(candidate,'execute',lambda record:pytest.fail('must not launch by default'))
    assert candidate.main()==0


def execution_fixture(monkeypatch):
    events=[]
    monkeypatch.setattr(candidate,'preflight',lambda:dict(manifest_fingerprint='same'))
    monkeypatch.setattr(candidate,'run_collection',lambda:0)
    monkeypatch.setattr(candidate,'verify_completed_inputs',lambda prepared:None)
    monkeypatch.setattr(candidate.batch,'emit',lambda *args,**kwargs:None)
    monkeypatch.setattr(candidate.batch,'archive',lambda episode:events.append('archive'))
    monkeypatch.setattr(candidate.batch,'mirror_qualified',lambda *args:events.append('qualified_mirror'))
    monkeypatch.setattr(candidate.batch,'local_module',lambda module,*args:events.append(module))
    return events


def test_failed_candidate_archived_without_qualification_or_cleanup(monkeypatch):
    events=execution_fixture(monkeypatch)
    monkeypatch.setattr(candidate.batch,'completed_status',lambda *args:dict(passed=False,teardown_complete=True))
    monkeypatch.setattr(candidate.batch,'run_remote',lambda *args:pytest.fail('failed candidate must not qualify'))
    with pytest.raises(RuntimeError,match='Collection gate failed'):candidate.execute({'manifest_fingerprint':'same'})
    assert events==['archive']


def test_snapshot_mismatch_archives_then_stops(monkeypatch):
    events=execution_fixture(monkeypatch)
    monkeypatch.setattr(candidate.batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    def fail(_):raise ValueError('snapshot mismatch')
    monkeypatch.setattr(candidate,'verify_completed_inputs',fail)
    with pytest.raises(ValueError,match='snapshot mismatch'):candidate.execute({'manifest_fingerprint':'same'})
    assert events==['archive']


def test_copy_failure_prevents_cleanup(monkeypatch):
    events=execution_fixture(monkeypatch)
    monkeypatch.setattr(candidate.batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    monkeypatch.setattr(candidate.batch,'run_remote',lambda *args:0)
    def fail(module,*args):
        events.append(module)
        if module=='mirror_qualification':raise RuntimeError('copy mismatch')
    monkeypatch.setattr(candidate.batch,'local_module',fail)
    with pytest.raises(RuntimeError,match='copy mismatch'):candidate.execute({'manifest_fingerprint':'same'})
    assert events==['qualified_mirror','archive','mirror_qualification']
