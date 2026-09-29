import pytest
from sim.auscultation_pi05_v6 import batch_collect as batch


def fixture(monkeypatch):
    events=[]
    monkeypatch.setattr(batch,'preflight',lambda point:{'checked':True})
    monkeypatch.setattr(batch,'emit',lambda *args,**kwargs:None)
    monkeypatch.setattr(batch.subprocess,'call',lambda *args,**kwargs:0)
    monkeypatch.setattr(batch,'archive',lambda episode:events.append('archive'))
    monkeypatch.setattr(batch,'mirror_qualified',lambda *args:events.append('qualified_mirror'))
    monkeypatch.setattr(batch,'local_module',lambda name,*args:events.append(name))
    return events


def test_default_is_prepare_only(monkeypatch):
    monkeypatch.setattr(batch.sys,'argv',['batch_collect'])
    monkeypatch.setattr(batch,'run_episode',lambda point:pytest.fail('must not launch by default'))
    assert batch.main()==0
    assert batch.POINTS==('R1','R2','R3','L2','L3','L4','R5','L5')


def test_collection_failure_archives_without_qualification_or_cleanup(monkeypatch):
    events=fixture(monkeypatch)
    monkeypatch.setattr(batch,'completed_status',lambda *args:dict(passed=False,teardown_complete=True))
    monkeypatch.setattr(batch,'run_remote',lambda *args:pytest.fail('failed collection must not qualify'))
    with pytest.raises(RuntimeError,match='Collection gate failed'):batch.run_episode('R1')
    assert events==['archive']


def test_audit_failure_stops_before_qualification_or_cleanup(monkeypatch):
    events=fixture(monkeypatch)
    monkeypatch.setattr(batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    calls=[]
    def fail_audit(args):calls.append(args);return 1
    monkeypatch.setattr(batch,'run_remote',fail_audit)
    with pytest.raises(RuntimeError,match='supervisor replay failed'):batch.run_episode('R1')
    assert len(calls)==1 and events==['archive']


def test_copy_verification_failure_prevents_all_cleanup(monkeypatch):
    events=fixture(monkeypatch)
    monkeypatch.setattr(batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    monkeypatch.setattr(batch,'run_remote',lambda *args:0)
    def stop(name,*args):
        events.append(name)
        if name=='mirror_qualification':raise RuntimeError('copy mismatch')
    monkeypatch.setattr(batch,'local_module',stop)
    with pytest.raises(RuntimeError,match='copy mismatch'):batch.run_episode('R1')
    assert events==['qualified_mirror','archive','mirror_qualification']


def test_qualification_failure_preserves_raw_and_stops(monkeypatch):
    events=fixture(monkeypatch)
    monkeypatch.setattr(batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    outcomes=iter([0,1]);monkeypatch.setattr(batch,'run_remote',lambda *args:next(outcomes))
    with pytest.raises(RuntimeError,match='Qualification failed'):batch.run_episode('R1')
    assert events==['qualified_mirror','archive']


def test_pre_export_qualification_failure_still_archives_completed_trial(monkeypatch):
    events=fixture(monkeypatch)
    monkeypatch.setattr(batch,'completed_status',lambda *args:dict(passed=True,teardown_complete=True))
    outcomes=iter([0,1]);monkeypatch.setattr(batch,'run_remote',lambda *args:next(outcomes))
    def missing_report(*args):raise FileNotFoundError('qualifier stopped before output')
    monkeypatch.setattr(batch,'mirror_qualified',missing_report)
    with pytest.raises(RuntimeError,match='Qualification failed'):batch.run_episode('R1')
    assert events==['archive']
