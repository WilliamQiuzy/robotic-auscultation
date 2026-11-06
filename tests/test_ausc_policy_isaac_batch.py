import hashlib
import json

import pytest

from sim.ausc_policy import isaac_batch as B


def config(tmp_path):
    child = tmp_path / 'child.json'
    child.write_text(json.dumps(dict(experiment='geometry_qualification', base=str(tmp_path),
        asset_root=str(tmp_path / 'assets'), isaac_python='/bin/python',
        ld_library_path='/lib', time_limit_seconds=600)))
    return dict(experiment='isaac_batch', base=str(tmp_path), parallelism=2,
        time_limit_seconds=1200, jobs=[dict(id='l1', config=str(child),
        sha256=hashlib.sha256(child.read_bytes()).hexdigest())])


def test_batch_pins_each_config_and_preserves_worker_safety(tmp_path):
    c = config(tmp_path)
    jobs = B.validate_jobs(c)
    assert jobs[0]['id'] == 'l1'
    assert jobs[0]['settings']['experiment'] == 'geometry_qualification'
    assert jobs[0]['settings']['base'] == str(tmp_path)
    (tmp_path / 'child.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        B.validate_jobs(c)


def test_batch_rejects_duplicates_escaped_ids_and_unbounded_parallelism(tmp_path):
    c = config(tmp_path)
    for invalid in ('../escape', '/absolute', 'a/b', ''):
        with pytest.raises(ValueError):
            B.validate_jobs(dict(c, jobs=[dict(c['jobs'][0], id=invalid)]))
    with pytest.raises(ValueError):
        B.validate_jobs(dict(c, jobs=c['jobs'] * 2))
    for n in (0, 5, True):
        with pytest.raises(ValueError):
            B.validate_jobs(dict(c, parallelism=n))


def test_child_inherits_owner_session_and_is_never_new_session(tmp_path):
    job = B.validate_jobs(config(tmp_path))[0]
    command, kwargs = B.child_launch(job, tmp_path / 'out')
    assert command[:4] == ['/bin/python', '-u', '-B', '-m']
    assert command[4] == 'sim.ausc_policy.geometry_qualification'
    assert kwargs.get('start_new_session') is False
    assert kwargs['env']['AUSC_OWNED_RUN'] == '1'
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'


def test_result_does_not_count_failed_physics_as_success():
    assert not B.physical_pass(dict(completed=True, physical_validation=False))
    assert not B.physical_pass(dict(success=True, physical_validation=True, errors=['bad']))
    assert B.physical_pass(dict(success=True, physical_validation=True, errors=[]))


@pytest.mark.parametrize('failure', ['identity', 'launch_receipt'])
def test_startup_failure_reaps_child_before_releasing_gpu_lock(tmp_path, monkeypatch, failure):
    import fcntl
    import sys

    from sim.ausc_policy import process_guard as G

    c = config(tmp_path)
    config_path = tmp_path / 'batch.json'
    config_path.write_text(json.dumps(c))
    locks = tmp_path / 'locks'
    locks.mkdir()
    lock_path = locks / 'isaac_gpu.lock'
    output = tmp_path / 'out'
    events = []

    def assert_lock_held():
        with lock_path.open('a+') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    class Child:
        pid = 123456
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            assert_lock_held()
            events.append('terminate')

        def wait(self, timeout):
            assert_lock_held()
            events.append('reap')
            self.returncode = -15
            return self.returncode

    child = Child()
    monkeypatch.setenv('AUSC_OWNED_RUN', '1')
    monkeypatch.setattr(sys, 'argv', ['isaac_batch', '--config', str(config_path), '--output', str(output)])
    monkeypatch.setattr(G, 'gpu_pids', lambda: set())
    monkeypatch.setattr(B.time, 'sleep', lambda _: None)
    monkeypatch.setattr(B.subprocess, 'Popen', lambda *args, **kwargs: child)

    def fail(*args, **kwargs):
        raise OSError('injected startup failure')

    if failure == 'identity':
        monkeypatch.setattr(G, 'process_identity', fail)
    else:
        monkeypatch.setattr(G, 'process_identity', lambda pid: dict(pid=pid, pgid=77, sid=77, start_ticks='1'))
        monkeypatch.setattr(G, 'write_new', fail)
    with pytest.raises(OSError, match='injected startup failure'):
        B.main()
    assert events == ['terminate', 'reap']
    assert not (output / 'COMPLETE.json').exists()
    with lock_path.open('a+') as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender.fileno(), fcntl.LOCK_UN)


def test_worker_infrastructure_failure_has_results_without_complete(tmp_path, monkeypatch):
    import sys

    from sim.ausc_policy import process_guard as G

    c = config(tmp_path)
    config_path = tmp_path / 'batch.json'
    config_path.write_text(json.dumps(c))
    (tmp_path / 'locks').mkdir()
    output = tmp_path / 'out'

    class Child:
        pid = 123456
        returncode = 1

        def poll(self):
            return self.returncode

    monkeypatch.setenv('AUSC_OWNED_RUN', '1')
    monkeypatch.setattr(sys, 'argv', ['isaac_batch', '--config', str(config_path), '--output', str(output)])
    monkeypatch.setattr(G, 'gpu_pids', lambda: set())
    monkeypatch.setattr(G, 'process_identity', lambda pid: dict(pid=pid, pgid=77, sid=77, start_ticks='1'))
    monkeypatch.setattr(B.time, 'sleep', lambda _: None)
    monkeypatch.setattr(B.subprocess, 'Popen', lambda *args, **kwargs: Child())
    with pytest.raises(RuntimeError, match='Worker infrastructure failed'):
        B.main()
    report = json.loads((output / 'BATCH_RESULTS.json').read_text())
    assert report['completed'] is False
    assert report['attempted'] == 1
    assert report['jobs'][0]['worker_complete'] is False
    assert report['physical_successes'] == 0
    assert not (output / 'COMPLETE.json').exists()
