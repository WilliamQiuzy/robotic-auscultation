"""Lifecycle regressions; process/GPU snapshots are explicit test inputs."""
import pytest

from sim.ausc_policy import process_guard as G


def member(pid=22, ticks='200', state='S', pgid=11, sid=11):
    return dict(pid=pid, start_ticks=ticks, state=state, pgid=pgid, sid=sid)


def identity():
    return dict(pid=11, start_ticks='100', pgid=11, sid=11,
                members={'11': '100', '22': '200'})


def missing(pid):
    raise FileNotFoundError(str(pid))


def sample(ownership):
    return dict(sampling_error=None, gpu_pids=[22], owned_group_pids=[11],
                gpu_ownership=ownership, ram_available_bytes=3*G.GIB,
                disk_free_bytes=3*G.GIB, gpu_free_bytes=2*G.GIB)


def test_container_memory_respects_parent_and_leaf_limits(tmp_path):
    for relative,limit,used in [('',100,25),('pod',50,40),('pod/worker',80,60)]:
        folder=tmp_path/relative;folder.mkdir(exist_ok=True,parents=True)
        (folder/'memory.max').write_text(str(limit*G.GIB))
        (folder/'memory.current').write_text(str(used*G.GIB))
    record=G.cgroup_memory_available(1000*G.GIB,root=tmp_path,cgroup_path='/pod/worker')
    assert record['available_bytes']==10*G.GIB
    assert len(record['constraints'])==3
    assert G.cgroup_memory_available(2*G.GIB,root=tmp_path,cgroup_path='/pod/worker')['available_bytes']==2*G.GIB


def test_container_root_namespace_and_unlimited_memory(tmp_path):
    (tmp_path/'memory.max').write_text('max');(tmp_path/'memory.current').write_text('500')
    assert G.cgroup_memory_available(1234,root=tmp_path,cgroup_path='/not-visible/child')['available_bytes']==1234
    (tmp_path/'memory.max').write_text('1000')
    assert G.cgroup_memory_available(1234,root=tmp_path,cgroup_path='/not-visible/child')['available_bytes']==500
    (tmp_path/'memory.current').write_text('1001')
    assert G.cgroup_memory_available(1234,root=tmp_path,cgroup_path='/')['available_bytes']==0


def test_container_memory_incomplete_or_corrupt_measurement_fails(tmp_path):
    (tmp_path/'memory.max').write_text('1000')
    with pytest.raises(FileNotFoundError):G.cgroup_memory_available(5000,root=tmp_path,cgroup_path='/')
    (tmp_path/'memory.current').write_text('-1')
    with pytest.raises(ValueError):G.cgroup_memory_available(5000,root=tmp_path,cgroup_path='/')


def test_missing_leaf_still_applies_visible_ancestor_and_subtree_mapping(tmp_path):
    folder=tmp_path/'pod';folder.mkdir()
    (folder/'memory.max').write_text('1000');(folder/'memory.current').write_text('950')
    assert G.cgroup_memory_available(100000,root=tmp_path,cgroup_path='/pod/missing')['available_bytes']==50
    assert G.cgroup_memory_available(100000,root=tmp_path,cgroup_path='/host/pod',mount_root='/host')['available_bytes']==50
    with pytest.raises(ValueError,match='outside visible'):
        G.cgroup_memory_available(100000,root=tmp_path,cgroup_path='/other/pod',mount_root='/host')


def test_auto_membership_requires_matching_cgroup_mount(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=='/proc/self/cgroup':return '0::/pod\n'
        if str(path)=='/proc/self/mountinfo':return f'1 2 0:30 / {tmp_path}/unified rw - cgroup2 cgroup rw\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='could not be mapped'):
        G.cgroup_memory_available(100000,root=tmp_path)


def test_exited_known_gpu_pid_gets_bounded_grace_without_signal(monkeypatch):
    monkeypatch.setattr(G, 'process_identity', missing)
    owner=identity()
    result=G.classify_gpu_ownership(owner, {22}, 100.)
    assert result['exiting_owned_pids']==[22]
    assert G.guard_violation(sample(result), 1., 3600.) is None
    expired=G.classify_gpu_ownership(owner, {22}, 115.)
    assert expired['rejected_pids']==[22]
    assert 'outside' in G.guard_violation(sample(expired), 2., 3600.)


def test_unknown_disappeared_pid_is_not_an_exit_exception(monkeypatch):
    monkeypatch.setattr(G, 'process_identity', missing)
    assert G.classify_gpu_ownership(identity(), {77}, 100.)['rejected_pids']==[77]


@pytest.mark.parametrize('current', [member(ticks='201',pgid=80,sid=80),
    member(pgid=80,sid=80),member(ticks='201',state='Z'),member(pgid=80,sid=80,state='Z')])
def test_recycled_or_escaped_known_pid_is_rejected(monkeypatch,current):
    monkeypatch.setattr(G, 'process_identity', lambda pid: current)
    assert G.classify_gpu_ownership(identity(), {22}, 100.)['rejected_pids']==[22]


def test_known_same_identity_zombie_gets_same_bounded_grace(monkeypatch):
    monkeypatch.setattr(G, 'process_identity', lambda pid: member(state='Z'))
    owner=identity()
    assert G.classify_gpu_ownership(owner,{22},100.)['exiting_owned_pids']==[22]
    assert G.classify_gpu_ownership(owner,{22},115.)['rejected_pids']==[22]


def test_new_child_created_between_group_and_gpu_samples_is_owned(monkeypatch):
    monkeypatch.setattr(G, 'process_identity', lambda pid: member(pid=33,ticks='300'))
    owner=identity()
    result=G.classify_gpu_ownership(owner,{33},100.)
    assert result['verified_owned_pids']==[33]
    assert owner['members']['33']=='300'


def test_genuine_foreign_gpu_is_rejected_even_with_known_exit(monkeypatch):
    def lookup(pid):
        if pid==22:return missing(pid)
        return member(pid=77,ticks='900',pgid=77,sid=77)
    monkeypatch.setattr(G,'process_identity',lookup)
    result=G.classify_gpu_ownership(identity(),{22,77},100.)
    row=sample(result);row['gpu_pids']=[22,77]
    assert result['exiting_owned_pids']==[22] and result['rejected_pids']==[77]
    assert 'outside' in G.guard_violation(row,1.,3600.)


def test_permission_error_fails_sampling_not_treated_as_exit(monkeypatch):
    def denied(pid):raise PermissionError('denied')
    monkeypatch.setattr(G,'process_identity',denied)
    with pytest.raises(PermissionError):G.classify_gpu_ownership(identity(),{22},100.)


def test_gpu_disappearance_does_not_restart_an_existing_exit_budget(monkeypatch):
    monkeypatch.setattr(G,'process_identity',missing);owner=identity()
    G.classify_gpu_ownership(owner,{22},100.)
    G.classify_gpu_ownership(owner,set(),101.)
    assert G.classify_gpu_ownership(owner,{22},116.)['rejected_pids']==[22]


def test_existing_resource_and_wall_limits_still_fail(monkeypatch):
    monkeypatch.setattr(G,'process_identity',missing)
    row=sample(G.classify_gpu_ownership(identity(),{22},100.))
    assert G.guard_violation(row,3600.,3600.)=='Stage wall-clock limit reached'
    for key in ('ram_available_bytes','disk_free_bytes','gpu_free_bytes'):
        assert 'below emergency margin' in G.guard_violation({**row,key:0},1.,3600.)
    assert 'sampling failed' in G.guard_violation({**row,'sampling_error':'unreadable'},1.,3600.)


def test_legacy_record_without_new_identity_evidence_still_rejects():
    row=sample({});row.pop('gpu_ownership')
    assert 'outside' in G.guard_violation(row,1.,3600.)


def test_guard_does_not_accept_incomplete_classification():
    row=sample(dict(verified_owned_pids=[],exiting_owned_pids=[],rejected_pids=[]))
    assert 'outside' in G.guard_violation(row,1.,3600.)


def test_remember_group_does_not_erase_pid_reuse_evidence(monkeypatch):
    owner=identity()
    monkeypatch.setattr(G,'group_members',lambda ident:[member(ticks='201')])
    monkeypatch.setattr(G,'process_identity',lambda pid:member(ticks='201'))
    G.remember_group(owner)
    assert owner['members']['22']=='200'
    assert G.classify_gpu_ownership(owner,{22},100.)['rejected_pids']==[22]


def test_leader_identity_is_known_even_if_first_group_snapshot_is_empty(monkeypatch):
    owner=dict(pid=11,start_ticks='100',pgid=11,sid=11,state='S')
    monkeypatch.setattr(G,'group_members',lambda ident:[])
    monkeypatch.setattr(G,'process_identity',missing)
    G.remember_group(owner)
    assert G.classify_gpu_ownership(owner,{11},100.)['exiting_owned_pids']==[11]
