import hashlib

import pytest

from sim.ausc_policy import storage_policy as storage


def test_mac_export_allows_only_selected_small_evidence(tmp_path):
    (tmp_path / 'summary.json').write_text('{"success": false}')
    digest = hashlib.sha256((tmp_path / 'summary.json').read_bytes()).hexdigest()
    rows = storage.checked_mac_exports(tmp_path, [dict(path='summary.json', kind='summary', sha256=digest)])
    assert rows[0]['bytes'] == 18
    assert rows[0]['sha256'] == digest


@pytest.mark.parametrize('name,kind', [('raw/summary.json', 'summary'), ('rgbd/one.png', 'chart'),
    ('trace.jsonl', 'summary'), ('body.npz', 'summary'), ('candidate.usdc', 'summary'),
    ('frames/one.png', 'chart'), ('../escape.json', 'summary')])
def test_raw_and_escape_paths_cannot_be_mac_exports(tmp_path, name, kind):
    with pytest.raises(ValueError):
        storage.checked_mac_exports(tmp_path, [dict(path=name, kind=kind, sha256='0'*64)])


def test_symlink_escape_and_changed_file_rejected(tmp_path):
    outside=tmp_path.parent/'outside.json'; outside.write_text('{}')
    (tmp_path/'summary.json').symlink_to(outside)
    with pytest.raises(ValueError):
        storage.checked_mac_exports(tmp_path, [dict(path='summary.json',kind='summary',sha256=hashlib.sha256(b'{}').hexdigest())])
    (tmp_path/'summary.json').unlink(); (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(ValueError):
        storage.checked_mac_exports(tmp_path, [dict(path='summary.json',kind='summary',sha256='0'*64)])


def test_in_root_symlink_cannot_disguise_raw_stream(tmp_path):
    (tmp_path/'raw').mkdir(); (tmp_path/'raw/trace.jsonl').write_text('{}')
    (tmp_path/'summary.json').symlink_to(tmp_path/'raw/trace.jsonl')
    with pytest.raises(ValueError,match='symlink'):
        storage.checked_mac_exports(tmp_path,[dict(path='summary.json',kind='summary',sha256=hashlib.sha256(b'{}').hexdigest())])


def test_tier_b_is_preassigned_two_percent_and_order_independent():
    episodes=['episode-'+str(i) for i in range(100)]
    one=storage.assign_capture_tiers(episodes, seed='frozen-batch-v1')
    assert sum(x=='tier_b' for x in one.values())==2
    assert storage.assign_capture_tiers(list(reversed(episodes)),seed='frozen-batch-v1')==one
    assert all(x in ('training','tier_b') for x in one.values())


def test_tier_selection_rejects_duplicate_or_invalid_rate():
    with pytest.raises(ValueError): storage.assign_capture_tiers(['a','a'],seed='x')
    with pytest.raises(ValueError): storage.assign_capture_tiers(['a'],seed='x',percent=100)


def test_training_byte_budget_includes_every_file(tmp_path):
    (tmp_path/'video.mp4').write_bytes(b'1234')
    (tmp_path/'numbers.npz').write_bytes(b'5678')
    assert storage.episode_size(tmp_path,limit_bytes=8)['within_budget']
    report=storage.episode_size(tmp_path,limit_bytes=7)
    assert report['total_bytes']==8 and not report['within_budget']
    assert report['training_eligible'] is False


def test_export_root_cannot_alias_or_be_a_raw_directory(tmp_path):
    raw=tmp_path/'frames'; raw.mkdir(); (raw/'chart.png').write_bytes(b'fake')
    alias=tmp_path/'alias'; alias.symlink_to(raw,target_is_directory=True)
    entry=dict(path='chart.png',kind='chart',sha256=hashlib.sha256(b'fake').hexdigest())
    for root in [raw,alias]:
        with pytest.raises(ValueError): storage.checked_mac_exports(root,[entry])


def test_byte_budget_requires_episode_directory(tmp_path):
    p=tmp_path/'file.npz';p.write_bytes(b'123')
    with pytest.raises(ValueError):storage.episode_size(p)
