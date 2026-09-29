"""Release accounting and safe review serving; no server or media jobs."""
import importlib
import json
import subprocess

import pytest


def module(name):
    return importlib.import_module('sim.auscultation_skill_v1.' + name)


def test_missing_and_assisted_trials_remain_in_full_denominator():
    report = module('report')
    good = dict(condition_id='a', requested_points=list(report.TARGETS),
                complete_requested_sequence=True, learned_task_success=True,
                assisted_system_success=True, evidence_verified=True, mode='policy')
    assisted = dict(good, condition_id='b', learned_task_success=False)
    result = report.summarize_heldout([good, assisted], ['a', 'b', 'c', 'd'])
    assert result['requested_trials'] == 4
    assert result['learned_successes'] == 1
    assert result['assisted_successes'] == 2
    assert result['missing_conditions'] == ['c', 'd']
    assert result['learned_success_rate'] == .25
    with pytest.raises(ValueError, match='duplicate'):
        report.summarize_heldout([good, good], ['a'])


def test_unverified_or_partial_claim_never_counts_as_learned_success():
    report = module('report')
    for extra in ({'evidence_verified': False}, {'requested_points': ['L4']}, {'mode': 'expert'}):
        row = dict(condition_id='a', requested_points=list(report.TARGETS),
                   complete_requested_sequence=True, learned_task_success=True,
                   assisted_system_success=True, evidence_verified=True, mode='policy')
        row.update(extra)
        assert report.summarize_heldout([row], ['a'])['learned_successes'] == 0


def test_empty_review_is_explicitly_incomplete_with_reserved_denominator(tmp_path):
    report = module('report')
    summary = report.build_report(tmp_path / 'review')
    assert not summary['evidence_complete']
    assert not summary['simulation_v1_success']
    assert summary['heldout']['requested_trials'] == 4
    assert summary['heldout']['learned_successes'] == 0
    assert summary['external_acceptance'] == 'pending_not_evaluated'
    assert summary['scope']['image_encoder_trained'] is False
    assert (tmp_path / 'review/summary.json').is_file()
    assert (tmp_path / 'review/serve_manifest.json').is_file()
    with pytest.raises(FileExistsError):
        report.build_report(tmp_path / 'review')


def test_video_requires_original_time_and_whole_ten_point_coverage():
    report = module('report')
    with pytest.raises(ValueError, match='original|1x'):
        report.video_metadata({'playback_speed': 2.}, {})


@pytest.mark.parametrize('error', [RuntimeError('invalid runtime evidence'),
                                 subprocess.TimeoutExpired(['decoder'], 180)])
def test_runtime_stage_failure_produces_incomplete_report(tmp_path, monkeypatch, error):
    report, quality = module('report'), module('quality')
    qualify = tmp_path/'qualify'; qualify.mkdir()
    (qualify/'repetitions.json').write_text('{}')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(quality, 'verify_repetitions', fail)
    result = report.build_report(tmp_path/'review', qualify=qualify)
    assert result['components']['qualification']['status'] == 'incomplete'
    assert result['components']['qualification']['reason'] == str(error)
    assert not result['simulation_v1_success']
    assert result['heldout']['requested_trials'] == 4


@pytest.mark.parametrize('error', [RuntimeError('invalid checkpoint'),
                                 subprocess.TimeoutExpired(['decoder'], 180)])
def test_runtime_trial_failure_preserves_denominator(tmp_path, monkeypatch, error):
    report = module('report')
    conditions = json.loads((report.ROOT/'conditions.json').read_text())['conditions']
    name = next(row['condition_id'] for row in conditions if row['split'] == 'test')
    evaluate = tmp_path/'evaluate'; (evaluate/'evaluations').mkdir(parents=True)
    (evaluate/'evaluations'/(name+'.json')).write_text('{}')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(report.Evidence, 'evaluation', fail)
    result = report.build_report(tmp_path/'review', evaluate=evaluate)
    assert result['heldout']['requested_trials'] == 4
    assert result['heldout']['learned_successes'] == 0
    assert not result['simulation_v1_success']
    trial = next(row for row in result['heldout']['trials'] if row['condition_id'] == name)
    assert not trial['evidence_verified']
    assert trial['state'] == 'invalid'
    assert trial['reason'] == str(error)


def video_fixture():
    report = module('report')
    points = [dict(target=name, start_physics_row=i*16, end_physics_row=(i+1)*16)
              for i, name in enumerate(report.TARGETS)]
    frames = [dict(frame_index=i, video_time_s=i/15, measurement_row=(i+1)*16-1,
                   subphase='home_settle' if i == 9 else 'contact_cycle') for i in range(10)]
    outcome = dict(requested_points=list(report.TARGETS), physics_steps=160, points=points)
    manifest = dict(playback_speed=1., point_intervals=points, source_physics_rows=160,
                    fps=15, resolution=[1920, 720], source_physics_dt_s=1/240,
                    frames=frames, frame_count=10, source_duration_s=160/240, duration_s=10/15)
    return manifest, outcome


def test_video_navigation_uses_source_time_and_rejects_omitted_intervals():
    report = module('report')
    manifest, outcome = video_fixture()
    data = report.video_metadata(manifest, outcome)
    assert data['points'][1]['time_s'] == 16/240
    manifest['frames'][4]['video_time_s'] = 0
    with pytest.raises(ValueError, match='frame times'):
        report.video_metadata(manifest, outcome)


def test_verified_bundle_rejects_tampering_and_arbitrary_manifest_files(tmp_path):
    report, serve = module('report'), module('serve')
    package = tmp_path / 'skill'; package.mkdir()
    output = package / 'review'
    report.build_report(output)
    files, _ = serve.verified_bundle(output, package_root=package)
    assert set(files) == {'review.html', 'summary.json'}
    (output / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        serve.verified_bundle(output, package_root=package)


def test_review_uses_named_custom_controls_without_native_video_controls():
    from html.parser import HTMLParser
    class Controls(HTMLParser):
        def __init__(self):
            super().__init__(); self.buttons = []; self.videos = []
        def handle_starttag(self, tag, attrs):
            if tag == 'button': self.buttons.append(dict(attrs))
            if tag == 'video': self.videos.append(dict(attrs))
    parser = Controls(); parser.feed((module('report').ROOT / 'review.html').read_text())
    assert len(parser.videos) == 2 and all('controls' not in attrs for attrs in parser.videos)
    assert len(parser.buttons) == 6 and all(attrs.get('aria-label') for attrs in parser.buttons)


def test_failed_attribution_is_reported_as_failure_not_hidden_or_promoted(tmp_path, monkeypatch):
    report = module('report')
    quality_path = tmp_path / 'quality.json'; quality_path.write_text('{}')
    artifacts = {}
    for name in ('expert_trace.jsonl', 'control_commands.json', 'outcome.json',
                 'runner_summary.json', 'worker_completion.json'):
        path = tmp_path / name; path.write_text('{}'); artifacts[name] = report.sha(path)
    quality = dict(episode=str(tmp_path), condition_id='test_a', requested_points=list(report.TARGETS),
                   artifact_sha256=artifacts, episode_simulation_gates_passed=False,
                   hard_simulation_gates_passed=False)
    monkeypatch.setattr(report.Evidence, 'quality', lambda self, path: (quality, {}))
    monkeypatch.setattr(report.Evidence, 'recheck_evaluation', lambda self, result, path: result)
    result = dict(schema='auscultation_scalar_evaluation_v1', episode=str(tmp_path), condition_id='test_a',
        requested_points=list(report.TARGETS), artifact_sha256=artifacts, mode='policy',
        quality_report={'path': str(quality_path), 'sha256': report.sha(quality_path)},
        checkpoint_sha256='checkpoint', model_manifest_sha256='model', student_identity_valid=True,
        learned_task_success=False, assisted_system_success=False,
        action_attribution={'action_attribution_passed': False, 'decisions_replayed': 2, 'hard_intervention_records': 1})
    path = tmp_path / 'evaluation.json'; path.write_text(json.dumps(result))
    model = dict(checkpoint_sha256='checkpoint', manifest_sha256='model')
    verified = report.Evidence().evaluation(path, model)
    assert verified['evidence_verified'] and not verified['learned_task_success']
    result['learned_task_success'] = True; path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='unsupported learned'):
        report.Evidence().evaluation(path, model)
    result['learned_task_success'] = False
    path.write_text(json.dumps(result))
    monkeypatch.setattr(report.Evidence, 'recheck_evaluation',
                        lambda self, result, path: dict(result, student_identity_valid=False))
    with pytest.raises(ValueError, match='fresh model/source/lifecycle'):
        report.Evidence().evaluation(path, model)
    del result['artifact_sha256']['worker_completion.json']; path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='lifecycle binding'):
        report.Evidence().evaluation(path, model)


@pytest.mark.parametrize('kind', ['render', 'rgbd'])
def test_media_receipt_must_bind_current_verifier_decoder_and_process(tmp_path, monkeypatch, kind):
    report = module('report')
    verifier = module('verify_media')
    path = tmp_path / 'media.json'
    manifest = tmp_path / 'rgbd_manifest.json'; manifest.write_text('{}')
    guard = tmp_path / 'guard.json'; guard.write_text('{"returncode": 0}')
    launcher = tmp_path / 'launcher.sh'; launcher.write_text('# test fixture only\n')
    result = dict(manifest=str(manifest), manifest_sha256=report.sha(manifest),
                  process_completion=dict(guard_path=str(guard), guard_sha256=report.sha(guard),
                                          launcher_path=str(launcher), launcher_sha256=report.sha(launcher)))
    calls = []
    def fresh(directory, *, guard_path):
        calls.append((directory, guard_path))
        return result
    monkeypatch.setattr(verifier, 'verify_'+kind, fresh)
    value = dict(schema='auscultation_skill_media_verification_v1', status='passed',
                 reports={kind: result},
                 verifier_sha256=report.sha(report.ROOT / 'verify_media.py'),
                 decoder_sha256=report.sha(report.PROJECT / 'sim/upper_pair_posture_20260924/verify_video.py'))
    path.write_text(json.dumps(value))
    assert report.Evidence().media_receipt(path) == value['reports']
    assert calls == [(tmp_path, guard)]
    monkeypatch.setattr(verifier, 'verify_'+kind, lambda *args, **kwargs: dict(result, status='failed'))
    with pytest.raises(ValueError, match='fresh source/process/content'):
        report.Evidence().media_receipt(path)
    monkeypatch.setattr(verifier, 'verify_'+kind, fresh)
    guard.write_text('{"returncode": -11}')
    with pytest.raises(ValueError, match='hash mismatch'):
        report.Evidence().media_receipt(path)
    guard.write_text('{"returncode": 0}')
    value['verifier_sha256'] = '0'*64; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash mismatch'):
        report.Evidence().media_receipt(path)


def test_report_rechecks_renderer_implementation_not_only_prior_receipt():
    report = module('report')
    inventory = {'sim/auscultation_skill_v1/' + name: report.sha(report.ROOT / name)
                 for name in ('render.py', 'quality.py', 'provenance.py', 'contract.py')}
    report.bind_renderer_inventory(report.Evidence(), {'renderer_implementation_sha256': inventory})
    inventory['sim/auscultation_skill_v1/render.py'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        report.bind_renderer_inventory(report.Evidence(), {'renderer_implementation_sha256': inventory})
    with pytest.raises(ValueError, match='binding incomplete'):
        report.bind_renderer_inventory(report.Evidence(), {})


def test_server_rejects_dotfiles_traversal_symlinks_and_unlisted_files(tmp_path):
    serve = module('serve')
    root = tmp_path / 'package'; root.mkdir()
    (root / 'review.html').write_text('review')
    (root / 'secret.json').write_text('not served')
    outside = tmp_path / 'outside.mp4'; outside.write_bytes(b'outside')
    (root / 'escape.mp4').symlink_to(outside)
    permitted = {'review.html', 'escape.mp4'}
    assert serve.resolve_request(root, '/review.html', permitted) == root / 'review.html'
    for request in ('/secret.json', '/.env', '/%2e%2e/outside.mp4', '/escape.mp4', '/%2freview.html', '/nested/../review.html'):
        with pytest.raises(ValueError):
            serve.resolve_request(root, request, permitted)


@pytest.mark.parametrize('header,expected', [('bytes=0-4', (0, 4)), ('bytes=5-', (5, 9)),
                                          ('bytes=-3', (7, 9)), ('bytes=7-99', (7, 9))])
def test_video_ranges_support_original_time_seeking(header, expected):
    assert module('serve').range_bounds(header, 10) == expected


@pytest.mark.parametrize('header', ['bytes=20-', 'bytes=9-3', 'bytes=-0', 'bytes=0-2,4-6', 'garbage'])
def test_invalid_ranges_are_rejected(header):
    with pytest.raises(ValueError):
        module('serve').range_bounds(header, 10)
