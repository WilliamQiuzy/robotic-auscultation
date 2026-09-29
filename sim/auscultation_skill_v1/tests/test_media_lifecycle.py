"""Final media manifests require returning Kit shutdown, without loading Isaac."""
import json
import sys
from types import SimpleNamespace

import pytest

from sim.auscultation_skill_v1 import render, rgbd


def scoped_receipt(products):
    return dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
        replicator_wait_completed=True, annotators_detached=2, render_products_destroyed=products,
        capture_scope_ended=True, gc_before_stage_close=True, pre_stage_close_updates=5,
        explicit_stage_close_checked=True, stage_absent_after_drain=True,
        post_stage_close_updates=5, gc_before_app_close=True, close_returned=True, errors=[])


def fake_isaac(monkeypatch, events, failure=None):
    def factory(configuration):
        events.append(('create', dict(configuration)))
        if failure == 'create':
            raise RuntimeError('construction failed')
        def close():
            events.append('close')
            if failure == 'close':
                raise RuntimeError('close failed')
        return SimpleNamespace(close=close)
    module = SimpleNamespace(SimulationApp=factory)
    monkeypatch.setitem(sys.modules, 'isaacsim', module)
    monkeypatch.setattr(sys, 'platform', 'linux')
    return module, factory


@pytest.mark.parametrize('failure', [None, 'create', 'close', 'no_app', 'two_apps', 'cleanup'])
def test_video_wrapper_publishes_its_schema_only_after_exactly_one_app_closes(tmp_path, monkeypatch, failure):
    events = []
    isaac, original = fake_isaac(monkeypatch, events, failure)
    episode, output = tmp_path/'episode', tmp_path/'render'
    episode.mkdir()
    (episode/'outcome.json').write_text(json.dumps(dict(requested_points=list(render.TARGETS),
        executed_controller_class='sim.auscultation_skill_v1.controller.ExpertController')))
    monkeypatch.setattr(render, 'PROJECT', tmp_path)
    monkeypatch.setattr(render, 'validate_episode_provenance', lambda path: {'status': 'passed'})
    from sim.auscultation_skill_v1 import provenance
    monkeypatch.setattr(provenance, 'implementation_sources', lambda: {})
    def inherited_main():
        apps = [] if failure == 'no_app' else [isaac.SimulationApp({'headless': True})]
        if failure == 'two_apps':
            apps.append(isaac.SimulationApp({'headless': True}))
        output.mkdir()
        (output/'review_manifest.json').write_text(json.dumps({'schema': 'continuous_recorded_route_v1'}))
        for app in apps:
            app.close()
        if failure == 'cleanup':
            raise RuntimeError('resource cleanup failed')
        return scoped_receipt(2)
    adapted = SimpleNamespace(main=inherited_main)
    monkeypatch.setattr(render.importlib.util, 'spec_from_file_location',
        lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(render.importlib.util, 'module_from_spec', lambda spec: adapted)
    argv_before = list(sys.argv)
    if failure:
        with pytest.raises(RuntimeError):
            render.main(['--episode', str(episode), '--output', str(output)])
        if (output/'review_manifest.json').exists():
            assert json.loads((output/'review_manifest.json').read_text())['schema'] == 'continuous_recorded_route_v1'
    else:
        assert render.main(['--episode', str(episode), '--output', str(output)]) == 0
        final = json.loads((output/'review_manifest.json').read_text())
        assert final['schema'] == 'auscultation_skill_recorded_full10_review_v1'
        assert final['render_lifecycle'] == {'apps_created': 1, 'close_returned': True,
            'completion_stage': 'post_app_close_pre_process_exit',
            'exit_policy': 'controlled_os_exit_after_close'}
        assert events[-1] == 'close'
        assert events[0][1]['fast_shutdown'] is False
        assert final['executed_renderer'] == 'sim/auscultation_skill_v1/private_renderer.py'
        assert final['inherited_renderer'] == 'sim/upper_pair_posture_20260924/render_episode.py'
        assert final['resource_teardown']['annotators_detached'] == 2
        assert final['resource_teardown']['render_products_destroyed'] == 2
    assert isaac.SimulationApp is original
    assert sys.argv == argv_before


@pytest.mark.parametrize('failure', [None, 'create', 'close', 'no_app', 'two_apps', 'cleanup', 'no_cleanup_evidence'])
def test_rgbd_manifest_is_unavailable_until_exactly_one_app_has_closed(tmp_path, monkeypatch, failure):
    events = []
    isaac, original = fake_isaac(monkeypatch, events, failure)
    output = tmp_path/'rgbd'
    def capture(args):
        apps = [] if failure == 'no_app' else [isaac.SimulationApp({'headless': True})]
        if failure == 'two_apps':
            apps.append(isaac.SimulationApp({'headless': True}))
        output.mkdir()
        assert not (output/'rgbd_manifest.json').exists()
        for app in apps:
            app.close()
        if failure == 'cleanup':
            raise RuntimeError('resource cleanup failed')
        manifest = {'schema': 'auscultation_saved_pose_rgbd_v1', 'samples': []}
        if failure != 'no_cleanup_evidence':
            manifest['resource_teardown'] = scoped_receipt(1)
        return output, manifest, {'artifact_sha256': {}}, {}
    monkeypatch.setattr(rgbd, '_capture', capture, raising=False)
    if failure:
        with pytest.raises(RuntimeError):
            rgbd.main(['--episode', str(tmp_path/'episode'), '--output', str(output)])
        assert not (output/'rgbd_manifest.json').exists()
    else:
        assert rgbd.main(['--episode', str(tmp_path/'episode'), '--output', str(output)]) == 0
        final = json.loads((output/'rgbd_manifest.json').read_text())
        assert final['capture_lifecycle'] == {'apps_created': 1, 'close_returned': True,
            'completion_stage': 'post_app_close_pre_process_exit',
            'exit_policy': 'controlled_os_exit_after_close'}
        assert events[-1] == 'close'
        assert events[0][1]['fast_shutdown'] is False
    assert isaac.SimulationApp is original
