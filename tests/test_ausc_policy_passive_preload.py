"""Opt-in removal of one duplicated preload term; no controller/force-gate edit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/ausc_policy/retainer_r3_contact_v2/output/review/candidate_physics.usdc'
FRESH = {'passive_preload_N': 1.5, 'passive_preload_model': {'mode': 'single_explicit_v1'}}


def module():
    assert importlib.util.find_spec('sim.ausc_policy.passive_preload') is not None, 'Single explicit preload authoring is missing'
    from sim.ausc_policy import passive_preload
    return passive_preload


def test_absence_is_legacy_noop_without_usd_or_stage_access():
    p = module()
    assert p.passive_preload_settings({}) is None
    assert p.apply_single_explicit_preload(None, {}) is None
    assert p.passive_preload_settings(FRESH) == {'mode': 'single_explicit_v1', 'nominal_preload_N': 1.5}


@pytest.mark.parametrize('setting', [None, False, True, 'single_explicit_v1', {},
    {'mode': 'unknown'}, {'mode': 'single_explicit_v1', 'enabled': True}, {'mode': True}])
def test_unknown_or_malformed_explicit_model_is_rejected(setting):
    with pytest.raises(ValueError, match='passive_preload'):
        module().passive_preload_settings({**FRESH, 'passive_preload_model': setting})


@pytest.mark.parametrize('value', [None, True, '1.5', float('nan'), float('inf'), 0, 3, [1.5], np.float64(1.5)])
def test_nominal_preload_is_strict_json_number_and_remains_one_point_five(value):
    with pytest.raises(ValueError, match='1.5'):
        module().passive_preload_settings({**FRESH, 'passive_preload_N': value})


@pytest.fixture
def stage():
    pytest.importorskip('pxr')
    from pxr import Usd
    return Usd.Stage.Open(Usd.Stage.Open(str(SOURCE)).Flatten())


def test_real_source_removes_only_drive_target_and_preserves_nominal_explicit_preload(stage, tmp_path):
    p = module()
    from pxr import Usd, UsdPhysics
    source_hash = p.file_sha256(SOURCE)
    before = p.protected_stage_sha256(stage)
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(p.JOINT_PATH), 'linear')
    assert drive.GetTypeAttr().Get() == 'force'
    assert drive.GetTargetPositionAttr().Get() == float(np.float32(-.0015))
    receipt = p.apply_single_explicit_preload(stage, FRESH)
    assert drive.GetTargetPositionAttr().Get() == 0.
    assert p.protected_stage_sha256(stage) == before
    assert receipt['changed_attributes'] == [p.TARGET_ATTRIBUTE_PATH]
    assert receipt['before']['target_position_m'] == float(np.float32(-.0015))
    assert receipt['after']['target_position_m'] == 0.
    assert receipt['nominal_preload_N'] == 1.5
    assert receipt['explicit_effort_N'] == -1.5
    assert receipt['other_stage_attributes_unchanged'] is True
    assert receipt['applied_effort_measured'] is False
    assert receipt['protected_before_sha256'] == receipt['protected_after_sha256'] == before
    json.dumps(receipt, allow_nan=False)
    filename = tmp_path/'candidate.usdc'
    stage.GetRootLayer().Export(str(filename))
    saved = Usd.Stage.Open(str(filename))
    readback = p.verify_preload_readback(saved, FRESH, receipt)
    assert readback['scope'] == 'saved_or_preinitialization_usd'
    assert readback['drive']['target_position_m'] == 0.
    assert readback['applied_effort_measured'] is False
    assert p.file_sha256(SOURCE) == source_hash
    with pytest.raises(ValueError, match='target_position'):
        p.apply_single_explicit_preload(stage, FRESH)


@pytest.mark.parametrize('name,value', [('type', 'acceleration'), ('stiffness', 900.),
    ('damping', 3.1), ('targetPosition', 0.), ('maxForce', float('nan')), ('targetVelocity', .001)])
def test_unknown_source_drive_is_rejected_before_authoring(stage, name, value):
    p = module()
    from pxr import UsdPhysics
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(p.JOINT_PATH), 'linear')
    attribute = getattr(drive, 'Get'+name[0].upper()+name[1:]+'Attr')()
    attribute.Set(value)
    target = drive.GetTargetPositionAttr().Get()
    before = p.protected_stage_sha256(stage)
    with pytest.raises(ValueError):
        p.apply_single_explicit_preload(stage, FRESH)
    assert drive.GetTargetPositionAttr().Get() == target
    assert p.protected_stage_sha256(stage) == before


def test_runtime_readback_checks_native_slider_mapping_without_authoring(stage):
    p = module()
    receipt = p.apply_single_explicit_preload(stage, FRESH)
    robot = SimpleNamespace(get_dof_index=lambda name: 7,
        _articulation_view=SimpleNamespace(_dof_paths=np.array([['/unused']*7+[p.JOINT_PATH]])))
    before = p.protected_stage_sha256(stage)
    readback = p.verify_preload_readback(stage, FRESH, receipt, robot=robot, slider=7)
    assert readback['scope'] == 'live_usd_after_native_initialization'
    assert readback['native_dof_path'] == p.JOINT_PATH
    assert readback['native_dof_index'] == 7
    assert p.protected_stage_sha256(stage) == before
    robot._articulation_view._dof_paths[0, 7] = '/wrong'
    with pytest.raises(ValueError, match='binding'):
        p.verify_preload_readback(stage, FRESH, receipt, robot=robot, slider=7)


def test_runtime_readback_rejects_missing_or_changed_authoring_receipt(stage):
    p = module()
    receipt = p.apply_single_explicit_preload(stage, FRESH)
    with pytest.raises(ValueError, match='receipt'):
        p.verify_preload_readback(stage, FRESH, {})
    for key, value in [('nominal_preload_N', 0), ('other_stage_attributes_unchanged', False),
                       ('protected_after_sha256', '0'*64), ('source_sha256', '0'*64)]:
        with pytest.raises(ValueError, match='receipt'):
            p.verify_preload_readback(stage, FRESH, {**receipt, key: value})


def test_new_diagnostic_config_explicitly_selects_model_and_unique_episode():
    filename = ROOT/'configs/ausc_policy/fresh_expert_retainer_r3_single_preload_v3.json'
    assert filename.is_file(), 'Explicit single-preload diagnostic configuration is missing'
    data = json.loads(filename.read_text())
    assert data['fresh_expert']['passive_preload_model'] == {'mode': 'single_explicit_v1'}
    assert data['fresh_expert']['passive_preload_N'] == 1.5
    assert data['fresh_expert']['raw_multimodal_capture']['episode_id'] == 'retainer-r3-single-preload-20260930-v3'
    old = json.loads((ROOT/'configs/ausc_policy/fresh_expert_retainer_r3_v2.json').read_text())
    old['fresh_expert']['passive_preload_model'] = data['fresh_expert']['passive_preload_model']
    old['fresh_expert']['raw_multimodal_capture']['episode_id'] = data['fresh_expert']['raw_multimodal_capture']['episode_id']
    assert old == data


def test_qualification_authors_after_profile_and_exports_before_fresh_dispatch(stage, tmp_path, monkeypatch):
    import hashlib
    import sys
    from pxr import Usd, UsdPhysics
    from sim.ausc_policy import geometry_qualification as q, approved_camera, geometry_export, fresh_expert
    from sim.auscultation_repair_v3 import physics_profile
    p = module()
    events = []
    config = dict(geometry_source=str(SOURCE), geometry_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        component_priors={}, physics={}, fresh_expert=FRESH)
    config_file = tmp_path/'config.json'; config_file.write_text(json.dumps(config))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['qualification', '--config', str(config_file), '--output', str(output)])
    monkeypatch.setattr(sys, 'platform', 'linux'); monkeypatch.setenv('AUSC_OWNED_RUN', '1')
    monkeypatch.setitem(sys.modules, 'isaacsim', SimpleNamespace(SimulationApp=lambda *a, **k: SimpleNamespace(close=lambda: events.append('close'))))
    monkeypatch.setattr(q, 'require_current', lambda config: None)
    monkeypatch.setattr(q, 'moving_inertia', lambda *args: {'mass_kg': .067})
    monkeypatch.setattr(approved_camera, 'apply_approved_camera', lambda *args: {})
    def profile(s, config):
        drive = UsdPhysics.DriveAPI(s.GetPrimAtPath(p.JOINT_PATH), 'linear')
        assert drive.GetTargetPositionAttr().Get() == float(np.float32(-.0015))
        events.append('profile'); return {}
    monkeypatch.setattr(physics_profile, 'apply_profile', profile)
    def export(s, directory):
        saved = Usd.Stage.Open(str(output/'candidate_physics.usdc'))
        drive = UsdPhysics.DriveAPI(saved.GetPrimAtPath(p.JOINT_PATH), 'linear')
        assert drive.GetTargetPositionAttr().Get() == 0., 'Opt-in authoring must precede candidate export'
        assert (output/'passive_preload_authoring.json').is_file()
        events.append('export'); return []
    monkeypatch.setattr(geometry_export, 'export_geometry', export)
    monkeypatch.setattr(fresh_expert, 'run_probe', lambda *args: events.append('dispatch') or {'software_test_only': True})
    q.main()
    assert events == ['profile', 'export', 'dispatch', 'close']
    readback = json.loads((output/'passive_preload_saved_usd_readback.json').read_text())
    assert readback['drive']['target_position_m'] == 0.


def test_fresh_dispatch_checks_native_binding_and_reports_conditional_source(stage, tmp_path, monkeypatch):
    import sys
    from sim.ausc_policy import fresh_expert as fresh, contact_collision
    p = module()
    receipt = p.apply_single_explicit_preload(stage, FRESH)
    (tmp_path/'passive_preload_authoring.json').write_text(json.dumps(receipt))
    robot = SimpleNamespace(get_dof_index=lambda name: 7,
        _articulation_view=SimpleNamespace(_dof_paths=np.array([['/unused']*7+[p.JOINT_PATH]])))
    class Env:
        def __init__(self, output, **kwargs):
            self.rows = []; self.base = np.eye(4); self.out = output; self.stage = stage
            self.robot = robot; self.left = np.arange(7); self.slider = 7
            self.world = SimpleNamespace(stop=lambda: None)
            output.mkdir()
        def state(self): return dict(q=np.zeros(7))
        def save(self, name): (self.out/'fresh_expert_trace.jsonl').write_text('')
    class Controller:
        def __init__(self, *args, **kwargs): self.control_rows = []
        def run(self, points): return dict(success=False, points=[], failure_reason='software unit test, no physics')
    class Recorder:
        def __init__(self, *args): pass
        def close(self): return None
    metadata = dict(geometry_id=fresh.GEOMETRY_ID, freshly_generated=True,
        geometry_artifacts={}, world_from_left_base=np.eye(4).tolist(), plans=[])
    monkeypatch.setattr(sys, 'platform', 'linux'); monkeypatch.setenv('AUSC_OWNED_RUN', '1')
    monkeypatch.setattr(fresh, 'read_plans', lambda *args: ([], metadata))
    monkeypatch.setattr(contact_collision, 'ContactCollisionChecker', lambda *args: SimpleNamespace(static_bundle={}))
    monkeypatch.setattr(fresh, 'FreshContactEnv', Env); monkeypatch.setattr(fresh, 'FreshExpert', Controller)
    monkeypatch.setattr(fresh, '_VideoRecorder', Recorder); monkeypatch.setattr(fresh, 'require_geometry_artifacts', lambda *args: {})
    result = fresh.run_probe(None, stage, tmp_path, dict(geometry_source_sha256='unit_test', fresh_expert=FRESH), dict(mass_kg=.067))
    assert 'passive_preload_model' in result, 'Fresh result must include the authoring and startup USD readback receipt'
    model = result['passive_preload_model']
    assert model['authoring'] == receipt
    assert model['startup_usd_readback']['native_dof_path'] == p.JOINT_PATH
    assert model['startup_usd_readback']['applied_effort_measured'] is False
    assert any(path.endswith('/passive_preload.py') for path in result['runtime_sources_sha256'])
    assert result['physics_steps'] == 0
    assert not result['physical_validation']
