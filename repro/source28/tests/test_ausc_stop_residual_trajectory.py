"""Pure CPU interface fixtures; no native/geometry qualification claim."""
from copy import deepcopy
import importlib
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_pi05_v6.execution import JointTrajectory, FinishedStationaryRoundoffError
from sim.ausc_policy import m2_given_contact_handoff as handoff


def adapter():
    name = 'sim.ausc_policy.m2_stop_residual_trajectory'
    assert importlib.util.find_spec(name) is not None, 'Opt-in STOP residual adapter is missing'
    return importlib.import_module(name)


@pytest.fixture
def native(monkeypatch):
    """Ruckig public interface, deterministic Finished residual, reset-only cache."""
    class Input:
        def __init__(self, dof):
            for key in handoff.INPUT_FIELDS:
                setattr(self, key, None)
            self.enabled = [True] * dof
            self.intermediate_positions = []
            self.control_interface = 'Position'
            self.synchronization = 'Time'
            self.duration_discretization = 'Continuous'

    class Output:
        def __init__(self, dof):
            self.new_position = [0.] * dof
            self.new_velocity = [0.] * dof
            self.new_acceleration = [0.] * dof
            self.time = 0.
        def pass_to_input(self, inp):
            inp.current_position = self.new_position.copy()
            inp.current_velocity = self.new_velocity.copy()
            inp.current_acceleration = self.new_acceleration.copy()

    class Generator:
        def __init__(self, dof, dt):
            self.dt = dt
            self.resets = 0
            self.failures = 1
            self.inputs = []
            self.reset_observer = None
            self.mutate_on_reset = False
            self.local_ticks = 0
        def update(self, inp, out):
            self.inputs.append(deepcopy(vars(inp)))
            self.local_ticks += 1
            target = np.asarray(inp.target_position)
            if self.resets < self.failures:
                out.new_position = (target + 1e-9).tolist()
                out.new_velocity = [1e-17] * 7
                out.new_acceleration = [-1e-18] * 7
            else:
                out.new_position = target.tolist()
                out.new_velocity = [0.] * 7
                out.new_acceleration = [0.] * 7
            out.time = self.local_ticks * self.dt
            return 'Finished'
        def reset(self):
            if self.reset_observer:
                self.reset_observer()
            self.resets += 1
            self.local_ticks = 0
    module = SimpleNamespace(__version__='0.15.3', Ruckig=Generator,
        InputParameter=Input, OutputParameter=Output,
        Result=SimpleNamespace(Working='Working', Finished='Finished'))
    monkeypatch.setitem(sys.modules, 'ruckig', module)
    return module


def trajectory(native, *, active=True):
    t = adapter().StopResidualTrajectory(np.zeros(7), dt=handoff.DT, **handoff.PLANNING_LIMITS)
    if active:
        before = handoff.input_snapshot(t)
        handoff.apply_stop_profile(t, expected_input=before)
        t.set_target(np.zeros(7))
        t.begin_stop_residual_replanning()
    return t


def bits(values):
    return np.asarray(values, dtype=np.float64).tobytes()


def test_raw_finished_command_and_derivatives_pass_before_cache_reset(native):
    t = trajectory(native)
    objects = (t, t.input, t.output, t.generator)
    seen = []
    t.generator.reset_observer = lambda: seen.append((
        bits(t.input.current_position), bits(t.input.current_velocity), bits(t.input.current_acceleration),
        handoff.output_snapshot(t), tuple(map(id, (t, t.input, t.output, t.generator)))))
    q = t.step()
    assert bits(q) == bits([1e-9] * 7)
    assert bits(t.velocity) == bits([1e-17] * 7)
    assert bits(t.acceleration) == bits([-1e-18] * 7)
    assert seen[0][:3] == (bits(q), bits(t.velocity), bits(t.acceleration))
    assert seen[0][3] == handoff.output_snapshot(t)
    assert seen[0][4] == tuple(map(id, objects))
    assert all(a is b for a, b in zip(objects, (t, t.input, t.output, t.generator)))
    assert t.input.target_position == [0.] * 7 and t.terminal_normalizations == []
    first = t.stop_residual_report()
    assert first['cache_recalculations'] == 1 and first['ticks'] == 1
    assert first['events'][0]['canonicalized'] is False
    assert first['events'][0]['raw_output'] == handoff.output_snapshot(t)
    assert first['events'][0]['cache_reset_input_before'] == first['events'][0]['cache_reset_input_after']
    t.step()
    assert bits(t.generator.inputs[1]['current_position']) == bits(q)
    assert bits(t.generator.inputs[1]['current_velocity']) == bits([1e-17] * 7)
    assert bits(t.generator.inputs[1]['current_acceleration']) == bits([-1e-18] * 7)
    assert t.stop_residual_report()['events'][1]['tick'] == 2


def test_legacy_and_unarmed_adapter_keep_original_finished_error(native):
    for cls in (JointTrajectory, adapter().StopResidualTrajectory):
        t = cls(np.zeros(7), dt=handoff.DT, **handoff.STOP_PLANNING_LIMITS)
        before = bits(t.input.current_position)
        with pytest.raises(FinishedStationaryRoundoffError, match='roundoff bound'):
            t.step()
        assert bits(t.input.current_position) == before
        assert t.generator.resets == 0


@pytest.mark.parametrize('fault', ['target', 'jerk', 'dt', 'derivative', 'twice', 'version'])
def test_adapter_activation_refuses_non_original_stop(native, fault):
    t = trajectory(native, active=False)
    handoff.apply_stop_profile(t, expected_input=handoff.input_snapshot(t))
    if fault == 'target': t.set_target(np.ones(7) * .001)
    elif fault == 'jerk': t.input.max_jerk = [1.5] * 7
    elif fault == 'dt': t.dt = 1 / 120.
    elif fault == 'derivative': t.input.target_velocity = [1e-18] * 7
    elif fault == 'version': t.module.__version__ = 'other'
    elif fault == 'twice': t.begin_stop_residual_replanning()
    with pytest.raises((ValueError, RuntimeError)):
        t.begin_stop_residual_replanning()
    assert t.generator.resets == 0


def test_eight_cache_recalculations_are_a_hard_failure_without_numeric_retry(native):
    m = adapter(); t = trajectory(native); t.generator.failures = 100
    for _ in range(8):
        t.step()
    with pytest.raises(m.StopResidualReplanError, match='eight') as caught:
        t.step()
    assert not isinstance(caught.value, FinishedStationaryRoundoffError)
    assert t.generator.resets == 8
    assert caught.value.stop_residual_report['cache_recalculations'] == 8
    assert caught.value.stop_residual_report['failed'] is True


def test_active_adapter_rejects_changed_target_or_profile_before_update(native):
    for field in ('target_position', 'max_jerk'):
        t = trajectory(native)
        setattr(t.input, field, [1.] * 7)
        with pytest.raises(adapter().StopResidualReplanError, match='changed'):
            t.step()
        assert t.generator.inputs == [] and t.generator.resets == 0


def test_preview_and_live_preserve_full_seam_stationary_tail_and_event_parity(native):
    m = adapter(); t = trajectory(native, active=False)
    history = np.zeros((3, 7)); checked = []
    def live():
        return dict(input=handoff.input_snapshot(t), output=handoff.output_snapshot(t), physics_rows=0)
    before = deepcopy(live())
    preview = handoff.preview_stop(t, history, lambda q: checked.append(q.copy()), live)
    assert live() == before
    assert len(checked) == len(preview.commands) == 4
    assert preview.report['schema'] == m.PREVIEW_SCHEMA != handoff.PREVIEW_SCHEMA
    assert preview.report['command_derivative_maxima']['jerk'] <= 1.5 + 1e-8
    assert preview.report['reserved_settle_intervals'] == 24
    report = preview.report['stop_residual_adapter']
    assert report['ticks'] == 4 and report['elapsed_command_s'] == 4 * handoff.DT
    assert report['events'][0]['raw_output']['new_position'] == [1e-9] * 7
    assert np.array_equal(preview.commands[-3:], np.zeros((3, 7)))
    handoff.apply_stop_profile(t, expected_input=preview.report['input_state'])
    t.set_target(history[-1]); t.begin_stop_residual_replanning()
    actual = np.asarray([t.step() for _ in preview.commands])
    np.testing.assert_array_equal(actual, preview.commands)
    live_report = t.stop_residual_report()
    assert live_report['events'] == report['events']
    assert live_report['events_sha256'] == report['events_sha256']
    assert live_report['object_ids'] != report['object_ids']
    m.verify_report(report, preview.commands)
    m.verify_report(live_report, actual)
    reconstructed = m.replay_report(preview.report['input_state'], history, report, lambda q: None)
    assert reconstructed['events_sha256'] == report['events_sha256']


@pytest.mark.parametrize('fault', ['tick', 'digest', 'target', 'command', 'object', 'cap', 'dt', 'source'])
def test_evidence_verifier_refuses_mutated_or_resigned_protocol(native, fault):
    m = adapter(); t = trajectory(native)
    commands = np.asarray([t.step() for _ in range(4)])
    report = t.stop_residual_report()
    if fault == 'tick': report['events'][0]['tick'] = 0
    elif fault == 'digest': report['events_sha256'] = '0' * 64
    elif fault == 'target': report['events'][0]['target_q'][0] = .01
    elif fault == 'command': commands[0, 0] += .01
    elif fault == 'object': report['objects_unchanged'] = False
    elif fault == 'cap': report['max_cache_recalculations'] = 9
    elif fault == 'dt': report['dt'] = 0.
    elif fault == 'source': report['source_sha256'] = {}
    if fault != 'digest': report['events_sha256'] = handoff.digest(report['events'])
    with pytest.raises((ValueError, RuntimeError)):
        m.verify_report(report, commands)


def test_explicit_native_selection_is_strict_and_legacy_selection_stays_empty(native):
    m = adapter()
    assert m.native_options({}, metadata={}) == {}
    config = {m.CONFIG_KEY: deepcopy(m.CONFIG)}
    assert m.native_options(config, metadata={'handoff_policy_id': handoff.POLICY_ID}) == {
        'trajectory_factory': m.StopResidualTrajectory}
    assert m.source_paths(config) and m.source_paths({}) == []
    for bad in ({}, dict(m.CONFIG, max_cache_recalculations=9), dict(m.CONFIG, enabled=False)):
        with pytest.raises(ValueError):
            m.native_options({m.CONFIG_KEY: bad}, metadata={'handoff_policy_id': handoff.POLICY_ID})
    with pytest.raises(ValueError):
        m.native_options(config, metadata={})


def test_controller_actual_ticks_match_adapter_preview_and_keep_25_pose_settle(native, tmp_path):
    from test_ausc_given_contact_handoff import boundary
    c = boundary(tmp_path)
    c.trajectory = trajectory(native, active=False)
    original = (c.trajectory, c.trajectory.input, c.trajectory.output, c.trajectory.generator)
    assert c._handoff_ready(c._reference_observation())
    start = len(c.env.rows)
    assert c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))['passed']
    receipt = c.l2_handoffs[-1]
    preview = receipt['parking_preview']['stop_residual_adapter']
    actual = receipt['parking_stop_residual_adapter']
    assert preview['events'] == actual['events'] and preview['events_sha256'] == actual['events_sha256']
    assert receipt['parking_settle']['samples'] == 25 and receipt['parking_settle']['intervals'] == 24
    assert receipt['parking_steps'] == len(c.env.rows) - start == 4 + 24
    assert len(c.trajectory.generator.inputs) == 4  # Constant settle tail never steps the planner.
    assert all(a is b for a, b in zip(original, (c.trajectory, c.trajectory.input, c.trajectory.output, c.trajectory.generator)))
    assert receipt['parking_preview']['source_sha256'].items() >= actual['source_sha256'].items()
    assert 'parking_numeric_rejections' not in receipt


def test_cache_reset_must_leave_output_and_input_bit_identical(native):
    t = trajectory(native)
    def mutate():
        t.output.new_velocity = [0.] * 7
    t.generator.reset_observer = mutate
    with pytest.raises(adapter().StopResidualReplanError, match='changed raw state'):
        t.step()
    assert t.stop_residual_report()['failed'] is True


def test_adapter_cannot_exceed_original_stop_tick_budget(native):
    t = trajectory(native)
    t.generator.failures = 0
    for _ in range(handoff.MAX_STOP_STEPS):
        t.step()
    before = len(t.generator.inputs)
    with pytest.raises(adapter().StopResidualReplanError, match='1176'):
        t.step()
    assert len(t.generator.inputs) == before


def test_evidence_requires_real_three_stationary_commands(native):
    t = trajectory(native)
    q = t.step()
    report = t.stop_residual_report()
    report['three_stationary_commands'] = True
    with pytest.raises(ValueError, match='stationary'):
        adapter().verify_report(report, [q])


def test_native_and_deployment_source_inventories_include_adapter(native):
    from sim.ausc_policy import m2_given_contact_pilot_deployment as pilot
    m = adapter()
    assert m.source_paths({m.CONFIG_KEY: deepcopy(m.CONFIG)})[0].name in pilot.sources()['files']


def test_only_within_original_64ulp_uses_existing_native_normalizer(native):
    t = trajectory(native)
    original = t.generator.update
    def near_then_exact(inp, out):
        if t.generator.resets and inp.current_position == inp.target_position and inp.current_velocity == [0.] * 7:
            return original(inp, out)
        status = original(inp, out)
        if t.generator.resets:
            out.new_position = (np.asarray(inp.target_position) + 1e-15).tolist()
            out.new_velocity = [1e-17] * 7
            out.new_acceleration = [-1e-18] * 7
        return status
    t.generator.update = near_then_exact
    t.step()
    assert t.terminal_normalizations == []
    q = t.step()
    assert bits(q) == bits([0.] * 7) and len(t.terminal_normalizations) == 1
    event = t.last_stop_residual_event()
    assert event['canonicalized'] is True and event['cache_recalculated'] is False
    assert event['raw_output']['new_position'] == [1e-15] * 7
    assert event['output']['new_position'] == [0.] * 7


def test_signed_zero_target_change_is_not_bit_identical_original_target(native):
    t = trajectory(native)
    t.input.target_position = [-0.] * 7
    with pytest.raises(adapter().StopResidualReplanError, match='changed'):
        t.step()
    assert t.generator.inputs == []


def test_evidence_binds_unchanged_objects_inside_every_tick(native):
    t = trajectory(native)
    commands = np.asarray([t.step() for _ in range(4)])
    report = t.stop_residual_report()
    assert len(report['object_checks']) == len(report['events'])
    for check in report['object_checks']:
        assert check['before'] == check['after'] == report['object_ids']
        if 'after_cache_reset' in check:
            assert check['after_cache_reset'] == report['object_ids']
    report['object_checks'][0]['after']['input'] += 1
    report['object_checks_sha256'] = handoff.digest(report['object_checks'])
    with pytest.raises(ValueError, match='object'):
        adapter().verify_report(report, commands)


def test_original_preview_cpu_deadline_is_not_restarted_by_adapter(native):
    t = trajectory(native, active=False)
    history = np.zeros((3, 7)); ticks = iter([0., 0., 0., 31.])
    def live():
        return dict(input=handoff.input_snapshot(t), output=handoff.output_snapshot(t))
    before = deepcopy(live())
    with pytest.raises(ValueError, match='CPU deadline'):
        handoff.preview_stop(t, history, lambda q: None, live, clock=lambda: next(ticks))
    assert live() == before and t.generator.inputs == []


def test_non_dictionary_or_bool_count_config_is_refused(native):
    m = adapter()
    for value in (None, True, [], dict(m.CONFIG, max_cache_recalculations=True)):
        with pytest.raises(ValueError):
            m.native_options({m.CONFIG_KEY: value}, metadata={'handoff_policy_id': handoff.POLICY_ID})


def native_parking(native, tmp_path):
    from test_ausc_given_contact_handoff import boundary, append
    c = boundary(tmp_path)
    c.metadata.update(episode_id='e', target_id='L1')
    c.trajectory = trajectory(native, active=False)
    def block(commands, label, **kwargs):
        for q in commands:
            append(c, q)
            row = c.env.rows[-1]
            row['raw_intent'] = deepcopy(c._current_intent_audit)
            if c._current_stop_residual_audit is not None:
                row['stop_residual_adapter_tick'] = deepcopy(c._current_stop_residual_audit)
            row['actual_control_owner'] = c.control_owner
            row['interpolator'] = dict(input=handoff.input_snapshot(c.trajectory, profile='stop'),
                output=handoff.output_snapshot(c.trajectory), planning_profile='stop',
                objects={key: id(getattr(c.trajectory, key)) for key in ('input', 'output', 'generator')},
                trajectory_object=id(c.trajectory))
    c._block = block
    assert c._handoff_ready(c._reference_observation())
    result = c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))
    receipt = c.l2_handoffs[-1]; receipt['parked_entry'] = result
    run = dict(handoff_policy_id=handoff.POLICY_ID,
        runtime_sources_sha256=receipt['parking_preview']['source_sha256'])
    config = {adapter().CONFIG_KEY: deepcopy(adapter().CONFIG)}
    return c, run, receipt, config


def test_explicit_native_verifier_checks_new_report_while_training_reader_rejects(native, tmp_path):
    from sim.ausc_policy import m2_given_contact_dataset as dataset
    c, run, receipt, config = native_parking(native, tmp_path)
    adapter().verify_native_parking(run, receipt, c.env.rows, c.metadata, c.reference,
        c.manager.config, config=config, kin=c.kin)
    with pytest.raises(ValueError, match='preview bounds'):
        dataset._interior_parking(run, receipt, c.env.rows, c.metadata, c.reference, c.manager.config, kin=c.kin)


@pytest.mark.parametrize('fault', ['missing_config', 'config_cap', 'source', 'live_event', 'event_digest',
    'live_objects', 'interpolator', 'preview_count', 'settle', 'pose', 'tail', 'early_stationary', 'resigned_cache_state'])
def test_native_verifier_retains_physical_and_adapter_evidence_gates(native, tmp_path, fault):
    c, run, receipt, config = native_parking(native, tmp_path)
    start = receipt['parking_start_physics_row']
    report = receipt['parking_stop_residual_adapter']
    if fault == 'missing_config': config = {}
    elif fault == 'config_cap': config[adapter().CONFIG_KEY]['max_cache_recalculations'] = 9
    elif fault == 'source': run['runtime_sources_sha256'] = {}
    elif fault == 'live_event': c.env.rows[start]['stop_residual_adapter_tick']['event']['command_q'][0] += .1
    elif fault == 'event_digest': c.env.rows[start]['stop_residual_adapter_tick']['event_sha256'] = '0' * 64
    elif fault == 'live_objects': c.env.rows[start]['stop_residual_adapter_tick']['objects']['after']['input'] += 1
    elif fault == 'interpolator': c.env.rows[start]['interpolator']['input']['fields']['current_velocity'][0] = 0.
    elif fault == 'preview_count': receipt['parking_preview']['stop_steps'] -= 1
    elif fault == 'settle': receipt['parking_settle']['samples'] = 24
    elif fault == 'pose': c.env.rows[-1]['parking_pose_after']['tcp_speed_m_s'] = .1
    elif fault == 'tail': receipt['parking_tail']['object_ids']['generator'] += 1
    elif fault == 'early_stationary': report['three_stationary_commands'] = False
    elif fault == 'resigned_cache_state':
        report['events'][0]['cache_reset_input_after']['fields']['current_velocity'] = [0.] * 7
        report['events_sha256'] = handoff.digest(report['events'])
    with pytest.raises((ValueError, RuntimeError)):
        adapter().verify_native_parking(run, receipt, c.env.rows, c.metadata, c.reference,
            c.manager.config, config=config, kin=c.kin)


def test_adapter_opt_in_selects_existing_lossless_xz_without_changing_legacy_gzip(native):
    m = adapter()
    assert m.native_trace_filename({}, metadata={}, legacy_filename='given_control_native.jsonl.gz') == 'given_control_native.jsonl.gz'
    config = {m.CONFIG_KEY: deepcopy(m.CONFIG)}
    assert m.native_trace_filename(config, metadata={'handoff_policy_id': handoff.POLICY_ID},
        legacy_filename='given_control_native.jsonl.gz') == 'given_control_native.jsonl.xz'


def test_closed_pilot_qualifier_selects_fixed_native_verifier_on_explicit_config(native, tmp_path, monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import closed_run
    from sim.ausc_policy import m2_given_contact_runtime as rt
    import json
    audit, c, config, run = closed_run(tmp_path, monkeypatch)
    config[adapter().CONFIG_KEY] = deepcopy(adapter().CONFIG)
    run['neutral_native_audit']['pilot_binding']['configuration_sha256'] = rt.digest(config)
    index = run['neutral_native_audit']['index']['path']
    from pathlib import Path
    Path(index).write_text(json.dumps({key: value for key, value in run['neutral_native_audit'].items() if key != 'index'}))
    run['neutral_native_audit']['index'] = rt.reference(index)
    for name in ('fresh_expert_result.json', 'COMPLETE.json'):
        (tmp_path / name).write_text(json.dumps(run))
    calls = []
    monkeypatch.setattr(audit, '_ipc', lambda *a: {'normal_exit': True})
    monkeypatch.setattr(rt.dataset, '_physical', lambda *a: pytest.fail('new native trace entered legacy training validator'))
    monkeypatch.setattr(adapter(), 'verify_native_physical', lambda *args, **kwargs: calls.append((args, kwargs)))
    result = rt.read_bound(audit.qualify(tmp_path, config))
    assert result['passed'] is True and len(calls) == 1 and calls[0][1]['config'] == config


def test_activation_requires_original_target_bit_pattern(native):
    t = trajectory(native, active=False)
    handoff.apply_stop_profile(t, expected_input=handoff.input_snapshot(t))
    t.input.target_position = [-0.] * 7
    with pytest.raises(ValueError, match='current-position'):
        t.begin_stop_residual_replanning()


def test_finished_and_working_events_are_explicit_and_working_never_resets(native):
    t = trajectory(native)
    update = t.generator.update
    def working(inp, out):
        update(inp, out)
        return 'Working'
    t.generator.update = working
    assert bits(t.step()) == bits([1e-9] * 7)
    event = t.last_stop_residual_event()
    assert event['native_result'] == 'Working'
    assert event['cache_recalculated'] is False and event['canonicalized'] is False
    assert t.generator.resets == 0


def test_legacy_reader_refuses_adapter_marker_even_if_schema_is_relabelled(native, tmp_path):
    from sim.ausc_policy import m2_given_contact_dataset as dataset
    c, run, receipt, config = native_parking(native, tmp_path)
    receipt['parking_preview']['schema'] = handoff.PREVIEW_SCHEMA
    with pytest.raises(ValueError, match='adapter'):
        dataset._interior_parking(run, receipt, c.env.rows, c.metadata, c.reference,
            c.manager.config, kin=c.kin)


def test_three_identical_residual_commands_cannot_finish_outside_original_bound(native, monkeypatch):
    original_generator = native.Ruckig
    class PersistentResidual(original_generator):
        def __init__(self, *args):
            super().__init__(*args)
            self.failures = 100
    monkeypatch.setattr(native, 'Ruckig', PersistentResidual)
    t = trajectory(native, active=False)
    history = np.zeros((3, 7))
    live = lambda: dict(input=handoff.input_snapshot(t), output=handoff.output_snapshot(t))
    before = deepcopy(live())
    with pytest.raises(adapter().StopResidualReplanError, match='eight'):
        handoff.preview_stop(t, history, lambda q: None, live)
    assert live() == before


def test_actual_terminal_readback_must_remain_certified_before_constant_tail(native):
    t = trajectory(native)
    for _ in range(4): t.step()
    assert t.stop_residual_terminal_ready()
    t.output.new_velocity = [1e-17] * 7
    t.input.current_velocity = [1e-17] * 7
    t.velocity = np.asarray([1e-17] * 7)
    assert not t.stop_residual_terminal_ready()


def test_controller_refuses_tail_if_actual_terminal_native_readback_changes(native, tmp_path):
    from test_ausc_given_contact_handoff import boundary
    c = boundary(tmp_path); c.trajectory = trajectory(native, active=False)
    block = c._block
    def mutate_after_final_tick(commands, label, **kwargs):
        result = block(commands, label, **kwargs)
        if len(c.env.rows) == 244:
            c.trajectory.output.new_velocity = [1e-17] * 7
            c.trajectory.input.current_velocity = [1e-17] * 7
            c.trajectory.velocity = np.asarray([1e-17] * 7)
        return result
    c._block = mutate_after_final_tick
    assert c._handoff_ready(c._reference_observation())
    with pytest.raises(ValueError, match='terminal'):
        c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))
    assert 'parking_tail' not in c.l2_handoffs[-1]


def test_adapter_evidence_keeps_the_exact_existing_raw_intent_contract(native, tmp_path):
    c, run, receipt, config = native_parking(native, tmp_path)
    start = receipt['parking_start_physics_row']
    for row in c.env.rows[start:]:
        assert set(row['raw_intent']) == {'schema', 'action_semantics_id', 'setter',
            'native_tick_in_cell', 'live_target_q', 'live_target_after_step_q'}


def test_native_observer_records_separate_sidecar_without_changing_raw_intent(tmp_path):
    from test_ausc_given_contact_evaluation import native_tick_fixture
    import json
    m, audit, c, state = native_tick_fixture(tmp_path)
    c._current_stop_residual_audit = dict(adapter_id='CPU-interface-fixture', event={'tick': 1})
    def tick():
        state['time_s'] = 1 / 240.
        c.env.rows.append(dict(time_s=state['time_s'], q=[0.] * 7, q_command=[0.] * 7,
            phase='FREE', subphase='staging', physical_violations=[]))
    audit.tick(c, [0.] * 7, 'staging', tick)
    audit.close()
    event = json.loads((tmp_path / 'native.jsonl').read_text())
    assert event['raw_intent'] is None
    assert event['stop_residual_adapter_tick'] == c._current_stop_residual_audit
    assert event['controller_before']['stop_residual_adapter_tick'] == event['controller_after']['stop_residual_adapter_tick']


def test_complete_adapter_evidence_verification_stays_inside_original_cpu_deadline(native, monkeypatch):
    m = adapter(); t = trajectory(native, active=False)
    clock = [0.]
    verify = m.verify_report
    def slow_evidence(*args):
        result = verify(*args)
        clock[0] = 31.
        return result
    monkeypatch.setattr(m, 'verify_report', slow_evidence)
    live = lambda: dict(input=handoff.input_snapshot(t), output=handoff.output_snapshot(t))
    before = deepcopy(live())
    with pytest.raises(ValueError, match='CPU deadline'):
        handoff.preview_stop(t, np.zeros((3, 7)), lambda q: None, live, clock=lambda: clock[0])
    assert live() == before


def actual_preflight_case(tmp_path, module_name, *, original_opt_in=False):
    """Hash-bound local packets; actual deployment preflight is never mocked."""
    m = importlib.import_module('sim.ausc_policy.' + module_name)
    from test_ausc_given_contact_pose import reference, metadata
    from test_ausc_given_contact_pilot_deployment import complete_packet
    ref = reference(tmp_path)
    if module_name == 'm2_given_contact_pilot_deployment':
        complete = complete_packet()
        subject, episode = 'p000449', 'source-train'
        weights = tmp_path / 'weights.pt'; weights.write_bytes(b'hash-only local fixture')
        complete['best'] = m.runtime.reference(weights)
    else:
        from test_ausc_given_contact_expanded_pilot import trained_run
        trained = tmp_path / 'trained'; trained.mkdir()
        complete = trained_run(trained)
        preparation = m.runtime.read_bound(complete['inputs']['dataset'])
        row = preparation['episodes'][0]
        subject, episode = row['subject_id'], row['episode_id']
    meta = dict(metadata(ref), subject_id=subject, split='train', episode_id=episode,
        target_id='L1', handoff_policy_id=handoff.POLICY_ID)
    original = dict(isaac_python=m.NATIVE_PYTHON,
        physics=dict(position_iterations=64, velocity_iterations=1, armature_kg_m2=.02),
        fresh_expert=dict(plans=[ref.snapshot()['sources']['fresh_plan']],
            force_feedback={'frozen': True}, raw_multimodal_capture={'enabled': True}),
        m2_given_contact_control=dict(metadata=meta, episode_id=episode))
    if original_opt_in: original[adapter().CONFIG_KEY] = deepcopy(adapter().CONFIG)
    source = m.runtime.write_new(tmp_path / 'source.json', original)
    if module_name != 'm2_given_contact_pilot_deployment':
        preparation['episodes'][0]['config'] = source
        dataset = m.runtime.write_new(tmp_path / 'prepared.json', preparation)
        complete['inputs']['dataset'] = complete['schedule']['dataset'] = dataset
        complete['inputs']['config'] = m.runtime.write_new(tmp_path / 'schedule.json', complete['schedule'])
        from sim.ausc_policy import m2_given_contact_expanded_pilot_train as trainer
        complete['protocol'] = trainer.training_protocol(complete['schedule'])
        cpu = m.runtime.read_bound(complete['cpu_equivalence'])
        selection = m.runtime.read_bound(cpu['selection'])
        selection.update(inputs=deepcopy(complete['inputs']), protocol=deepcopy(complete['protocol']))
        cpu['selection'] = m.runtime.write_new(tmp_path / 'selected.json', selection)
        complete['cpu_equivalence'] = m.runtime.write_new(tmp_path / 'cpu.json', cpu)
    complete_ref = m.runtime.write_new(tmp_path / 'COMPLETE.json', complete)
    config = m.derive_config(original, source_config=source,
        model=dict(checkpoint=complete['best'], complete=complete_ref), run_id='residual-fixture',
        source_inventory=m.sources(), scope=complete['scope'])
    return m, original, source, config


@pytest.mark.parametrize('module_name', ['m2_given_contact_pilot_deployment', 'm2_given_contact_expanded_pilot_deployment'])
@pytest.mark.parametrize('original_opt_in', [False, True])
def test_actual_preflight_accepts_only_validated_stop_opt_in_without_physics_projection_loophole(tmp_path, module_name, original_opt_in):
    m, original, source, config = actual_preflight_case(tmp_path, module_name, original_opt_in=original_opt_in)
    assert m.preflight(config)['deployment'] == config[m.ROUTE_KEY]
    assert (adapter().CONFIG_KEY in config) is original_opt_in
    configured = deepcopy(config)
    configured[adapter().CONFIG_KEY] = deepcopy(adapter().CONFIG)
    configured[m.ROUTE_KEY]['runtime_config_sha256'] = m.config_digest(configured)
    assert m.preflight(configured)['deployment'] == configured[m.ROUTE_KEY]
    assert m.runtime.read_bound(source) == original
    for field, value in [('enabled', False), ('max_cache_recalculations', 9), ('unknown', True)]:
        bad = deepcopy(configured); bad[adapter().CONFIG_KEY][field] = value
        bad[m.ROUTE_KEY]['runtime_config_sha256'] = m.config_digest(bad)
        with pytest.raises(ValueError): m.preflight(bad)
    for fault in ('physics', 'force', 'unknown_key', 'missing_adapter_inventory'):
        bad = deepcopy(configured)
        if fault == 'physics': bad['physics']['position_iterations'] = 32
        elif fault == 'force': bad['fresh_expert']['force_feedback']['frozen'] = False
        elif fault == 'unknown_key': bad['unreviewed_native_option'] = True
        else: del bad[m.ROUTE_KEY]['source_inventory']['files']['m2_stop_residual_trajectory.py']
        bad[m.ROUTE_KEY]['runtime_config_sha256'] = m.config_digest(bad)
        with pytest.raises(ValueError): m.preflight(bad)


@pytest.mark.parametrize('module_name', ['m2_given_contact_pilot_deployment', 'm2_given_contact_expanded_pilot_deployment'])
def test_derive_explicit_stop_option_is_bound_before_digest_and_preserves_source(tmp_path, module_name):
    m, original, source, baseline = actual_preflight_case(tmp_path, module_name)
    dep = baseline[m.ROUTE_KEY]
    _, complete = m.model_binding(dep['model']['complete'])
    kwargs = dict(source_config=source, model=dep['model'], run_id='residual-fixture',
        source_inventory=m.sources(), scope=complete['scope'])
    configured = m.derive_config(original, stop_residual_trajectory=deepcopy(adapter().CONFIG), **kwargs)
    assert configured[adapter().CONFIG_KEY] == adapter().CONFIG
    assert configured[m.ROUTE_KEY]['runtime_config_sha256'] == m.config_digest(configured)
    assert m.preflight(configured)['deployment'] == configured[m.ROUTE_KEY]
    assert m.runtime.read_bound(source) == original and adapter().CONFIG_KEY not in original
    with pytest.raises(ValueError):
        m.derive_config(original, stop_residual_trajectory=dict(adapter().CONFIG, max_cache_recalculations=9), **kwargs)


@pytest.mark.parametrize('arm', ['expert', 'act'])
@pytest.mark.parametrize('original_opt_in', [False, True])
def test_paired_arms_derive_and_preflight_exact_option_from_legacy_and_opted_in_sources(tmp_path, monkeypatch, arm, original_opt_in):
    from test_ausc_rebuild_paired_eval import setup
    m, source, model, complete, cases = setup(tmp_path, monkeypatch)
    if original_opt_in:
        packet = m.runtime.read_bound(source)
        for row in packet['cases']:
            original = m.runtime.read_bound(row['config'])
            original[adapter().CONFIG_KEY] = deepcopy(adapter().CONFIG)
            row['config'] = m.runtime.write_new(tmp_path / (row['scenario_id'] + '-opt-in.json'), original)
        source = m.runtime.write_new(tmp_path / 'opt-in-cases.json', packet)
    prepared = m.runtime.read_bound(m.prepare(model['complete']['path'], source['path'], tmp_path / 'paired',
        astra_settings=m.DEFAULT_ASTRA, arms=('expert', 'act')))
    protocol, _ = m.load_protocol(prepared['protocol'])
    case = protocol['cases'][0]
    original = m.runtime.read_bound(case['config'])
    configured = m.derive_config(original, case, prepared['protocol'], protocol, arm,
        stop_residual_trajectory=deepcopy(adapter().CONFIG))
    assert configured[adapter().CONFIG_KEY] == adapter().CONFIG
    assert m.preflight_arm(configured)['arm'] == arm
    assert m.runtime.read_bound(case['config']) == original
    for fault in ('cap', 'physics', 'unknown_key'):
        bad = deepcopy(configured)
        if fault == 'cap': bad[adapter().CONFIG_KEY]['max_cache_recalculations'] = 9
        elif fault == 'physics': bad['physics']['position_iterations'] = 32
        else: bad['unreviewed_native_option'] = True
        if arm == 'act': bad[m.ROUTE_KEY]['runtime_config_sha256'] = m.runtime.digest({k: v for k, v in bad.items() if k != m.ROUTE_KEY})
        with pytest.raises(ValueError): m.preflight_arm(bad)


def test_legacy_stop_tick_without_optional_handoff_receipts_remains_compatible(tmp_path):
    from test_ausc_given_contact_handoff import boundary
    c = boundary(tmp_path)
    handoff.apply_stop_profile(c.trajectory, expected_input=handoff.input_snapshot(c.trajectory))
    c.control_owner = 'L2_STOP'
    c._issue_intent(c.q.copy(), ('e', 'L1', 0, 'APPROACH'), 'given_l2_park', stop=True)
    del c.l2_handoffs
    start = len(c.env.rows)
    c._advance_tick('given_l2_park')
    assert len(c.env.rows) == start + 1
    assert c._current_stop_residual_audit is None


@pytest.mark.parametrize('missing', ['attribute', 'empty', 'receipt', 'preview'])
@pytest.mark.parametrize('selection', ['trajectory', 'factory'])
def test_configured_adapter_cannot_skip_proof_when_handoff_receipts_are_missing(native, tmp_path, missing, selection):
    from test_ausc_given_contact_handoff import boundary
    c = boundary(tmp_path)
    if selection == 'trajectory':
        c.trajectory = trajectory(native)
    else:
        c.trajectory_factory = adapter().StopResidualTrajectory
        handoff.apply_stop_profile(c.trajectory, expected_input=handoff.input_snapshot(c.trajectory))
    c.control_owner = 'L2_STOP'
    c._issue_intent(c.q.copy(), ('e', 'L1', 0, 'APPROACH'), 'given_l2_park', stop=True)
    if missing == 'attribute': del c.l2_handoffs
    elif missing == 'empty': c.l2_handoffs = []
    elif missing == 'receipt': c.l2_handoffs = [{}]
    else: c.l2_handoffs = [{'parking_preview': {}}]
    called = []
    step = c.trajectory.step
    c.trajectory.step = lambda: called.append(True) or step()
    start = len(c.env.rows)
    with pytest.raises(ValueError, match='residual.*preview'):
        c._advance_tick('given_l2_park')
    assert called == [] and len(c.env.rows) == start
