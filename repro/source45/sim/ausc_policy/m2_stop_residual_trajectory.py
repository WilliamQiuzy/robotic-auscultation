"""Explicit STOP-only native planner-cache recalculation, protocol v1.

This adapter never changes the original STOP target or kinematic state. A
Finished output outside the legacy 64-ULP bound remains an authentic command;
only the native planner cache is cleared before the next actual 240-Hz tick.
Legacy JointTrajectory and legacy demonstration replay remain unchanged.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from sim.auscultation_pi05_v6.execution import (
    JointTrajectory, FinishedStationaryRoundoffError, FINISHED_ROUNDOFF_ULPS)

ADAPTER_ID = 'm2_stop_residual_native_cache_replan_v1'
SCHEMA = 'm2_stop_residual_trajectory_evidence_v1'
PREVIEW_SCHEMA = 'm2_given_current_position_stop_residual_preview_v1'
CONFIG_KEY = 'stop_residual_trajectory'
MAX_CACHE_RECALCULATIONS = 8
CONFIG = dict(schema=ADAPTER_ID, enabled=True, max_cache_recalculations=8)
RUCKIG_VERSION = '0.15.3'


def _handoff():
    from . import m2_given_contact_handoff
    return m2_given_contact_handoff


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths(config):
    if CONFIG_KEY not in config:
        return []
    _need(isinstance(config[CONFIG_KEY], dict) and config[CONFIG_KEY] == CONFIG and type(config[CONFIG_KEY].get('enabled')) is bool and
          type(config[CONFIG_KEY].get('max_cache_recalculations')) is int,
          'Exact opt-in STOP residual adapter configuration required')
    return [Path(__file__).resolve()]


def native_options(config, *, metadata):
    if not source_paths(config):
        return {}
    _need(metadata.get('handoff_policy_id') == _handoff().POLICY_ID,
          'STOP residual adapter requires the original explicit interior handoff policy')
    return dict(trajectory_factory=StopResidualTrajectory)


def native_trace_filename(config, *, metadata, legacy_filename):
    """Reuse the existing full-JSON XZ transport only for explicit new runs."""
    return 'given_control_native.jsonl.xz' if native_options(config, metadata=metadata) else legacy_filename


def configuration_projection(original_projection, actual, *, metadata):
    """Allow exactly the validated new option beside unchanged native settings."""
    projected = deepcopy(original_projection)
    if CONFIG_KEY in actual:
        native_options(actual, metadata=metadata)
        projected[CONFIG_KEY] = deepcopy(actual[CONFIG_KEY])
    return projected


def _source_hashes():
    from sim.auscultation_pi05_v6 import execution
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (__file__, execution.__file__, _handoff().__file__)}


def _objects(trajectory):
    return dict(trajectory=trajectory, input=trajectory.input,
                output=trajectory.output, generator=trajectory.generator)


def _qva_bits(trajectory):
    return tuple(np.asarray(value, dtype=np.float64).tobytes() for value in (
        trajectory.input.current_position, trajectory.input.current_velocity,
        trajectory.input.current_acceleration, trajectory.output.new_position,
        trajectory.output.new_velocity, trajectory.output.new_acceleration,
        trajectory.velocity, trajectory.acceleration))


def _within_roundoff(output, target):
    tolerance = FINISHED_ROUNDOFF_ULPS * np.finfo(np.float64).eps
    target = np.asarray(target, dtype=np.float64)
    return bool(np.all(abs(np.asarray(output['new_position']) - target) <= tolerance * np.maximum(1., abs(target))) and
                np.all(abs(np.asarray(output['new_velocity'])) <= tolerance) and
                np.all(abs(np.asarray(output['new_acceleration'])) <= tolerance))


class StopResidualReplanError(RuntimeError):
    """Fatal bounded-adapter failure; never a retryable legacy numeric rejection."""


class StopResidualTrajectory(JointTrajectory):
    """Legacy L2 execution until explicitly armed after the original STOP setter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_binding = None
        self._stop_events = []
        self._stop_object_checks = []
        self._stop_recalculations = 0
        self._stop_failed = False
        self._stop_failure = None
        self._stop_raw = None
        self._stop_pending = False
        self._stop_history = []

    def begin_stop_residual_replanning(self, *, history=None):
        h = _handoff()
        _need(self._stop_binding is None, 'STOP residual adapter is already active')
        snapshot = h.input_snapshot(self, profile='stop')
        _need(self.module.__version__ == RUCKIG_VERSION, 'Pinned native Ruckig version required')
        _need(callable(getattr(self.generator, 'reset', None)), 'Native planner cache reset() is required')
        _need(np.asarray(snapshot['fields']['current_position'], dtype=np.float64).tobytes() ==
              np.asarray(snapshot['fields']['target_position'], dtype=np.float64).tobytes(),
              'Only the original current-position STOP target can activate residual replanning')
        if history is not None:
            values = np.asarray(history, dtype=np.float64)
            _need(values.shape == (3, 7) and np.isfinite(values).all() and
                  values[-1].tobytes() == np.asarray(self.input.current_position, dtype=np.float64).tobytes(),
                  'Real three-command STOP seam is required')
            self._stop_history = values.tolist()
        self._stop_binding = deepcopy(snapshot)
        self._stop_objects = _objects(self)
        self._stop_sources = _source_hashes()
        self._stop_initial_history = deepcopy(self._stop_history)

    def _check_binding(self):
        h = _handoff()
        if not all(value is _objects(self)[name] for name, value in self._stop_objects.items()):
            raise StopResidualReplanError('STOP trajectory/input/output/generator object changed')
        snapshot = h.input_snapshot(self, profile='stop')
        mutable = {'current_position', 'current_velocity', 'current_acceleration'}
        fixed = lambda state: {key: value for key, value in state['fields'].items() if key not in mutable}
        if (self.module.__version__ != RUCKIG_VERSION or snapshot['dt'] != self._stop_binding['dt'] or
                h.digest(fixed(snapshot)) != h.digest(fixed(self._stop_binding))):
            raise StopResidualReplanError('Original STOP target/profile/settings changed')
        return snapshot

    def _normalize_finished_stationary(self, q):
        if self._stop_binding is None:
            return super()._normalize_finished_stationary(q)
        self._stop_raw = _handoff().output_snapshot(self)
        try:
            return super()._normalize_finished_stationary(q)
        except FinishedStationaryRoundoffError:
            # Refuse normalization. JointTrajectory.step then passes this exact
            # raw output into the same input before step() clears ONLY cache.
            if self._stop_recalculations >= MAX_CACHE_RECALCULATIONS:
                raise StopResidualReplanError('STOP residual planner exhausted eight cache recalculations')
            self._stop_pending = True

    def step(self):
        if self._stop_binding is None:
            return super().step()
        if self._stop_failed:
            raise StopResidualReplanError('Failed STOP residual adapter cannot resume')
        h = _handoff()
        try:
            if len(self._stop_events) >= h.MAX_STOP_STEPS:
                raise StopResidualReplanError('STOP residual adapter exceeded original 1176 tick budget')
            before = self._check_binding()
            objects = dict(tick=len(self._stop_events) + 1,
                before={name: id(value) for name, value in _objects(self).items()})
            self._stop_pending = False
            self._stop_raw = None
            normalizations = len(self.terminal_normalizations)
            q = super().step()
            raw = self._stop_raw or h.output_snapshot(self)
            after = self._check_binding()
            event = dict(tick=len(self._stop_events) + 1, dt=h.DT,
                elapsed_command_s=(len(self._stop_events) + 1) * h.DT,
                input_before=before, raw_output=raw, output=h.output_snapshot(self),
                native_result='Finished' if self._stop_raw is not None else 'Working',
                input_after=after, command_q=q.tolist(), target_q=self._stop_binding['fields']['target_position'],
                canonicalized=len(self.terminal_normalizations) > normalizations,
                cache_recalculated=self._stop_pending, objects_unchanged=True)
            if self._stop_pending:
                # All readbacks precede reset; reset() must not write q/v/a,
                # output clock, target, settings, or replace any live object.
                bits = _qva_bits(self)
                output = h.output_snapshot(self)
                event['cache_reset_input_before'] = after
                self.generator.reset()
                self._stop_recalculations += 1
                readback = self._check_binding()
                if bits != _qva_bits(self) or readback != after or h.output_snapshot(self) != output:
                    raise StopResidualReplanError('Native planner cache reset changed raw state/output')
                event['cache_reset_input_after'] = readback
                event['cache_reset_output_after'] = h.output_snapshot(self)
                objects['after_cache_reset'] = {name: id(value) for name, value in _objects(self).items()}
            event['cache_recalculations'] = self._stop_recalculations
            objects['after'] = {name: id(value) for name, value in _objects(self).items()}
            self._stop_events.append(event)
            self._stop_object_checks.append(objects)
            self._stop_history = (self._stop_history + [q.tolist()])[-3:]
            return q
        except Exception as error:
            self._stop_failed = True
            self._stop_failure = dict(exception_type=type(error).__name__, reason=str(error),
                attempted_tick=len(self._stop_events) + 1, raw_output=h.output_snapshot(self))
            if not isinstance(error, StopResidualReplanError):
                error = StopResidualReplanError(str(error))
            error.stop_residual_report = self.stop_residual_report()
            raise error

    def last_stop_residual_event(self):
        _need(bool(self._stop_events), 'No actual STOP residual command event exists')
        return deepcopy(self._stop_events[-1])

    def last_stop_residual_object_check(self):
        _need(bool(self._stop_object_checks), 'No actual STOP residual object check exists')
        return deepcopy(self._stop_object_checks[-1])

    def stop_residual_terminal_ready(self):
        """Stationarity alone cannot accept a repeated outside-bound residual."""
        if not self._stop_events or self._stop_failed:
            return False
        event = self._stop_events[-1]
        h = _handoff()
        try:
            if (h.digest(self._check_binding()) != h.digest(event['input_after']) or
                    h.digest(h.output_snapshot(self)) != h.digest(event['output'])):
                return False
        except (ValueError, RuntimeError):
            return False
        target = self._stop_binding['fields']['target_position']
        return bool(event['native_result'] == 'Finished' and not event['cache_recalculated'] and
            _within_roundoff(event['raw_output'], target) and
            np.asarray(event['output']['new_position'], dtype=np.float64).tobytes() ==
                np.asarray(target, dtype=np.float64).tobytes() and
            event['output']['new_velocity'] == event['output']['new_acceleration'] == [0.] * 7 and
            len(self._stop_history) == 3 and
            len({np.asarray(q, dtype=np.float64).tobytes() for q in self._stop_history}) == 1)

    def stop_residual_report(self):
        _need(self._stop_binding is not None, 'STOP residual adapter has not been activated')
        h = _handoff()
        stationary = bool(len(self._stop_history) == 3 and
            len({np.asarray(q, dtype=np.float64).tobytes() for q in self._stop_history}) == 1 and
            np.max(abs(self.velocity)) <= 1e-10 and np.max(abs(self.acceleration)) <= 1e-10)
        return dict(schema=SCHEMA, adapter_id=ADAPTER_ID, ruckig_version=RUCKIG_VERSION,
            dt=h.DT, max_cache_recalculations=MAX_CACHE_RECALCULATIONS,
            roundoff_ulps=FINISHED_ROUNDOFF_ULPS, max_stop_ticks=h.MAX_STOP_STEPS,
            planning_limits=h.STOP_PLANNING_LIMITS.copy(), initial_stop_input=deepcopy(self._stop_binding),
            initial_command_history=deepcopy(self._stop_initial_history),
            target_q=deepcopy(self._stop_binding['fields']['target_position']),
            ticks=len(self._stop_events), elapsed_command_s=len(self._stop_events) * h.DT,
            cache_recalculations=self._stop_recalculations, events=deepcopy(self._stop_events),
            events_sha256=h.digest(self._stop_events), source_sha256=deepcopy(self._stop_sources),
            object_checks=deepcopy(self._stop_object_checks),
            object_checks_sha256=h.digest(self._stop_object_checks),
            object_ids={name: id(value) for name, value in self._stop_objects.items()},
            objects_unchanged=all(value is _objects(self)[name] for name, value in self._stop_objects.items()),
            failed=self._stop_failed, failure=deepcopy(self._stop_failure),
            three_stationary_commands=stationary, terminal_ready=self.stop_residual_terminal_ready(), training_eligible=False)


def verify_report(report, commands):
    """Check evidence without trusting its event digest or reported counters."""
    return _verify_report(report, commands, expected_sources=_source_hashes())


def _verify_report(report, commands, *, expected_sources):
    """Check evidence without trusting its event digest or reported counters."""
    h = _handoff()
    _need(report.get('schema') == SCHEMA and report.get('adapter_id') == ADAPTER_ID and
          report.get('ruckig_version') == RUCKIG_VERSION and report.get('dt') == h.DT and
          report.get('max_cache_recalculations') == MAX_CACHE_RECALCULATIONS and
          report.get('roundoff_ulps') == FINISHED_ROUNDOFF_ULPS and
          report.get('planning_limits') == h.STOP_PLANNING_LIMITS and
          report.get('max_stop_ticks') == h.MAX_STOP_STEPS and
          report.get('objects_unchanged') is True and report.get('failed') is False and
          report.get('failure') is None and report.get('training_eligible') is False and
          report.get('source_sha256') == expected_sources, 'STOP residual adapter protocol/source changed')
    events = report.get('events', [])
    commands = np.asarray(commands, dtype=np.float64)
    _need(commands.shape == (len(events), 7) and np.isfinite(commands).all() and
          1 <= len(events) <= h.MAX_STOP_STEPS and report.get('ticks') == len(events) and
          report.get('elapsed_command_s') == len(events) * h.DT and
          report.get('events_sha256') == h.digest(events), 'STOP residual tick/command/event digest differs')
    ids = report.get('object_ids', {})
    _need(set(ids) == {'trajectory', 'input', 'output', 'generator'} and
          all(type(i) is int and i > 0 for i in ids.values()) and len(set(ids.values())) == 4,
          'Distinct unchanged STOP objects required')
    checks = report.get('object_checks', [])
    _need(len(checks) == len(events) and report.get('object_checks_sha256') == h.digest(checks),
          'STOP per-tick object evidence digest/count differs')
    for tick, (check, event) in enumerate(zip(checks, events), 1):
        _need(check.get('tick') == tick and check.get('before') == check.get('after') == ids and
              (check.get('after_cache_reset') == ids if event['cache_recalculated'] else
               'after_cache_reset' not in check), 'STOP object identity changed within a command/cache reset')
    initial = report['initial_stop_input']
    target = report['target_q']
    _need(initial['fields']['target_position'] == initial['fields']['current_position'] == target,
          'Original STOP current-position target differs')
    count = 0
    prior = initial
    for tick, (event, q) in enumerate(zip(events, commands), 1):
        _need(event.get('tick') == tick and event.get('dt') == h.DT and
              event.get('elapsed_command_s') == tick * h.DT and event.get('objects_unchanged') is True and
              event.get('target_q') == target and event.get('input_before') == prior and
              np.asarray(event.get('command_q'), dtype=np.float64).tobytes() == q.tobytes(),
              'STOP residual event tick/target/command continuity differs')
        out = event['output']; after = event['input_after']; raw = event['raw_output']
        _need(out['new_position'] == q.tolist() and out['time'] > 0 and raw['time'] > 0,
              'STOP residual command is not a positive-time native output')
        for current, new in [('current_position', 'new_position'), ('current_velocity', 'new_velocity'),
                             ('current_acceleration', 'new_acceleration')]:
            _need(np.asarray(after['fields'][current], dtype=np.float64).tobytes() ==
                  np.asarray(out[new], dtype=np.float64).tobytes(), 'STOP native raw output was not passed to input')
        _need(all(after['fields'][key] == value for key, value in initial['fields'].items()
                  if key not in ('current_position', 'current_velocity', 'current_acceleration')),
              'Original STOP input settings changed')
        within = _within_roundoff(raw, target)
        if event['canonicalized']:
            _need(event['native_result'] == 'Finished' and within and out['new_position'] == target and
                  out['new_velocity'] == out['new_acceleration'] == [0.] * 7,
                  'STOP canonicalization exceeded the unchanged 64-ULP bound')
        if event['cache_recalculated']:
            count += 1
            _need(event['native_result'] == 'Finished' and not within and event['canonicalized'] is False and raw == out and
                  event.get('cache_reset_input_before') == event.get('cache_reset_input_after') == after and
                  event.get('cache_reset_output_after') == out, 'Cache reset changed authentic raw STOP state')
        _need(event['cache_recalculations'] == count <= MAX_CACHE_RECALCULATIONS,
              'STOP cache recalculation count/cap differs')
        prior = after
    _need(report.get('cache_recalculations') == count, 'STOP cache recalculation total differs')
    history = (report.get('initial_command_history', []) + commands.tolist())[-3:]
    stationary = bool(len(history) == 3 and
        len({np.asarray(q, dtype=np.float64).tobytes() for q in history}) == 1 and
        np.max(abs(np.asarray(events[-1]['output']['new_velocity']))) <= 1e-10 and
        np.max(abs(np.asarray(events[-1]['output']['new_acceleration']))) <= 1e-10)
    _need(report.get('three_stationary_commands') is True and stationary,
          'Original three bit-identical stationary commands are required')
    last = events[-1]
    _need(report.get('terminal_ready') is True and last['native_result'] == 'Finished' and
          not last['cache_recalculated'] and _within_roundoff(last['raw_output'], target) and
          np.asarray(last['output']['new_position'], dtype=np.float64).tobytes() == np.asarray(target, dtype=np.float64).tobytes() and
          last['output']['new_velocity'] == last['output']['new_acceleration'] == [0.] * 7,
          'Native STOP endpoint has not met the original 64-ULP bound/exact readback')
    return report


def replay_report(input_state, history, report, verify_command):
    """Independently recompute commands/events with the pinned native planner.

    The caller supplies the original geometry/posture/coarse-tube verifier.
    This is read-only numerical replay, never physics or training admission.
    """
    h = _handoff()
    _need(np.array_equal(report.get('initial_command_history'), history), 'Original STOP replay history differs')
    live = h.restore_trajectory(input_state, StopResidualTrajectory)
    h.apply_stop_profile(live, expected_input=input_state)
    live.set_target(input_state['fields']['current_position'])
    live.begin_stop_residual_replanning(history=history)
    seam = np.asarray(history, dtype=np.float64)
    commands = []
    from sim.auscultation_skill_v1.executor import command_derivatives
    for _ in range(report['ticks']):
        q = live.step()
        values = command_derivatives(seam, q)
        _need(values['velocity'] <= .175 and values['acceleration'] <= .5 + 1e-8 and
              values['jerk'] <= 1.5 + 1e-8, 'Original STOP derivative seam failed during replay')
        verify_command(q)
        commands.append(q.copy())
        seam = np.vstack([seam[1:], q])
    verify_report(report, commands)
    actual = live.stop_residual_report()
    _need(actual['events'] == report['events'] and actual['events_sha256'] == report['events_sha256'] and
          actual['three_stationary_commands'] == report['three_stationary_commands'],
          'Independent native STOP residual replay differs')
    return actual


def _native_configuration_and_sources(config, run, metadata):
    _need(CONFIG_KEY in config, 'Explicit native STOP residual configuration is required')
    native_options(config, metadata=metadata)
    actual = run.get('runtime_sources_sha256', {})
    _need(all(actual.get(path) == sha for path, sha in _source_hashes().items()),
          'Actual native STOP adapter source inventory differs')


def verify_native_parking(run, receipt, audit, metadata, reference, phase_config, *, config, kin=None):
    """Native-only opt-in: original physical parking checks plus exact adapter proof.

    Legacy dataset readers call their unchanged strict entrypoint. This route
    supplies a single known native adapter, never a caller-provided callback.
    """
    _native_configuration_and_sources(config, run, metadata)
    h = _handoff()
    preview = receipt.get('parking_preview', {})
    _need(preview.get('schema') == PREVIEW_SCHEMA, 'Explicit residual STOP preview schema required')
    proposed = preview.get('stop_residual_adapter', {})
    actual = receipt.get('parking_stop_residual_adapter', {})
    commands = np.asarray(preview.get('commands'), dtype=np.float64)
    verify_report(proposed, commands)
    verify_report(actual, commands)
    _need(proposed['events'] == actual['events'] and proposed['events_sha256'] == actual['events_sha256'] and
          proposed['initial_stop_input'] == actual['initial_stop_input'] == preview.get('stop_input_state') and
          proposed['initial_command_history'] == actual['initial_command_history'] == preview.get('frozen_command_history'),
          'Preview/live native STOP residual state/event proof differs')
    ids = actual['object_ids']
    _need(ids == preview.get('live_state_before', {}).get('interpolator_object_ids') ==
          receipt.get('parking_tail', {}).get('object_ids') and
          not set(ids.values()) & set(proposed['object_ids'].values()),
          'Native STOP objects differ from the original live objects or alias preview objects')
    start = receipt.get('parking_start_physics_row')
    _need(type(start) is int and 0 <= start and start + len(commands) <= len(audit),
          'Actual native STOP event interval differs')
    for index, (event, objects) in enumerate(zip(actual['events'], actual['object_checks'])):
        row = audit[start + index]
        raw = row.get('raw_intent') or {}
        evidence = row.get('stop_residual_adapter_tick') or {}
        interpolator = row.get('interpolator') or {}
        _need(row.get('actual_control_owner') == 'L2_STOP' and raw.get('setter', {}).get('offgrid_stop') is True and
              evidence.get('adapter_id') == ADAPTER_ID and evidence.get('event') == event and
              evidence.get('event_sha256') == h.digest(event) and evidence.get('objects') == objects,
              'Actual native STOP event/digest/object/owner evidence differs')
        _need(interpolator.get('input') == event['input_after'] and interpolator.get('output') == event['output'] and
              interpolator.get('trajectory_object') == ids['trajectory'] and
              interpolator.get('objects') == {key: ids[key] for key in ('input', 'output', 'generator')} and
              interpolator.get('planning_profile') == 'stop',
              'Actual native STOP interpolator readback differs from residual event')
    from . import m2_given_contact_dataset as dataset
    dataset._interior_parking_checks(run, receipt, audit, metadata, reference, phase_config,
                                    kin=kin, _native_stop_residual=True)
    return dict(adapter_id=ADAPTER_ID, preview_live_events_equal=True,
                events_sha256=actual['events_sha256'], ticks=actual['ticks'],
                cache_recalculations=actual['cache_recalculations'], original_parking_gates_passed=True)


def verify_native_physical(run, audit, metadata, *, config):
    """All original dataset physical gates with the explicit native adapter route."""
    _native_configuration_and_sources(config, run, metadata)
    from . import m2_given_contact_dataset as dataset
    return dataset._physical_checks(run, audit, metadata, _native_stop_config=config)
