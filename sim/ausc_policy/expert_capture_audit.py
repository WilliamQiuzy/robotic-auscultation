"""Read-only raw expert evidence audit, with explicit prefix or complete scope.

Example (prints JSON; never rewrites receipts or creates dataset eligibility)::

    python -B -m sim.ausc_policy.expert_capture_audit RUN --mode complete \
        --recorded-project-root /recorded/project --source-snapshot-root SNAPSHOT

An optional --baseline compares bound current-geometry physics, including all
seven force channels and the recorded retainer front audit. Differences are
reported independently of raw capture consistency. Neither mode qualifies
contact physics, demonstrations, sensor use by a controller, or training.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .geometry_epoch import require_current, require_geometry_artifacts
from .m2_capture import APPROVED_CAMERA_SHA256, SCHEMA as RGBD_SCHEMA
from .m2_sensor_probe import CALIBRATION_SHA256
from .m2_sensors import ARTIFACTS, SCHEMA as HALL_SCHEMA
from .retainer_contact import ALL_FORCE_GROUPS, CONTACT_SURFACE


PHYSICS_FIELDS = ('time_s', 'q', 'dq', 'q_command', 'compression_m', 'tcp_world', 'axis_world') + tuple(
    key+'_N' for key in ALL_FORCE_GROUPS)
CONTEXT_FIELDS = ('phase', 'target', 'subphase', 'retainer_front_audit', 'contact_surface')
SOURCE_MODULES = dict(
    hall=dict(stream='sim.ausc_policy.m2_sensors', calibration='sim.auscultation_vla_v2.hall',
              simulator='sim.auscultation_vla_v2.hall',
              host_calibration='auscultation.mechanical_sensor.force_feedback.force_sensor'),
    rgbd=dict(capture='sim.ausc_policy.m2_capture', backend='sim.ausc_policy.m2_capture',
              environment='sim.ausc_policy.fresh_expert', hall_stream='sim.ausc_policy.m2_sensors',
              calibration='sim.auscultation_vla_v2.hall',
              host_calibration='auscultation.mechanical_sensor.force_feedback.force_sensor',
              geometry_gate='sim.ausc_policy.geometry_epoch'))


def _check(value, message):
    if not value:
        raise ValueError(message)


def _decode(text):
    def unique(items):
        result = {}
        for key, value in items:
            _check(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def _read(path):
    value = _decode(Path(path).read_text())
    _check(isinstance(value, dict), 'JSON record required: '+str(path))
    return value


def _lines(path):
    with Path(path).open() as stream:
        result = [_decode(line) for line in stream]
    _check(all(isinstance(row, dict) for row in result), 'JSONL records required')
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file(root, name):
    _check(isinstance(name, str) and Path(name).name == name, 'Invalid evidence filename')
    path = (root/name).resolve()
    _check(root.resolve() in path.parents and path.is_file(), 'Evidence path outside directory or missing')
    return path


def _unqualified(record):
    _check(record.get('training_eligible') is False, 'Unexpected training eligibility')
    if 'physical_qualification' in record:
        _check(record['physical_qualification'] is False, 'Unexpected physical qualification')


class _Sources:
    """Explicit lexical recorded-root mapping; no fallback to changed live files."""
    def __init__(self, recorded_root, snapshot_root):
        _check((recorded_root is None) == (snapshot_root is None), 'Both source mapping roots are required')
        self.recorded = None if recorded_root is None else Path(recorded_root)
        self.snapshot = None if snapshot_root is None else Path(snapshot_root).resolve()
        if self.recorded is not None:
            _check(self.recorded.is_absolute() and '..' not in self.recorded.parts,
                   'Recorded project root must be absolute')
            _check(self.snapshot.is_dir(), 'Source snapshot root is missing')
        self.verified = {}

    def verify(self, recorded, expected, module=None):
        _check(isinstance(recorded, str) and Path(recorded).is_absolute() and '..' not in Path(recorded).parts,
               'Invalid recorded source path')
        path = Path(recorded)
        if self.snapshot is not None:
            try:
                path = self.snapshot/path.relative_to(self.recorded)
            except ValueError as error:
                raise ValueError('Recorded source outside mapped project root: '+recorded) from error
            _check(self.snapshot in path.resolve().parents, 'Mapped source escapes snapshot')
        if module is not None:
            _check(isinstance(module, str) and module and all(part.isidentifier() for part in module.split('.')),
                   'Invalid source module identity')
            suffix = module.replace('.', '/')
            _check(path.as_posix().endswith('/'+suffix+'.py') or
                   path.as_posix().endswith('/'+suffix+'/__init__.py'), 'Source module/path identity mismatch')
        try:
            actual = _sha(path)
        except OSError as error:
            raise ValueError('Source unavailable; provide explicit source snapshot mapping: '+recorded) from error
        _check(actual == expected, 'Source SHA mismatch: '+recorded)
        self.verified[recorded] = dict(path=str(path), sha256=actual, module=module)

    def bindings(self, manifest, category):
        bindings = manifest.get('source_bindings')
        modules = SOURCE_MODULES[category]
        _check(isinstance(bindings, dict) and set(bindings) == set(modules),
               'Missing/unexpected acquisition source roles')
        _check(type(manifest.get('injected_backend')) is bool, 'Missing acquisition backend identity')
        for role, record in bindings.items():
            # The recorder deliberately accepts injected camera/environment
            # implementations. Fixed recorder roles and native sources cannot
            # exchange otherwise self-consistent module/path/hash records.
            injected = category == 'rgbd' and manifest['injected_backend'] and role in ('backend', 'environment')
            _check(isinstance(record, dict) and (injected or record.get('module') == modules[role]),
                   'Acquisition source role/module mismatch: '+role)
            self.verify(record['path'], record['sha256'], record['module'])

    def runtime(self, bindings):
        _check(isinstance(bindings, dict) and bindings, 'Missing runtime source bindings')
        for path, digest in bindings.items():
            self.verify(path, digest)


def _replay_hall(directory, source_root=None):
    """Uses archived implementations in mapped mode; this function writes nothing."""
    from sim.ausc_policy.m2_sensors import (
        HallCalibration, HallSimulator, _source_identity, _sources, audit_hall_stream)
    if source_root is not None:
        # An incomplete snapshot must not fall through to an editable install
        # or another checkout on site-packages' search path, even with matching
        # entrypoint bytes. Check every imported project dependency as well.
        root = Path(source_root).resolve()
        for name, module in list(sys.modules.items()):
            if name.startswith(('sim.', 'auscultation.')) and getattr(module, '__file__', None):
                _check(root in Path(module.__file__).resolve().parents,
                       'Project replay import outside source snapshot: '+name)
    directory = Path(directory)
    metadata = _read(directory/'manifest.json')
    receipt = _read(directory/'receipt.json')
    calibration = HallCalibration((directory/'calibration.json').read_bytes())
    _check(_source_identity(_sources(HallCalibration, HallSimulator, calibration)) ==
           _source_identity(metadata['source_bindings']), 'Hall replay source identity differs from acquisition')
    # Reuse the existing complete-stream audit without altering failed receipts.
    if receipt.get('completed') is True:
        audit_hall_stream(directory)
    simulator = HallSimulator(calibration, **metadata['parameters'])
    physical, hall, adc = [_lines(directory/name) for name in
                           ('physics.jsonl', 'hall.jsonl', 'adc_acquisition.jsonl')]
    _check(len(physical) == len(hall) == receipt['physics_rows'] == receipt['hall_rows'],
           'Unequal physics/Hall ledger lengths')
    _check(len(adc) == receipt['unique_packets'] == receipt['adc_acquisitions'], 'ADC receipt count mismatch')
    last = None
    count = 0
    packet_id = None
    for i, (row, observed) in enumerate(zip(physical, hall)):
        _check(row['physics_row'] == i and abs(row['time_s']-(i+1)/240) < 1e-8,
               'Hall physics clock mismatch')
        _check(row.get('training_eligible') is False, 'Hall physics promotion')
        simulator.update(row['time_s'], row['compression_m'])
        packet = simulator.read(row['time_s'])
        actual = None if packet is None else asdict(packet)
        new = actual is not None and actual != last
        if new:
            packet_id = 'adc'+str(count).zfill(8)
            expected = dict(acquisition_id=packet_id, physics_row=i, time_s=row['time_s'],
                            packet=actual, valid_json_packet=True, training_eligible=False)
            _check(count < len(adc) and adc[count] == expected, 'ADC replay mismatch')
            count += 1
        decoded = None if packet is None else asdict(calibration.decode(packet, now_s=row['time_s']))
        expected = dict(physics_row=i, time_s=row['time_s'], packet_id=packet_id if actual is not None else None,
                        acquisition_id=packet_id if actual is not None else None, packet=actual if new else None,
                        is_new_packet=new, decoded=decoded, status='pending' if actual is None else 'available',
                        error=None, training_eligible=False)
        _check(observed == expected, 'Decoded Hall replay mismatch')
        last = actual
    _check(count == len(adc), 'Unreferenced ADC packets')
    return dict(passed=True, physics_rows=len(physical), adc_packets=len(adc), training_eligible=False)


def _hall_replay(directory, sources):
    if sources.snapshot is None:
        return dict(_replay_hall(directory), implementation='current_hash_verified_sources')
    # No module substitution in the parent process, imports from live checkout,
    # bytecode cache writes, receipt rewriting, GPU or native simulator imports.
    script = 'import json, sys\nfrom pathlib import Path\nfrom dataclasses import asdict\n'
    script += '\n'.join(inspect.getsource(fn) for fn in (_check, _decode, _read, _lines, _replay_hall))
    script += '\nprint(json.dumps(_replay_hall(sys.argv[1], source_root=sys.argv[2]), allow_nan=False))\n'
    env = dict(os.environ, PYTHONPATH=str(sources.snapshot), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(directory.resolve()), str(sources.snapshot)],
                            cwd=sources.snapshot, env=env, text=True, capture_output=True)
    _check(result.returncode == 0, 'Snapshot Hall replay failed: '+result.stderr.strip())
    return dict(_decode(result.stdout), implementation='source_snapshot')


def _run_trace(run):
    run = Path(run).resolve()
    owner = _read(run/'result.json')
    _check(owner.get('completed') is True and owner.get('gpu_released') is True, 'Run owner is not finished/released')
    review = run/'output/review'
    metadata = _read(review/'fresh_expert_inputs/manifest.json')
    require_current(metadata)
    _check(metadata.get('contact_surface') == CONTACT_SURFACE, 'Wrong recorded contact surface')
    artifacts = metadata['geometry_artifacts']
    require_geometry_artifacts(review/'geometry', dict(metadata, **artifacts))
    result = _read(review/'fresh_expert_result.json')
    for key in ('geometry_id', 'geometry_source_sha256', 'geometry_artifacts', 'contact_surface'):
        _check(result.get(key) == metadata.get(key), 'Result geometry binding mismatch: '+key)
    path = review/'fresh_expert/fresh_expert_trace.jsonl'
    digest = _sha(path)
    _check(result.get('trace_sha256') == digest, 'Physics trace hash mismatch')
    trace = _lines(path)
    _check(bool(trace) and len(trace) == result.get('physics_steps'), 'Empty or incorrect physics trace count')
    for row in trace:
        for key in PHYSICS_FIELDS:
            _check(key in row and np.isfinite(np.asarray(row[key], float)).all(), 'Missing/nonfinite physics field: '+key)
        _check(row.get('contact_surface') == CONTACT_SURFACE, 'Trace contact surface mismatch')
        _check(isinstance(row.get('retainer_front_audit'), dict), 'Missing retainer front audit record')
        _check(all(key in row for key in CONTEXT_FIELDS), 'Missing physics context field')
    return review, metadata, result, trace, digest


def _compare(run, baseline):
    _, left_geometry, _, left, _ = _run_trace(run)
    _, right_geometry, _, right, digest = _run_trace(baseline)
    for key in ('geometry_id', 'geometry_source_sha256', 'geometry_artifacts', 'contact_surface'):
        _check(left_geometry[key] == right_geometry[key], 'Baseline geometry/artifact/contact binding differs: '+key)
    fields = PHYSICS_FIELDS + CONTEXT_FIELDS
    differing = set()
    maxima = {key: 0. for key in PHYSICS_FIELDS}
    for a, b in zip(left, right):
        for key in fields:
            if a[key] != b[key]:
                differing.add(key)
            if key in maxima:
                x, y = np.asarray(a[key], float), np.asarray(b[key], float)
                _check(x.shape == y.shape, 'Baseline physics shape mismatch: '+key)
                maxima[key] = max(maxima[key], float(np.max(np.abs(x-y))))
    return dict(matched_physics_rows=len(left) == len(right), compared_rows=min(len(left), len(right)),
                compared_fields=list(fields), differing_fields=sorted(differing),
                maximum_absolute_differences=maxima,
                all_compared_physics_exact=len(left) == len(right) and not differing,
                baseline_trace_sha256=digest)


def _attempt_time(text):
    """Read the recorder's numeric repr without evaluating arbitrary code.

    Recent NumPy versions wrap scalar reprs as np.float64(...). The archived
    attempt preserves repr rather than dtype, so use the existing acquisition
    clock tolerance when comparing its numeric value with the saved frame.
    """
    try:
        node = ast.parse(text, mode='eval').body
        if isinstance(node, ast.Call):
            _check(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
                   node.func.value.id in ('np', 'numpy') and node.func.attr in
                   ('float16', 'float32', 'float64', 'longdouble', 'int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64') and len(node.args) == 1 and not node.keywords,
                   'Invalid camera attempt time representation')
            node = node.args[0]
        value = ast.literal_eval(node)
        _check(type(value) in (int, float) and np.isfinite(value), 'Invalid camera attempt time')
        return float(value)
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError('Invalid camera attempt time representation') from error


def _attempts(raw, frames, receipt, complete):
    attempts = _lines(raw/'rgbd/attempts.jsonl')
    groups = {}
    for record in attempts:
        _unqualified(record)
        index = record['attempt']
        _check(type(index) is int and index >= 0, 'Invalid camera attempt index')
        if index not in groups:
            _check(index == len(groups) and
                   (not groups or list(groups.values())[-1][-1]['status'] == 'complete'),
                   'Camera attempt cannot follow an unfinished/failed attempt')
        group = groups.setdefault(index, [])
        _check((not group and record['status'] == 'started') or
               (len(group) == 1 and record['status'] in ('complete', 'error')), 'Invalid camera attempt lifecycle')
        group.append(record)
    _check(list(groups) == list(range(receipt['attempts'])), 'Camera attempt receipt count mismatch')
    if complete:
        _check(len(groups) == len(frames) and all(len(g) == 2 and g[1]['status'] == 'complete' for g in groups.values()),
               'complete audit requires every camera attempt to complete')
    bound_frames = set()
    identity = ('attempt', 'index', 'physics_row', 'supplied_time_s', 'phase', 'target')
    for ordinal, group in enumerate(groups.values()):
        started = group[0]
        last = group[-1]
        _check(all(key in started for key in identity) and type(started['index']) is int and
               started['index'] == ordinal, 'Camera attempt/frame index is not consecutive')
        _check(all(key in last and last[key] == started[key] for key in identity) and
               type(last['index']) is int, 'Camera attempt started/terminal identity differs')
        index = started['index']
        if last['status'] == 'complete':
            _check(index < len(frames), 'Completed attempt has no frame')
        else:
            _check(not complete and ordinal == len(groups)-1 and index in (len(frames)-1, len(frames)),
                   'Only the last prefix attempt may be incomplete')
        if index < len(frames):
            _check(index not in bound_frames, 'Multiple camera attempts claim the same frame')
            bound_frames.add(index)
            frame = frames[index]
            _check(started['physics_row'] == repr(frame['physics_row']) and
                   abs(_attempt_time(started['supplied_time_s'])-frame['time_s']) <= 1e-8 and
                   started['phase'] == repr(frame['phase']) and started['target'] == repr(frame['target']),
                   'Camera attempt/frame physics identity mismatch')
            for key in ('raw_file', 'raw_sha256'):
                if last['status'] == 'complete' or key in last:
                    _check(last.get(key) == frame[key], 'Camera attempt/frame hash mismatch')
    _check(bound_frames == set(range(len(frames))), 'Saved frame has no unique camera attempt')
    return len(groups)


def _error_markers(record):
    """Known recorder error fields only; do not infer errors from other data."""
    markers = {}
    for prefix, values in (('', record), ('context.', record.get('context', {}))):
        if not isinstance(values, dict):
            continue
        if values.get('status') == 'error':
            markers[prefix+'status'] = 'error'
        if values.get('tick_error') is not None:
            markers[prefix+'tick_error'] = values['tick_error']
    return markers


def _events(raw, trace, frames, complete):
    """Check all tick records, including ticks without a scheduled camera frame.

    The acquisition schema does not bind this ledger in a receipt hash. Its
    current digest is reported as audit provenance, never an original receipt.
    Prefixes retain error records and may lack finalization of a physics tick.
    """
    path = raw/'events.jsonl'
    _check(path.is_file() or not complete, 'complete audit requires the tick events ledger')
    records = _lines(path) if path.is_file() else []
    ticks, errors, tick_errors = {}, [], []
    for record in records:
        _unqualified(record)
        if record.get('status') == 'error' and 'physics_row' not in record:
            _check(isinstance(record.get('stage'), str) and isinstance(record.get('error'), str),
                   'Invalid recorded error event')
            _check(not complete, 'complete audit rejects recorded error events')
            errors.append(record)
            continue
        _check('status' not in record or record['status'] == 'error', 'Unexpected tick event status')
        index = record.get('physics_row')
        _check(type(index) is int and 0 <= index < len(trace) and
               index == len(ticks), 'Invalid/nonconsecutive tick event physics row')
        measured = trace[index]
        _check(record.get('time_s') == measured['time_s'] and
               type(record.get('camera_due')) is bool and record['camera_due'] == (index % 12 == 11),
               'Tick event clock/camera schedule mismatch')
        context = record.get('context')
        _check(isinstance(context, dict) and 'tick_error' in context, 'Missing tick event context/error marker')
        before, after = context.get('controller_before'), context.get('controller_after')
        _check(isinstance(before, dict) and isinstance(after, dict) and
               context.get('physics_labels') == {key: measured.get(key) for key in ('phase', 'target', 'subphase')} and
               all(before.get(key) == after.get(key) == measured[key] for key in ('target', 'subphase')),
               'Tick event physics context mismatch')
        if _error_markers(record):
            _check(not complete, 'complete audit rejects recorded tick errors')
            tick_errors.append(index)
            errors.append(dict(physics_row=index, markers=_error_markers(record)))
        ticks[index] = record
    if complete:
        _check(list(ticks) == list(range(len(trace))), 'complete audit requires every physics tick event')
    for frame in frames:
        event = ticks.get(frame['physics_row'])
        _check(event is not None and event['camera_due'] and frame['context'] == event['context'],
               'RGB-D frame/tick event context mismatch')
    return dict(tick_records=len(ticks), missing_physics_tick_count=len(trace)-len(ticks),
                tick_error_rows=tick_errors, error_records=errors,
                sha256=_sha(path) if path.is_file() else None, original_receipt_hash_bound=False,
                scope='Recorded tick consistency; this schema supplies no acquisition receipt hash for events.jsonl')


def audit_capture(run, *, mode, baseline=None, recorded_project_root=None, source_snapshot_root=None):
    """Return a new audit report; original bytes and eligibility remain unchanged.

    Prefix audits may retain caller failures and missing final camera attempts.
    They still require deterministic replay of every saved Hall/physics row.
    Complete audits additionally require the original complete raw
    capture receipts and all scheduled frames. Neither reruns physical gates.
    """
    _check(mode in ('prefix', 'complete'), 'mode must be explicitly prefix or complete')
    complete = mode == 'complete'
    sources = _Sources(recorded_project_root, source_snapshot_root)
    review, geometry, result, trace, trace_digest = _run_trace(run)
    raw = review/'raw_multimodal'
    manifest, top = _read(raw/'manifest.json'), _read(raw/'receipt.json')
    _check(manifest['geometry'] == geometry, 'Raw capture geometry binding mismatch')
    _check(top == result.get('raw_multimodal_capture'), 'Top capture receipt disagrees with original result')
    _check(result.get('raw_capture_complete') == top.get('raw_capture_complete'), 'Top completion flags disagree')
    config = _read(Path(run)/'run.json')['config']['fresh_expert']['raw_multimodal_capture']
    _check(manifest['binding'] == config and top['episode_id'] == config['episode_id'], 'Capture run/episode binding mismatch')
    _check(top['runtime_sources_sha256'] == manifest['runtime_sources_sha256'], 'Top runtime source bindings disagree')
    sources.runtime(manifest['runtime_sources_sha256'])
    if result.get('runtime_sources_sha256'):
        sources.runtime(result['runtime_sources_sha256'])
    for name, expected in [('approved_camera.json', APPROVED_CAMERA_SHA256), ('calibration.json', CALIBRATION_SHA256)]:
        _check(_sha(raw/name) == expected, 'Pinned raw source bytes mismatch: '+name)
    _unqualified(manifest)
    _unqualified(top)
    _unqualified(result)
    receipts, manifests = {}, {}
    for category, top_key in [('hall', 'hall'), ('rgbd', 'camera')]:
        directory = raw/category
        receipt, stream = _read(directory/'receipt.json'), _read(directory/'manifest.json')
        receipts[category], manifests[category] = receipt, stream
        schema = HALL_SCHEMA if category == 'hall' else RGBD_SCHEMA
        _check(receipt.get('schema') == stream.get('schema') == schema, 'Unsupported acquisition stream schema')
        _check(top['receipts'].get(top_key) == receipt, 'Original stream receipt differs from top receipt')
        expected_names = set(ARTIFACTS if category == 'hall' else
                             ('manifest.json', 'approved_camera.json', 'calibration.json', 'frames.jsonl', 'attempts.jsonl'))
        _check(set(receipt['artifacts_sha256']) == expected_names, 'Missing/unexpected stream receipt artifacts')
        for name, digest in receipt['artifacts_sha256'].items():
            _check(_sha(_file(directory, name)) == digest, 'Stream artifact hash mismatch: '+name)
        _unqualified(receipt)
        _unqualified(stream)
        _check(stream['episode_id'] == config['episode_id'] and stream['geometry'] == geometry,
               'Stream episode/geometry mismatch')
        _check(stream['calibration_sha256'] == CALIBRATION_SHA256 and
               _sha(directory/'calibration.json') == CALIBRATION_SHA256, 'Stream calibration mismatch')
        sources.bindings(stream, category)
    hall_receipt, camera_receipt = receipts['hall'], receipts['rgbd']
    if complete:
        _check(top.get('raw_capture_complete') is True and
               top.get('errors') == [] and result.get('errors') == [] and
               hall_receipt.get('completed') is True and hall_receipt.get('failure') is None and
               hall_receipt.get('close_errors') == [] and camera_receipt.get('completed') is True and
               camera_receipt.get('failures') == [] and camera_receipt.get('receipt_written') is True and
               top.get('receipts', {}).get('hall_replay', {}).get('passed') is True,
               'complete audit requires original complete/error-free top, Hall and camera receipts')
    hall_hash = _sha(raw/'hall/manifest.json')
    _check(manifests['rgbd']['hall_stream'] == dict(snapshot=manifests['hall'], manifest_sha256=hall_hash),
           'RGB-D Hall source identity mismatch')
    for camera_role, hall_role in (('hall_stream', 'stream'), ('calibration', 'calibration'),
                                   ('host_calibration', 'host_calibration')):
        camera_source = manifests['rgbd']['source_bindings'][camera_role]
        hall_source = manifests['hall']['source_bindings'][hall_role]
        _check(all(camera_source[key] == hall_source[key] for key in ('module', 'sha256')),
               'RGB-D Hall acquisition source identity mismatch: '+camera_role)
    _check(manifests['rgbd']['camera_sha256'] == APPROVED_CAMERA_SHA256 and
           _sha(raw/'rgbd/approved_camera.json') == APPROVED_CAMERA_SHA256, 'RGB-D camera source mismatch')
    replay = _hall_replay(raw/'hall', sources)
    physical, hall, adc = [_lines(raw/'hall'/name) for name in ('physics.jsonl', 'hall.jsonl', 'adc_acquisition.jsonl')]
    _check(len(physical) == len(trace) == top['physics_steps'], 'Physics/Hall/trace count mismatch')
    for i, (row, measured) in enumerate(zip(physical, trace)):
        _check(row['physics_row'] == i and row['time_s'] == measured['time_s'] and
               row['compression_m'] == measured['compression_m'], 'Hall/trace physics binding mismatch')
    packets = {row['acquisition_id']: row['packet'] for row in adc}
    frames = _lines(raw/'rgbd/frames.jsonl')
    _check(len(frames) == camera_receipt['frames'] and len(frames) <= len(trace)//12 and
           top['expected_frames'] == len(trace)//12, 'RGB-D count mismatch')
    if complete:
        _check(len(frames) == len(trace)//12, 'complete audit requires all scheduled RGB-D frames')
    ratios, raw_files, frame_errors = [], set(), []
    for j, frame in enumerate(frames):
        i = 11+12*j
        measured, h = trace[i], hall[i]
        context = frame['context']
        markers = _error_markers(frame)
        if markers:
            _check(not complete, 'complete audit rejects recorded frame/tick/context errors')
            frame_errors.append(dict(physics_row=i, markers=markers))
        before, after = context['controller_before'], context['controller_after']
        phase = after['phase'] if context['state_audit'] == 'completed' and not before['recovery'] else measured['phase']
        _check(type(frame['index']) is int and frame['index'] == j and
               frame['physics_row'] == i and frame['time_s'] == measured['time_s'],
               'RGB-D physics clock mismatch')
        _check(frame['hall_stream_manifest_sha256'] == hall_hash, 'Frame Hall stream identity mismatch')
        _check(context['physics_labels'] == {key: measured.get(key) for key in ('phase', 'target', 'subphase')} and
               all(before[key] == after[key] == measured[key] for key in ('target', 'subphase')), 'RGB-D context mismatch')
        _check(frame['q'] == measured['q'] and frame['dq'] == measured['dq'] and
               frame['phase'] == phase and frame['target'] == measured['target'], 'RGB-D measured state mismatch')
        _check(frame['render_physics_delta_s'] == 0 and frame['hall_physics'] == physical[i], 'RGB-D render/Hall step mismatch')
        expected = dict(h)
        if h['status'] == 'available':
            expected['packet'] = packets[h['packet_id']]
        _check(frame['hall'] == expected and frame['packet_id'] == h['packet_id'] and
               frame['acquisition_id'] == h['acquisition_id'], 'RGB-D full Hall packet mismatch')
        _unqualified(frame)
        _check(frame['raw_file'] == 'frame-'+str(j).zfill(8)+'.npz', 'RGB-D raw filename/frame index mismatch')
        path = _file(raw/'rgbd', frame['raw_file'])
        _check(path not in raw_files, 'Multiple RGB-D frames reference the same raw file')
        raw_files.add(path)
        _check(_sha(path) == frame['raw_sha256'], 'Raw RGB-D bytes mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            rgb, depth, valid = arrays['rgb'], arrays['depth_m'], arrays['depth_valid']
            _check(rgb.shape in ((360, 640, 3), (360, 640, 4)) and rgb.dtype == np.uint8, 'RGB format mismatch')
            _check(depth.shape == valid.shape == (360, 640) and depth.dtype.kind == 'f' and valid.dtype == bool,
                   'Depth format mismatch')
            _check(np.array_equal(valid, np.isfinite(depth) & (depth > 0)), 'Depth validity mask mismatch')
            for key in ('training_eligible', 'physical_qualification'):
                _check(arrays[key].shape == () and arrays[key].dtype == bool and not bool(arrays[key]), 'Raw NPZ promotion')
            ratios.append(float(valid.mean()))
    attempts = _attempts(raw, frames, camera_receipt, complete)
    events = _events(raw, trace, frames, complete)
    return dict(schema='expert_raw_capture_audit_v2', mode=mode, consistency_passed=True,
                complete_capture_audited=complete, physics_rows=len(trace), adc_packets=len(adc), rgbd_frames=len(frames),
                camera_attempts=attempts, depth_valid_fraction_range=[min(ratios), max(ratios)] if ratios else None,
                original=dict(success=result.get('success'), physical_validation=result.get('physical_validation'),
                              failure_reason=result.get('failure_reason'),
                              raw_capture_complete=top['raw_capture_complete'], result_errors=result.get('errors'),
                              capture_errors=top['errors'], hall_receipt=hall_receipt, camera_receipt=camera_receipt),
                hall_replay=replay, events=events, frame_error_records=frame_errors, geometry=geometry,
                sources=dict(mapped=sources.snapshot is not None, verified=sources.verified,
                             rgbd_injected_backend=manifests['rgbd']['injected_backend']),
                baseline_comparison=None if baseline is None else _compare(run, baseline),
                source_trace_sha256=trace_digest, audit_script_sha256=_sha(__file__),
                training_eligible=False, physical_qualification=False,
                scope=('Original complete raw capture consistency' if complete else 'Recorded prefix consistency; original failure flags retained')+
                      '; no contact physics, episode, expert, randomization, sensor-use or model qualification.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--mode', required=True, choices=('prefix', 'complete'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--recorded-project-root', type=Path)
    parser.add_argument('--source-snapshot-root', type=Path)
    args = parser.parse_args(argv)
    report = audit_capture(args.run, mode=args.mode, baseline=args.baseline,
                           recorded_project_root=args.recorded_project_root, source_snapshot_root=args.source_snapshot_root)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
