"""Opt-in raw expert acquisition; successful recording grants no training gate."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re

from .geometry_epoch import require_current
from .m2_capture import M2RGBDCapture, APPROVED_CAMERA_SHA256, _state, _same_state, _scalar
from .m2_sensors import M2HallStream, audit_hall_stream
from .m2_sensor_probe import CALIBRATION_SHA256, _sha


def capture_settings(fresh_config):
    if 'raw_multimodal_capture' not in fresh_config:
        return None
    value = fresh_config['raw_multimodal_capture']
    if isinstance(value, dict) and value == {'enabled': False} and type(value['enabled']) is bool:
        return None
    keys = {'enabled', 'episode_id', 'camera_file', 'camera_sha256',
            'calibration_file', 'calibration_sha256'}
    if (not isinstance(value, dict) or set(value) != keys or value.get('enabled') is not True
            or not isinstance(value.get('episode_id'), str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value['episode_id']) is None
            or any(not isinstance(value.get(k), str) or not value[k].strip()
                   for k in ('camera_file', 'calibration_file'))
            or value.get('camera_sha256') != APPROVED_CAMERA_SHA256
            or value.get('calibration_sha256') != CALIBRATION_SHA256):
        raise ValueError('fresh_expert.raw_multimodal_capture requires an explicit episode and pinned sources')
    return deepcopy(value)


def checked_render(env, operation, label):
    """Rendering must preserve finite clocks and measured q/dq/displacement."""
    before = _state(env.state())
    clock = _scalar(env.world.current_time, label+' world clock')
    result = operation()
    _same_state(before, _state(env.state()), label)
    if abs(_scalar(env.world.current_time, label+' world clock')-clock) > 1e-8:
        raise ValueError(label+' advanced world physics clock')
    return result


def controller_labels(controller, subphase, active, recovery):
    return dict(phase=controller.manager.state.phase.value, target=controller.target.name,
                subphase=subphase, active=bool(active), recovery=bool(recovery))


class ExpertCapture:
    """Small lifecycle owner. Holds no exception objects; close releases env refs.

    The constructor pins inputs before native setup. ``start`` registers each
    resource before constructing the next so partial setup can be finalized.
    Every actual step is appended before any post-step readback or video.
    Camera calls belong to the inherited tick's finalization, after its audits.
    """
    def __init__(self, output, *, binding, geometry_metadata, source_paths=()):
        self.binding = capture_settings({'raw_multimodal_capture': binding})
        if self.binding is None: raise ValueError('Raw capture must be explicitly enabled')
        require_current(geometry_metadata)
        self.geometry = deepcopy(geometry_metadata)
        for key in ('collision_geometry_sha256', 'collision_arrays_sha256'):
            digest = self.geometry.get('geometry_artifacts', {}).get(key)
            if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
                raise ValueError('Raw capture requires complete geometry artifact hashes')
        self.camera_bytes = Path(binding['camera_file']).read_bytes()
        self.calibration_bytes = Path(binding['calibration_file']).read_bytes()
        import hashlib
        if (hashlib.sha256(self.camera_bytes).hexdigest() != APPROVED_CAMERA_SHA256 or
                hashlib.sha256(self.calibration_bytes).hexdigest() != CALIBRATION_SHA256):
            raise ValueError('Pinned camera or Hall calibration changed')
        paths = [*source_paths, inspect.getfile(ExpertCapture), inspect.getfile(M2HallStream),
                 inspect.getfile(M2RGBDCapture), inspect.getfile(_sha)]
        self.sources = {str(Path(path).resolve()): _sha(path) for path in paths}
        self.sources.update({str(Path(binding['camera_file']).resolve()): APPROVED_CAMERA_SHA256,
                             str(Path(binding['calibration_file']).resolve()): CALIBRATION_SHA256})
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.hall = self.camera = self._events = None
        self.errors = []; self.receipts = {}; self.latest_hall = None; self._closed = False
        (self.output/'approved_camera.json').write_bytes(self.camera_bytes)
        (self.output/'calibration.json').write_bytes(self.calibration_bytes)
        self._write('manifest.json', dict(schema='m2_expert_raw_capture_v1', binding=self.binding,
            geometry=self.geometry, runtime_sources_sha256=self.sources, audio='absent',
            physics_hz=240, camera_hz=20, hall_adc_nominal_hz=100,
            controller_hall_proxy='unchanged; recorded ADC is not controller feedback',
            training_eligible=False, physical_qualification=False, randomization_qualified=False))
        self._events = (self.output/'events.jsonl').open('x')

    def _write(self, name, value):
        (self.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def _event(self, value):
        self._events.write(json.dumps(value, allow_nan=False)+'\n'); self._events.flush()

    def failure(self, stage, error):
        record = dict(stage=stage, error=str(error))
        self.errors.append(record)
        try: self._event(dict(record, status='error', training_eligible=False))
        except BaseException as write_error:
            self.errors.append(dict(stage='error_event_write', error=repr(write_error)))

    def start(self, env, *, backend_factory=None):
        if self.hall is not None or self._closed: raise RuntimeError('Raw capture already started or closed')
        if env.rows: raise ValueError('Raw expert capture must start before the first physics row')
        try:
            self.hall = M2HallStream(self.output/'hall', geometry_metadata=self.geometry,
                calibration_bytes=self.calibration_bytes, episode_id=self.binding['episode_id'])
            self.camera = M2RGBDCapture(env, self.output/'rgbd', geometry_metadata=self.geometry,
                calibration_bytes=self.calibration_bytes, camera_bytes=self.camera_bytes,
                camera_sha256=APPROVED_CAMERA_SHA256, episode_id=self.binding['episode_id'],
                hall_stream=self.hall, backend_factory=backend_factory)
        except BaseException as error:
            self.failure('setup', repr(error)); raise

    def after_physics(self, env, rows_before):
        if len(env.rows) == rows_before: return
        try:
            if len(env.rows) != rows_before+1: raise ValueError('Expected exactly one actual physics row')
            row = env.rows[-1]
            self.latest_hall = self.hall.append(rows_before, row['time_s'], row['compression_m'])
        except BaseException as error:
            self.failure('hall_append', repr(error)); raise

    def finish_tick(self, env, rows_before, before, after, *, error=None):
        if error is not None: self.failure('controller_tick', error)
        if len(env.rows) == rows_before: return
        try:
            if len(env.rows) != rows_before+1: raise ValueError('Tick must contain one actual physics step')
            index = len(env.rows)-1; row = env.rows[-1]
            if any(before[key] != row.get(key) or after[key] != row.get(key)
                   for key in ('target','subphase')):
                raise ValueError('Controller labels differ from the actual physics row')
            audited = 'physical_violations' in row
            context = dict(controller_before=deepcopy(before), controller_after=deepcopy(after),
                physics_labels={key:row.get(key) for key in ('phase','target','subphase')},
                manager_update=('not_requested' if not before['active'] else
                                'completed' if audited else 'not_completed'),
                state_audit='completed' if audited else 'not_completed', tick_error=error)
            due = index % 12 == 11
            self._event(dict(physics_row=index, time_s=row['time_s'], context=context,
                             camera_due=due, training_eligible=False))
            if due:
                phase = after['phase'] if audited and not before['recovery'] else row['phase']
                self.camera.capture(env.state(), physics_row=index, phase=phase, target=row['target'],
                                    hall=self.latest_hall, context=context)
        except BaseException as caught:
            self.failure('tick_capture', repr(caught)); raise

    def close(self, error=None):
        """Camera first, Hall next, then audit; failures never skip a release."""
        if self._closed: return
        self._closed = True
        for name in ('camera','hall'):
            resource = getattr(self, name)
            if resource is not None:
                try: self.receipts[name] = resource.finalize(error)
                except BaseException as caught: self.failure(name+'_finalize', repr(caught))
                finally: setattr(self, name, None)
            elif name == 'camera' and (self.output/'rgbd/receipt.json').exists():
                # M2RGBDCapture finalizes itself if construction fails.
                try: self.receipts[name] = json.loads((self.output/'rgbd/receipt.json').read_text())
                except BaseException as caught: self.failure('camera_partial_receipt', repr(caught))
        if self.receipts.get('hall', {}).get('completed'):
            try: self.receipts['hall_replay'] = audit_hall_stream(self.output/'hall')
            except BaseException as caught: self.failure('hall_replay', repr(caught))
        if self._events is not None:
            try: self._events.close()
            except BaseException as caught: self.errors.append(dict(stage='events_close',error=repr(caught)))
            self._events = None

    def result(self, physics_rows, *, execution_completed, errors=()):
        failures = deepcopy([*self.errors, *errors])
        for path, expected in self.sources.items():
            try:
                if _sha(path) != expected: raise ValueError('Source changed during acquisition')
            except BaseException as caught: failures.append(dict(stage='source',path=path,error=repr(caught)))
        hall = self.receipts.get('hall', {}); camera = self.receipts.get('camera', {})
        counts_match = (type(physics_rows) is int and physics_rows > 0 and
            hall.get('physics_rows') == hall.get('hall_rows') == physics_rows and
            camera.get('frames') == physics_rows//12)
        if not counts_match: failures.append(dict(stage='counts',error='Raw Hall/frame counts differ from actual physics'))
        complete = bool(self._closed and execution_completed and not failures and counts_match and
            camera.get('completed') and hall.get('completed') and self.receipts.get('hall_replay',{}).get('passed'))
        result = dict(schema='m2_expert_raw_capture_v1', raw_capture_complete=complete,
            episode_id=self.binding['episode_id'], physics_steps=physics_rows,
            expected_frames=physics_rows//12, receipts=deepcopy(self.receipts), errors=failures,
            runtime_sources_sha256=self.sources, training_eligible=False, physical_qualification=False,
            randomization_qualified=False, audio='absent',
            scope='Expert raw RGB-D/Hall acquisition diagnostic; no dataset promotion or training acceptance')
        try: self._write('receipt.json', result)
        except BaseException as caught:
            result['raw_capture_complete'] = False
            result['errors'].append(dict(stage='receipt_write',error=repr(caught)))
        return result
