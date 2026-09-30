"""Raw 20 Hz RGB-D / same-step Hall evidence for the current 5 mm tool.

Call after the controller phase update at physics rows 11, 23, ... . The
episode clock is state['time_s']; world.current_time may have a setup offset.
This module attaches to the approved camera without changing its transform,
body or optics. It does not encode policy inputs, track targets, certify
contact physics, or grant training eligibility. Native rendering needs an
owned Linux Isaac process; CPU tests inject the small backend protocol below.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

from .geometry_epoch import require_current
from .m2_sensors import M2HallStream
from ..auscultation_vla_v2.hall import HallCalibration, HallPacket


SCHEMA = 'm2_raw_rgbd20_hall100_physics240_5mm_v1'
CAMERA_PATH = '/World/RobotCam'
APPROVED_CAMERA_SHA256 = 'aeac00eb5a963520e869c60efff52a300484bd1c4a49232067a669c525ad07f0'
CALIBRATION_PATH = Path(__file__).resolve().parents[1]/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
WIDTH, HEIGHT = 640, 360
ANNOTATORS = ('rgb', 'distance_to_image_plane')
TARGETS = frozenset(side+str(i) for side in ('L','R') for i in range(1,6))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(_json(value)+'\n')
        stream.flush()


def _append(stream, value):
    stream.write(_json(value)+'\n')
    stream.flush()


def _scalar(value, label):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value)):
        raise ValueError(label+' must be finite numeric')
    return float(value)


def _state(state):
    stamp = _scalar(state['time_s'], 'physics time')
    result = []
    for name in ('q','dq'):
        value = np.asarray(state[name])
        if value.shape != (7,) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Measured '+name+' must contain seven finite numbers')
        result.append(value.copy())
    return stamp, *result, _scalar(state['compression_m'],'measured slider compression')


def _same_state(before, after, label):
    if (abs(before[0]-after[0]) > 1e-8 or before[3] != after[3] or
            any(not np.allclose(a,b,atol=1e-8,rtol=0) for a,b in zip(before[1:3],after[1:3]))):
        raise ValueError(label+' changed measured physics time, joints or slider compression')


def _source(obj):
    path = Path(inspect.getsourcefile(obj)).resolve()
    return dict(module=obj.__module__, path=str(path), sha256=_sha256(path))


class _NativeBackend:
    """Read-only camera snapshot, one owned product, two owned annotators.

    Injected CPU backends implement snapshot_camera, setup, render, read,
    detach(name), destroy_product. Native construction owns no resources;
    setup registers each resource before the next operation can fail.
    """
    def __init__(self, env):
        if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
            raise RuntimeError('Native M2 capture requires an owned Linux AUSC_OWNED_RUN=1 process')
        from pxr import UsdGeom
        import omni.replicator.core as rep
        self.env, self.UsdGeom, self.rep = env, UsdGeom, rep
        self.product = None
        self.annotators = {}

    def snapshot_camera(self):
        geom = self.UsdGeom
        prim = self.env.stage.GetPrimAtPath(CAMERA_PATH)
        if not prim or not prim.IsA(geom.Camera): raise ValueError('Approved camera is missing')
        camera = geom.Camera(prim)
        optics = (
            camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(),
            camera.GetVerticalApertureAttr(), camera.GetHorizontalApertureOffsetAttr(),
            camera.GetVerticalApertureOffsetAttr())
        f, ha, va, ho, vo = [float(attr.Get()) for attr in optics]
        if not np.isfinite([f,ha,va,ho,vo]).all() or min(f,ha,va) <= 0:
            raise ValueError('Invalid camera optics')
        # The approved camera is static. Reject authored time samples rather
        # than mistake DefaultTime for the currently evaluated animation pose.
        time_varying = any(attr.GetNumTimeSamples() for attr in (*optics,camera.GetProjectionAttr()))
        parent = prim
        while parent and not parent.IsPseudoRoot():
            time_varying |= any(attr.GetNumTimeSamples() for attr in parent.GetAttributes()
                                if attr.GetName().startswith('xformOp:'))
            parent = parent.GetParent()
        # USD matrices use row-vector convention; evidence uses column vectors.
        transform = np.asarray(geom.XformCache().GetLocalToWorldTransform(prim)).T
        return dict(camera_path=str(prim.GetPath()), world_from_camera_usd=transform.tolist(),
            intrinsics=dict(fx=f/ha*WIDTH, fy=f/va*HEIGHT,
                            cx=WIDTH/2-ho/ha*WIDTH, cy=HEIGHT/2+vo/va*HEIGHT),
            width=WIDTH, height=HEIGHT, meters_per_unit=float(geom.GetStageMetersPerUnit(self.env.stage)),
            projection=str(camera.GetProjectionAttr().Get()), time_varying=bool(time_varying))

    def setup(self):
        # A video recorder may already use this camera and resolution. Never
        # reuse its product: finalization destroys only this capture's product.
        self.product = self.rep.create.render_product(CAMERA_PATH, (WIDTH,HEIGHT), force_new=True)
        for name in ANNOTATORS:
            self.annotators[name] = self.rep.AnnotatorRegistry.get_annotator(name)
            self.annotators[name].attach([self.product])

    def render(self):
        self.rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=False)

    def read(self):
        return tuple(self.annotators[name].get_data() for name in ANNOTATORS)

    def detach(self, name):
        annotator = self.annotators.pop(name, None)
        if annotator is not None: annotator.detach()

    def destroy_product(self):
        product, self.product = self.product, None
        if product is not None: product.destroy()


class M2RGBDCapture:
    """Fail-closed append-only raw acquisition; caller must always finalize.

    Constructor failures also finalize their partial output and native scope.
    A returned frame already has an atomically published NPZ, its streaming
    SHA-256 and a flushed ledger row. Failed attempts retain an error receipt,
    and partial files are retained for diagnosis. Flushes are not a promise of
    power-loss durability. Finalization returns all errors even if the receipt
    cannot be written. Neither successful capture nor teardown qualifies the
    episode for training or physics acceptance.

    ``hall_stream`` is the caller-owned live M2HallStream for this episode.
    Append its 240 Hz raw displacement first, then pass that exact returned
    Hall value to capture. Finalize the camera before finalizing the Hall
    stream. Its raw displacement, not noisy decoded displacement, must equal
    the measured state's ``compression_m``.
    """
    def __init__(self, env, output, *, geometry_metadata, camera_bytes,
                 camera_sha256, episode_id, hall_stream, backend_factory=None, calibration_bytes=None):
        require_current(geometry_metadata)
        if not isinstance(episode_id,str) or not episode_id.strip():
            raise ValueError('Explicit episode identity required')
        if (not isinstance(camera_bytes,bytes) or camera_sha256 != APPROVED_CAMERA_SHA256 or
                hashlib.sha256(camera_bytes).hexdigest() != APPROVED_CAMERA_SHA256):
            raise ValueError('Exact approved camera bytes and pinned hash required')
        self._camera = json.loads(camera_bytes)
        if self._camera['camera_path'] != CAMERA_PATH: raise ValueError('Wrong approved camera path')
        if calibration_bytes is None: calibration_bytes = CALIBRATION_PATH.read_bytes()
        if not isinstance(calibration_bytes,bytes): raise ValueError('Exact calibration bytes required')
        self.calibration = HallCalibration(calibration_bytes)
        if type(hall_stream) is not M2HallStream:
            raise ValueError('Hall stream must be a concrete M2HallStream instance')
        hall_snapshot = hall_stream.snapshot()
        if (hall_snapshot['episode_id'] != episode_id or
                _json(hall_snapshot['geometry']) != _json(geometry_metadata) or
                hall_snapshot['calibration_sha256'] != self.calibration.sha256):
            raise ValueError('Hall stream episode, geometry or calibration does not match capture')
        hall_manifest_sha256 = _sha256(hall_stream.output/'manifest.json')
        if hall_manifest_sha256 != hashlib.sha256((_json(hall_snapshot)+'\n').encode()).hexdigest():
            raise ValueError('Hall stream manifest does not match its live identity')
        self._hall_stream = hall_stream
        self.env, self.output = env, Path(output)
        self._backend = self._frames = self._attempts = None
        self._count = self._attempt_count = 0
        self._failures = []
        self._closed = False
        self._receipt = None
        self._last_hall = None
        self._metadata = dict(schema=SCHEMA, episode_id=episode_id, geometry=deepcopy(geometry_metadata),
            camera_sha256=camera_sha256, calibration_sha256=self.calibration.sha256,
            camera_path=CAMERA_PATH, width=WIDTH, height=HEIGHT, physics_hz=240, capture_hz=20,
            hall_adc_nominal_hz=100, camera_bytes_file='approved_camera.json',
            calibration_bytes_file='calibration.json', audio='absent', depth_semantics='metric_optical_z',
            phase_timing='caller phase after controller update on each twelfth physics step',
            timestamp_semantics='measured episode physics time; zero-delta render preserves world clock and joints',
            injected_backend=backend_factory is not None, target_data='target identifier only',
            hall_stream=dict(snapshot=hall_snapshot,manifest_sha256=hall_manifest_sha256),
            training_eligible=False, physical_qualification=False)
        self.output.mkdir(parents=True, exist_ok=False)
        stage = 'setup'
        try:
            (self.output/'approved_camera.json').write_bytes(camera_bytes)
            (self.output/'calibration.json').write_bytes(calibration_bytes)
            self._frames = (self.output/'frames.jsonl').open('x')
            self._attempts = (self.output/'attempts.jsonl').open('x')
            self._backend = (backend_factory or _NativeBackend)(env)
            self._metadata['source_bindings'] = dict(capture=_source(M2RGBDCapture),
                backend=_source(type(self._backend)), environment=_source(type(env)),
                hall_stream=_source(M2HallStream), calibration=_source(HallCalibration),
                host_calibration=_source(type(self.calibration._host)), geometry_gate=_source(require_current))
            stage = 'camera_before_setup'
            self._metadata['actual_camera_at_setup'] = self._verify_camera()
            before = _state(env.state())
            clock = _scalar(env.world.current_time, 'world clock')
            stage = 'backend_setup'
            self._backend.setup()
            _same_state(before, _state(env.state()), 'camera setup')
            if abs(_scalar(env.world.current_time,'world clock')-clock) > 1e-8:
                raise ValueError('camera setup changed world clock')
            self._verify_camera()
            stage = 'manifest_write'
            _write_json(self.output/'manifest.json', self._metadata)
        except BaseException as error:
            self._failure(stage,error)
            self.finalize()
            raise

    def snapshot(self):
        return deepcopy(self._metadata)

    def _failure(self, stage, error):
        self._failures.append(dict(stage=stage,error=type(error).__name__+': '+str(error)))

    def _verify_camera(self):
        actual = self._backend.snapshot_camera()
        pose = np.asarray(actual['world_from_camera_usd'],dtype=float)
        expected = np.asarray(self._camera['world_from_camera_usd'],dtype=float)
        if (actual.get('camera_path') != CAMERA_PATH or pose.shape != (4,4) or
                not np.allclose(pose,expected,atol=1e-7,rtol=0) or
                actual.get('width') != WIDTH or actual.get('height') != HEIGHT or
                actual.get('meters_per_unit') != 1. or actual.get('projection') != 'perspective' or
                actual.get('time_varying',False)):
            raise ValueError('Approved camera transform, projection or metric units changed')
        original = self._camera['intrinsics']
        for key, dimension in (('fx','width'),('fy','height'),('cx','width'),('cy','height')):
            # Compare ratios, not focal/aperture values tied to USD length units.
            ratio = _scalar(actual['intrinsics'][key],'camera intrinsics')/actual[dimension]
            if abs(ratio-original[key]/original[dimension]) > 1e-7:
                raise ValueError('Approved camera intrinsics changed')
        return deepcopy(actual)

    def _hall(self, hall, physics_row, stamp, compression):
        hall = deepcopy(hall)
        evidence = self._hall_stream.latest_evidence()
        identity = dict(snapshot=evidence['snapshot'],manifest_sha256=evidence['manifest_sha256'])
        if _json(identity) != _json(self._metadata['hall_stream']):
            raise ValueError('Bound Hall stream identity changed')
        if _json(hall) != _json(evidence['hall']):
            raise ValueError('Hall result differs from the bound stream latest evidence')
        physical = evidence['physics']
        if (physical['physics_row'] != physics_row or
                abs(physical['time_s']-stamp) > 1e-8 or physical['compression_m'] != compression):
            raise ValueError('Bound Hall raw physics row, time or compression differs from measured state')
        if (not isinstance(hall,dict) or type(hall.get('physics_row')) is not int or
                hall['physics_row'] != physics_row or
                abs(_scalar(hall.get('time_s'),'Hall time')-stamp) > 1e-8 or
                hall.get('training_eligible') is not False or hall.get('error') is not None or
                type(hall.get('is_new_packet')) is not bool):
            raise ValueError('Hall result must bind the same physics row and time')
        if hall.get('status') == 'pending':
            if any(hall.get(key) is not None for key in ('packet_id','acquisition_id','packet','decoded')) or hall['is_new_packet']:
                raise ValueError('Pending Hall result contains invented packet data')
        elif hall.get('status') == 'available':
            if (not isinstance(hall.get('packet_id'),str) or not hall['packet_id'] or
                    hall.get('acquisition_id') != hall['packet_id'] or not isinstance(hall.get('packet'),dict)):
                raise ValueError('Available Hall result needs its full packet and acquisition ID')
            packet = HallPacket(**hall['packet'])
            if packet.source != 'simulated_adc': raise ValueError('M2 requires simulated ADC source')
            decoded = asdict(self.calibration.decode(packet,now_s=stamp))
            if hall.get('decoded') != decoded: raise ValueError('Hall decode does not match archived calibration')
            if self._last_hall is not None:
                previous = self._last_hall
                if hall['acquisition_id'] == previous['acquisition_id']:
                    if hall['packet'] != previous['packet'] or hall['is_new_packet']:
                        raise ValueError('Repeated ADC acquisition cannot become a new packet')
                elif packet.sample_time_s <= previous['packet']['sample_time_s']:
                    raise ValueError('Distinct ADC acquisition must advance sample time')
        else:
            raise ValueError('Hall result must be available or pending')
        _json(hall)
        return hall, evidence

    def capture(self, state, *, physics_row, phase, target, hall, context=None):
        if self._closed: raise RuntimeError('RGB-D acquisition is finalized')
        if self._failures: raise RuntimeError('RGB-D acquisition already failed; preserve this episode')
        attempt = dict(attempt=self._attempt_count, index=self._count, physics_row=repr(physics_row),
            supplied_time_s=repr(state.get('time_s')) if isinstance(state,dict) else repr(state),
            phase=repr(phase), target=repr(target), status='started',
            training_eligible=False, physical_qualification=False)
        self._attempt_count += 1
        stage = 'attempt_write'
        try:
            _append(self._attempts,attempt)
            stage = 'validate_physics'
            if type(physics_row) is not int or physics_row != 11+12*self._count:
                raise ValueError('Missing, duplicate or invalid 20 Hz physics row')
            before = _state(state)
            stamp, q, dq, compression = before
            if abs(stamp-(physics_row+1)/240.) > 1e-8:
                raise ValueError('Frame time must be the actual consecutive 240 Hz physics time')
            _same_state(before, _state(self.env.state()), 'caller state')
            if not isinstance(phase,str) or not phase.strip() or target not in TARGETS:
                raise ValueError('Explicit phase and target L1-L5/R1-R5 required')
            if context is not None and not isinstance(context,dict): raise ValueError('Context must be an explicit record')
            context = deepcopy(context)
            _json(context)
            stage = 'validate_hall'
            hall, hall_evidence = self._hall(hall,physics_row,stamp,compression)
            stage = 'camera_before_render'
            self._verify_camera()
            clock = _scalar(self.env.world.current_time,'world clock')
            stage = 'render'
            self._backend.render()
            _same_state(before, _state(self.env.state()), 'render')
            after_clock = _scalar(self.env.world.current_time,'world clock')
            if abs(after_clock-clock) > 1e-8: raise ValueError('render advanced world physics clock')
            self._verify_camera()
            stage = 'read_raw'
            rgb, depth = (np.asarray(value).copy() for value in self._backend.read())
            stage = 'validate_after_read'
            _same_state(before, _state(self.env.state()), 'RGB-D read')
            after_clock = _scalar(self.env.world.current_time,'world clock')
            if abs(after_clock-clock) > 1e-8: raise ValueError('RGB-D read advanced world physics clock')
            self._verify_camera()
            self._hall(hall,physics_row,stamp,compression)
            stage = 'validate_raw'
            if rgb.dtype != np.uint8 or rgb.shape not in ((HEIGHT,WIDTH,3),(HEIGHT,WIDTH,4)):
                raise ValueError('RGB must be uint8 (360,640,3/4)')
            if depth.shape != (HEIGHT,WIDTH) or depth.dtype.kind != 'f':
                raise ValueError('Depth must be floating metric optical Z (360,640)')
            valid = np.isfinite(depth)&(depth>0)
            filename = 'frame-'+str(self._count).zfill(8)+'.npz'
            attempt.update(raw_file=filename, partial_file=filename+'.partial')
            stage = 'raw_write'
            with (self.output/attempt['partial_file']).open('xb') as stream:
                np.savez_compressed(stream, rgb=rgb, depth_m=depth, depth_valid=valid,
                    training_eligible=np.array(False), physical_qualification=np.array(False))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(self.output/attempt['partial_file'], self.output/filename)
            stage = 'raw_hash'
            raw_hash = _sha256(self.output/filename)
            attempt['raw_sha256'] = raw_hash
            row = dict(index=self._count,physics_row=physics_row,time_s=stamp,phase=phase,target=target,
                context=context,q=q.tolist(),dq=dq.tolist(),raw_file=filename,raw_sha256=raw_hash,
                hall=hall,packet_id=hall['packet_id'],acquisition_id=hall['acquisition_id'],
                hall_physics=hall_evidence['physics'],
                hall_stream_manifest_sha256=hall_evidence['manifest_sha256'],
                render_world_time_s=clock,render_physics_delta_s=after_clock-clock,
                training_eligible=False,physical_qualification=False)
            stage = 'frame_write'
            _append(self._frames,row)
            self._count += 1
            if hall['status'] == 'available': self._last_hall = deepcopy(hall)
            stage = 'attempt_complete_write'
            _append(self._attempts,dict(attempt,status='complete'))
            return deepcopy(row)
        except BaseException as error:
            self._failure(stage,error)
            try: _append(self._attempts,dict(attempt,status='error',failure=self._failures[-1]))
            except BaseException as write_error: self._failure('error_attempt_write',write_error)
            raise

    def finalize(self, error=None):
        """Always release all owned resources; return aggregated write/close errors."""
        if self._receipt is not None: return deepcopy(self._receipt)
        self._closed = True
        if error is not None: self._failure('caller',error)
        if not self._count and not self._failures: self._failure('capture',ValueError('No captured frames'))
        if self._backend is not None:
            for name in ANNOTATORS:
                try: self._backend.detach(name)
                except BaseException as detach_error: self._failure('detach:'+name,detach_error)
            try: self._backend.destroy_product()
            except BaseException as destroy_error: self._failure('destroy_product',destroy_error)
        for label, stream in (('frames',self._frames),('attempts',self._attempts)):
            if stream is not None:
                try: stream.close()
                except BaseException as close_error: self._failure('close:'+label,close_error)
        artifacts = {}
        for name in ('manifest.json','approved_camera.json','calibration.json','frames.jsonl','attempts.jsonl'):
            try: artifacts[name] = _sha256(self.output/name)
            except BaseException as hash_error: self._failure('hash:'+name,hash_error)
        for source in self._metadata.get('source_bindings',{}).values():
            try:
                if _sha256(source['path']) != source['sha256']: raise ValueError('Source changed during acquisition')
            except BaseException as source_error: self._failure('source_binding',source_error)
        receipt = dict(schema=SCHEMA,completed=not self._failures,frames=self._count,
            attempts=self._attempt_count,failures=deepcopy(self._failures),artifacts_sha256=artifacts,
            receipt_written=True,training_eligible=False,physical_qualification=False)
        try: _write_json(self.output/'receipt.json',receipt)
        except BaseException as write_error:
            self._failure('receipt_write',write_error)
            receipt.update(completed=False,receipt_written=False,failures=deepcopy(self._failures))
        self._receipt = receipt
        return deepcopy(receipt)
