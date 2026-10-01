"""Bounded, single-target compact recording; NOT an Isaac runtime adapter.

Only Linux may write real/replay sources. CPU fixtures must label themselves
``cpu_synthetic_test``; every qualification flag remains false. The runtime
must supply a real state17 adapter, an independently sourced target estimate,
causal raw Hall ADC packets, and actual command endpoint readback evidence.
This module verifies consistency of supplied evidence, not its authenticity.

Integration contract:
* Reuse the native M2 camera backend at 640x360, with its before/after camera
  and zero-delta physics guards. Do not instantiate the raw M2RGBDCapture or
  M2HallStream in training mode: they persist 240 Hz/raw ledgers.
* At episode rows 11,23,... snapshot RGB-D, q, and the causal target estimate;
  encode state17 with m2_contract.encode_state. Keep exactly one pending
  observation until the actual next 20 Hz command endpoint has been observed.
* Feed a bounded in-memory HallSimulator/real-ADC buffer at nominal 100 Hz.
  Supply only newly received packets available at that observation, never
  packets first received while waiting for the future action endpoint.
* Supply command issue and endpoint/readback times separately. Retain all
  intervening 240 Hz phase/owner/target boundary events and physics/safety
  audits in a separate bounded in-memory audit. This endpoint stream alone
  cannot qualify H20 chunks or show absence of intervening events.
* Predeclare either an exact offline frame count or an adaptive maximum.
  Fixed offline recordings retain the v3 all-paired contract. Adaptive v4
  episodes ending after N=12*M+r actual native steps have M observations,
  M-1 complete pairs (or zero if M=0), one final observation without any
  future-action fields when M>0, and an explicit <=11-tick terminal ledger.
  Append the pending last observation with append_terminal, then close with
  the actual final count/time and every trailing tick. Never add, repeat or
  round native steps for capture. N<12 retains only the supplied native ticks.
  Abort on missing modalities, estimator failure, changed target, early errors
  or budget failure, passing actual terminal evidence when available.
  This terminal ledger is NOT the separate full-episode event/physics audit;
  v4 cannot qualify H20 chunks or training without that missing outer layer.
  A standalone writer is not a
  completed native recording/training pipeline. Linux replay size/fidelity
  benchmarking and a native integration probe remain separate requirements.

RGB: full-FOV 256x144 JPEG quality 85, no native camera modification.
Depth: nearest sampled metric optical Z, uint32 units of 0.0001 m, separate
bool validity; rounding error <=0.00005 m. Temporal storage XORs each
quantized frame with its predecessor, with absolute keyframes at index 0 and
every 20 frames. This is lossless relative to the metric quantization.
Out-of-range valid depth fails;
invalid pixels become 0 with validity false, never a normalized depth image.
Numbers are lossless little-endian float64/int64 in one compressed NPZ per
record. Both files stream: <uint32 length,uint32 CRC32,payload> frames, RGB
payload JPEG and numeric payload NPZ. No per-physics JSONL/raw frame folders.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import zipfile
import zlib

import numpy as np
from PIL import Image

from . import m2_contract
from .m2_capture import APPROVED_CAMERA_SHA256, TARGETS
from .storage_policy import EPISODE_LIMIT_BYTES
from ..auscultation_vla_v2.hall import HallCalibration, HallPacket

SCHEMA = 'm2_compact_single_target_rgbd256x144_state17_hall20_v3'
ADAPTIVE_SCHEMA = 'm2_compact_single_target_rgbd256x144_state17_hall20_v4'
TERMINAL_SCHEMA = 'm2_compact_terminal_native_ticks_v1'
POLICY_SIZE = (256, 144)
NATIVE_SIZE = (640, 360)
DEPTH_UNIT_M = .0001
DEPTH_DTYPE = np.dtype('<u4')
DEPTH_MAX_M = np.iinfo(np.uint32).max*DEPTH_UNIT_M
DEPTH_CODEC = 'uint32_xor_previous_with_periodic_absolute_keyframes'
DEPTH_KEYFRAME_INTERVAL = 20
HEADER = struct.Struct('<II')
RECEIPT_RESERVE = 8192
MAX_RECORD_BYTES = 2_000_000
MAX_FRAMES = (2**63-1)//12-1  # Final native step count must also fit signed int64.
QUALIFICATIONS = dict(training_eligible=False, physical_qualification=False,
                      artifact_hashes_verified=False)
FILES = ('manifest.json', 'approved_camera.json', 'calibration.json', 'rgb.mjpeg', 'numeric.bin')
ACTION_FIELDS = {'q_command','action_issued_at_s','action_endpoint_at_s',
                 'action_readback_at_s','action_source_id'}
TICK_FIELDS = {'episode_id','target_id','step_count','physics_row','time_s','phase',
               'control_owner','measured_q','q_command','command_issued_at_s',
               'command_readback_at_s','command_source_id','observation_source_id'}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _bounded_failure(reason):
    """Bound the JSON-serialized field, including escaping, within the reserve."""
    if reason is None: return None
    value = str(reason)[:2048]
    lo,hi = 0,len(value)
    while lo < hi:
        middle = (lo+hi+1)//2
        if len(json.dumps(value[:middle]).encode()) <= 2048: lo = middle
        else: hi = middle-1
    return value[:lo]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing SHA256 binding: '+name)
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(name+' must be a bounded nonempty string')
    return value


def _scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating))
            or not np.isfinite(value)):
        raise ValueError(name+' must be finite numeric')
    return float(value)


def _array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError(name+' must be finite numeric '+str(shape))
    return value.astype('<f8', copy=True)


def _framed(payload):
    if not 0 < len(payload) <= MAX_RECORD_BYTES:
        raise ValueError('Encoded frame exceeds bounded record size')
    return HEADER.pack(len(payload), zlib.crc32(payload))+payload


def _require_episode_metadata(meta):
    """Shared writer/reader provenance contract; hashes are attestations only."""
    m2_contract.require_metadata(meta)
    for key in ('episode_id','subject_id'): _text(meta.get(key),key)
    if meta.get('target_id') not in TARGETS: raise ValueError('One explicit target_id required')
    if (meta.get('split') not in ('train','validation','test') or
            meta.get('identity_split_sha256') != m2_contract.IDENTITY_SPLIT_SHA256):
        raise ValueError('Frozen identity split binding required')
    for key in ('collision_geometry_sha256','collision_arrays_sha256'): _hash(meta.get(key),key)
    if meta.get('source_kind') not in ('cpu_synthetic_test','isaac_live_physics','replay_transcode'):
        raise ValueError('Explicit acquisition source_kind required')


def _terminal_evidence(metadata, final_native_step_count, final_native_time_s, ticks, observed_frames):
    """Validate completed native ticks, never synthesize observations or actions."""
    n = final_native_step_count
    t = _scalar(final_native_time_s,'final native time')
    if (type(n) is not int or not 0 <= n <= 12*MAX_FRAMES+11 or
            abs(t-n/240.) > 1e-9):
        raise ValueError('Final native step count/time must describe the actual 240 Hz clock')
    count,remainder = divmod(n,12)
    if type(observed_frames) is not int or not 0 <= observed_frames <= count:
        raise ValueError('Recorded observations cannot exceed the actual native observation clock')
    if not isinstance(ticks,(tuple,list)) or len(ticks) != remainder:
        raise ValueError('Explicit terminal native suffix must contain every trailing tick (0..11)')
    result = []
    for expected,tick in enumerate(ticks,start=count*12+1):
        if not isinstance(tick,dict) or set(tick) != TICK_FIELDS:
            raise ValueError('Terminal native tick requires exact context, state, command and source fields')
        if (tick['episode_id'] != metadata['episode_id'] or tick['target_id'] != metadata['target_id'] or
                type(tick['step_count']) is not int or tick['step_count'] != expected or
                type(tick['physics_row']) is not int or tick['physics_row'] != expected-1):
            raise ValueError('Terminal native tick context/count/physics row mismatch')
        at = _scalar(tick['time_s'],'terminal native time')
        issue = _scalar(tick['command_issued_at_s'],'terminal command issue')
        readback = _scalar(tick['command_readback_at_s'],'terminal command readback')
        if abs(at-expected/240.) > 1e-9 or not 0 <= issue <= at or abs(readback-at) > 1e-9:
            raise ValueError('Terminal native tick must retain causal command issue and actual readback time')
        for key in ('phase','command_source_id','observation_source_id'): _text(tick[key],'terminal '+key)
        if tick['control_owner'] not in ('L2','L3'):
            raise ValueError('Terminal native tick requires explicit L2/L3 control owner')
        saved = dict(tick,time_s=at,command_issued_at_s=issue,command_readback_at_s=readback)
        for key in ('measured_q','q_command'): saved[key] = _array(tick[key],(7,),'terminal '+key).tolist()
        result.append(saved)
    return dict(schema=TERMINAL_SCHEMA,episode_id=metadata['episode_id'],target_id=metadata['target_id'],
        final_native_step_count=n,final_native_time_s=t,terminal_tick_count=remainder,
        suffix_boundary_step_count=count*12,suffix_boundary_time_s=count/20.,
        last_observation_step_count=observed_frames*12 if observed_frames else None,
        last_observation_time_s=observed_frames/20. if observed_frames else None,
        missing_observations=count-observed_frames,ticks=result,
        scope='completed trailing native ticks; not policy action labels or full-episode event/physics audit',
        h20_eligible=False,**QUALIFICATIONS)


class CompactTrainingCapture:
    """Append one observation and its next actual 20 Hz command endpoint.

    Choose exactly one duration mode: ``expected_frames`` for fixed offline
    recordings, or ``max_frames`` for adaptive native episodes. Both retain the
    byte cap and reject missing/nonconsecutive frames. Adaptive callers retain
    the final observed frame with append_terminal and explicitly close with
    actual episode-local native step count/time plus terminal_ticks;
    exiting a context without that final evidence aborts completion. A valid
    close checks record completeness, not task success or physics acceptance.
    Source bindings/counts supplied by the caller are attestations only; all
    eligibility flags remain false. No flush promises power-loss durability.
    """
    def __init__(self, output, *, metadata, camera_bytes, calibration_bytes,
                 code_bindings, expected_frames=None, max_frames=None, limit_bytes=EPISODE_LIMIT_BYTES):
        _require_episode_metadata(metadata)
        if type(limit_bytes) is not int or not RECEIPT_RESERVE < limit_bytes <= EPISODE_LIMIT_BYTES:
            raise ValueError('Byte limit must be <=100000000 and reserve a close receipt')
        if (expected_frames is None) == (max_frames is None):
            raise ValueError('Choose exactly one positive frame bound: expected_frames or max_frames')
        bound = expected_frames if expected_frames is not None else max_frames
        if type(bound) is not int or not 0 < bound <= MAX_FRAMES:
            raise ValueError('Positive integer frame bound must fit the supported signed int64 native counter')
        duration_mode = 'fixed' if expected_frames is not None else 'adaptive'
        meta = deepcopy(metadata)
        if meta['source_kind'] != 'cpu_synthetic_test' and sys.platform != 'linux':
            raise RuntimeError('Real/replay acquisition storage is Linux-only')
        if not isinstance(camera_bytes,bytes) or hashlib.sha256(camera_bytes).hexdigest() != APPROVED_CAMERA_SHA256:
            raise ValueError('Exact approved camera bytes required')
        if not isinstance(calibration_bytes,bytes): raise ValueError('Exact calibration bytes required')
        self.calibration = HallCalibration(calibration_bytes)
        code = deepcopy(code_bindings)
        required = ['native_backend','state_adapter','target_estimator','action_source','body']
        if meta['target_source'] == 'gt_plus_measured_noise': required.append('noise_model')
        for key in required: _hash(code.get(key),key)
        code.update(writer=_sha(__file__), action_contract=_sha(m2_contract.__file__))
        meta.update(QUALIFICATIONS)
        camera = json.loads(camera_bytes)
        intrinsics = camera['intrinsics']
        sx,sy = POLICY_SIZE[0]/intrinsics['width'],POLICY_SIZE[1]/intrinsics['height']
        # Camera JSON uses pixel-boundary/principal-center coordinates. Full
        # FOV resize scales them directly; no cropped/recentered intrinsics.
        policy_intrinsics = {key:intrinsics[key]*scale for key,scale in
                             (('fx',sx),('fy',sy),('cx',sx),('cy',sy))}
        policy_intrinsics.update(width=POLICY_SIZE[0],height=POLICY_SIZE[1])
        self._manifest = dict(schema=SCHEMA if duration_mode == 'fixed' else ADAPTIVE_SCHEMA,metadata=meta,code_bindings=code,
            camera_sha256=APPROVED_CAMERA_SHA256,calibration_sha256=self.calibration.sha256,
            policy_image_size=list(POLICY_SIZE),native_image_size=list(NATIVE_SIZE),
            policy_intrinsics=policy_intrinsics,camera_transform_unchanged=True,
            capture_hz=20,hall_adc_nominal_hz=100,depth_unit_m=DEPTH_UNIT_M,
            depth_storage_dtype=DEPTH_DTYPE.str,depth_max_m=DEPTH_MAX_M,
            depth_codec=DEPTH_CODEC,depth_keyframe_interval=DEPTH_KEYFRAME_INTERVAL,
            depth_semantics='metric_optical_z',depth_resize='nearest pixel center; no invalid interpolation',
            depth_invalid='zero with false validity; raw invalid sentinel value is not retained',
            rgb_encoding='JPEG quality85 subsampling0; full-FOV BOX resize',
            numeric_encoding='framed ZIP-deflated NPZ; little-endian float64/int64; no pickle',
            hall_downsampling='latest causally received packet; no averaging',
            hall_batch_window='first observation includes episode start; later receive times strictly after prior observation with 1e-9s tolerance',
            hall_role='separate L3 sensor input; never state17; force is calibrated spring-load estimate',
            action_semantics='absolute raw-radian actual command endpoint t+0.05; not an H20 chunk',
            source_binding_scope='caller attestations plus writer hashes; no authenticity/physics qualification',
            duration_mode=duration_mode,expected_frames=expected_frames,max_frames=bound,
            native_clock='episode-local 240 Hz steps; one-based step count, not zero-based row index',
            limit_bytes=limit_bytes,**QUALIFICATIONS)
        self._files = FILES if duration_mode == 'fixed' else FILES+('terminal.json',)
        if duration_mode == 'adaptive':
            self._manifest.update(terminal_schema=TERMINAL_SCHEMA,h20_eligible=False,
                terminal_observation='last native 20 Hz observation; action fields absent',
                terminal_tick_sources='native_backend/state_adapter/action_source bindings; caller attestations',
                terminal_audit_scope='suffix only; full-episode intervening event/physics audit still required')
        manifest = _json_bytes(self._manifest)
        initial = len(manifest)+len(camera_bytes)+len(calibration_bytes)
        if initial+RECEIPT_RESERVE > limit_bytes: raise ValueError('Metadata exceeds byte budget')
        self.output = Path(output)
        self._count = 0
        self._last_hall_time = -np.inf
        self._last_observed_time = None
        self._previous_depth = None
        self._failed = None
        self._closed = False
        self._receipt = None
        self._receipt_staging_owned = False
        self._receipt_staging_identity = None
        self._native_endpoint = None
        self._terminal_observation = False
        self._terminal_evidence = None
        self._rgb = self._numeric = self._terminal = None
        self.output.mkdir(parents=True,exist_ok=False)
        try:
            for name,data in (('manifest.json',manifest),('approved_camera.json',camera_bytes),
                              ('calibration.json',calibration_bytes)):
                with (self.output/name).open('xb') as stream: stream.write(data)
            self._rgb = (self.output/'rgb.mjpeg').open('xb')
            self._numeric = (self.output/'numeric.bin').open('xb')
            if duration_mode == 'adaptive': self._terminal = (self.output/'terminal.json').open('xb')
        except BaseException as error:
            self.abort('setup: '+str(error))
            raise

    def _encode(self, *, episode_id, target_id, observed_at_s, rgb, depth_m, depth_valid, state17, measured_q,
                estimated_target_base, target_confidence, target_source, target_estimate_time_s,
                observation_source_id, target_estimate_source_id, phase, control_owner,
                action, hall_packets, terminal=False):
        if (episode_id != self._manifest['metadata']['episode_id'] or
                target_id != self._manifest['metadata']['target_id']):
            raise ValueError('Frame crosses single-target episode context')
        t = _scalar(observed_at_s,'20 Hz observation time')
        if abs(t-(self._count+1)*.05) > 1e-9:
            raise ValueError('Observations must use consecutive episode 20 Hz clock .05,.10,...')
        if self._count >= self._manifest['max_frames']: raise ValueError('Extra frame exceeds predeclared maximum count')
        if target_source != self._manifest['metadata']['target_source']:
            raise ValueError('Frame target source differs from bound estimate provenance')
        estimate_time = _scalar(target_estimate_time_s,'target estimate timestamp')
        if not 0 <= estimate_time <= t or t-estimate_time > .05+1e-9:
            raise ValueError('Target estimate must be causal and at most one frame old')
        q = _array(measured_q,(7,),'measured_q')
        state = _array(state17,(17,),'state17')
        expected = m2_contract.encode_state(q,estimated_target_base,target_confidence,
                                            metadata=self._manifest['metadata'])
        if not np.allclose(state,expected,rtol=0,atol=1e-10):
            raise ValueError('state17 differs from measured joints and supplied estimated target adapter output')
        for value,label in ((observation_source_id,'observation source'),
                            (target_estimate_source_id,'target estimate source'),(phase,'phase')):
            _text(value,label)
        if control_owner not in ('L2','L3'): raise ValueError('Explicit L2/L3 control owner required')
        action_data = {}
        if not terminal:
            if not isinstance(action,dict) or set(action) != {'q_command','issued_at_s','endpoint_at_s','readback_at_s','source_id'}:
                raise ValueError('Explicit action command, issue, endpoint, readback and source required')
            command = _array(action['q_command'],(7,),'action command')
            issue = _scalar(action['issued_at_s'],'action issue')
            endpoint = _scalar(action['endpoint_at_s'],'action endpoint')
            readback = _scalar(action['readback_at_s'],'action readback')
            _text(action['source_id'],'action source')
            if (abs(endpoint-(t+.05)) > 1e-9 or not t <= issue <= endpoint or
                    abs(readback-endpoint) > 1e-9):
                raise ValueError('action requires causal issue and actual readback at future 20 Hz endpoint')
            action_data = dict(q_command=command,action_issued_at_s=np.float64(issue),
                action_endpoint_at_s=np.float64(endpoint),action_readback_at_s=np.float64(readback),
                action_source_id=action['source_id'])
        if not isinstance(hall_packets,(tuple,list)) or not 1 <= len(hall_packets) <= 6:
            raise ValueError('Hall requires a bounded list of causally received 100 Hz ADC packets')
        last = self._last_hall_time
        received = -np.inf
        for packet in hall_packets:
            if type(packet) is not HallPacket: raise ValueError('Hall input must be raw ADC packet, never a force label')
            if packet.sample_time_s <= last or packet.received_time_s < received:
                raise ValueError('Hall packets must have new ordered sample and receive timestamps')
            decoded = self.calibration.decode(packet,now_s=t,max_age_s=.05)
            if (self._last_observed_time is not None and
                    packet.received_time_s <= self._last_observed_time+1e-9):
                raise ValueError('Hall batch contains a packet already available at the prior observation')
            last,received = packet.sample_time_s,packet.received_time_s
        if not np.isfinite([decoded.hall_voltage_v,decoded.spring_force_estimate_n,decoded.displacement_estimate_m]).all():
            raise ValueError('Hall calibration produced nonfinite voltage/force/displacement')
        image = np.asarray(rgb)
        if image.shape not in ((360,640,3),(360,640,4)) or image.dtype != np.uint8:
            raise ValueError('RGB must be native 640x360 uint8 RGB/RGBA')
        depth = np.asarray(depth_m)
        validity = np.asarray(depth_valid)
        if depth.shape != (360,640) or depth.dtype.kind not in 'fiu': raise ValueError('Native metric depth shape/type mismatch')
        if validity.shape != depth.shape or validity.dtype != bool: raise ValueError('Explicit bool depth validity required')
        truth = np.isfinite(depth)&(depth>0)
        if not np.array_equal(validity,truth): raise ValueError('Depth validity disagrees with positive finite metric depth')
        if not truth.any() or np.any(depth[truth].astype(np.float64) > DEPTH_MAX_M):
            raise ValueError('Valid depth is empty or outside metric uint32 range; no saturation permitted')
        ys = ((np.arange(POLICY_SIZE[1])+.5)*NATIVE_SIZE[1]/POLICY_SIZE[1]).astype(int)
        xs = ((np.arange(POLICY_SIZE[0])+.5)*NATIVE_SIZE[0]/POLICY_SIZE[0]).astype(int)
        small_depth = depth[np.ix_(ys,xs)]
        valid = validity[np.ix_(ys,xs)]
        quantized = np.zeros(valid.shape,dtype=DEPTH_DTYPE)
        values = np.rint(small_depth[valid].astype(np.float64)/DEPTH_UNIT_M)
        if not valid.any() or np.any(values < 1): raise ValueError('Valid depth underflows metric quantization')
        quantized[valid] = values.astype(DEPTH_DTYPE)
        keyframe = self._count%DEPTH_KEYFRAME_INTERVAL == 0
        if not keyframe and self._previous_depth is None:
            raise ValueError('Temporal depth state is missing its preceding frame')
        depth_code = quantized if keyframe else np.bitwise_xor(quantized,self._previous_depth)
        jpeg = io.BytesIO()
        Image.fromarray(image[:,:,:3]).resize(POLICY_SIZE,resample=Image.Resampling.BOX).save(
            jpeg,format='JPEG',quality=85,subsampling=0)
        rgb_payload = jpeg.getvalue()
        data = dict(index=np.int64(self._count),episode_id=episode_id,target_id=target_id,
            observed_at_s=np.float64(t),state17=state,measured_q=q,
            estimated_target_base=np.asarray(estimated_target_base,dtype='<f8'),
            target_confidence=np.float64(target_confidence),target_estimate_time_s=np.float64(estimate_time),
            target_source=target_source,observation_source_id=observation_source_id,
            target_estimate_source_id=target_estimate_source_id,phase=phase,control_owner=control_owner,
            depth_code=depth_code,depth_keyframe=np.bool_(keyframe),
            depth_valid=valid,hall_adc=np.int64(packet.raw),
            hall_sample_time_s=np.float64(packet.sample_time_s),hall_received_time_s=np.float64(packet.received_time_s),
            hall_packets_consumed=np.int64(len(hall_packets)),hall_source=packet.source,
            hall_voltage_v=np.float64(decoded.hall_voltage_v),
            hall_spring_force_estimate_n=np.float64(decoded.spring_force_estimate_n),
            hall_displacement_estimate_m=np.float64(decoded.displacement_estimate_m),
            rgb_bytes=np.int64(len(rgb_payload)),rgb_crc32=np.int64(zlib.crc32(rgb_payload)))
        data.update(action_data)
        if self._manifest['duration_mode'] == 'adaptive':
            data['record_kind'] = 'terminal_observation' if terminal else 'paired'
        numeric = io.BytesIO(); np.savez_compressed(numeric,**data)
        return _framed(rgb_payload),_framed(numeric.getvalue()),last,quantized

    def append(self, **sample):
        """Append an observation paired with the actual next command endpoint."""
        return self._append(sample,terminal=False)

    def append_terminal(self, **observation):
        """Append the final native observation, with no future action argument."""
        return self._append(observation,terminal=True)

    def _append(self, sample, *, terminal):
        if self._failed is not None: raise RuntimeError('Capture already failed')
        if self._closed: raise RuntimeError('Capture is closed')
        try:
            if self._terminal_observation: raise ValueError('No frame may follow the terminal observation')
            if terminal:
                if self._manifest['duration_mode'] != 'adaptive':
                    raise ValueError('Terminal observations require adaptive mode')
                if 'action' in sample: raise ValueError('Terminal observation cannot contain an action label')
                sample = dict(sample,action=None)
            try: rgb,numeric,last,quantized = self._encode(**sample,terminal=terminal)
            except TypeError as error: raise ValueError('Missing/invalid recording modality: '+str(error)) from error
            current = sum(path.stat().st_size for path in self.output.iterdir() if path.is_file())
            if current+len(rgb)+len(numeric)+RECEIPT_RESERVE > self._manifest['limit_bytes']:
                raise ValueError('Single-target episode byte budget exceeded; no successful truncated recording')
            for stream,payload in ((self._rgb,rgb),(self._numeric,numeric)):
                if stream.write(payload) != len(payload): raise OSError('Short capture write')
                stream.flush()
            self._previous_depth = quantized
            self._last_hall_time = last
            self._last_observed_time = float(sample['observed_at_s'])
            self._count += 1
            self._terminal_observation = terminal
        except BaseException as error:
            self._failed = (type(error).__name__+': '+str(error))[:2048]
            raise

    def _check_owned_staging(self, staging):
        value = staging.lstat()
        if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or
                (value.st_dev,value.st_ino) != self._receipt_staging_identity):
            raise ValueError('Owned receipt staging was replaced, linked or is not a regular file')

    def _open_receipt_staging(self, staging):
        if not self._receipt_staging_owned:
            stream = staging.open('xb')
            value = os.fstat(stream.fileno())
            self._receipt_staging_identity = (value.st_dev,value.st_ino)
            self._receipt_staging_owned = True
            return stream
        self._check_owned_staging(staging)
        # Do not truncate until the opened descriptor is proven to be the
        # exclusively owned inode; never follow a substituted symlink.
        descriptor = os.open(staging,os.O_WRONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            value = os.fstat(descriptor)
            if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or
                    (value.st_dev,value.st_ino) != self._receipt_staging_identity):
                raise ValueError('Owned receipt staging inode changed before retry')
            os.ftruncate(descriptor,0)
            return os.fdopen(descriptor,'wb')
        except BaseException:
            os.close(descriptor)
            raise

    def _finish(self):
        if self._closed: return self._receipt
        for stream in (self._rgb,self._numeric,self._terminal):
            if stream is not None:
                try: stream.close()
                except Exception as error: self._failed = 'close: '+str(error)[:2048]
        if self._failed is None:
            if self._manifest['duration_mode'] == 'fixed' and self._count != self._manifest['expected_frames']:
                self._failed = 'Partial recording: expected '+str(self._manifest['expected_frames'])+' frames'
            elif self._manifest['duration_mode'] == 'adaptive':
                if self._terminal_evidence is None:
                    self._failed = 'Adaptive recording requires explicit final native evidence'
                elif (self._count != self._native_endpoint[0]//12 or
                        self._terminal_observation != (self._count > 0)):
                    self._failed = 'Adaptive native clock requires every observation and its final terminal observation'
        staging = self.output/'.receipt.json.tmp'
        owned_staging = {staging.name} if self._receipt_staging_owned and staging.exists() else set()
        if ({path.name for path in self.output.iterdir()} != set(self._files)|owned_staging or
                any(path.is_symlink() or not path.is_file() for path in self.output.iterdir())):
            self._failed = 'Unexpected, missing or symlink recording artifact'
        artifacts = {}
        for name in self._files:
            path = self.output/name
            if path.is_file() and not path.is_symlink(): artifacts[name] = dict(bytes=path.stat().st_size,sha256=_sha(path))
            else: self._failed = 'Missing recording artifact: '+name
        self._failed = _bounded_failure(self._failed)
        receipt = dict(schema=self._manifest['schema'],completed=self._failed is None,frames=self._count,
            duration_mode=self._manifest['duration_mode'],expected_frames=self._manifest['expected_frames'],
            max_frames=self._manifest['max_frames'],failure=self._failed,artifacts=artifacts,
            final_native_step_count=None if self._native_endpoint is None else self._native_endpoint[0],
            final_native_time_s=None if self._native_endpoint is None else self._native_endpoint[1],
            limit_bytes=self._manifest['limit_bytes'],total_bytes=0,**QUALIFICATIONS)
        if self._manifest['duration_mode'] == 'adaptive':
            receipt.update(complete_pairs=self._count-int(self._terminal_observation),
                terminal_observations=int(self._terminal_observation),h20_eligible=False,
                terminal_tick_count=None if self._native_endpoint is None else self._native_endpoint[0]%12,
                terminal_evidence_complete=self._terminal_evidence is not None,
                recording_scope='observations_with_terminal_native_suffix' if self._count else
                    'native_ticks_only_no_policy_observations')
        initial = sum(path.stat().st_size for path in self.output.iterdir()
                      if path.is_file() and path.name not in owned_staging)
        while receipt['total_bytes'] != initial+len(_json_bytes(receipt)):
            receipt['total_bytes'] = initial+len(_json_bytes(receipt))
        try:
            if receipt['total_bytes'] > receipt['limit_bytes']:
                raise RuntimeError('Capture byte budget exceeded during receipt finalization')
            payload = _json_bytes(receipt)
            if 2*len(payload) > RECEIPT_RESERVE:
                raise RuntimeError('Receipt exceeds reserved serialized byte budget')
            with self._open_receipt_staging(staging) as stream:
                if stream.write(payload) != len(payload): raise OSError('Short receipt write')
                stream.flush()
            self._check_owned_staging(staging)
            # Atomic no-clobber publication: an existing receipt is never
            # overwritten. The reserve covers both names before unlinking.
            os.link(staging,self.output/'receipt.json',follow_symlinks=False)
            staging.unlink()
        except Exception as error:
            self._failed = _bounded_failure('receipt finalization: '+str(error))
            raise
        self._closed = True
        self._receipt = receipt
        return deepcopy(receipt)

    def _save_terminal(self, final_native_step_count, final_native_time_s, terminal_ticks):
        evidence = _terminal_evidence(self._manifest['metadata'],final_native_step_count,
                                      final_native_time_s,terminal_ticks,self._count)
        if self._terminal_evidence is not None:
            if self._terminal_evidence != evidence: raise ValueError('Final native evidence cannot change on retry')
            return
        # Validated clock evidence fits the reserved receipt even when the
        # suffix payload cannot be persisted. That failure remains incomplete.
        self._native_endpoint = (final_native_step_count,evidence['final_native_time_s'])
        payload = _json_bytes(evidence)
        current = sum(path.stat().st_size for path in self.output.iterdir() if path.is_file())
        if current+len(payload)+RECEIPT_RESERVE > self._manifest['limit_bytes']:
            raise ValueError('Terminal native payload exceeds single-target episode byte budget')
        if self._terminal.write(payload) != len(payload): raise OSError('Short terminal native evidence write')
        self._terminal.flush()
        self._terminal_evidence = evidence

    def close(self, *, final_native_step_count=None, final_native_time_s=None, terminal_ticks=None):
        """Close with actual, unrounded episode-local native endpoint evidence.

        Fixed v3 retains the original all-paired endpoint contract. Adaptive
        v4 requires N//12 observations (last one terminal) plus N%12 actual
        trailing ticks, including explicit [] for zero ticks. Ending before
        step 12 can complete a ticks-only recording with zero policy pairs.
        Completion concerns supplied recording evidence, never task success.
        """
        supplied = any(value is not None for value in (final_native_step_count,final_native_time_s,terminal_ticks))
        if not self._closed and (supplied or
                (self._failed is None and self._manifest['duration_mode'] == 'adaptive')):
            try:
                if self._manifest['duration_mode'] == 'adaptive':
                    self._save_terminal(final_native_step_count,final_native_time_s,terminal_ticks)
                else:
                    if terminal_ticks is not None: raise ValueError('Terminal native ticks require adaptive mode')
                    stamp = _scalar(final_native_time_s,'final native time')
                    if (type(final_native_step_count) is not int or self._count == 0 or
                            final_native_step_count != 12*(self._count+1) or
                            abs(stamp-final_native_step_count/240.) > 1e-9):
                        raise ValueError('Final native step count/time must equal the last paired action endpoint')
                    self._native_endpoint = (final_native_step_count,stamp)
            except (ValueError,TypeError,KeyError,OSError) as error:
                self._failed = _bounded_failure((self._failed+'; ' if self._failed else '')+
                                                'Final native evidence: '+str(error))
        receipt = self._finish()
        if not receipt['completed']: raise RuntimeError('Capture incomplete: '+str(receipt['failure']))
        return deepcopy(receipt)

    def abort(self, reason, *, final_native_step_count=None, final_native_time_s=None, terminal_ticks=None):
        """Keep valid partial native evidence if available; never mark complete."""
        if not self._closed:
            self._failed = ('Aborted: '+str(reason))[:2048]
            if any(value is not None for value in (final_native_step_count,final_native_time_s,terminal_ticks)):
                try:
                    if self._manifest['duration_mode'] != 'adaptive':
                        raise ValueError('Abort terminal native evidence requires adaptive mode')
                    self._save_terminal(final_native_step_count,final_native_time_s,terminal_ticks)
                except (ValueError,TypeError,KeyError,OSError) as error:
                    self._failed = _bounded_failure(self._failed+'; terminal native evidence invalid: '+str(error))
        return self._finish()

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None: self.close()
        else: self.abort(str(exc))
        return False


def _read_frame(stream):
    header = stream.read(HEADER.size)
    if len(header) != HEADER.size: raise ValueError('Recording integrity: truncated frame header')
    length,expected_crc = HEADER.unpack(header)
    if not 0 < length <= MAX_RECORD_BYTES: raise ValueError('Recording integrity: invalid bounded frame length')
    data = stream.read(length)
    if len(data) != length or zlib.crc32(data) != expected_crc:
        raise ValueError('Recording integrity: truncated/corrupt frame payload')
    return data


def _read_numeric_payload(payload):
    """Read the bounded, comment-free ZIP/NPY container emitted by the writer.

    NumPy/zipfile intentionally tolerate appended bytes and duplicate names.
    A capture record cannot: every framed byte must belong to one member or
    the ZIP directory, and every member must contain exactly one NPY array.
    """
    try:
        if len(payload) < 22 or not payload.startswith(b'PK\x03\x04'):
            raise ValueError('numeric payload is not a ZIP container')
        signature,disk,central_disk,on_disk,count,directory_size,directory_at,comment = struct.unpack(
            '<4s4H2IH',payload[-22:])
        if (signature != b'PK\x05\x06' or disk or central_disk or comment or on_disk != count or
                not 0 < count <= 64 or directory_at+directory_size != len(payload)-22):
            raise ValueError('numeric ZIP directory or trailing bytes mismatch')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if (len(members) != count or len(set(names)) != count or
                    sum(item.file_size for item in members) > MAX_RECORD_BYTES):
                raise ValueError('numeric ZIP has duplicate members or exceeds decoded byte bound')
            arrays = {}
            offset = 0
            for member in members:
                if (member.header_offset != offset or member.compress_type != zipfile.ZIP_DEFLATED or
                        member.flag_bits or member.comment or not member.filename.endswith('.npy') or
                        '/' in member.filename or '\\' in member.filename):
                    raise ValueError('numeric ZIP member format or contiguous boundary mismatch')
                header = payload[offset:offset+30]
                if len(header) != 30: raise ValueError('truncated numeric ZIP member header')
                fields = struct.unpack('<4s5H3I2H',header)
                if fields[0] != b'PK\x03\x04' or fields[2] or fields[3] != zipfile.ZIP_DEFLATED:
                    raise ValueError('numeric ZIP local header mismatch')
                compressed_at = offset+30+fields[-2]+fields[-1]
                offset = compressed_at+member.compress_size
                if offset > directory_at: raise ValueError('numeric ZIP member exceeds payload boundary')
                decoder = zlib.decompressobj(-15)
                decoded = decoder.decompress(payload[compressed_at:offset],member.file_size+1)
                if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or
                        len(decoded) != member.file_size):
                    raise ValueError('numeric DEFLATE member boundary or decoded size mismatch')
                # zipfile additionally verifies the local filename and CRC.
                raw = archive.read(member)
                if not raw.startswith(b'\x93NUMPY'): raise ValueError('numeric ZIP member is not NPY')
                stream = io.BytesIO(raw)
                value = np.load(stream,allow_pickle=False)
                if not isinstance(value,np.ndarray) or stream.tell() != len(raw):
                    raise ValueError('numeric NPY member contains trailing bytes')
                arrays[member.filename[:-4]] = value
            if offset != directory_at: raise ValueError('numeric ZIP has unbound bytes before directory')
            return arrays
    except (ValueError,TypeError,KeyError,OSError,EOFError,zipfile.BadZipFile,struct.error,zlib.error) as error:
        raise ValueError('Recording integrity: invalid numeric container: '+str(error)) from error


def _check_jpeg_payload(payload):
    """Locate the first real EOI marker; reject PNG and bytes after one JPEG."""
    if not payload.startswith(b'\xff\xd8'):
        raise ValueError('Recording integrity: RGB payload must be JPEG')
    offset = 2
    scan = False
    while offset < len(payload):
        if scan:
            offset = payload.find(b'\xff',offset)
            if offset < 0: break
        if payload[offset] != 0xff: break
        while offset < len(payload) and payload[offset] == 0xff: offset += 1
        if offset == len(payload): break
        marker = payload[offset]
        offset += 1
        if scan and (marker == 0 or 0xd0 <= marker <= 0xd7): continue
        if marker == 0xd9:
            if offset != len(payload): raise ValueError('Recording integrity: trailing bytes after JPEG')
            return
        if marker in (0,0xd8) or 0xd0 <= marker <= 0xd7 or offset+2 > len(payload): break
        length = int.from_bytes(payload[offset:offset+2],'big')
        if length < 2 or offset+length > len(payload): break
        offset += length
        scan = marker == 0xda
    raise ValueError('Recording integrity: incomplete or malformed JPEG container')


def _read_bundle(output):
    root = Path(output)
    receipt = json.loads((root/'receipt.json').read_text())
    if receipt.get('completed') is not True: raise RuntimeError('Recording is incomplete')
    v4 = receipt.get('schema') == ADAPTIVE_SCHEMA
    files = FILES+('terminal.json',) if v4 else FILES
    names = {p.name for p in root.iterdir()}
    if names != set(files)|{'receipt.json'} or any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        raise ValueError('Recording integrity: unexpected/symlink/missing file')
    actual_size = sum(p.stat().st_size for p in root.iterdir())
    if actual_size != receipt['total_bytes'] or actual_size > min(receipt['limit_bytes'],EPISODE_LIMIT_BYTES):
        raise ValueError('Recording integrity: total byte count/budget mismatch')
    for name in files:
        expected = receipt['artifacts'].get(name,{})
        if (root/name).stat().st_size != expected.get('bytes') or _sha(root/name) != expected.get('sha256'):
            raise ValueError('Recording integrity: artifact hash/size changed: '+name)
    manifest = json.loads((root/'manifest.json').read_text())
    try: _require_episode_metadata(manifest['metadata'])
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError('Recording integrity: malformed episode provenance: '+str(error)) from error
    frames = receipt['frames']
    mode = manifest.get('duration_mode')
    duration_valid = (type(frames) is int and (0 <= frames if v4 else 0 < frames) and
        type(manifest.get('max_frames')) is int and 0 < manifest['max_frames'] <= MAX_FRAMES and
        frames <= manifest['max_frames'] and
        receipt.get('max_frames') == manifest.get('max_frames') and receipt.get('duration_mode') == mode and
        receipt.get('expected_frames') == manifest.get('expected_frames'))
    terminal = None
    if mode == 'fixed':
        duration_valid &= not v4 and frames == manifest['expected_frames']
    elif mode == 'adaptive' and v4:
        try:
            terminal = json.loads((root/'terminal.json').read_text())
            expected = _terminal_evidence(manifest['metadata'],receipt['final_native_step_count'],
                                          receipt['final_native_time_s'],terminal['ticks'],frames)
            # Canonical JSON comparison keeps bool/int/float types distinct;
            # Python dict equality alone would accept False == 0 == 0.0.
            duration_valid &= (manifest['expected_frames'] is None and _json_bytes(terminal) == _json_bytes(expected) and
                receipt.get('failure') is None and type(receipt.get('limit_bytes')) is int and
                receipt.get('terminal_evidence_complete') is True and
                type(manifest.get('limit_bytes')) is int and
                RECEIPT_RESERVE < receipt['limit_bytes'] == manifest['limit_bytes'] <= EPISODE_LIMIT_BYTES and
                frames == expected['final_native_step_count']//12 and
                type(receipt.get('terminal_tick_count')) is int and
                receipt['terminal_tick_count'] == expected['terminal_tick_count'] and
                type(receipt.get('complete_pairs')) is int and receipt['complete_pairs'] == max(frames-1,0) and
                type(receipt.get('terminal_observations')) is int and receipt['terminal_observations'] == int(frames > 0) and
                receipt.get('recording_scope') == ('observations_with_terminal_native_suffix' if frames else
                    'native_ticks_only_no_policy_observations') and
                manifest.get('terminal_schema') == TERMINAL_SCHEMA and
                manifest.get('h20_eligible') is False and receipt.get('h20_eligible') is False)
            if any(manifest['metadata'].get(key) is not False for key in QUALIFICATIONS):
                raise ValueError('false metadata qualification')
            for key in ('native_backend','state_adapter','target_estimator','action_source','body','writer','action_contract'):
                _hash(manifest['code_bindings'].get(key),key)
            if manifest['metadata']['target_source'] == 'gt_plus_measured_noise':
                _hash(manifest['code_bindings'].get('noise_model'),'noise_model')
            if (manifest['camera_sha256'] != APPROVED_CAMERA_SHA256 or
                    _sha(root/'approved_camera.json') != manifest['camera_sha256'] or
                    _sha(root/'calibration.json') != manifest['calibration_sha256']):
                raise ValueError('camera/calibration source binding mismatch')
        except (ValueError,TypeError,KeyError) as error:
            raise ValueError('Recording integrity: malformed terminal native evidence: '+str(error)) from error
    elif mode == 'adaptive':
        duration_valid &= (manifest['expected_frames'] is None and
            receipt.get('final_native_step_count') == 12*(frames+1) and
            isinstance(receipt.get('final_native_time_s'),(int,float)) and
            np.isfinite(receipt['final_native_time_s']) and
            abs(receipt['final_native_time_s']-(frames+1)*.05) <= 1e-9)
    else:
        duration_valid = False
    if (manifest.get('schema') not in (SCHEMA,ADAPTIVE_SCHEMA) or
            receipt.get('schema') != manifest.get('schema') or not duration_valid or
            manifest.get('depth_storage_dtype') != DEPTH_DTYPE.str or
            manifest.get('depth_codec') != DEPTH_CODEC or
            manifest.get('depth_keyframe_interval') != DEPTH_KEYFRAME_INTERVAL or
            manifest.get('depth_unit_m') != DEPTH_UNIT_M or manifest.get('depth_max_m') != DEPTH_MAX_M or
            any(manifest.get(key) is not False or receipt.get(key) is not False for key in QUALIFICATIONS)):
        raise ValueError('Recording integrity: unsupported contract or false qualification')
    return root,receipt,manifest,terminal


def _record_value(data, key, *, dtype=None):
    value = data[key]
    if not isinstance(value,np.ndarray) or value.shape != () or (dtype is not None and value.dtype != np.dtype(dtype)):
        raise ValueError('Invalid numeric scalar '+key)
    return value.item()


def _validate_v4_record(data, index, frames, manifest, calibration, last_hall):
    """Check terminal/action absence, source context and observation-time Hall."""
    try:
        terminal = index == frames-1
        kind = _record_value(data,'record_kind')
        if kind != ('terminal_observation' if terminal else 'paired'):
            raise ValueError('terminal observation position/kind mismatch')
        common = {'index','episode_id','target_id','observed_at_s','state17','measured_q',
            'estimated_target_base','target_confidence','target_estimate_time_s','target_source',
            'observation_source_id','target_estimate_source_id','phase','control_owner','depth_code',
            'depth_keyframe','depth_valid','hall_adc','hall_sample_time_s','hall_received_time_s',
            'hall_packets_consumed','hall_source','hall_voltage_v','hall_spring_force_estimate_n',
            'hall_displacement_estimate_m','rgb_bytes','rgb_crc32','record_kind'}
        if set(data) != common | (set() if terminal else ACTION_FIELDS):
            raise ValueError('terminal/action record fields mismatch')
        meta = manifest['metadata']
        for key in ('episode_id','target_id','target_source'):
            if _record_value(data,key) != meta[key]: raise ValueError('record source context mismatch')
        for key in ('observation_source_id','target_estimate_source_id','phase'):
            _text(_record_value(data,key),key)
        if _record_value(data,'control_owner') not in ('L2','L3'): raise ValueError('invalid control owner')
        for key in ('index','rgb_bytes','rgb_crc32','hall_adc','hall_packets_consumed'):
            _record_value(data,key,dtype='<i8')
        times = {key:_scalar(_record_value(data,key,dtype='<f8'),key) for key in
                 ('observed_at_s','target_estimate_time_s','hall_sample_time_s','hall_received_time_s')}
        t = times['observed_at_s']
        if not 0 <= times['target_estimate_time_s'] <= t or t-times['target_estimate_time_s'] > .05+1e-9:
            raise ValueError('target estimate is not causal')
        for key,shape in (('measured_q',(7,)),('state17',(17,)),('estimated_target_base',(4,4))):
            _array(data[key],shape,key)
            if data[key].dtype != np.dtype('<f8'): raise ValueError('invalid '+key+' dtype')
        confidence = _scalar(_record_value(data,'target_confidence',dtype='<f8'),'target confidence')
        expected = m2_contract.encode_state(data['measured_q'],data['estimated_target_base'],confidence,metadata=meta)
        if not np.allclose(data['state17'],expected,rtol=0,atol=1e-10): raise ValueError('state17 source mismatch')
        if not terminal:
            _array(data['q_command'],(7,),'action command')
            if data['q_command'].dtype != np.dtype('<f8'): raise ValueError('invalid action dtype')
            issue,endpoint,readback = [_scalar(_record_value(data,key,dtype='<f8'),key) for key in
                ('action_issued_at_s','action_endpoint_at_s','action_readback_at_s')]
            _text(_record_value(data,'action_source_id'),'action source')
            if abs(endpoint-t-.05) > 1e-9 or not t <= issue <= endpoint or abs(readback-endpoint) > 1e-9:
                raise ValueError('action endpoint/readback mismatch')
        packet = HallPacket(_record_value(data,'hall_adc'),times['hall_sample_time_s'],
                           times['hall_received_time_s'],calibration.sha256,_record_value(data,'hall_source'))
        decoded = calibration.decode(packet,now_s=t,max_age_s=.05)
        if (packet.sample_time_s <= last_hall or not 1 <= _record_value(data,'hall_packets_consumed') <= 6 or
                (index > 0 and packet.received_time_s <= index*.05+1e-9)):
            raise ValueError('Hall packet reuses previous observation window')
        for key,expected in (('hall_voltage_v',decoded.hall_voltage_v),
            ('hall_spring_force_estimate_n',decoded.spring_force_estimate_n),
            ('hall_displacement_estimate_m',decoded.displacement_estimate_m)):
            if _scalar(_record_value(data,key,dtype='<f8'),key) != expected: raise ValueError('Hall calibration mismatch')
        return packet.sample_time_s
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError('Recording integrity: malformed v4 observation: '+str(error)) from error


def read_terminal_evidence(output):
    """Verify a complete v4 bundle and return its explicit native suffix ledger.

    Incomplete/aborted receipts are rejected by public fidelity readers; their
    terminal.json still retains any valid partial evidence supplied to abort.
    """
    for _ in iter_training_records(output): pass
    _,_,_,terminal = _read_bundle(output)
    if terminal is None: raise ValueError('Terminal native evidence requires a v4 adaptive recording')
    return terminal


def iter_training_records(output):
    """Verify bytes, then stream v3 pairs or v4 pairs plus a terminal observation.

    V4's last observation has record_kind=terminal_observation and NO action
    keys. This is a fidelity reader, never a qualification or H20 exporter.
    Exhaust the iterator to verify absence of trailing records. Hash receipts
    are not signatures; independent trusted source hashes remain necessary.
    """
    root,receipt,manifest,_ = _read_bundle(output)
    v4 = manifest['schema'] == ADAPTIVE_SCHEMA
    calibration = HallCalibration((root/'calibration.json').read_bytes()) if v4 else None
    last_hall = -np.inf
    with (root/'rgb.mjpeg').open('rb') as rgb, (root/'numeric.bin').open('rb') as numeric:
        previous_depth = None
        for index in range(receipt['frames']):
            image = _read_frame(rgb)
            data = _read_numeric_payload(_read_frame(numeric))
            if v4: last_hall = _validate_v4_record(data,index,receipt['frames'],manifest,calibration,last_hall)
            if (data['index'] != index or data['rgb_bytes'] != len(image) or
                    data['rgb_crc32'] != zlib.crc32(image) or
                    abs(float(data['observed_at_s'])-(index+1)*.05) > 1e-9):
                raise ValueError('Recording integrity: frame alignment mismatch')
            code,valid = data.get('depth_code'),data.get('depth_valid')
            keyframe = data.get('depth_keyframe')
            if (not isinstance(code,np.ndarray) or not isinstance(valid,np.ndarray) or
                    not isinstance(keyframe,np.ndarray) or keyframe.shape != () or keyframe.dtype != np.dtype(bool) or
                    bool(keyframe) != (index%DEPTH_KEYFRAME_INTERVAL == 0) or
                    code.shape != (POLICY_SIZE[1],POLICY_SIZE[0]) or valid.shape != code.shape or
                    code.dtype != DEPTH_DTYPE or valid.dtype != np.dtype(bool) or not valid.any()):
                raise ValueError('Recording integrity: temporal depth keyframe, dtype, shape or validity mismatch')
            if not bool(keyframe) and previous_depth is None:
                raise ValueError('Recording integrity: first depth frame cannot be a delta')
            depth = code.copy() if bool(keyframe) else np.bitwise_xor(code,previous_depth)
            if np.any(depth[valid] == 0) or np.any(depth[~valid] != 0):
                raise ValueError('Recording integrity: reconstructed metric depth validity mismatch')
            # Downstream callers may transform their yielded depth in place.
            # The next delta must use a distinct internal, unmodified baseline.
            previous_depth = depth.copy()
            data['depth_q'] = depth
            _check_jpeg_payload(image)
            with Image.open(io.BytesIO(image)) as decoded:
                if decoded.format != 'JPEG' or decoded.size != POLICY_SIZE or decoded.mode != 'RGB':
                    raise ValueError('Recording integrity: RGB format mismatch')
                data['rgb'] = np.asarray(decoded).copy()
            yield data
        if rgb.read(1) or numeric.read(1): raise ValueError('Recording integrity: trailing frames')
