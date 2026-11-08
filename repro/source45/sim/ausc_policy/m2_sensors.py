"""Append-only 5 mm M2 Hall acquisition, independent of policy state encoding.

Caller supplies each actual 240 Hz slider sample. The pinned calibration maps
travel to voltage, then the existing RP2040 model produces nominal 100 Hz ADC
packets. This records a sensor simulation, not bench-verified patient pressure
or proof that the caller's samples originated in Isaac. Episode-level evidence
must bind these rows to actual physics. All training eligibility stays false.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np

from .geometry_epoch import require_current
from ..auscultation_vla_v2.hall import HallCalibration, HallPacket, HallSimulator
from ..auscultation_pi05_v6.hall_conditions import validate_condition

SCHEMA = 'm2_hall_adc100_physics240_5mm_v1'
DT = 1./240.
ADC_PERIOD_S = .01
ARTIFACTS = ('manifest.json','calibration.json','physics.jsonl',
             'adc_acquisition.jsonl','hall.jsonl')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    """Bound memory while hashing long episode ledgers or source files."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _write(path, value):
    with Path(path).open('x') as stream:
        stream.write(_json(value)+'\n')


def _append(stream, value):
    stream.write(_json(value)+'\n')
    stream.flush()


def _raw_packet_record(packet):
    """Represent even invalid packet fields without converting them to valid ADC.

    Nonfinite Python floats retain their exact IEEE754 bits (including NaN
    payload), while unusual NumPy scalars retain dtype and bytes. These tags are
    evidence only and must never be passed back to the normal Hall decoder.
    """
    valid_json = True
    def preserve(value):
        nonlocal valid_json
        if type(value) is float and not np.isfinite(value):
            valid_json = False
            return dict(invalid_type='nonfinite_float',repr=repr(value),
                        ieee754_binary64_be=struct.pack('>d',value).hex())
        if isinstance(value,np.generic):
            valid_json = False
            return dict(invalid_type='numpy_scalar',repr=repr(value),
                        dtype=value.dtype.str,bytes_hex=value.tobytes().hex())
        if value is None or type(value) in (str,int,float,bool): return value
        valid_json = False
        if isinstance(value,bytes):
            return dict(invalid_type='bytes',bytes_hex=value.hex())
        # A malformed non-scalar packet is never eligible. Preserve its exact
        # diagnostic repr/type rather than coercing it into a numeric field.
        return dict(invalid_type=type(value).__module__+'.'+type(value).__qualname__,
                    repr=repr(value))
    record = {key:preserve(value) for key,value in asdict(packet).items()}
    return record,valid_json


def _sources(calibration_factory, simulator_factory, calibration):
    objects = dict(stream=M2HallStream, calibration=calibration_factory,
                   simulator=simulator_factory, host_calibration=type(calibration._host))
    sources = {}
    for label, obj in objects.items():
        path = Path(inspect.getsourcefile(obj)).resolve()
        sources[label] = dict(module=obj.__module__, path=str(path), sha256=_file_sha256(path))
    return sources


def _source_identity(sources):
    # Recorded host paths aid provenance but do not prevent replay on the Mac
    # from identical Linux source bytes. Module identity and bytes must match.
    return {label: {key: record[key] for key in ('module', 'sha256')}
            for label, record in sources.items()}


def _validate_step(physics_row, time_s, compression_m, expected_row, previous_time):
    if type(physics_row) is not int or physics_row != expected_row:
        raise ValueError('Missing, duplicated or invalid 240 Hz physics row')
    for value in (time_s, compression_m):
        if (not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_)) or not np.isfinite(value)):
            raise ValueError('Physics time and compression must be finite numeric scalars')
    if (abs(time_s-(physics_row+1)*DT) > 1e-8 or time_s <= previous_time
            or abs(time_s-previous_time-DT) > 1e-8):
        raise ValueError('Physics time must be consecutive actual 240 Hz time from the first step')


class M2HallStream:
    """One new output directory per episode; a failure permanently closes input.

    ``calibration_bytes`` are authoritative and saved verbatim. Injectable
    factories receive those bytes / that calibration plus the explicit model
    parameters, and their source hashes are bound too. Append flushes each line
    but does not claim power-loss durability. ``finalize`` closes both ledgers
    even after an acquisition failure and never grants training eligibility.
    """
    def __init__(self, output_dir, *, geometry_metadata, calibration_bytes, episode_id,
                 condition=None, calibration_factory=HallCalibration,
                 simulator_factory=HallSimulator):
        require_current(geometry_metadata)
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Explicit episode identity required')
        if not isinstance(calibration_bytes, bytes):
            raise ValueError('Exact calibration bytes required')
        parameters = dict(sample_period_s=ADC_PERIOD_S, drift_v_s=0.,
                          **validate_condition(condition))
        self.calibration = calibration_factory(calibration_bytes)
        if self.calibration.sha256 != _digest(calibration_bytes):
            raise ValueError('Injected calibration does not bind exact source bytes')
        self.sensor = simulator_factory(self.calibration, **parameters)
        sources = _sources(calibration_factory, simulator_factory, self.calibration)
        self._metadata = dict(schema=SCHEMA, episode_id=episode_id,
            geometry=deepcopy(geometry_metadata), physics_hz=240, adc_nominal_hz=100,
            parameters=parameters, calibration_sha256=self.calibration.sha256,
            bench_verified=bool(self.calibration.bench_verified), source_bindings=sources,
            injected_backend=calibration_factory is not HallCalibration or simulator_factory is not HallSimulator,
            input='caller-supplied consecutive slider displacement and physics time',
            adc_source='simulated_adc', force_label_used=False,
            scope='Hall sensor simulation; spring estimate is not bench-verified patient pressure',
            training_eligible=False)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'calibration.json').write_bytes(calibration_bytes)
        _write(self.output/'manifest.json', self._metadata)
        self._manifest_sha256 = _file_sha256(self.output/'manifest.json')
        self._physics = self._adc = self._hall = None
        try:
            self._physics = (self.output/'physics.jsonl').open('x')
            self._adc = (self.output/'adc_acquisition.jsonl').open('x')
            self._hall = (self.output/'hall.jsonl').open('x')
        except Exception:
            if self._physics is not None: self._physics.close()
            if self._adc is not None: self._adc.close()
            raise
        self._physics_rows = self._hall_rows = self._packet_count = self._adc_rows = 0
        self._previous_time = 0.
        self._last_packet = None
        self._packet_id = None
        self._failure = None
        self._closed = False
        self._receipt = None
        self._latest = None

    def snapshot(self):
        return deepcopy(self._metadata)

    def latest_evidence(self):
        """Copy the latest completed live append, never a decoder estimate alone.

        No disk schema or append return changes. A bounded manifest hash binds
        the stream identity, while the raw displacement and complete returned
        packet are retained from the same successfully flushed append.
        """
        if (self._closed or self._failure is not None or self._latest is None or
                self._physics_rows != self._hall_rows or self._packet_count != self._adc_rows or
                self._latest['physics']['physics_row'] != self._physics_rows-1):
            raise RuntimeError('Live Hall evidence requires an active complete successful append')
        if _file_sha256(self.output/'manifest.json') != self._manifest_sha256:
            raise RuntimeError('Live Hall source manifest changed')
        return deepcopy(dict(self._latest, snapshot=self._metadata, manifest_sha256=self._manifest_sha256))

    def append(self, physics_row, time_s, compression_m):
        """Persist displacement first, then acquired ADC and same-calibration decode.

        Every new ADC packet is independently flushed to adc_acquisition.jsonl
        BEFORE decode. Repeated reads and decoded rows reference that acquisition
        ID. The return also includes packet bytes for a camera consumer. Latency yields
        an explicit pending observation, never an invented ADC or force value.
        """
        if self._failure is not None:
            raise RuntimeError('Hall acquisition already failed; preserve this episode')
        if self._closed:
            raise RuntimeError('Hall acquisition is finalized')
        if self._physics_rows != self._hall_rows:
            self._failure = dict(stage='incomplete_ledgers',error='Previous physics step has no completed Hall row')
            raise RuntimeError('Hall acquisition already failed during an interrupted step')
        hall_row = None
        stage = 'validate_physics'
        try:
            _validate_step(physics_row,time_s,compression_m,self._physics_rows,self._previous_time)
            physical = dict(physics_row=physics_row,time_s=float(time_s),compression_m=float(compression_m),
                            training_eligible=False)
            stage = 'physics_write'
            _append(self._physics, physical)
            self._physics_rows += 1
            self._previous_time = float(time_s)
            hall_row = dict(physics_row=physics_row,time_s=float(time_s),packet_id=None,acquisition_id=None,packet=None,
                is_new_packet=False,decoded=None,status='pending',error=None,training_eligible=False)
            stage = 'sensor_update'
            self.sensor.update(float(time_s),float(compression_m))
            stage = 'sensor_read'
            packet = self.sensor.read(float(time_s))
            if packet is not None:
                if not isinstance(packet, HallPacket):
                    raise ValueError('Simulator must return a raw HallPacket')
                packet_dict = asdict(packet)
                raw_record,valid_json = _raw_packet_record(packet)
                # Python dict equality aliases float/int/NumPy values. Compare
                # encoded representations, and always archive an invalid one,
                # before refusing it; equal numeric values must not erase type
                # evidence behind a reference to a previously valid packet.
                new = (not valid_json or self._last_packet is None
                       or _json(raw_record) != _json(self._last_packet))
                if new:
                    self._packet_id = 'adc'+str(self._packet_count).zfill(8)
                    self._packet_count += 1
                hall_row.update(packet_id=self._packet_id,acquisition_id=self._packet_id,
                                packet=raw_record if new else None,
                                is_new_packet=new)
                if new:
                    stage = 'adc_acquisition_write'
                    _append(self._adc,dict(acquisition_id=self._packet_id,physics_row=physics_row,
                        time_s=float(time_s),packet=raw_record,valid_json_packet=valid_json,
                        training_eligible=False))
                    self._adc_rows += 1
                stage = 'decode_adc'
                if not valid_json:
                    raise ValueError('Invalid Hall packet clock or scalar representation; raw acquisition preserved')
                # Source, calibration SHA, timestamp order and saturation are
                # checked by the same decoder that consumes hardware packets.
                if packet.source != 'simulated_adc':
                    raise ValueError('M2 simulation requires simulated_adc source')
                decoded = self.calibration.decode(packet,now_s=float(time_s))
                if new and self._last_packet is not None and packet.sample_time_s <= self._last_packet['sample_time_s']:
                    raise ValueError('New ADC packet did not advance acquisition time')
                hall_row.update(decoded=asdict(decoded),status='available')
                self._last_packet = packet_dict
            stage = 'hall_write'
            _append(self._hall, hall_row)
            self._hall_rows += 1
            result = deepcopy(hall_row)
            result['packet'] = deepcopy(self._last_packet) if packet is not None else None
            self._latest = deepcopy(dict(physics=physical,hall=result))
            return result
        except Exception as error:
            self._failure = dict(stage=stage,error=type(error).__name__+': '+str(error),
                                 physics_rows=self._physics_rows,hall_rows=self._hall_rows)
            if hall_row is not None and stage != 'hall_write':
                hall_row.update(status='error',error=deepcopy(self._failure))
                try:
                    _append(self._hall,hall_row)
                    self._hall_rows += 1
                except Exception as write_error:
                    self._failure['error_ledger_write'] = repr(write_error)
            raise

    def finalize(self, error=None):
        if self._receipt is not None:
            return deepcopy(self._receipt)
        if error is not None and self._failure is None:
            self._failure = dict(stage='caller',error=str(error))
        if not self._physics_rows and self._failure is None:
            self._failure = dict(stage='acquisition',error='No acquired physics samples')
        if (self._failure is None and
                (self._physics_rows != self._hall_rows or self._packet_count != self._adc_rows)):
            self._failure = dict(stage='incomplete_ledgers',error='Interrupted acquisition or decode left unmatched rows')
        close_errors = []
        for label, stream in (('physics',self._physics),('adc',self._adc),('hall',self._hall)):
            try: stream.close()
            except Exception as close_error: close_errors.append(label+': '+repr(close_error))
        self._closed = True
        artifacts = {}
        for name in ARTIFACTS:
            try: artifacts[name] = _file_sha256(self.output/name)
            except Exception as hash_error: close_errors.append(name+': '+repr(hash_error))
        for source in self._metadata['source_bindings'].values():
            try:
                if _file_sha256(source['path']) != source['sha256']:
                    raise ValueError('Source changed during acquisition')
            except Exception as source_error: close_errors.append(repr(source_error))
        receipt = dict(schema=SCHEMA,completed=self._failure is None and not close_errors,
            physics_rows=self._physics_rows,hall_rows=self._hall_rows,unique_packets=self._packet_count,
            adc_acquisitions=self._adc_rows,
            failure=self._failure,close_errors=close_errors,artifacts_sha256=artifacts,
            physical_qualification=False,training_eligible=False)
        _write(self.output/'receipt.json',receipt)
        self._receipt = receipt
        return deepcopy(receipt)


def audit_hall_stream(directory, *, calibration_factory=HallCalibration, simulator_factory=HallSimulator):
    """Read-only byte binding and complete 240 Hz replay of a finished stream.

    A sensor-only replay pass grants no episode, physics or training acceptance.
    Failed/incomplete streams retain evidence but cannot pass this audit.
    """
    directory = Path(directory)
    receipt = json.loads((directory/'receipt.json').read_bytes())
    if (receipt.get('schema') != SCHEMA or receipt.get('completed') is not True
            or receipt.get('training_eligible') is not False):
        raise ValueError('Failed or incomplete sensor stream; original evidence retained')
    for name in ARTIFACTS:
        if _file_sha256(directory/name) != receipt['artifacts_sha256'].get(name):
            raise ValueError('Sensor artifact changed: '+name)
    metadata = json.loads((directory/'manifest.json').read_bytes())
    require_current(metadata['geometry'])
    calibration_bytes = (directory/'calibration.json').read_bytes()
    cal = calibration_factory(calibration_bytes)
    if cal.sha256 != _digest(calibration_bytes) or cal.sha256 != metadata['calibration_sha256']:
        raise ValueError('Calibration source SHA changed')
    expected_parameters = dict(sample_period_s=ADC_PERIOD_S,drift_v_s=0.,
        **validate_condition({k:v for k,v in metadata['parameters'].items()
                              if k not in ('sample_period_s','drift_v_s')}))
    if (metadata.get('schema') != SCHEMA or metadata.get('physics_hz') != 240
            or metadata.get('adc_nominal_hz') != 100 or metadata.get('training_eligible') is not False
            or metadata['parameters'] != expected_parameters
            or _source_identity(_sources(calibration_factory,simulator_factory,cal))
               != _source_identity(metadata['source_bindings'])):
        raise ValueError('Sensor model, rates or source bindings changed')
    simulator = simulator_factory(cal,**expected_parameters)
    previous = 0.;last_packet = None;packet_id = None;packets = count = 0
    from itertools import zip_longest
    with (directory/'physics.jsonl').open() as physical, (directory/'hall.jsonl').open() as hall, \
            (directory/'adc_acquisition.jsonl').open() as adc:
        for a,b in zip_longest(physical,hall):
            if a is None or b is None: raise ValueError('Incomplete physics/Hall row pair')
            row, observed = json.loads(a),json.loads(b)
            _validate_step(row['physics_row'],row['time_s'],row['compression_m'],count,previous)
            previous = row['time_s']
            if (row.get('training_eligible') is not False or observed.get('training_eligible') is not False
                    or observed.get('physics_row') != count or observed.get('time_s') != previous
                    or observed.get('error') is not None):
                raise ValueError('Hall observation does not bind the actual physics row')
            simulator.update(previous,row['compression_m'])
            packet = simulator.read(previous)
            actual = None if packet is None else asdict(packet)
            new = actual is not None and actual != last_packet
            if new:
                packet_id = 'adc'+str(packets).zfill(8);packets += 1
                raw_line = next(adc,None)
                expected_acquisition = dict(acquisition_id=packet_id,physics_row=count,time_s=previous,
                    packet=actual,valid_json_packet=True,training_eligible=False)
                if raw_line is None or json.loads(raw_line) != expected_acquisition:
                    raise ValueError('Raw ADC acquisition differs from 240 Hz replay')
            decoded = None if packet is None else asdict(cal.decode(packet,now_s=previous))
            expected = dict(physics_row=count,time_s=previous,packet_id=packet_id if actual is not None else None,
                acquisition_id=packet_id if actual is not None else None,
                packet=actual if new else None,is_new_packet=new,decoded=decoded,
                status='pending' if actual is None else 'available',error=None,training_eligible=False)
            if observed != expected: raise ValueError('ADC packet, reference or decode differs from 240 Hz replay')
            last_packet = actual;count += 1
        if next(adc,None) is not None: raise ValueError('Unreferenced raw ADC acquisition')
    if (count != receipt['physics_rows'] or count != receipt['hall_rows']
            or packets != receipt['unique_packets'] or packets != receipt['adc_acquisitions']):
        raise ValueError('Sensor receipt row or packet counts differ')
    return dict(passed=True,physics_rows=count,unique_packets=packets,physics_hz=240,
                adc_nominal_hz=100,calibration_sha256=cal.sha256,training_eligible=False,
                scope='sensor-only deterministic replay; no physical or model qualification')
