"""Opt-in raw expert acquisition; successful recording grants no training gate."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re
import gzip

import numpy as np

from .geometry_epoch import require_current
from .m2_capture import (M2RGBDCapture, M2CameraSampler, APPROVED_CAMERA_SHA256,
                         COMPACT_IMAGE_SIZE, image_size, _state, _same_state, _scalar)
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
    if isinstance(value, dict) and 'profile' in value:
        keys.add('profile')
        if value['profile'] == 'compact': keys.add('compact')
        elif value['profile'] != 'full_raw':
            raise ValueError('fresh_expert.raw_multimodal_capture profile must be compact or full_raw')
    if (not isinstance(value, dict) or set(value) != keys or value.get('enabled') is not True
            or not isinstance(value.get('episode_id'), str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value['episode_id']) is None
            or any(not isinstance(value.get(k), str) or not value[k].strip()
                   for k in ('camera_file', 'calibration_file'))
            or value.get('camera_sha256') != APPROVED_CAMERA_SHA256
            or value.get('calibration_sha256') != CALIBRATION_SHA256):
        raise ValueError('fresh_expert.raw_multimodal_capture requires an explicit episode and pinned sources')
    if value.get('profile') == 'compact':
        _compact_settings(value)
    return deepcopy(value)


def _compact_settings(binding):
    from . import m2_contract
    from .training_capture import _require_episode_metadata, _require_numeric_container_codec
    settings = binding.get('compact')
    required = {'metadata','code_bindings','max_frames','estimated_target_base',
                'target_confidence','target_estimate_source_id'}
    if not isinstance(settings, dict) or set(settings)-{'limit_bytes','native_image_size','numeric_container_codec'} != required:
        raise ValueError('Compact profile requires explicit metadata, source bindings and target estimate')
    _require_numeric_container_codec(settings.get('numeric_container_codec'))
    image_size(settings.get('native_image_size',COMPACT_IMAGE_SIZE))
    _require_episode_metadata(settings['metadata'])
    split_bytes = m2_contract.IDENTITY_SPLIT_PATH.read_bytes()
    import hashlib
    if hashlib.sha256(split_bytes).hexdigest() != m2_contract.IDENTITY_SPLIT_SHA256:
        raise ValueError('Frozen identity split bytes changed')
    splits = json.loads(split_bytes)['splits']
    subject = settings['metadata']['subject_id']
    memberships = [name for name, subjects in splits.items() if subject in subjects]
    if (memberships != [settings['metadata']['split']]
            or subject in m2_contract.QUARANTINED_IDENTITIES):
        raise ValueError('Subject identity is outside declared split or quarantined')
    if settings['metadata']['episode_id'] != binding['episode_id']:
        raise ValueError('Compact episode identity differs from capture binding')
    if (type(settings['max_frames']) is not int or settings['max_frames'] <= 0 or
            not isinstance(settings['target_estimate_source_id'],str) or
            not 0 < len(settings['target_estimate_source_id']) <= 256):
        raise ValueError('Compact maximum frames and target estimate source required')
    m2_contract.encode_state(np.zeros(7),settings['estimated_target_base'],
                             settings['target_confidence'],metadata=settings['metadata'])
    if settings['metadata']['target_source']=='given_contact_pose':
        from .m2_given_contact_pose import require_route
        from .m2_given_contact_handoff import POLICY_ID
        if settings['metadata'].get('handoff_policy_id') not in (None,POLICY_ID):
            raise ValueError('Unknown given-contact handoff policy')
        ref=require_route(settings['metadata'])
        if settings['target_estimate_source_id']!='given-contact-'+ref.sha256:
            raise ValueError('Given-reference capture source mismatch')
    return settings


def create_expert_capture(output, *, binding, geometry_metadata, source_paths=()):
    """Choose compact production storage or unchanged full_raw evidence."""
    settings = capture_settings({'raw_multimodal_capture':binding})
    if settings is None: raise ValueError('Capture must be explicitly enabled')
    cls = CompactExpertCapture if settings.get('profile') == 'compact' else ExpertCapture
    return cls(output,binding=settings,geometry_metadata=geometry_metadata,source_paths=source_paths)


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
    labels=dict(phase=controller.manager.state.phase.value, target=controller.target.name,
                subphase=subphase, active=bool(active), recovery=bool(recovery))
    capture=getattr(controller.env,'raw_capture',None)
    if isinstance(capture,CompactExpertCapture):
        labels['actor']=capture.actor_context(controller,labels)
        if capture.metadata['target_source']=='given_contact_pose':
            labels['raw_intent']=deepcopy(getattr(controller,'_current_intent_audit',None))
            labels['actual_control_owner']=controller.control_owner
            from .m2_given_contact_handoff import POLICY_ID,measured_sample
            if capture.metadata.get('handoff_policy_id')==POLICY_ID and subphase=='given_l2_park':
                rows=controller.env.rows
                labels['parking_observed_pose']=measured_sample(controller.env.state(),controller.env.base,len(rows),
                    rows[-1].get('tcp_speed_m_s') if rows else None)
                labels['parking_interpolator_hold']=deepcopy(getattr(controller,'_parking_tail_audit',None))
    return labels


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
        if self.binding.get('profile') == 'compact':
            raise ValueError('Use create_expert_capture factory for the compact profile')
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


class CompactExpertCapture:
    """Native single-target hook with one pending observation and clean labels.

    ``finish_tick`` requires the issued seven-joint command and its real
    post-step readback on *every* native tick. Observation t is paired only
    when the command at t+0.05 actually executes. The final observation has
    no invented action. Per-tick context/audit records are compressed and
    separately bounded; Hall packets remain in memory until each observation.
    """
    # Reserve audit capacity inside the same 100 MB total, not in addition.
    # Actual native 240 Hz records exhausted the previous 2 MB reservation
    # near 16k ticks, before an otherwise normal single-point episode ended.
    AUXILIARY_LIMIT_BYTES = 10_000_000
    AUXILIARY_RESERVE_BYTES = 65_536

    def __init__(self, output, *, binding, geometry_metadata, source_paths=()):
        from . import m2_contract, m2_stage
        from .training_capture import CompactTrainingCapture
        from .storage_policy import EPISODE_LIMIT_BYTES
        from ..auscultation_vla_v2.hall import HallCalibration, HallSimulator
        self.binding = capture_settings({'raw_multimodal_capture':binding})
        settings = self.binding['compact']
        self.native_image_size=image_size(settings.get('native_image_size',COMPACT_IMAGE_SIZE))
        require_current(geometry_metadata)
        self.metadata = deepcopy(settings['metadata'])
        runtime_audit=geometry_metadata.get('native_retainer_audit')
        if self.metadata.get('native_retainer_audit')!=runtime_audit:
            raise ValueError('Compact native retainer audit differs from current runtime')
        if runtime_audit is not None:
            from .retainer_witness import load_contract
            load_contract(runtime_audit)
        if 'observation_contract' in self.metadata:
            m2_contract.require_observation_contract(self.metadata)
        self.metadata['observation_contract']=m2_contract.observation_contract(self.metadata)
        self._actor_context=None
        self._actor_stage=None
        for key in ('geometry_id','geometry_source_sha256'):
            if self.metadata[key] != geometry_metadata[key]:
                raise ValueError('Compact geometry differs from current runtime')
        for key in ('collision_geometry_sha256','collision_arrays_sha256'):
            if self.metadata[key] != geometry_metadata.get('geometry_artifacts',{}).get(key):
                raise ValueError('Compact collision artifacts differ from current runtime')
        patient=geometry_metadata.get('patient_provenance')
        if patient is not None:
            if (patient.get('identity')!=self.metadata['subject_id'] or
                    patient.get('split')!=self.metadata['split']):
                raise ValueError('Compact identity/split differs from installed patient')
            if (not isinstance(patient.get('asset_sha256'),str) or len(patient['asset_sha256'])!=64
                    or not isinstance(patient.get('receipt_sha256'),str) or len(patient['receipt_sha256'])!=64):
                raise ValueError('Compact installed patient lacks authenticated receipt')
            self.metadata['patient_asset_sha256']=patient['asset_sha256']
            self.metadata['installed_patient_receipt_sha256']=patient['receipt_sha256']
        elif self.metadata['source_kind']=='isaac_live_physics':
            raise ValueError('Production compact capture requires authenticated installed patient')
        m2_contract.require_metadata(self.metadata)
        self.camera_bytes = Path(binding['camera_file']).read_bytes()
        self.calibration_bytes = Path(binding['calibration_file']).read_bytes()
        if (_sha(binding['camera_file']) != APPROVED_CAMERA_SHA256 or
                _sha(binding['calibration_file']) != CALIBRATION_SHA256):
            raise ValueError('Pinned camera or Hall calibration changed')
        self.limit_bytes = settings.get('limit_bytes',EPISODE_LIMIT_BYTES)
        if type(self.limit_bytes) is not int or not 262144 <= self.limit_bytes <= EPISODE_LIMIT_BYTES:
            raise ValueError('Compact profile byte limit must be 262144..100000000')
        self.auxiliary_limit = min(self.AUXILIARY_LIMIT_BYTES,self.limit_bytes//2)
        self.calibration = HallCalibration(self.calibration_bytes)
        self.sensor = HallSimulator(self.calibration,sample_period_s=.01,noise_std_v=0.,
                                    voltage_bias_v=0.,drift_v_s=0.,curve_warp=0.)
        from .hall_adc_boundary import lower_endpoint_policy,sensor_input
        self.hall_zero_policy=lower_endpoint_policy(self.calibration)
        self.hall_zero_tolerance_m=self.hall_zero_policy['lower_tolerance_m']
        self.estimate = np.asarray(settings['estimated_target_base'],dtype=float)
        self.confidence = settings['target_confidence']
        self.estimate_source = settings['target_estimate_source_id']
        paths = [*source_paths,__file__,inspect.getfile(M2CameraSampler),
                 inspect.getfile(CompactTrainingCapture),inspect.getfile(HallSimulator),
                 m2_contract.__file__,m2_stage.__file__,binding['camera_file'],binding['calibration_file'],
                 inspect.getfile(sensor_input)]
        from . import m2_given_contact_handoff as handoff
        if self.metadata.get('handoff_policy_id')==handoff.POLICY_ID:paths.append(handoff.__file__)
        self.sources = {str(Path(path).resolve()):_sha(path) for path in paths}
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.camera = self.writer = self._audit = None
        self.hall = None  # No M2HallStream is constructed in compact mode.
        self.errors = []; self.receipts = {}
        self._closed = False
        self._physics_rows = self._audit_rows = self._frames = 0
        self._last_time = 0.
        self._packets = []; self._last_packet_time = -np.inf
        self._pending = None; self._terminal_ticks = []
        self._hall_input = None
        self._hall_zero_count = 0; self._hall_zero_max_delta_m = 0.
        self._last_parking_pose = None
        capture_schema='m2_compact_expert_capture_v2'
        if self.metadata['target_source']=='given_contact_pose':
            from .m2_given_contact_pose import CAPTURE_SCHEMA
            capture_schema=CAPTURE_SCHEMA
        manifest = dict(schema=capture_schema,capture_profile='compact',
            observation_contract=m2_contract.observation_contract(self.metadata),
            binding=self.binding,geometry=deepcopy(geometry_metadata),
            native_image_size=list(self.native_image_size),
            runtime_sources_sha256=self.sources,physics_hz=240,capture_hz=20,
            hall_adc_nominal_hz=100,target_estimate='fixed per-episode estimate read causally at each observation',
            hall_zero_conversion=deepcopy(self.hall_zero_policy),
            clean_command='actual unperturbed joint command; no measured-q substitution',
            audit='every native tick context and physical audit; no raw state mirror',
            auxiliary_limit_bytes=self.auxiliary_limit,
            compact_payload_limit_bytes=self.limit_bytes-self.auxiliary_limit,
            limit_bytes=self.limit_bytes,training_eligible=False,physical_qualification=False)
        if self.metadata['target_source']=='given_contact_pose':
            manifest.update(action_semantics_id=self.metadata['action_semantics_id'],
                audit='every native tick: physical context, raw setter with live pre/post-step target and actual q_command')
        payload=json.dumps(manifest,sort_keys=True,allow_nan=False).encode()+b'\n'
        if len(payload)+self.AUXILIARY_RESERVE_BYTES >= self.auxiliary_limit:
            raise ValueError('Compact metadata exceeds auxiliary byte budget')
        (self.output/'manifest.json').write_bytes(payload)
        if self.metadata['target_source']=='given_contact_pose':
            from .m2_given_contact_pose import require_route
            ref=require_route(self.metadata);ref.verify_sources()
            reference_payload=ref.payload+b'\n'
            (self.output/'given_reference.json').write_bytes(reference_payload)
            self._auxiliary_reference_bytes=len(reference_payload)
        else:self._auxiliary_reference_bytes=0
        self._auxiliary_initial_bytes = len(payload)+self._auxiliary_reference_bytes
        self._audit = gzip.open(self.output/'native_audit.jsonl.gz','xb')

    def actor_context(self,controller,labels):
        """Native before/after hook: sample actual q/time without advancing physics."""
        from . import m2_contract, m2_stage
        stamp,q,_,_=_state(controller.env.state())
        identity=(self.metadata['episode_id'],labels['target'])
        if self._actor_context!=identity:
            self._actor_context=identity
            self._actor_stage=m2_stage.CausalStage()
        state=m2_contract.encode_state(q,self.estimate,self.confidence,metadata=self.metadata)
        distance=float(np.linalg.norm(state[7:10]))
        if self.metadata['target_source']=='given_contact_pose':
            from .m2_given_contact_pose import NATIVE_SOURCE,STAGE_RULE,L2_ROLES,require_route,TASK_ID
            ref=require_route(self.metadata)
            actual=getattr(controller,'control_owner',None)
            if getattr(controller,'task_id',None)!=TASK_ID or actual not in (
                    'PREPARATION','TEACHER_L2','ACT_L2','L2_STOP','L3_ENTRY',
                    'ANALYTIC_L3_ALIGN','ANALYTIC_L3','STOPPED'):
                raise ValueError('Given-contact capture requires the actual new controller boundary/owner')
            eligible=(actual in ('TEACHER_L2','ACT_L2') and labels['subphase'] in L2_ROLES and
                      labels['phase']=='APPROACH' and not labels['active'] and not labels['recovery'])
            if actual=='ANALYTIC_L3_ALIGN':owner,role='L3','L3_ALIGN'
            elif actual=='ANALYTIC_L3' or labels['recovery']:owner,role='L3','L3'
            elif eligible:owner,role='L2','L2'
            elif actual=='L2_STOP':owner,role='PREPARATION','L2_PARK'
            else:owner,role='PREPARATION','PREPARATION'
            # WAIT is excluded from labels, but its actual measured positions
            # still advance the same causal latch used by the dispatcher.
            if actual in ('TEACHER_L2','ACT_L2'):
                self._actor_stage.update(distance,stamp)
            phase=self._actor_stage.phase if eligible else None
            return dict(phase=phase,owner=owner,sampled_at_s=stamp,rule_id=STAGE_RULE,source=NATIVE_SOURCE,
                estimated_tcp_distance_m=distance,role=role,reference_sha256=ref.sha256,reference_revision=0)
        l3=bool(getattr(controller,'_contact_active',False) or labels['recovery'])
        eligible=(labels['phase']=='APPROACH' and labels['subphase'] in m2_stage.L2_EXECUTION_ROLES and
                  not labels['active'] and not l3)
        owner='L3' if l3 else 'L2' if eligible else 'PREPARATION'
        phase=self._actor_stage.update(distance,stamp) if eligible else None
        return dict(phase=phase,owner=owner,sampled_at_s=stamp,rule_id=m2_stage.RULE_ID,
                    source=m2_stage.NATIVE_SOURCE,estimated_tcp_distance_m=distance)

    def failure(self, stage, error):
        if len(self.errors) < 20:
            self.errors.append(dict(stage=stage,error=str(error)[:1024]))

    def start(self, env, *, backend_factory=None):
        from .training_capture import CompactTrainingCapture
        if self.camera is not None or self.writer is not None or self._closed:
            raise RuntimeError('Compact capture already started or closed')
        if env.rows: raise ValueError('Compact capture must start before the first physics row')
        if self.metadata['target_source']=='given_contact_pose':
            from .m2_given_contact_pose import require_route
            if require_route(self.metadata).snapshot()['available_at_s']>float(env.state()['time_s']):
                raise ValueError('Given reference unavailable before capture motion')
        if self.metadata['source_kind'] not in ('isaac_live_physics','cpu_synthetic_test'):
            raise ValueError('Native compact capture cannot label replay input as live')
        if self.metadata['source_kind']=='cpu_synthetic_test' and backend_factory is None:
            raise ValueError('Synthetic compact source requires an injected CPU backend')
        try:
            self.camera=M2CameraSampler(env,camera_bytes=self.camera_bytes,backend_factory=backend_factory,
                native_image_size=self.native_image_size)
            settings=self.binding['compact']
            self.writer=CompactTrainingCapture(self.output/'compact',metadata=self.metadata,
                camera_bytes=self.camera_bytes,calibration_bytes=self.calibration_bytes,
                code_bindings=settings['code_bindings'],max_frames=settings['max_frames'],
                limit_bytes=self.limit_bytes-self.auxiliary_limit,native_image_size=self.camera.native_image_size,
                numeric_container_codec=settings.get('numeric_container_codec'))
        except BaseException as error:
            self.failure('setup',repr(error)); raise

    def after_physics(self, env, rows_before):
        from .m2_sensors import _validate_step
        if len(env.rows)==rows_before: return
        try:
            if len(env.rows)!=rows_before+1 or rows_before!=self._physics_rows:
                raise ValueError('Expected one consecutive actual native physics row')
            row=env.rows[-1]
            _validate_step(rows_before,row['time_s'],row['compression_m'],self._physics_rows,self._last_time)
            self._physics_rows+=1; self._last_time=float(row['time_s'])
            from .hall_adc_boundary import sensor_input
            raw=float(row['compression_m']);self._hall_input=sensor_input(self.calibration,raw)
            mapped=self._hall_input['sensor_compression_m']
            if mapped!=raw:
                self._hall_zero_count+=1
                self._hall_zero_max_delta_m=max(self._hall_zero_max_delta_m,-raw)
            self.sensor.update(self._last_time,mapped)
            packet=self.sensor.read(self._last_time)
            if packet is not None and packet.sample_time_s > self._last_packet_time:
                self.calibration.decode(packet,now_s=self._last_time)
                if len(self._packets)>=6: raise ValueError('Hall buffer exceeded one observation window')
                self._packets.append(packet); self._last_packet_time=packet.sample_time_s
        except BaseException as error:
            self.failure('physics',repr(error)); raise

    def finish_tick(self, env, rows_before, before, after, *, error=None,
                    clean_command=None, control_owner=None, collection_controller=None,collection_stop_capture=None):
        from . import m2_contract
        if error is not None: self.failure('controller_tick',error)
        if len(env.rows)==rows_before: return
        try:
            if (self._closed or self.writer is None or len(env.rows)!=rows_before+1 or
                    rows_before!=self._audit_rows or self._physics_rows!=rows_before+1):
                raise ValueError('Compact tick must follow each actual physics append exactly once')
            row=env.rows[-1]; t=_scalar(row['time_s'],'native tick time')
            from .m2_stage import validate_actor
            actor_kwargs={}
            if self.metadata['target_source']=='given_contact_pose':
                from .m2_given_contact_pose import NATIVE_SOURCE
                actor_kwargs=dict(source=NATIVE_SOURCE,reference_sha256=self.metadata['reference_sha256'])
            validate_actor(before.get('actor'),t-1/240.,**actor_kwargs)
            validate_actor(after.get('actor'),t,**actor_kwargs)
            if any(before[key]!=row.get(key) or after[key]!=row.get(key) for key in ('target','subphase')):
                raise ValueError('Controller labels differ from actual physics row')
            if row['target']!=self.metadata['target_id']:
                raise ValueError('Compact capture crosses its single target')
            if control_owner not in ('L2','L3'): raise ValueError('Explicit compact L2/L3 control owner required')
            if (not isinstance(clean_command,dict) or set(clean_command)!=
                    {'q_command','issued_at_s','endpoint_at_s','readback_at_s','source_id'}):
                raise ValueError('Explicit clean command evidence required at every native tick')
            qcmd=np.asarray(clean_command['q_command'])
            applied=np.asarray(row.get('q_command'))
            if (qcmd.shape!=(7,) or qcmd.dtype.kind not in 'fiu' or not np.isfinite(qcmd).all() or
                    applied.shape!=(7,) or not np.array_equal(qcmd,applied)):
                raise ValueError('Clean command must contain seven actual unperturbed joint targets')
            issue=_scalar(clean_command['issued_at_s'],'command issue')
            endpoint=_scalar(clean_command['endpoint_at_s'],'command endpoint')
            readback=_scalar(clean_command['readback_at_s'],'command readback')
            source=clean_command['source_id']
            if (not t-1/240.-1e-9 <= issue <= t or abs(endpoint-t)>1e-9 or abs(readback-t)>1e-9 or
                    not isinstance(source,str) or not 0<len(source)<=256):
                raise ValueError('Clean command must bind this actual native issue and endpoint readback')
            state=env.state(); stamp,q,_,_=_state(state)
            if abs(stamp-t)>1e-9: raise ValueError('Native observation state time differs from tick')
            audited='physical_violations' in row
            context=dict(physics_row=rows_before,time_s=t,controller_before=deepcopy(before),
                controller_after=deepcopy(after),control_owner=control_owner,
                state_audit='completed' if audited else 'not_completed',
                physical_violations=deepcopy(row.get('physical_violations')),tick_error=error)
            if self.metadata['target_source']=='given_contact_pose':
                evidence=before.get('raw_intent')
                actual_owner=before.get('actual_control_owner')
                if actual_owner!=after.get('actual_control_owner'):
                    raise ValueError('Given actual owner changed during one native command')
                if evidence != after.get('raw_intent'):
                    raise ValueError('Raw target evidence changed during one native command')
                if evidence is not None and evidence['setter']['control_owner']!=actual_owner:
                    raise ValueError('Given actual owner differs from held raw setter')
                if before['subphase'] in ('given_l2_motion','given_l2_history_wait','given_l2_park') and evidence is None:
                    raise ValueError('Given L2 tick lacks an actual raw target setter')
                context.update(raw_intent=deepcopy(evidence),q_command=qcmd.tolist(),actual_control_owner=actual_owner)
                from . import m2_given_contact_handoff as handoff
                if self.metadata.get('handoff_policy_id')==handoff.POLICY_ID and before['subphase']=='given_l2_park':
                    observed=handoff.measured_sample(state,env.base,rows_before+1,row.get('tcp_speed_m_s'))
                    if after.get('parking_observed_pose')!=observed:
                        raise ValueError('Actual full parking observation differs from native state')
                    prior=before.get('parking_observed_pose');handoff.validate_sample(prior)
                    if prior['physics_row']!=rows_before:raise ValueError('Parking before-pose row mismatch')
                    context.update(parking_pose_before=deepcopy(prior),parking_pose_after=observed)
            # Tier A: native safety/force scalars and every phase boundary,
            # without a full raw pose/contact/command mirror. Allocation is
            # shared with this already bounded auxiliary audit stream.
            audit_keys={'compression_m','tcp_speed_m_s','gap_m','tangent_m','angle_deg',
                        'hold_elapsed_s','audit_hold_s','axis_world','contact_plane_audit'}
            context['measurements']={key:deepcopy(value) for key,value in row.items()
                if (key in audit_keys or (key.endswith('_N') and np.isscalar(value)))}
            if 'native_retainer_audit' in self.metadata:
                contract=self.metadata['native_retainer_audit']
                if row.get('native_retainer_audit')!=contract:
                    raise ValueError('Native compact tick downgraded retainer audit contract')
                front=row.get('retainer_front_audit',{})
                if front.get('native_retainer_audit')!=contract:
                    raise ValueError('Native compact tick lacks bound retainer witness audit')
                context['native_retainer_audit']=deepcopy(contract)
                context['retainer_front_audit']={key:deepcopy(value) for key,value in front.items() if key!='points'}
            if collection_controller is not None or source.startswith('onpolicy_collection_'):
                from . import m2_act_onpolicy_collection as collection
                from . import m2_given_contact_handoff as handoff
                collection._authenticated_controller(collection_controller)
                controller=collection_controller
                if (controller.env is not env or env.raw_capture is not self or controller.metadata!=self.metadata or
                        before.get('actual_control_owner')!=controller.control_owner or
                        after.get('actual_control_owner')!=controller.control_owner):
                    raise ValueError('Actual collection capture/controller/owner binding differs')
                expected_source=('onpolicy_collection_act_native_command_v1' if controller.control_owner=='ACT_L2' else
                    'onpolicy_collection_teacher_native_command_v1' if controller.control_owner=='TEACHER_L2' else
                    'onpolicy_collection_original_native_command_v1')
                if source!=expected_source:
                    raise ValueError('Actual collection command source differs from its owner')
                context.update(collection.finish_stop_capture(controller,collection_stop_capture,rows_before=rows_before,
                    before=before,after=after,clean_command=clean_command))
                if controller.control_owner=='ACT_L2' and (rows_before+1)%12==0:
                    # Read actual POST-step sensors and the live limiter here.
                    # No observation/provider call: their causal H2/latch update
                    # occurs after this capture returns on the same boundary.
                    dq=m2_contract._array(state.get('dq'),(7,),'actual captured collection dq')
                    row_dq=m2_contract._array(row.get('dq'),(7,),'actual native row collection dq')
                    if dq.tobytes()!=row_dq.tobytes():
                        raise ValueError('Actual collection sensor dq differs from its native row')
                    pose=np.eye(4)
                    base=m2_contract._pose(env.base,'actual collection base')
                    pose[:3,:3]=base[:3,:3].T@m2_contract._array(state['rotation_world'],(3,3),'actual collection full rotation')
                    pose[:3,3]=m2_contract._array(state['tcp_base'],(3,),'actual collection TCP')
                    objects=dict(trajectory=controller.trajectory,input=controller.trajectory.input,
                        output=controller.trajectory.output,generator=controller.trajectory.generator)
                    context['onpolicy_collection_boundary']=dict(schema=collection.CAPTURE_BOUNDARY_SCHEMA,
                        source_id='actual_native_post_step_collection_boundary',physics_row=rows_before+1,time_s=float(stamp),
                        context=[self.metadata['episode_id'],self.metadata['target_id'],0,self._actor_stage.phase],
                        actual_control_owner='ACT_L2',reference_sha256=self.metadata['reference_sha256'],
                        config_sha256=controller.collection_binding['config_sha256'],measured_q=q.tolist(),measured_dq=dq.tolist(),
                        measured_pose_base=m2_contract._pose(pose,'actual captured collection full TCP').tolist(),
                        input=handoff.input_snapshot(controller.trajectory),output=handoff.output_snapshot(controller.trajectory),
                        object_ids={name:id(value) for name,value in objects.items()},capture_id=id(self),history_id=id(controller._history))
            context['hall_sensor_input']=deepcopy(self._hall_input)
            if 'dq' in row:
                context['measurements']['max_abs_joint_velocity_rad_s']=float(np.max(np.abs(row['dq'])))
            payload=json.dumps(context,separators=(',',':'),allow_nan=False).encode()+b'\n'
            record_limit=16384
            if collection_controller is not None and collection_stop_capture is not None:
                record_limit=collection.STOP_CAPTURE_RECORD_BYTES  # Authenticated collection branch above only.
            if (len(payload)>record_limit or self._auxiliary_initial_bytes+self._audit.fileobj.tell()+
                    len(payload)+self.AUXILIARY_RESERVE_BYTES > self.auxiliary_limit):
                raise ValueError('Native compact audit exceeds auxiliary byte budget')
            self._audit.write(payload); self._audit_rows+=1
            if 'parking_pose_after' in context:self._last_parking_pose=deepcopy(context['parking_pose_after'])
            tick=dict(episode_id=self.metadata['episode_id'],target_id=row['target'],
                step_count=rows_before+1,physics_row=rows_before,time_s=t,
                phase=after['phase'],control_owner=control_owner,measured_q=q.tolist(),
                q_command=qcmd.tolist(),command_issued_at_s=issue,command_readback_at_s=readback,
                command_source_id=source,observation_source_id=f'native-{rows_before}')
            self._terminal_ticks.append(tick)
            if (rows_before+1)%12: return
            self._terminal_ticks=[]
            if self._frames>=self.binding['compact']['max_frames']:
                raise ValueError('Compact observation exceeds maximum frame bound')
            if self._pending is not None:
                self.writer.append(**self._pending,action=deepcopy(clean_command))
                self._pending=None
            rgb,depth,valid=self.camera.capture(state)
            self._pending=dict(episode_id=self.metadata['episode_id'],target_id=row['target'],
                observed_at_s=t,rgb=rgb,depth_m=depth,depth_valid=valid,measured_q=q,
                estimated_target_base=self.estimate.copy(),target_confidence=self.confidence,
                state17=m2_contract.encode_state(q,self.estimate,self.confidence,metadata=self.metadata),
                target_source=self.metadata['target_source'],target_estimate_time_s=t,
                observation_source_id=f'native-{rows_before}',target_estimate_source_id=self.estimate_source,
                phase=after['phase'],control_owner=control_owner,hall_packets=self._packets)
            self._packets=[]; self._frames+=1
            self._audit.flush()
        except BaseException as caught:
            self.failure('tick_capture',repr(caught)); raise

    def close(self, error=None):
        if self._closed: return
        self._closed=True
        if error is not None: self.failure('caller',error)
        if self._physics_rows!=self._audit_rows:
            self.failure('counts','Actual physics rows have missing native context audits')
        if self.camera is not None:
            try: self.receipts['camera']=self.camera.finalize(error)
            except BaseException as caught: self.failure('camera_finalize',repr(caught))
            finally: self.camera=None
        if self._audit is not None:
            try: self._audit.close()
            except BaseException as caught: self.failure('audit_close',repr(caught))
            finally: self._audit=None
        if self.writer is not None:
            endpoint=dict(final_native_step_count=self._physics_rows,
                final_native_time_s=self._last_time,terminal_ticks=self._terminal_ticks)
            try:
                if self.errors: self.receipts['compact']=self.writer.abort(self.errors[0]['error'],**endpoint)
                else:
                    if self._pending is not None: self.writer.append_terminal(**self._pending)
                    self.receipts['compact']=self.writer.close(**endpoint)
            except BaseException as caught:
                self.failure('compact_finalize',repr(caught))
                try: self.receipts['compact']=self.writer.abort(str(caught),**endpoint)
                except BaseException as second: self.failure('compact_abort',repr(second))
            finally: self._pending=None; self.writer=None

    def result(self, physics_rows, *, execution_completed, errors=()):
        failures=deepcopy([*self.errors,*errors])
        for path,expected in self.sources.items():
            try:
                if _sha(path)!=expected: raise ValueError('Source changed during acquisition')
            except BaseException as caught: failures.append(dict(stage='source',path=path,error=repr(caught)))
        counts=(type(physics_rows) is int and physics_rows==self._physics_rows==self._audit_rows and
                self._frames==physics_rows//12)
        if not counts: failures.append(dict(stage='counts',error='Compact native/frame counts differ'))
        complete=bool(self._closed and execution_completed and counts and not failures and
            self.receipts.get('camera',{}).get('completed') and self.receipts.get('compact',{}).get('completed'))
        from .m2_contract import observation_contract
        capture_schema='m2_compact_expert_capture_v2'
        if self.metadata['target_source']=='given_contact_pose':
            from .m2_given_contact_pose import CAPTURE_SCHEMA
            capture_schema=CAPTURE_SCHEMA
        result=dict(schema=capture_schema,capture_profile='compact',
            observation_contract=observation_contract(self.metadata),
            compact_capture_complete=complete,raw_capture_complete=False,
            episode_id=self.binding['episode_id'],physics_steps=physics_rows,
            expected_frames=physics_rows//12,receipts=deepcopy(self.receipts),errors=failures,
            native_audit_sha256=None,
            hall_zero_conversion=dict(**self.hall_zero_policy,
                normalized_ticks=self._hall_zero_count,max_delta_m=self._hall_zero_max_delta_m,
                ),
            training_eligible=False,physical_qualification=False,randomization_qualified=False,
            h20_eligible=False,limit_bytes=self.limit_bytes,total_bytes=0)
        if self.metadata['target_source']=='given_contact_pose':
            result.update(task_id=self.metadata['task_id'],target_source=self.metadata['target_source'],
                action_semantics_id=self.metadata['action_semantics_id'],
                reference_sha256=self.metadata['reference_sha256'],reference_revision=0,
                policy_input='state17 history2 only; archived RGB-D/Hall are not network inputs')
        receipt_path=self.output/'receipt.json'
        initial=0;result['total_bytes_complete']=True
        try:
            result['native_audit_sha256']=_sha(self.output/'native_audit.jsonl.gz')
        except Exception as error:
            result['compact_capture_complete']=False
            result['errors'].append(dict(stage='artifact_inventory',error=repr(error)[:1024]))
        try:
            initial=sum(path.stat().st_size for path in self.output.rglob('*')
                        if path.is_file() and path!=receipt_path)
        except Exception as error:
            result['compact_capture_complete']=False;result['total_bytes_complete']=False
            result['errors'].append(dict(stage='artifact_inventory',error=repr(error)[:1024]))
        while True:
            payload=json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n'
            total=initial+len(payload)
            if total>self.limit_bytes and not any(e['stage']=='budget' for e in result['errors']):
                result['compact_capture_complete']=False
                result['errors'].append(dict(stage='budget',error='Compact episode exceeded byte budget'))
                continue
            if total==result['total_bytes']: break
            result['total_bytes']=total
        try: receipt_path.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
        except BaseException as caught:
            result['compact_capture_complete']=False
            result['errors'].append(dict(stage='receipt_write',error=repr(caught)))
        return result
