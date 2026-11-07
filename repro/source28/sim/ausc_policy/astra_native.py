"""Opt-in Astra supervisor + unchanged GivenContactExpert local motor skills.

Remote inference occurs only between whole skills, with frozen native state.
It never supplies a joint action, force command, success label or safety gain.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .astra_supervisor import (MODEL, HTTPClient, canonical, need,
    request_payload, validate_decision, validate_observation)
from .m2_given_contact_controller import GivenContactExpert
from .m2_given_contact_pose import TASK_ID, GivenPoseProvider, require_route
from .m2_given_contact_handoff import POLICY_ID

SCHEMA = 'ausc_astra_given_contact_supervisor_v1'


def settings(config):
    value = config.get('astra_supervisor')
    need(isinstance(value, dict) and set(value) == {'schema', 'model', 'endpoint', 'token_env', 'timeout_s'}
         and value['schema'] == SCHEMA and value['model'] == MODEL, 'Explicit Astra supervisor configuration required')
    need(isinstance(value['token_env'], str) and value['token_env'].startswith('AUSC_ASTRA_'),
         'Astra per-run token environment name required')
    need(config.get('m2_given_contact_control') is not None, 'Astra requires the explicit GivenContactExpert route')
    conflicting = [k for k in config if k.startswith('m2_') and k != 'm2_given_contact_control']
    need(not conflicting, 'Astra arm cannot be relabeled as an ACT/evaluation/recovery arm')
    fresh = config['fresh_expert']
    need(fresh.get('raw_multimodal_capture') in (None, {'enabled':False}),
         'Expert training capture cannot label Astra-supervised commands')
    route = config['m2_given_contact_control']
    need(route.get('handoff_policy_id') == POLICY_ID and route['metadata'].get('handoff_policy_id') == POLICY_ID,
         'Unchanged strict interior parking policy required')
    HTTPClient(value['endpoint'], os.environ.get(value['token_env'], ''), value['timeout_s'])
    return deepcopy(value)


def _save(path, value):
    path.write_bytes(canonical(value)+b'\n')


def _native_state(env):
    value = env.state()
    if hasattr(env, 'world'): value = dict(value, world_physics_time_s=float(env.world.current_time))
    return canonical({k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in value.items()})


def preload_observation_context(output):
    """Describe the already-verified configured preload, never an applied-force label."""
    from . import passive_preload
    output=Path(output)
    names=dict(configuration='fresh_expert_inputs/runtime_config.json',
        authoring='passive_preload_authoring.json',startup='passive_preload_startup_usd_readback.json')
    raw={key:(output/name).read_bytes() for key,name in names.items()}
    config,receipt,startup=(json.loads(raw[key]) for key in ('configuration','authoring','startup'))
    settings=passive_preload.passive_preload_settings(config['fresh_expert'])
    need(settings is not None and receipt.get('schema') == settings['mode'] and
         receipt.get('mode') == settings['mode'] and startup.get('mode') == settings['mode'],
         'Bound approved preload authoring context required')
    need(receipt.get('nominal_preload_N') == startup.get('nominal_preload_N') == settings['nominal_preload_N'] and
         receipt.get('explicit_effort_N') == startup.get('explicit_effort_N') == -settings['nominal_preload_N'] and
         receipt.get('source_sha256') == passive_preload.file_sha256(passive_preload.__file__) and
         receipt.get('other_stage_attributes_unchanged') is True and
         receipt.get('protected_before_sha256') == receipt.get('protected_after_sha256') and
         startup.get('scope') == 'live_usd_after_native_initialization' and startup.get('read_only_usd') is True and
         startup.get('applied_effort_measured') is False,'Preload authoring/readback context differs')
    return dict(mode=settings['mode'],nominal_preload_N=settings['nominal_preload_N'],
        native_authoring_readback_verified=True,applied_effort_measured=False,
        source='configured nominal preload; authoring and read-only startup receipts, not an effort measurement',
        sources={key:dict(file=names[key],sha256=hashlib.sha256(value).hexdigest()) for key,value in raw.items()})


def native_observation(controller, boundary):
    """One native RGB-D frame and calibrated simulated ADC, with a given pose."""
    state = controller.env.state()
    sensors, rgb = controller.astra_sensors.capture(state,len(controller.env.rows),controller.reference.pose_base)
    hall = sensors['hall']['decoded']
    obs = dict(schema='ausc_astra_native_observation_v1', simulation_only=True,
        observation_id='pending', episode_id=controller.episode_id,
        physics_row=len(controller.env.rows), time_s=float(state['time_s']), boundary=boundary,
        phase=controller.manager.state.phase.value,
        q_rad=np.asarray(state['q']).tolist(), dq_rad_s=np.asarray(state['dq']).tolist(),
        hall_force_N=hall['spring_force_estimate_n'], hall_source='calibration_decoded_simulated_adc_spring_load',
        compression_m=float(state['compression_m']),
        target_id=controller.target.name, target_source='given_contact_pose',
        reference_sha256=controller.reference.sha256,
        reference_pose_base=controller.reference.pose_base.tolist(),
        rgb_sha256=hashlib.sha256(rgb).hexdigest(), camera='/World/RobotCam',
        local_executor='GivenContactExpert', local_motor_assistance=True,
        hall_voltage_v=hall['hall_voltage_v'], hall_voltage_source='simulated_adc',
        sensor_contract=sensors['schema'],sensor_bundle=sensors,
        depth_target_source='native optical-Z measurements; prescribed pose retained; no anatomical target estimator')
    validate_observation(obs)
    return obs, rgb


class AstraSupervisorMixin:
    def _init_astra(self, *, client, observation_provider, output):
        need(callable(client) and callable(observation_provider), 'Explicit client and native observation provider required')
        self.astra_client = client; self.astra_observation_provider = observation_provider
        self.astra_output = Path(output); self.astra_output.mkdir(parents=True, exist_ok=True)
        self.astra_decisions = []; self.astra_active_decision = None

    def _astra_dispatch(self, boundary, operation):
        need(not getattr(self, '_contact_active', False), 'Remote inference forbidden during ongoing force control')
        index = len(self.astra_decisions); folder = self.astra_output/f'{index:04d}'
        folder.mkdir(exist_ok=False)
        before = _native_state(self.env); start_rows = len(self.env.rows)
        row = dict(decision_id=index, boundary=boundary, model=MODEL,
            local_executor='GivenContactExpert', local_motor_assistance=True,
            execution=dict(status='requested', completed=False, physical_success=None,
                physics_row_start=start_rows), physics_paused_during_inference=True)
        self.astra_decisions.append(row); started = time.monotonic()
        try:
            obs, rgb = self.astra_observation_provider(self, boundary)
            obs = deepcopy(obs); obs['observation_id'] = obs['episode_id']+':'+str(index)
            need(obs['boundary'] == boundary, 'Observation boundary differs from dispatcher')
            _save(folder/'observation.json', obs)
            if rgb is not None: (folder/'rgb.png').write_bytes(rgb)
            _save(folder/'request.json', request_payload(obs, rgb))
            row.update(request_sha256=hashlib.sha256((folder/'request.json').read_bytes()).hexdigest(),
                observation_sha256=hashlib.sha256((folder/'observation.json').read_bytes()).hexdigest(),
                rgb_sha256=obs.get('rgb_sha256'))
            need(_native_state(self.env) == before and len(self.env.rows) == start_rows,
                 'Physics changed during native observation acquisition')
            row['inference_attempted'] = True
            proposal = self.astra_client(deepcopy(obs), rgb)
            _save(folder/'proposal.json', proposal)
            need(_native_state(self.env) == before and len(self.env.rows) == start_rows,
                 'Physics changed during Astra inference')
            need(proposal.get('model') == MODEL, 'Astra response model mismatch')
            selected = validate_decision(proposal.get('decision'), obs)
            row.update(observation_id=obs['observation_id'], proposal=deepcopy(proposal),
                proposal_admitted=True,
                inference_wall_s=time.monotonic()-started,
                inference_simulation_start_s=obs['time_s'], inference_simulation_end_s=float(self.env.state()['time_s']))
            if selected['skill'] == 'abort':
                row['execution']['status'] = 'aborted'
                raise RuntimeError('Astra requested abort: '+selected['reason'])
            row['execution'].update(status='dispatched', skill=selected['skill'])
            self.astra_active_decision = index
            _save(folder/'execution.json', row['execution'])
            result = operation()
            row['execution'].update(status='completed', completed=True)
            return result
        except BaseException as error:
            if row['execution']['status'] == 'requested': row['execution']['status'] = 'rejected'
            elif row['execution']['status'] == 'dispatched': row['execution']['status'] = 'failed'
            row['execution']['failure_reason'] = str(error)
            raise
        finally:
            row['wall_s'] = time.monotonic()-started
            row['execution'].update(physics_row_end_exclusive=len(self.env.rows),
                time_s_end=float(self.env.state()['time_s']))
            _save(folder/'execution.json', row['execution'])
            _save(folder/'record.json', row)

    def _move_to_prepose(self, row):
        return self._astra_dispatch('HOME', lambda:super(AstraSupervisorMixin,self)._move_to_prepose(row))

    def _contact_cycle(self):
        # The local skill includes measured ALIGN and complete HOLD/release.
        return self._astra_dispatch('PARKED', lambda:super(AstraSupervisorMixin,self)._contact_cycle())

    def _return_home(self, transfer):
        return self._astra_dispatch('RELEASED', lambda:super(AstraSupervisorMixin,self)._return_home(transfer))

    def _tick(self, *args, **kwargs):
        count = len(self.env.rows)
        control_count = len(getattr(self, 'control_rows', []))
        try:
            result = super()._tick(*args, **kwargs)
            if getattr(self,'astra_sensors',None) is not None:
                self.astra_sensors.observe_hall(self.env.state(),len(self.env.rows))
            return result
        finally:
            for row in [*self.env.rows[count:], *getattr(self, 'control_rows', [])[control_count:]]:
                row.update(astra_decision_id=self.astra_active_decision, supervisor_model=MODEL,
                    motor_actor='analytic_GivenContactExpert', local_motor_assistance=True,
                    expert_demonstration=False, training_accepted=False,
                    training_dataset_eligible=False, learned_policy=False)


class AstraGivenContactExpert(AstraSupervisorMixin, GivenContactExpert):
    def __init__(self, *args, astra_client, astra_output, preload_context, **kwargs):
        super().__init__(*args, **kwargs)
        need(getattr(self.env, 'raw_capture', None) is None, 'Expert capture forbidden for Astra arm')
        self._init_astra(client=astra_client, observation_provider=native_observation, output=astra_output)
        from .astra_sensors import AstraSensors
        self.astra_sensors=AstraSensors(self.env,self.astra_output/'sensors',preload_context=preload_context)

    def close_astra_sensors(self):
        receipt=self.astra_sensors.close()
        need(receipt['completed'],'Astra RGB-D/Hall sensor teardown failed: '+str(receipt['failures']))
        return receipt

    def run(self, points):
        points = list(points)
        need(len(points) == 1, 'One paired prescribed target per Astra episode required')
        from .fresh_expert import contact_plane_contract
        plane = contact_plane_contract(points[0][1])
        need(plane is not None and plane['contact_plane_radius_m'] == .023 and
             plane['plane_tilt_limit_deg'] == 3. and plane['tilt_relaxed'] is False,
             'Original strict 23 mm / 3 degree local plane required')
        return tag_result(super().run(points), self)


def make_controller(*args, route, supervisor, output, **kwargs):
    need(isinstance(route, dict) and set(route) == {'schema','metadata','episode_id','handoff_policy_id'}
         and route['schema'] == TASK_ID and route['handoff_policy_id'] == POLICY_ID
         and route['metadata'].get('handoff_policy_id') == POLICY_ID, 'Strict given-contact configuration required')
    ref = require_route(route['metadata'])
    provider = GivenPoseProvider(ref, route['episode_id'], route['metadata'])
    client = HTTPClient(supervisor['endpoint'], os.environ.get(supervisor['token_env'], ''), supervisor['timeout_s'])
    return AstraGivenContactExpert(*args, metadata=route['metadata'], episode_id=route['episode_id'],
        estimate_provider=provider, astra_client=client, astra_output=Path(output)/'astra',
        preload_context=preload_observation_context(output), **kwargs)


def tag_result(result, controller):
    decisions = deepcopy(getattr(controller, 'astra_decisions', []))
    executed = [r for r in decisions if r['execution']['status'] in ('completed','failed') and
                r['execution']['physics_row_end_exclusive'] > r['execution']['physics_row_start']]
    admitted = [r for r in decisions if r.get('proposal_admitted')]
    for row in [result, *result.get('points', [])]:
        row.update(expert_demonstration=False, learned_policy=False, local_motor_assistance=True,
            training_accepted=False, training_eligible=False, training_dataset_eligible=False)
    result.update(astra_schema=SCHEMA, supervisor_model=MODEL, supervisor_configured=True,
        supervisor_inference=any(r.get('proposal', {}).get('backend') == 'codex_exec' for r in admitted),
        astra_inference_attempt_count=sum(bool(r.get('inference_attempted')) for r in decisions),
        astra_admitted_decision_count=len(admitted),
        controller_source='Astra supervisor + unchanged GivenContactExpert local motor/contact skills',
        executed_controller_class=__name__+'.AstraGivenContactExpert',
        motor_actor='analytic_GivenContactExpert', astra_decisions=decisions,
        closed_loop_evaluated=bool(executed and len(admitted) >= 2),
        astra_native_skill_dispatch_count=len(executed),
        sensor_contract='ausc_astra_rgbd_hall_adc_v1',
        scope='Simulation-only Astra skill supervisor with single-camera RGB-D, calibrated simulated Hall ADC and analytic local motors; prescribed pose, no anatomical depth target estimator, learned motor or clinical claim')
    return result


def run_native_probe(app, stage, output, config, mass):
    """Own no GPU lifecycle; preserve Astra identity around the existing worker."""
    from .fresh_expert import _run_probe
    lifecycle = {}
    try:
        settings(config)
        return _run_probe(app, stage, output, config, mass, _learned_lifecycle=lifecycle)
    except Exception as error:
        env = lifecycle.get('env'); controller = lifecycle.get('controller')
        result = dict(success=False, physical_validation=False, diagnostic_completed=False,
            points=[], physics_steps=0 if env is None else len(env.rows),
            failure_reason=type(error).__name__+': '+str(error),
            errors=[dict(stage='astra_preflight_or_finalization',error=repr(error))],
            runtime_sources_sha256=lifecycle.get('source_hashes', {}))
        tag_result(result, controller)
        Path(output).mkdir(parents=True, exist_ok=True)
        _save(Path(output)/'fresh_expert_result.json', result)
        return result
