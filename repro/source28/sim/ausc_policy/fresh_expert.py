"""Actual home-to-home expert mechanics on hash-bound current 5 mm geometry.

The planner's paths are proposals. Every commanded free sample is checked again,
contact runs the Hall feedback controller, and physical acceptance uses the actual
membrane and device-retainer impulses independently. No raw multimodal training dataset is produced.
Imports are safe on the development Mac; only run_probe imports Isaac/Replicator.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .edge_timing import checked_edge_commands
from .expert_capture import create_expert_capture, capture_settings, checked_render, controller_labels
from .gravity_feedforward import (GravityFeedforwardRejected, NativeGravityFeedforward,
                                 gravity_feedforward_settings, read_native_angular_drive_modes)
from .force_feedback import hold_config, balance_receipt
from .velocity_feedforward import NativeVelocityFeedforward, velocity_feedforward_settings
from .retainer_contact import (CONTACT_SURFACE, RETAINER_PATHS, DIAPHRAGM_PATH, ALL_FORCE_GROUPS,
                               CONTACT_FORCE_GROUPS, is_retainer_patient_pair, audit_retainer_front)
from .retainer_witness import settings as retainer_audit_settings, NativeRetainerAudit
from .retainer_failure import latch_retainer_failure,save_retainer_failure
from .native_contact_evidence import (native_pose_sample,latch_first_positive_contact,
                                      save_first_positive_contact)
from .geometry_epoch import (GEOMETRY_ID, contact_frame, require_current,
                             require_geometry_artifacts, qualify_contact)
from sim.auscultation_expert_layout_v1.controller import CONTROL_STEPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.env import DT, IsaacContactEnv
from sim.auscultation_expert_layout_v1.targets import Target, _base_transform
from sim.auscultation_expert_layout_v1.posture import PostureLimits
from sim.auscultation_skill_v1.controller import ExpertController as SkillExpert


FORBIDDEN = ('lip', 'contact_rim', 'carriage_other', 'frame', 'forbidden')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def passive_effort(slider, preload):
    """Positive slider means compression; preload pushes outward continuously."""
    if (isinstance(preload, (bool, np.bool_)) or not np.isscalar(preload)
            or not np.isfinite(preload) or float(preload) != 1.5
            or isinstance(slider, (bool, np.bool_))
            or not isinstance(slider, (int, np.integer)) or slider < 0):
        raise ValueError('Fresh mechanics requires explicit nominal 1.5 N passive preload')
    return np.array([slider], dtype=int), np.array([-float(preload)])


def _vector(value, shape, label):
    result = np.asarray(value, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid '+label)
    return result


def _route(value, label):
    result = np.asarray(value, float)
    if result.ndim != 2 or result.shape[1:] != (7,) or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError('Invalid '+label)
    return result


def contact_plane_contract(row):
    """Optional for frozen nominal regression; mandatory for varied patients."""
    source=row.get('selected_fit',{}).get('first_contact',{})
    if 'contact_plane_normal_world' not in source:
        if row.get('patient_provenance') is not None:
            raise ValueError('Varied patient is missing the local contact plane')
        return None
    normal=_vector(source['contact_plane_normal_world'],(3,),'contact plane normal')
    limit=source.get('plane_tilt_limit_deg')
    if (not np.isclose(np.linalg.norm(normal),1.,atol=1e-8,rtol=0)
            or isinstance(limit,bool) or limit not in (3.,10.)
            or source.get('contact_plane_radius_m')!=.023
            or source.get('tilt_relaxed') is not (limit==10.)):
        raise ValueError('Invalid local contact plane contract')
    return dict(contact_plane_normal_world=normal.tolist(),contact_plane_radius_m=.023,
        plane_tilt_limit_deg=float(limit),tilt_relaxed=limit==10.)


def contact_plane_observation(contract,axis_world):
    axis=_vector(axis_world,(3,),'actual tool axis')
    if not np.isclose(np.linalg.norm(axis),1.,atol=1e-6,rtol=0):
        raise ValueError('Actual tool axis is not unit length')
    angle=float(np.rad2deg(np.arccos(np.clip(-axis@contract['contact_plane_normal_world'],-1.,1.))))
    return dict(tilt_deg=angle,limit_deg=contract['plane_tilt_limit_deg'],
        passed=angle<=contract['plane_tilt_limit_deg']+1e-9,
        tilt_relaxed=contract['tilt_relaxed'])


def require_near_contact(row, target, normal, center, fitted_gap):
    """Reject coarse noncontact fits and mismatched planner/controller lines."""
    fit=row.get('selected_fit',{})
    receipt=fit.get('first_contact',{})
    if any(value.get('contact_surface')!=CONTACT_SURFACE
           for value in (row,fit,receipt,receipt.get('fit',{}))):
        raise ValueError('Missing current B/C contact surface contract')
    if (row.get('gap_axis')!='command_normal' or fit.get('gap_axis')!='command_normal'
            or receipt.get('gap_axis')!='command_normal' or receipt.get('passed') is not True):
        raise ValueError('Missing command-line first-contact fit receipt')
    residual=receipt.get('contact_face_residual_distance_m',float('nan'))
    resolution=receipt.get('first_contact_search_resolution_m',float('nan'))
    bracket=_vector(receipt.get('contact_surface_gap_bracket_m'),(2,),'contact bracket')
    recorded_center=_vector(receipt.get('center_world_m'),(3,),'contact center')
    raw=_vector(receipt.get('raw_anchor_world_m'),(3,),'unchanged raw target')
    if (not np.isfinite([residual,resolution]).all() or not 1e-5<=residual<=3e-5
            or not 0<resolution<=1e-6 or not 0<=bracket[0]<=bracket[1]<=.005
            or bracket[1]-bracket[0]>resolution+1e-12
            or not bracket[1] < fitted_gap <= .005
            or not np.allclose(raw,target,atol=1e-10,rtol=0)
            or not np.allclose(recorded_center,center,atol=1e-10,rtol=0)
            or not np.allclose(center,target+normal*fitted_gap,atol=1e-10,rtol=0)):
        raise ValueError('First-contact position or resolution does not match command line')
    final=receipt.get('fit',{})
    distances=final.get('distances_m',{})
    if (final.get('passed') is not True or final.get('contact_face_collision') is not False
            or not np.isclose(distances.get('contact_face',float('nan')),residual,rtol=0,atol=1e-12)
            or any(not np.isfinite(distances.get(k,float('nan'))) or distances[k]<minimum
                   for k,minimum in [('fixed',.003),('clamp',.003),('shell',.0003)])):
        raise ValueError('First-contact fit lacks unchanged noncontact clearances')


def read_plans(config, geometry, kin):
    """Verify exact new plan bytes, collider bytes, safety profile and FK identity.

    A loaded fit has a measured micrometre contact-interface gap on the *command* normal
    through the unchanged raw task point, the same line used by the controller.
    The fitted command normal remains distinct from the raw surface normal.
    """
    require_current(config)
    requested = config['fresh_expert']
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    hold_config(PhaseConfig(), requested.get('force_feedback'))
    inter_point_budget_settings(requested.get('inter_point_budget'))
    gravity_feedforward_settings(requested)
    velocity_feedforward_settings(requested)
    passive_effort(0, requested.get('passive_preload_N'))
    installed_patient=None
    geometry_file=Path(geometry)/'collision_geometry.json'
    smpl_surface=geometry_file.is_file() and any(
        c.get('path')=='/World/Patient/SMPLSurface'
        for c in json.loads(geometry_file.read_text()).get('colliders',[]))
    if config.get('patient_asset') or smpl_surface:
        if not config.get('patient_asset'):
            raise ValueError('Varied patient has no authenticated asset specification')
        from .patient_scene import load_candidate
        _,expected_patient=load_candidate(config['patient_asset'],**(
            {'formal_config':config} if 'm2_given_contact_evaluation' in config else {}))
        patient_file=Path(geometry).parent/'patient_receipt.json'
        patient_bytes=patient_file.read_bytes();installed_patient=json.loads(patient_bytes)
        if (installed_patient.get('non_patient_scene_unchanged') is not True or any(
                installed_patient.get(k)!=expected_patient[k]
                for k in ('identity','split','asset_sha256','identity_split_sha256','targets'))):
            raise ValueError('Installed patient differs from authenticated asset')
        installed_patient=dict(installed_patient,receipt_sha256=hashlib.sha256(patient_bytes).hexdigest())
    bindings = requested.get('plans')
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('Fresh plan bindings are required')
    reference_limits = PostureLimits.from_dict(config['posture_limits'], kin).to_dict()
    records=[]; points=[]; base=None; seen=set(); geometry_meta=None
    for binding in bindings:
        path = Path(binding['file']); data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != binding.get('sha256'):
            raise ValueError('Fresh plan bytes changed: '+str(path))
        row = json.loads(data); require_current(row)
        if installed_patient is not None:
            bound=row.get('patient_provenance',{})
            if (bound.get('identity')!=installed_patient['identity'] or
                    bound.get('asset_sha256')!=installed_patient['asset_sha256'] or
                    bound.get('binding',{}).get('sha256')!=installed_patient['receipt_sha256']):
                raise ValueError('Fresh plan missing or mismatched installed patient binding')
        metadata = {**row, **row.get('geometry_artifacts', {})}
        artifacts = require_geometry_artifacts(geometry, metadata)
        if geometry_meta is not None and artifacts != geometry_meta['geometry_artifacts']:
            raise ValueError('Plan sequence geometry differs')
        geometry_meta = dict(geometry_id=row['geometry_id'], freshly_generated=True,
                            geometry_source_sha256=row['geometry_source_sha256'],
                            geometry_artifacts=artifacts,contact_surface=CONTACT_SURFACE)
        current_base = _base_transform(row['world_from_left_base'])
        if base is not None and not np.allclose(current_base, base, rtol=0, atol=1e-9):
            raise ValueError('Plan sequence robot base differs')
        base=current_base
        name=row.get('name')
        if name not in [s+str(i) for s in 'LR' for i in range(1,6)] or name in seen:
            raise ValueError('Unknown or duplicate target')
        seen.add(name)
        if row.get('success') is not True:
            raise ValueError('Fresh plan has no complete checked free route')
        limits = PostureLimits.from_dict(row['posture_limits'], kin).to_dict()
        if limits != reference_limits:
            raise ValueError('Plan safety posture differs from runtime configuration')
        target = _vector(row['position_world'], (3,), 'surface target')
        raw_normal = _vector(row['normal_world'], (3,), 'surface normal')
        normal = _vector(row['command_normal_world'], (3,), 'command normal')
        rotation = _vector(row['command_rotation_world'], (3,3), 'command rotation')
        if (not np.isclose(np.linalg.norm(raw_normal),1.,atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.norm(normal),1.,atol=1e-6,rtol=0)
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0)
                or not np.allclose(rotation[:,2],-normal,atol=1e-6,rtol=0)):
            raise ValueError('Invalid fitted contact frame')
        compression=float(row['loaded_compression_m']); gap=float(row['free_approach_start_gap_m'])
        fitted_gap=float(row['expected_contact_gap_m'])
        if (not np.isfinite([compression,gap,fitted_gap]).all()
                or not 0 < compression <= .005 or not .025 <= gap <= .06
                or not 0 <= fitted_gap < .015):
            raise ValueError('Unsupported contact or free domain')
        center = _vector(row['expected_contact_center_world'],(3,),'fitted center')
        require_near_contact(row,target,normal,center,fitted_gap)
        plane=contact_plane_contract(row)
        if plane is not None:
            if not contact_plane_observation(plane,rotation[:,2])['passed']:
                raise ValueError('Planned orientation exceeds local contact plane limit')
            if row.get('patient_provenance') is not None:
                binding=row['patient_provenance']['binding']
                patient_bytes=Path(binding['file']).read_bytes()
                if hashlib.sha256(patient_bytes).hexdigest()!=binding['sha256']:
                    raise ValueError('Local contact plane patient binding changed')
                actual_target=next(t for t in json.loads(patient_bytes)['targets'] if t['name']==name)
                if (actual_target.get('contact_plane_radius_m')!=.023 or not np.allclose(
                        actual_target['contact_plane_normal_world'],plane['contact_plane_normal_world'],
                        atol=1e-10,rtol=0) or not np.allclose(
                        actual_target['position_world_m'],target,atol=1e-10,rtol=0) or not np.allclose(
                        actual_target['normal_world'],raw_normal,atol=1e-10,rtol=0)):
                    raise ValueError('Local contact plane differs from installed patient')
        transfer=_route(row['transfer_q'],'home transfer'); approach=_route(row['approach_q'],'free approach')
        loaded=_vector(row['loaded_q'],(7,),'loaded joint pose')
        staging=_vector(row['lower_staging_q'],(7,),'lower staging joint pose')
        matches=np.flatnonzero(np.all(np.isclose(transfer,staging,atol=1e-9,rtol=0),axis=1))
        if (not np.allclose(transfer[0],0.,atol=1e-9,rtol=0)
                or not np.allclose(transfer[-1],approach[0],atol=1e-9,rtol=0)
                or len(matches)!=1 or not 0<matches[0]<len(transfer)-1):
            raise ValueError('Disconnected home/approach route or missing unique lower stage')
        if (row.get('geometric_clearance_m') != .003
                or row.get('loaded_noncontact_clearance_m') != .0003
                or not 0 < row.get('joint_edge_sample_step_rad',1) <= .015):
            raise ValueError('Planner clearance criteria differ')
        for q,c,expected,label in ((transfer[-1],0.,target+normal*gap,'prepose'),
                                  (loaded,compression,center,'loaded')):
            pose=base@kin.tcp(q,c)
            if (np.linalg.norm(pose[:3,3]-expected)>3e-5
                    or np.linalg.norm(pose[:3,:3]-rotation)>4e-4):
                raise ValueError('Fresh '+label+' FK identity differs from 132 mm TCP')
        provenance=dict(plan_file=str(path),plan_sha256=actual,geometry_artifacts=artifacts,
                        target_definition='unchanged fresh surface point',
                        fitted_normal_deviation_deg=float(np.rad2deg(np.arccos(np.clip(normal@raw_normal,-1.,1.)))))
        points.append((Target(name,target.copy(),raw_normal.copy(),base[:3,:3].T@(target-base[:3,3]),
                              base[:3,:3].T@raw_normal,provenance),row))
        records.append(dict(file=str(path),sha256=actual,name=name))
    return points, dict(**geometry_meta,world_from_left_base=base.tolist(),plans=records,
        patient_provenance=installed_patient)


def inter_point_budget_settings(value):
    if value is None:
        return None
    if type(value) is not dict or value != {'mode': 'separate_search_validation_v1'}:
        raise ValueError('Explicit separate_search_validation_v1 inter-point budget required')
    return dict(value)


def near_surface_settings(value, inward_speed):
    """Optional slower inward motion, never an increased actuator/safety limit."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'distance_m', 'inward_speed_m_s'}:
        raise ValueError('Explicit near-surface distance and inward speed required')
    distance, speed = value['distance_m'], value['inward_speed_m_s']
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int,float,np.integer,np.floating))
            or not np.isfinite(v) for v in (distance, speed))
            or not 0 < distance <= .005 or not 0 < speed < inward_speed):
        raise ValueError('Near-surface motion must reduce inward speed within 5 mm')
    return dict(distance_m=float(distance), inward_speed_m_s=float(speed))


def contact_timing_preflight(kin, target, row, config, inward_speed, near_surface=None):
    """Reject inadequate contact schedules without changing physical watchdogs.

    The verified loaded FK locates the finite face's first-contact center. Its
    projection can differ from the target point plane when the head is tilted.
    Keep the mathematical optimistic bound separate from the nominal two-speed
    and loaded-pose schedule. Both must fit. The nominal estimate is a planning
    screen, not a proof: acceleration, tracking and feedback delay remain unknown.
    read_plans remains responsible for the contact-geometry evidence.
    """
    base = _base_transform(row['world_from_left_base'])
    normal = _vector(row['command_normal_world'], (3,), 'command normal')
    if not np.isclose(np.linalg.norm(normal), 1., atol=1e-6, rtol=0):
        raise ValueError('Contact timing requires a unit command normal')
    pose = base @ kin.tcp(_vector(row['loaded_q'], (7,), 'loaded joint pose'),
                          float(row['loaded_compression_m']))
    surface_gap = float((pose[:3,3]-target.position_world) @ normal)
    pre_gap = float(row['free_approach_start_gap_m'])
    values = [surface_gap, pre_gap, inward_speed, config.settle_duration_s,
              config.contact_timeout_s]
    if not np.isfinite(values).all() or min(values[1:]) <= 0:
        raise ValueError('Invalid contact timing inputs')
    settings = near_surface_settings(near_surface, inward_speed)
    distance = max(0., pre_gap-surface_gap)
    slow_distance = 0. if settings is None else min(distance, settings['distance_m'])
    travel = distance/inward_speed
    nominal_travel = (distance-slow_distance)/inward_speed
    if settings is not None:
        nominal_travel += slow_distance/settings['inward_speed_m_s']
    # The inner speed is a desired velocity, not an instantaneous bound:
    # retained v/a and the 20 Hz actor delay can cross part of that region more
    # quickly. Only the original global hard inward ceiling gives a necessary
    # lower bound. Keep the two-speed schedule as a separately named estimate.
    compression = float(row['loaded_compression_m'])
    if not np.isfinite(compression) or compression < 0:
        raise ValueError('Invalid loaded compression for contact timing')
    loading = compression/(inward_speed if settings is None else settings['inward_speed_m_s'])
    necessary = travel+config.settle_duration_s
    nominal = nominal_travel+loading+config.settle_duration_s
    necessary_passed = bool(necessary < config.contact_timeout_s)
    schedule_passed = bool(nominal < config.contact_timeout_s)
    return dict(target=target.name, precontact_gap_m=pre_gap, surface_gap_m=surface_gap,
        inward_speed_limit_m_s=float(inward_speed), minimum_travel_s=travel,
        minimum_settle_s=float(config.settle_duration_s), necessary_lower_bound_s=necessary,
        contact_timeout_s=float(config.contact_timeout_s),
        passed=necessary_passed and schedule_passed, success_guaranteed=False,
        necessary_bound_passed=necessary_passed, nominal_schedule_passed=schedule_passed,
        near_surface_motion=settings,
        nominal_two_speed_travel_s=nominal_travel,
        nominal_loaded_pose_travel_s=loading,
        nominal_schedule_total_s=nominal,
        nominal_schedule_margin_s=float(config.contact_timeout_s)-nominal,
        nominal_schedule_exclusions='retained v/a, actor delay, feedback and changing joint tracking',
        excluded_time='acceleration, loading, tracking and feedback delay',
        scope='necessary bound and nominal planning screen; no physical or model qualification')


class FreshExpert(SkillExpert):
    """Retain the original contact executor; use stop-at-waypoint free timing."""
    # Printed lip/rim remain included in the total stop and strict forbidden
    # checks below, while the true device ring contributes to contact evidence.
    front_contact_groups = CONTACT_FORCE_GROUPS
    total_contact_groups = ALL_FORCE_GROUPS
    def __init__(self, *args, near_surface_motion=None, force_feedback=None,
                 inter_point_budget=None, **kwargs):
        super().__init__(*args, speed_scale=.7,
            executor_config=dict(joint_velocity_limit_rad_s=.175), **kwargs)
        self.config=hold_config(self.config,force_feedback)
        self.force_feedback=force_feedback
        self.inter_point_budget=inter_point_budget_settings(inter_point_budget)
        # .25*.7 is mathematically .175 but retain the explicit contract value.
        self.joint_speed_limit=.175
        self.near_surface_motion=near_surface_settings(near_surface_motion,
            self.executor_config['normal_inward_speed_limit_m_s']*self.speed_scale)
        self.contact_preflights=[]
        self.phase_failure_diagnostics=[]
        self.collision_failure_diagnostics=[]

    def _inter_point_validation_deadline(self, search_deadline, record):
        if inter_point_budget_settings(getattr(self,'inter_point_budget',None)) is None:
            return super()._inter_point_validation_deadline(search_deadline,record)
        now=time.monotonic()
        record.update(search_wall_budget_s=30., search_wall_s=now-(search_deadline-30.),
                      validation_wall_budget_s=30.,
                      budget_scope='30 s bounded search then 30 s complete mesh/posture verification; no physics advance')
        return now+30.

    def _tick(self, command, subphase, active=False, recovery=False):
        # The inherited tick performs exactly one command check, then one
        # observed check. Preserve its gates and counter increments verbatim.
        self._fresh_collision_context=dict(stage='command',subphase=subphase,
                                           next_physics_row_index=len(self.env.rows))
        if not hasattr(self,'collision_failure_diagnostics'):
            self.collision_failure_diagnostics=[]
        capture=getattr(self.env,'raw_capture',None)
        rows_before=len(self.env.rows)
        before=controller_labels(self,subphase,active,recovery) if capture is not None else None
        capture_evidence={}
        if is_compact_capture(capture):
            issued=float(self.env.state()['time_s'])
            capture_evidence=dict(clean_command=dict(q_command=np.asarray(command,float).tolist(),
                issued_at_s=issued,endpoint_at_s=issued+DT,readback_at_s=issued+DT,
                source_id='fresh_expert_clean_native_command_v1'),
                control_owner='L3' if getattr(self,'_contact_active',False) or recovery else 'L2')
            if getattr(self,'task_id',None)=='m2_given_contact_pose_control_v1':
                capture_evidence['control_owner']=('L3' if self.control_owner in
                    ('ANALYTIC_L3_ALIGN','ANALYTIC_L3') or recovery else 'L2')
                capture_evidence['clean_command']['source_id']='given_contact_teacher_native_command_v1'
        try:
            try:
                state=super()._tick(command,subphase,active=active,recovery=recovery)
            except BaseException as error:
                if capture is not None:
                    try:
                        capture.finish_tick(self.env,rows_before,before,
                            controller_labels(self,subphase,active,recovery),error=repr(error),**capture_evidence)
                    except BaseException:
                        # The capture owns its failure evidence; retain the
                        # initiating controller exception without storing it.
                        pass
                raise
            if capture is not None:
                capture.finish_tick(self.env,rows_before,before,
                                    controller_labels(self,subphase,active,recovery),**capture_evidence)
            return state
        finally:
            del self._fresh_collision_context

    def _check_collision(self, q, compression, contact):
        context=getattr(self,'_fresh_collision_context',{})
        try:
            if context.get('stage')=='observation':
                if not np.isfinite(compression):
                    raise EpisodeRejected('invalid compression for collision check')
                clearance=self.executor_config['contact_clearance_m'] if contact else .003
                result=self.checker.check_observed(q,contact=self._contact_posture_required(),
                    clearance=clearance,allow_patient_contact=contact,compression=max(0.,float(compression)))
                distance=float(result.get('minimum_distance_m',float('nan')))
                if result.get('clear') is not True or not np.isfinite(distance) or distance<clearance-1e-9:
                    raise EpisodeRejected('command/observation mesh collision rejected: '+str(result))
            else:
                result=super()._check_collision(q,compression,contact)
        except Exception as error:
            def finite(value):
                try:array=np.asarray(value,float)
                except (ValueError,TypeError):return None
                return np.where(np.isfinite(array),array,None).tolist()
            diagnostic=dict(stage=context.get('stage','outside_tick'),
                target=self.target.name,subphase=context.get('subphase'),
                query_q_rad=finite(q),query_compression_m=finite(compression),
                allow_patient_contact=bool(contact),physics_rows=len(self.env.rows),
                physics_row_index=(len(self.env.rows)-1 if context.get('stage')=='observation' else None),
                next_physics_row_index=context.get('next_physics_row_index'),error=str(error))
            if not hasattr(self,'collision_failure_diagnostics'):
                self.collision_failure_diagnostics=[]
            self.collision_failure_diagnostics.append(diagnostic)
            raise
        if context.get('stage')=='command':
            context['stage']='observation'
        return result

    def desired_normal_velocity(self, state):
        # The inherited contact loop refreshes this hook on every phase change,
        # before executor.step. Preserve the initiating fault before a finite
        # acceleration reversal could produce a secondary braking rejection.
        phase = self.manager.state
        if phase.failure_reason is not None:
            def finite_or_none(value):
                try: return float(value) if np.isfinite(value) else None
                except (ValueError, TypeError): return None
            executor = self.contact_executor
            diagnostic = dict(target=self.target.name, phase=phase.phase.value,
                phase_failure_reason=phase.failure_reason, physics_rows=len(self.env.rows),
                time_s=finite_or_none(state.get('time_s')),
                observed_gap_m=finite_or_none(self.metrics(state)['gap_m']),
                executor_gap_m=finite_or_none(executor.gap),
                executor_velocity_m_s=finite_or_none(executor.velocity),
                executor_acceleration_m_s2=finite_or_none(executor.acceleration),
                rejected_before_next_executor_step=True,
                response='stop simulation at original supervisor fault; no alternate motion')
            self.phase_failure_diagnostics.append(diagnostic)
            raise EpisodeRejected('contact phase supervision failed: '+phase.failure_reason)
        desired = super().desired_normal_velocity(state)
        settings = getattr(self, 'near_surface_motion', None)
        if settings is not None and desired < 0.:
            # A commanded wrist pose paired with the currently measured passive
            # slider can reach the unchanged shell gate before actual contact.
            # Slow the approach/loading; retain BOTH command and observed mesh
            # checks, original velocity/acceleration/jerk and all force gates.
            center = float(self._point_plan['expected_contact_gap_m'])
            gaps = [float(self.metrics(state)['gap_m']), float(self.contact_executor.gap), center]
            if not np.isfinite(gaps).all():
                raise EpisodeRejected('Nonfinite near-surface speed observation')
            if min(gaps[:2])-center <= settings['distance_m']:
                desired = max(desired, -settings['inward_speed_m_s'])
        return desired

    def _continuous_path(self, route, subphase):
        route=_route(route,'free route')
        if not np.allclose(route[0],self.q,rtol=0,atol=1e-9):
            raise EpisodeRejected('free route disconnected from retained command')
        commands,report=checked_edge_commands(route,maximum_seconds=300.)
        started=time.monotonic(); deadline=started+max(180.,min(1200.,30.+.02*len(commands)))
        for index,command in enumerate(commands):
            if time.monotonic()>deadline:
                raise EpisodeRejected('free command verification deadline at '+str(index))
            checked=self.checker.check(command,clearance=.003,allow_patient_contact=False,compression=0.)
            distance=float(checked.get('minimum_distance_m',float('nan')))
            if checked.get('clear') is not True or not np.isfinite(distance) or distance < .003-1e-9:
                raise EpisodeRejected('free command failed mesh/posture check: '+str(checked))
        self.free_collision_samples+=len(commands)
        report.update(subphase=subphase,target=self.target.name,start_physics_row=len(self.env.rows),
                      motion_complete=False,all_240hz_commands_mesh_checked=True,
                      collision_verification_samples=len(commands),
                      collision_verification_seconds=time.monotonic()-started)
        self.free_motion_timing.append(report)
        try:
            for index in range(0,len(commands),CONTROL_STEPS):
                self._block(commands[index:index+CONTROL_STEPS],subphase)
            report['motion_complete']=True
        finally:
            report['end_physics_row_exclusive']=len(self.env.rows)

    def _violations(self, state):
        violations,total=super()._violations(state)
        if getattr(self,'_contact_active',False):
            plane=contact_plane_contract(getattr(self,'_point_plan',{}))
            if plane is not None:
                observed=contact_plane_observation(plane,state['axis_world'])
                if self.env.rows:self.env.rows[-1]['contact_plane_audit']=observed
                if not observed['passed']:violations.append('local contact plane tilt')
        for group in FORBIDDEN:
            force=float(self.env.last_contacts.get(group,0.))
            if not np.isfinite(force) or force!=0.:
                violations.append('forbidden non-interface physical contact: '+group)
        front=getattr(self.env,'retainer_front_audit',None)
        if front is not None and front.get('passed') is not True:
            violations.append('device retainer side/back or unreconciled contact')
        return violations,total

    def run(self,points):
        self.contact_preflights=[]
        self.phase_failure_diagnostics=[]
        self.collision_failure_diagnostics=[]
        # Screen the entire sequence before the first home/transfer command.
        inward=self.executor_config['normal_inward_speed_limit_m_s']*self.speed_scale
        for target,row in points:
            report=contact_timing_preflight(self.kin,target,row,self.config,inward,
                                           getattr(self,'near_surface_motion',None))
            self.contact_preflights.append(report)
            if not report['passed']:
                raise EpisodeRejected('contact timing planning screen exceeds watchdog: '
                                      +str(report))
        outcome=super().run(points)
        # Parent acceptance belongs to its older episode-data contract. These
        # traces deliberately lack synchronized RGB-D/ADC training observations.
        outcome['training_accepted']=False
        for row in [*self.env.rows,*self.control_rows,*outcome.get('points',[])]:
            row['training_accepted']=False
        for point in outcome.get('points',[]):
            failures=[item for item in self.phase_failure_diagnostics
                      if item['target']==point.get('target')]
            if failures:
                point['phase_failure_reason']=failures[-1]['phase_failure_reason']
                point['phase_failure_diagnostic']=failures[-1]
        outcome.update(executed_controller_class=__name__+'.FreshExpert',
            contact_plane_contracts={target.name:contact_plane_contract(row) for target,row in points},
            contact_surface=CONTACT_SURFACE,
            candidate_method=__name__+'.FreshExpert',
            candidate_timing='each retained free edge uses quintic time, with a stationary waypoint; contact uses bounded normal executor',
            free_motion_command_bounds=dict(joint_velocity_rad_s=.175,joint_acceleration_rad_s2=.5,
                                            joint_jerk_rad_s3=1.5),
            training_eligible=False,learned_policy=False,
            contact_preflights=self.contact_preflights,
            phase_failure_diagnostics=self.phase_failure_diagnostics,
            collision_failure_diagnostics=self.collision_failure_diagnostics,
            near_surface_motion=getattr(self,'near_surface_motion',None),
            scope='Fresh full-route expert mechanics diagnostic; no raw multimodal training dataset')
        if getattr(self,'near_surface_motion',None) is not None:
            outcome['actual_contact_policy']['near_surface_motion']=self.near_surface_motion
            outcome['actual_contact_policy']['near_surface_rule']='reduce inward desired velocity within distance of fitted first-contact center; outward force correction and withdrawal unchanged'
            outcome['variation']['free_approach_near_speed_m_s']=self.near_surface_motion['inward_speed_m_s']
        if getattr(self,'inter_point_budget',None) is not None:
            outcome.update(inter_point_budget=self.inter_point_budget,
                inter_point_planning_wall_budget_s=60., inter_point_search_wall_budget_s=30.,
                inter_point_validation_wall_budget_s=30.)
        return outcome


class FreshContactEnv(IsaacContactEnv):
    contact_groups = ALL_FORCE_GROUPS

    def __init__(self,*args,native_retainer_audit=None,**kwargs):
        # Verify native binary before World initialization or any movement.
        self.native_retainer_audit=deepcopy(native_retainer_audit)
        self._native_retainer_auditor=(None if native_retainer_audit is None else
                                       NativeRetainerAudit(native_retainer_audit))
        super().__init__(*args,**kwargs)

    def _sample_native_carriage(self,label):
        def read(path):
            from omni.physx import get_physx_interface
            return get_physx_interface().get_rigidbody_transformation(path)
        return native_pose_sample(read,'/World/Stethoscope/Carriage',label)

    def on_native_contact_report(self,headers):
        evidence=self.contact_tick_evidence
        index=evidence['callback_count'];evidence['callback_count']+=1
        if len(evidence['callbacks'])<16:
            evidence['callbacks'].append(dict(callback_index=index,
                header_count=len(headers),contact_data_count=sum(h.num_contact_data for h in headers),
                callback_current_pose=self._sample_native_carriage('callback_current')))
        else:evidence['callbacks_omitted']+=1
        return index

    def contact_group(self, a, b):
        if is_retainer_patient_pair(a, b):return 'retainer'
        label=super().contact_group(a, b)
        if label=='diaphragm' and DIAPHRAGM_PATH not in (a,b):return 'carriage_other'
        return label

    def step(self,q_target,label='',target='',meta=None):
        velocity=getattr(self,'velocity_feedforward',None)
        if velocity is None:return self._position_step(q_target,label,target,meta)
        with velocity.command(self,q_target,label,target,meta) as receipt:
            return self._position_step(q_target,label,target,dict(meta or {},velocity_feedforward=receipt))

    def _position_step(self,q_target,label='',target='',meta=None):
        declared=getattr(self,'native_retainer_audit',None)
        auditor=getattr(self,'_native_retainer_auditor',None)
        if (declared is not None or auditor is not None) and (
                auditor is None or declared!=auditor.contract):
            raise ValueError('Configured v3 native retainer auditor is missing or mismatched')
        from isaacsim.core.utils.types import ArticulationAction
        indices,efforts=passive_effort(self.slider,self.passive_preload_N)
        feedforward=getattr(self,'gravity_feedforward',None)
        metadata=dict(meta or {},explicit_passive_preload_N=self.passive_preload_N,
                      training_eligible=False)
        if feedforward is not None:
            rows_before=len(self.rows)
            try:
                desired=(metadata.get('velocity_feedforward') or {}).get('velocity_target_rad_s')
                indices,efforts,receipt=(feedforward.prepare(q_target) if desired is None else
                    feedforward.prepare(q_target,desired_velocity=desired))
            except GravityFeedforwardRejected as error:
                # A rejected preparation advances no physics. Preserve it in
                # an independent ledger, never by inventing an executed row.
                if not hasattr(self,'gravity_feedforward_failure_receipts'):
                    self.gravity_feedforward_failure_receipts=[]
                self.gravity_feedforward_failure_receipts.append(dict(
                    stage='prepare',phase=label,target=target,subphase=metadata.get('subphase'),
                    physics_rows_before_attempt=rows_before,physics_rows_after_attempt=len(self.rows),
                    executed_physics_row_index=None,effort_action_applied=False,
                    physics_step_executed=False,receipt=deepcopy(error.receipt),error=repr(error)))
                raise
            metadata['gravity_feedforward']=receipt
        self.robot.apply_action(ArticulationAction(joint_efforts=efforts,joint_indices=indices))
        capture=getattr(self,'raw_capture',None)
        rows_before=len(self.rows)
        self.contact_tick_evidence=dict(schema='native_contact_tick_v1',physics_row=rows_before,
            interval_start_s=float(getattr(self,'t',0.)),physics_dt_s=DT,
            pre_step=self._sample_native_carriage('pre_step_current'),callbacks=[],
            callback_count=0,callbacks_omitted=0,contact_generation_pose_available=False,
            callback_contract='PhysX110.1.11 callback delivered after simulation step; current pose is not an impact pose')
        self.native_contact_observer_error=None
        try:
            state=super().step(q_target,label,target,metadata)
            self.contact_tick_evidence['post_step']=self._sample_native_carriage('post_step_current')
            self.contact_tick_evidence['observer_error']=self.native_contact_observer_error
            auditor=getattr(self,'_native_retainer_auditor',None)
            if auditor is not None:
                self.retainer_front_audit=auditor.audit(self.contact_points,self.contact_tick_evidence,
                    self.last_contacts.get('retainer',0.),dt=DT,physics_row=rows_before,
                    time_s=self.rows[-1]['time_s'])
                self.rows[-1]['native_retainer_audit']=deepcopy(auditor.contract)
                self.rows[-1]['native_retainer_tick']=deepcopy(self.contact_tick_evidence)
                # Snapshot metadata only; audit never substitutes an unavailable
                # native frame with SingleRigidPrim, callback pose or FK.
                world=self.contact_tick_evidence['post_step'].get('world_from_carriage')
            else:
                if getattr(self,'native_retainer_audit',None) is not None:
                    raise ValueError('Configured v3 native retainer auditor is missing')
                from scipy.spatial.transform import Rotation
                carriage=next(body for path,body in self.bodies if path=='/World/Stethoscope/Carriage')
                position,quaternion=carriage.get_world_pose()
                world=np.eye(4)
                world[:3,:3]=Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix()
                world[:3,3]=position
                self.contact_tick_evidence['post_step_single_rigid_prim_world_from_carriage']=world.tolist()
                self.retainer_front_audit=audit_retainer_front(
                    self.contact_points,world,self.last_contacts.get('retainer',0.))
            self.rows[-1]['retainer_front_audit']=deepcopy(self.retainer_front_audit)
            self.rows[-1]['contact_surface']=CONTACT_SURFACE
            latch_first_positive_contact(self,world)
            latch_retainer_failure(self,world)
            if getattr(self,'force_balance_enabled',False) and label in ('CONTACT','HOLD','RETRACT'):
                from omni.physx import get_physx_simulation_interface
                from pxr import PhysicsSchemaTools
                from .retainer_contact import CONTACT_PATHS
                report=get_physx_simulation_interface().get_full_contact_report()
                balance=balance_receipt(report,
                    lambda value:str(PhysicsSchemaTools.intToSdfPath(value)),
                    lambda a,b:((a in CONTACT_PATHS and b.startswith('/World/Patient/'))
                                or (b in CONTACT_PATHS and a.startswith('/World/Patient/'))),
                    lambda path:path in CONTACT_PATHS,state,
                    float(self.robot.get_joint_velocities()[self.slider]),self.moving_mass_kg,DT)
                expected=sum(self.last_contacts.get(group,0.) for group in CONTACT_FORCE_GROUPS)
                if not np.isclose(balance['normal_force_N'],expected,atol=1e-8,rtol=1e-8):
                    raise ValueError('Full contact diagnostic differs from original contact ledger')
                self.rows[-1]['force_balance']=balance
        except BaseException:
            if capture is not None:
                try:capture.after_physics(self,rows_before)
                except BaseException:pass  # Preserve the initiating step failure.
            raise
        if capture is not None:capture.after_physics(self,rows_before)
        if feedforward is not None:
            feedforward.verify_applied(receipt)
        if self.video_recorder is not None:
            if capture is None:self.video_recorder.after_step()
            else:checked_render(self,self.video_recorder.after_step,'expert video render')
        return state


def audit_physics(outcome,rows, *, retainer_witness_contract=None):
    """Each target must independently complete ≥3 s of actual B+C contact load."""
    result=deepcopy(outcome)
    if retainer_witness_contract is not None:
        result['native_retainer_audit']=deepcopy(retainer_witness_contract)
    result['training_accepted']=result['training_eligible']=False
    valid=bool(result.get('success') and result.get('points'))
    for point in result.get('points',[]):
        start=point.get('start_physics_row');end=point.get('end_physics_row')
        if (not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(rows)
                or any(row.get('target')!=point.get('target') for row in rows[start:end])):
            audit=dict(passed=False,reason='missing or inconsistent per-point physics range')
        else:
            try:
                audit=qualify_contact(rows[start:end],retainer_witness_contract=retainer_witness_contract)
            except (KeyError, ValueError, TypeError):
                audit=dict(passed=False,reason='missing or malformed physical contact samples')
        point['force_audit']=audit;point['training_accepted']=False
        plane=result.get('contact_plane_contracts',{}).get(point.get('target'))
        if plane is not None:
            try:
                # Independently recompute from measured native axes. Do not
                # trust a stored controller pass flag or nominal command pose.
                longest=count=0;angles=[]
                for row in rows[start:end]:
                    if row.get('phase')!='HOLD':
                        count=0
                        continue
                    observation=contact_plane_observation(plane,row['axis_world'])
                    force=sum(float(row.get(k+'_N',0.)) for k in CONTACT_FORCE_GROUPS)
                    good=observation['passed'] and 3.5<=force<=4.5
                    count=count+1 if good else 0;longest=max(longest,count)
                    angles.append(observation['tilt_deg'])
                plane_audit=dict(passed=longest*DT>=3.-1e-9,
                    continuous_force_and_plane_hold_s=longest*DT,
                    maximum_hold_tilt_deg=max(angles) if angles else None,**plane)
            except (KeyError,ValueError,TypeError):
                plane_audit=dict(passed=False,reason='missing or malformed measured plane evidence')
            point['contact_plane_audit']=plane_audit
            valid=bool(valid and plane_audit['passed'])
        valid=bool(valid and point.get('success') and audit['passed'])
    for row in rows: row['training_accepted']=False
    result['controller_motion_complete']=bool(outcome.get('success'))
    result['success']=result['physical_validation']=valid
    return result


class _VideoRecorder:
    """Stream actual fixed-camera frames, with their authoritative physics indices."""
    def __init__(self,env,output):
        import omni.replicator.core as rep
        self.env=env;self.output=Path(output);self.frames=[]
        self.product=rep.create.render_product('/World/RobotCam',(640,360))
        self.annotator=rep.AnnotatorRegistry.get_annotator('rgb')
        self.annotator.attach([self.product])
        # Rendering warmup advances no physics and never moves any joint.
        for _ in range(3): env.world.render()
        self.stderr=(self.output/'video_encoder.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','640x360','-r','10','-i','-','-an','-c:v','libx264',
            '-pix_fmt','yuv420p',str(self.output/'actual_fresh_expert.mp4')],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.stderr)

    def after_step(self):
        if len(self.env.rows)%24: return
        self.env.world.render()
        image=np.asarray(self.annotator.get_data())
        if image.shape!=(360,640,4) or image.dtype!=np.uint8:
            raise RuntimeError('Missing fixed-camera RGB render at requested physics frame')
        self.process.stdin.write(np.ascontiguousarray(image[:,:,:3]).tobytes())
        self.frames.append(dict(video_frame=len(self.frames),physics_row=len(self.env.rows)-1,
                                time_s=float(self.env.rows[-1]['time_s'])))

    def close(self):
        errors=[];code=None
        def attempt(label,operation):
            try:return operation()
            except Exception as error:
                errors.append(label+': '+repr(error))
                return None
        if getattr(self,'process',None) is not None:
            attempt('stdin_close',self.process.stdin.close)
            try:code=self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                attempt('encoder_kill',self.process.kill)
                attempt('encoder_reap',lambda:self.process.wait(timeout=5))
                errors.append('encoder timed out')
            except Exception as error:errors.append('encoder_wait: '+repr(error))
        # Releasing Replicator resources and preserving the frame index are
        # independent of ffmpeg or its stderr stream successfully closing.
        if getattr(self,'stderr',None) is not None:attempt('stderr_close',self.stderr.close)
        if getattr(self,'annotator',None) is not None:attempt('annotator_detach',self.annotator.detach)
        if getattr(self,'product',None) is not None:attempt('render_product_destroy',self.product.destroy)
        attempt('frame_index_save',lambda:_write(self.output/'video_frames.json',self.frames))
        if code!=0 or not self.frames:errors.append('encoding failed or no frames captured')
        if errors:raise RuntimeError('; '.join(errors))
        return dict(file=str(self.output/'actual_fresh_expert.mp4'),frames=len(self.frames),fps=10,
                    width=640,height=360,camera='/World/RobotCam')



def is_compact_capture(capture):
    return capture is not None and getattr(capture,'binding',{}).get('profile')=='compact'


def scene_factory(stage,scene_reference=None):
    """Keep legacy disk export; opted-in workers get a private anonymous root."""
    def factory(destination):
        if scene_reference is not None:
            from .scene_store import SDK_SCHEMA,SDK_LAYER_SCHEMA,runtime_stage,write_scene_ref,write_sdk_selection
            write_scene_ref(Path(destination).parent,scene_reference)
            private=runtime_stage(scene_reference)
            if scene_reference['binding']['schema'] in (SDK_SCHEMA,SDK_LAYER_SCHEMA):
                write_sdk_selection(private,Path(destination).parent)
            return private
        stage.GetRootLayer().Export(str(destination))
        return stage
    return factory


def finalize_probe(env,controller,recorder,output,*,capture=None,error=None):
    """Attempt every independent preservation/release operation after a failure.

    An encoder must close and the physics world must stop even when a disk write
    or strict JSON serialization fails. Return all errors to revoke acceptance.
    """
    errors=[];video=None;output=Path(output)
    from .m2_given_contact_pilot_native_audit import installed,close_checked
    pilot_light=installed(controller)
    def attempt(label,operation):
        try:return operation()
        except BaseException as caught:
            if capture is None and not isinstance(caught,Exception):raise
            errors.append(dict(stage=label,error=repr(caught)))
            return None
    if env is not None and getattr(env,'velocity_feedforward',None) is not None:
        attempt('velocity_feedforward_reset',lambda:env.velocity_feedforward.reset(env))
        attempt('velocity_feedforward_reset_save',lambda:_write(
            output/'fresh_expert_velocity_feedforward_reset.json',
            getattr(env,'velocity_feedforward_reset_receipt',None)))
    if capture is not None:attempt('raw_capture_finalize',lambda:capture.close(error))
    if controller is not None and hasattr(controller,'close_astra_sensors'):
        attempt('astra_sensor_finalize',controller.close_astra_sensors)
    if env is not None:
        if getattr(env,'_retainer_failure_payload',None) is not None:
            attempt('retainer_failure_save',lambda:save_retainer_failure(env,output,capture=capture))
        if getattr(env,'_retainer_first_contact_payload',None) is not None:
            attempt('retainer_first_contact_save',lambda:save_first_positive_contact(env,output,capture=capture))
        if capture is not None:attempt('raw_capture_hook_detach',lambda:setattr(env,'raw_capture',None))
        attempt('video_hook_detach',lambda:setattr(env,'video_recorder',None))
        def save_trace():
            for row in env.rows:row['training_accepted']=False
            # Compact's bounded native_audit already records safety/phase
            # scalars. Full poses, impulses and native command mirrors belong
            # to the explicitly selected full_raw/Tier B profile only.
            if not is_compact_capture(capture) and not pilot_light:env.save('fresh_expert')
        attempt('trace_save',save_trace)
        if hasattr(env,'gravity_feedforward_failure_receipts'):
            attempt('gravity_feedforward_failure_save',lambda:_write(
                output/'fresh_expert_gravity_feedforward_failures.json',
                env.gravity_feedforward_failure_receipts))
        if hasattr(env,'velocity_feedforward_failure_receipts'):
            attempt('velocity_feedforward_failure_save',lambda:_write(
                output/'fresh_expert_velocity_feedforward_failures.json',
                env.velocity_feedforward_failure_receipts))
    if controller is not None:
        def save_controls():
            if is_compact_capture(capture) or pilot_light:
                return
            with (output/'fresh_expert_control.jsonl').open('w') as stream:
                for row in controller.control_rows:
                    row['training_accepted']=False
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
        attempt('control_save',save_controls)
        attempt('diagnostics_save',lambda:_write(output/'fresh_expert_controller_diagnostics.json',dict(
            contact_curves=getattr(controller,'contact_curves',[]),
            free_motion_timing=getattr(controller,'free_motion_timing',[]),
            inter_point_routes=getattr(controller,'inter_point_routes',[]),
            contact_preflights=getattr(controller,'contact_preflights',[]),
            phase_failure_diagnostics=getattr(controller,'phase_failure_diagnostics',[]),
            collision_failure_diagnostics=getattr(controller,'collision_failure_diagnostics',[]))))
        if hasattr(controller,'_neutral_native_audit'):
            from .m2_given_contact_evaluation import close_native_audit
            attempt('neutral_native_audit_finalize',lambda:close_checked(controller) if pilot_light else close_native_audit(controller))
    if recorder is not None:video=attempt('video_finalize',lambda:recorder.close())
    if env is not None:
        attempt('world_stop',lambda:env.world.stop())
        if getattr(env,'scene_reference',None) is not None:
            from .scene_store import defer_runtime_stage_release
            def defer_cache_lease():
                env.private_scene_cache_lease_receipt=defer_runtime_stage_release(env.stage)
                _write(output/'private_scene_stage_cache_lease.json',env.private_scene_cache_lease_receipt)
            attempt('private_scene_cache_defer',defer_cache_lease)
    return video,errors


def _learned_failure_handoffs(config,reason,target=None):
    """A failed declared single-point attempt remains in its L2 denominator."""
    if target is None:
        deployment=config.get('m2_act_deployment')
        metadata=deployment.get('metadata') if isinstance(deployment,dict) else None
        target=metadata.get('target_id') if isinstance(metadata,dict) else None
    return [dict(target=target,l2_handoff_success=False,l3_started=False,l3_completed=False,
                 expert_l2_fallback=False,failure_reason=str(reason),learned_policy=True,
                 expert_demonstration=False,training_eligible=False)]


def run_probe(app,stage,output,config,mass):
    """Preserve learned identity even when preflight fails before env creation."""
    if 'm2_rebuild_paired_eval' in config:
        from .m2_rebuild_paired_eval import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'astra_supervisor' in config:
        from .astra_native import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_given_contact_expanded_val_scenarios' in config:
        from .m2_given_contact_expanded_val_scenarios import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_given_contact_expanded_pilot_deployment' in config:
        from .m2_given_contact_expanded_pilot_deployment import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_given_contact_pilot_deployment' in config:
        from .m2_given_contact_pilot_deployment import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_given_contact_evaluation' in config:
        from .m2_given_contact_evaluation import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_given_contact_deployment' in config:
        from .m2_given_contact_deployment import run_native_probe
        return run_native_probe(app,stage,output,config,mass)
    if 'm2_act_deployment' not in config:
        return _run_probe(app,stage,output,config,mass)
    lifecycle={}
    try:
        if isinstance(config['m2_act_deployment'],dict) and config['m2_act_deployment'].get('schema')=='m2_act_pilot_history2_deployment_v2':
            from .m2_pilot_deployment import preflight
            preflight(config)
        return _run_probe(app,stage,output,config,mass,_learned_lifecycle=lifecycle)
    except BaseException as error:
        from .m2_act_controller import tag_learned_result
        deployment=config.get('m2_act_deployment')
        receipt=dict(schema='m2_act_l2_failed_preflight_v1',controller_constructed=False,
            configured_deployment=deepcopy(deployment),checkpoint_validation_completed=False)
        env=lifecycle.get('env');controller=lifecycle.get('controller')
        result=dict(success=False,physical_validation=False,diagnostic_completed=False,
            physics_steps=len(env.rows) if env is not None else 0,video=lifecycle.get('video'),
            geometry_id=GEOMETRY_ID,geometry_source_sha256=config.get('geometry_source_sha256'),points=[],
            errors=[dict(stage='learned_preflight_or_finalization',error=repr(error))],failure_reason=str(error),
            raw_capture_complete=False,compact_capture_complete=False,raw_multimodal_training_capture=False)
        result['runtime_sources_sha256']=lifecycle.get('source_hashes',{})
        if env is not None and hasattr(env,'retainer_failure_evidence'):
            result['retainer_failure_evidence']=deepcopy(env.retainer_failure_evidence)
        if env is not None and hasattr(env,'retainer_first_contact_evidence'):
            result['retainer_first_contact_evidence']=deepcopy(env.retainer_first_contact_evidence)
        tag_learned_result(result,getattr(controller,'policy_receipt',receipt),
            getattr(controller,'l2_handoffs',[]) or _learned_failure_handoffs(config,error))
        try:_write(Path(output)/'fresh_expert_result.json',result)
        except Exception as write_error:
            result['errors'].append(dict(stage='result_save',error=repr(write_error)))
        if not isinstance(error,Exception):raise
        return result


def _run_probe(app,stage,output,config,mass,*,_learned_lifecycle=None):
    """Called only by the owned Linux geometry-qualification Isaac worker."""
    import time
    from .runtime_cost import runtime_cost
    probe_wall_started=time.monotonic()
    controller_wall_started=None;controller_wall_ended=None
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from .contact_collision import ContactCollisionChecker
    import os,sys
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':
        raise RuntimeError('Owned Linux Isaac runtime required')
    formal_test=None
    if 'm2_given_contact_evaluation' in config:
        from .m2_given_contact_evaluation import preflight
        formal_test=preflight(config,output=output)
        if formal_test['stage']!='native':raise ValueError('A preparation capability cannot execute native physics')
        import random
        random.seed(formal_test['row']['seed']);np.random.seed(formal_test['row']['seed'])
    native_retainer_contract=retainer_audit_settings(config['fresh_expert'])
    feedforward_settings=gravity_feedforward_settings(config['fresh_expert'])
    velocity_settings=velocity_feedforward_settings(config['fresh_expert'])
    raw_settings=capture_settings(config['fresh_expert'])
    learned='m2_act_deployment' in config
    formal_learned='m2_given_contact_deployment' in config
    validation_scenario='m2_given_contact_expanded_val_scenarios' in config
    expanded_pilot='m2_given_contact_expanded_pilot_deployment' in config
    rebuilt_paired='m2_rebuild_paired_eval' in config
    pilot_learned='m2_given_contact_pilot_deployment' in config or expanded_pilot or validation_scenario or rebuilt_paired
    given_learned=formal_learned or pilot_learned
    formal_binding=None
    if formal_learned:
        from . import m2_given_contact_deployment as formal_adapter
        formal_binding=formal_adapter.preflight(config,output=output)
    if pilot_learned:
        if rebuilt_paired:
            from . import m2_rebuild_paired_eval as pilot_adapter
        elif validation_scenario:
            from . import m2_given_contact_expanded_val_scenarios as pilot_adapter
        elif expanded_pilot:
            from . import m2_given_contact_expanded_pilot_deployment as pilot_adapter
        else:
            from . import m2_given_contact_pilot_deployment as pilot_adapter
        pilot_binding=pilot_adapter.preflight(config,output=output)
        if validation_scenario:
            import random
            random.seed(pilot_binding['case']['seed']);np.random.seed(pilot_binding['case']['seed'])
    given_route=config.get('m2_given_contact_control')
    given_recovery='m2_given_contact_recovery' in config
    if given_recovery:
        from . import m2_given_contact_recovery as recovery_adapter
        recovery_adapter.preflight(config)
    if given_route is not None:
        if learned:raise ValueError('Given-contact teacher and old learned route cannot be mixed')
        if raw_settings is not None and raw_settings.get('profile')!='compact':
            raise ValueError('Given-contact native capture requires its versioned compact metadata')
        from .m2_given_contact_pose import require_route
        given_ref=require_route(given_route['metadata']);given_ref.verify_sources()
        if (raw_settings is not None and (raw_settings['compact']['metadata'].get('reference_sha256')!=given_ref.sha256 or
                raw_settings['compact']['metadata'].get('task_id')!=given_route['schema'])):
            raise ValueError('Given-contact capture/control reference differs')
    elif raw_settings is not None and raw_settings.get('compact',{}).get('metadata',{}).get('target_source')=='given_contact_pose':
        raise ValueError('Given-contact capture cannot relabel the original precise-route teacher')
    estimate_provider=None;learned_adapter=None
    if given_learned:
        from .m2_given_contact_pose import GivenPoseProvider,require_route
        from . import m2_act_controller as learned_adapter
        binding=pilot_binding if pilot_learned else formal_binding
        deployment=deepcopy(binding['deployment']);given_ref=require_route(deployment['metadata'])
        estimate_provider=GivenPoseProvider(given_ref,deployment['episode_id'],deployment['metadata'])
    if learned and raw_settings is not None:
        raise ValueError('M2 ACT deployment cannot use expert raw/compact capture labels')
    residual_options={};residual_sources=[]
    if 'stop_residual_trajectory' in config:
        from . import m2_stop_residual_trajectory as residual
        given_meta=(given_route['metadata'] if given_route is not None else
                    deployment['metadata'] if given_learned else {})
        residual_options=residual.native_options(config,metadata=given_meta)
        residual_sources=residual.source_paths(config)
    from .passive_preload import passive_preload_settings, verify_preload_readback
    preload_settings=passive_preload_settings(config['fresh_expert'])
    preload_receipt=None;preload_startup=None
    output=Path(output);kin=OpenArmKinematics(contact_frame=contact_frame)
    points,metadata=read_plans(config,output/'geometry',kin)
    if given_route is not None:
        if len(points)!=1:raise ValueError('Given-contact control requires one declared target')
        metadata.update(task_id=given_route['schema'],given_contact_control=deepcopy(given_route))
    if native_retainer_contract is not None:
        metadata['native_retainer_audit']=deepcopy(native_retainer_contract)
    if given_learned:
        if len(points)!=1 or points[0][0].name!=deployment['metadata']['target_id']:
            raise ValueError('Given ACT requires its one frozen target')
        patient=metadata.get('patient_provenance') or {}
        if (patient.get('identity')!=deployment['metadata']['subject_id'] or
                patient.get('split')!=deployment['metadata']['split'] or
                patient.get('receipt_sha256')!=given_ref.snapshot()['sources']['patient_receipt']['sha256']):
            raise ValueError('Given ACT reference differs from actual installed patient')
        metadata.update(task_id=given_ref.snapshot()['task_id'],learned_policy=True,expert_demonstration=False,
            **{(pilot_adapter.ROUTE_KEY if pilot_learned else 'm2_given_contact_deployment'):deepcopy(deployment)})
    if learned:
        from . import m2_act_controller as learned_adapter
        from .m2_estimate_provider import FrozenEstimateProvider
        deployment=config['m2_act_deployment']
        if len(points)!=1:raise ValueError('M2 ACT deployment requires exactly one declared target')
        if _sha(deployment['checkpoint'])!=deployment['checkpoint_sha256']:
            raise ValueError('M2 deployment checkpoint hash changed before initialization')
        estimate_provider=FrozenEstimateProvider(config.get('m2_target_estimate'),
            metadata=deployment['metadata'],episode_id=deployment['episode_id'],target_id=points[0][0].name,
            patient=metadata.get('patient_provenance'))
        deployment=dict(deepcopy(deployment),metadata=deepcopy(estimate_provider.metadata))
        for key,value in metadata['geometry_artifacts'].items():
            if estimate_provider.snapshot()['source_metadata'].get(key)!=value:
                raise ValueError('Frozen estimate collision geometry differs from actual runtime: '+key)
    if preload_settings is not None:
        preload_receipt=json.loads((output/'passive_preload_authoring.json').read_text())
        metadata['passive_preload_model']=dict(authoring=preload_receipt,
            preinitialization_usd_readback=verify_preload_readback(stage,config['fresh_expert'],preload_receipt))
    base=np.asarray(metadata['world_from_left_base'])
    current={**metadata,**metadata['geometry_artifacts']}
    checker=ContactCollisionChecker(kin,base,output/'geometry',current,
        profile=config['fresh_expert'].get('collision_profile','full'))
    metadata['collision_profile']=checker.profile
    snapshot=output/'fresh_expert_inputs';snapshot.mkdir(exist_ok=False)
    for index,binding in enumerate(metadata['plans']):
        data=Path(binding['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=binding['sha256']:
            raise ValueError('Fresh plan changed before scene initialization')
        (snapshot/(str(index).zfill(2)+'_'+binding['name']+'.json')).write_bytes(data)
    source_paths=[Path(__file__),Path(__file__).with_name('runtime_cost.py'),Path(__file__).with_name('edge_timing.py'),
                  Path(__file__).with_name('geometry_epoch.py'),Path(__file__).with_name('contact_collision.py'),
                  Path(__file__).with_name('retainer_contact.py'),
                  Path(__file__).with_name('retainer_failure.py'),
                  Path(__file__).with_name('native_contact_evidence.py'),
                  Path(__file__).with_name('gravity_feedforward.py'),
                  Path(__file__).with_name('velocity_feedforward.py')]
    source_paths += residual_sources
    if native_retainer_contract is not None:
        source_paths += [Path(__file__).with_name('retainer_witness.py'),
            Path(__file__).with_name('retainer_geometry.py'),Path(native_retainer_contract['qualification']['file'])]
    source_paths.append(Path(__file__).with_name('force_feedback.py'))
    if 'matched_scene_eval' in config:
        from . import m2_rebuild_paired_eval
        source_paths += [Path(m2_rebuild_paired_eval.__file__),Path(m2_rebuild_paired_eval.audit.__file__),Path(m2_rebuild_paired_eval.neutral.__file__)]
    if 'astra_supervisor' in config:
        from . import astra_native,astra_supervisor,astra_service,astra_sensors
        source_paths += [Path(module.__file__) for module in (astra_native,astra_supervisor,astra_service)]
        source_paths += astra_sensors.source_paths()
        metadata['astra_supervisor']=deepcopy(config['astra_supervisor'])
        _write(snapshot/'astra_supervisor.json',config['astra_supervisor'])
    if given_route is not None or given_learned:
        from . import m2_given_contact_pose,m2_given_contact_controller,m2_given_contact_act,m2_given_contact_intent,m2_contract,m2_stage,m2_act_controller,m2_act
        from sim.auscultation_pi05_v6 import execution as given_execution
        source_paths += [Path(module.__file__) for module in (m2_given_contact_pose,m2_given_contact_controller,
            m2_given_contact_act,m2_given_contact_intent,m2_contract,m2_stage,m2_act_controller,m2_act,given_execution)]
        source_paths += [Path(ref['file']) for ref in given_ref.snapshot()['sources'].values()]
        from . import m2_given_contact_handoff
        given_meta=given_route['metadata'] if given_route is not None else deployment['metadata']
        if given_meta.get('handoff_policy_id')==m2_given_contact_handoff.POLICY_ID:
            source_paths.append(Path(m2_given_contact_handoff.__file__))
        if pilot_learned:
            from . import m2_given_contact_pilot_native_audit as pilot_native_audit
            source_paths += [Path(pilot_adapter.__file__),Path(pilot_native_audit.__file__),Path(pilot_native_audit.neutral.__file__)]
        if given_recovery:
            from sim.auscultation_expert_layout_v1 import plan_low_elbow,posture,pose_ik
            source_paths += [Path(module.__file__) for module in (recovery_adapter,recovery_adapter.runtime,
                recovery_adapter.runtime.dataset,plan_low_elbow,posture,pose_ik)]
    if 'compact_scene_storage' in config:
        source_paths.append(Path(__file__).with_name('scene_store.py'))
        from .scene_store import SDK_LAYER_SCHEMA
        if config['compact_scene_storage'].get('schema')==SDK_LAYER_SCHEMA:
            source_paths.append(Path(__file__).with_name('layered_scene_authoring.py'))
            source_paths.append(Path(__file__).with_name('m2_pilot_config.py'))
            source_paths.append(Path(__file__).with_name('patient_scene.py'))
            source_paths.append(Path(__file__).with_name('native_usd_preflight.py'))
            source_paths.append(Path(__file__).with_name('geometry_qualification.py'))
    if preload_settings is not None:source_paths.append(Path(__file__).with_name('passive_preload.py'))
    import inspect
    from sim.auscultation_skill_v1 import curve,executor
    from sim.curobo_expert_v1 import static_world
    from sim.auscultation_expert_layout_v1.kinematics import DEFAULT_URDF_PATH
    source_objects=[IsaacContactEnv,OpenArmKinematics,PostureLimits,curve,executor,static_world]
    source_paths.append(Path(static_world.__file__).with_name('fast_collision.py'))
    source_objects += [cls for cls in SkillExpert.__mro__ if cls.__module__.startswith('sim.')]
    source_paths += [Path(inspect.getfile(value)) for value in source_objects]+[Path(DEFAULT_URDF_PATH)]
    if learned:
        from . import m2_act,m2_contract,m2_estimate_provider,contact_entry
        from sim.auscultation_pi05_v6 import execution
        source_paths += [Path(module.__file__) for module in
            (learned_adapter,m2_act,m2_contract,m2_estimate_provider,contact_entry,execution)]
        source_paths += [Path(config['m2_act_deployment']['checkpoint']),*estimate_provider.source_paths]
        if deployment.get('schema')=='m2_act_pilot_history2_deployment_v2':
            from . import m2_pilot_deployment
            source_paths += [Path(m2_pilot_deployment.__file__),Path(deployment['diagnostic_receipt']['file']).resolve()]
        metadata.update(learned_policy=True,expert_demonstration=False,
            m2_act_deployment=deepcopy(deployment),
            m2_estimate_provider=estimate_provider.snapshot())
        for name,data in estimate_provider.artifact_bytes.items():(snapshot/name).write_bytes(data)
        _write(snapshot/'m2_estimate_provider.json',estimate_provider.snapshot())
    if formal_learned:
        permit=formal_binding['permit'];admission=formal_adapter.runtime.read_bound(permit['admission'])
        source_paths += [Path(ref['path']) for ref in permit['source_inventory'].values()]
        source_paths += [Path(ref['path']) for ref in (deployment['permit'],permit['admission'],
            permit['model']['checkpoint'],permit['model']['complete'],admission['normalization'],
            admission['reference_inventory'],admission['source_inventory'],permit['processing']['processing_freeze'])]
    if formal_learned or formal_test is not None:
        from . import m2_given_contact_evaluation,patient_scene,m2_pilot_config,geometry_plan,tool_geometry_review
        source_paths.extend(Path(module.__file__) for module in
            (m2_given_contact_evaluation,patient_scene,m2_pilot_config,geometry_plan,tool_geometry_review))
        if formal_test is not None:
            source_paths.extend(Path(ref['path']) for ref in formal_test['capability']['source_inventory'].values())
    source_hashes={str(path):_sha(path) for path in source_paths}
    if learned:
        expected=dict(estimate_provider.expected_source_hashes)
        expected[str(Path(config['m2_act_deployment']['checkpoint']))]=config['m2_act_deployment']['checkpoint_sha256']
        if deployment.get('schema')=='m2_act_pilot_history2_deployment_v2':
            expected[str(Path(deployment['diagnostic_receipt']['file']).resolve())]=deployment['diagnostic_receipt']['sha256']
        if any(source_hashes[path]!=digest for path,digest in expected.items()):
            raise ValueError('Learned checkpoint/estimate source changed while preparing snapshot')
        if _learned_lifecycle is not None:_learned_lifecycle['source_hashes']=source_hashes
    if formal_learned and _learned_lifecycle is not None:_learned_lifecycle['source_hashes']=source_hashes
    if 'astra_supervisor' in config and _learned_lifecycle is not None:_learned_lifecycle['source_hashes']=source_hashes
    metadata['runtime_sources_sha256']=source_hashes
    _write(snapshot/'manifest.json',metadata)
    _write(snapshot/'runtime_config.json',config)
    _write(snapshot/'current_mass_estimate.json',mass)
    mass_kg=float(mass['mass_kg'])
    if not np.isfinite(mass_kg) or mass_kg<=0:raise ValueError('Invalid current moving mass')
    scene_reference=None
    if 'compact_scene_storage' in config:
        from .scene_store import resolve_scene
        scene_reference=json.loads((output/'scene_ref.json').read_bytes())
        resolve_scene(scene_reference)
    factory=scene_factory(stage,scene_reference)
    env=None;controller=None;recorder=None;video=None;capture=None
    outcome=dict(success=False,points=[],training_accepted=False)
    errors=[]
    try:
        if raw_settings is not None:
            capture=create_expert_capture(output/('compact_multimodal' if raw_settings.get('profile')=='compact'
                                                 else 'raw_multimodal'),binding=raw_settings,
                geometry_metadata=metadata,source_paths=source_paths)
        scene_options={} if scene_reference is None else dict(scene_reference=scene_reference)
        env=FreshContactEnv(output/'fresh_expert',stage_factory=factory,
                            contact_offset_m=.132,moving_mass_kg=mass_kg,native_retainer_audit=native_retainer_contract,**scene_options)
        if _learned_lifecycle is not None:_learned_lifecycle['env']=env
        env.passive_preload_N=config['fresh_expert']['passive_preload_N'];env.video_recorder=None
        env.force_balance_enabled=config['fresh_expert'].get('force_feedback') is not None
        if preload_settings is not None:
            preload_startup=verify_preload_readback(env.stage,config['fresh_expert'],preload_receipt,
                robot=env.robot,slider=env.slider)
            _write(output/'passive_preload_startup_usd_readback.json',preload_startup)
        if velocity_settings is not None and velocity_settings['enabled']:
            env.velocity_feedforward=NativeVelocityFeedforward(env.robot,env.left,physics_dt=DT)
        if feedforward_settings is not None and feedforward_settings['enabled']:
            env.gravity_feedforward=NativeGravityFeedforward(env.robot,env.left,env.slider,
                drive_mode_provider=lambda:read_native_angular_drive_modes(env.stage,env.robot,env.left))
        if not np.allclose(env.base,base,atol=1e-7,rtol=0):raise ValueError('Actual robot base differs from fresh plan')
        # No joint setter, teleport, or contact-pose reset. The asset home state
        # must be dynamically followed through the complete canonical route.
        if np.max(np.abs(env.state()['q']))>.03:raise ValueError('Scene does not start at original home')
        def create_recorder():
            nonlocal recorder
            if capture is None:recorder=_VideoRecorder(env,output)
            else:
                # Register ownership before setup can fail after creating a
                # render product, annotator or encoder stream.
                recorder=_VideoRecorder.__new__(_VideoRecorder)
                recorder.__init__(env,output)
        if capture is None:create_recorder()
        elif not is_compact_capture(capture):checked_render(env,create_recorder,'expert video setup')
        env.video_recorder=recorder
        if capture is not None:
            env.raw_capture=capture
            capture.start(env)
        controller_options=dict(static_world_bundle=checker.static_bundle,static_geometry_dir=output/'geometry',
            near_surface_motion=config['fresh_expert'].get('near_surface_motion'),
            force_feedback=config['fresh_expert'].get('force_feedback'),
            inter_point_budget=config['fresh_expert'].get('inter_point_budget'))
        controller_options.update(residual_options)
        if given_learned:
            given_adapter=pilot_adapter if pilot_learned else formal_adapter
            controller=given_adapter.make_controller(env,kin,checker,deployment=deployment,
                estimate_provider=estimate_provider,output=output,**controller_options)
            _write(snapshot/'given_reference_before_motion.json',estimate_provider.snapshot())
        elif learned:
            controller_factory=learned_adapter.make_controller
            if deployment.get('schema')=='m2_act_pilot_history2_deployment_v2':
                from .m2_pilot_deployment import make_controller as controller_factory
            controller=controller_factory(env,kin,checker,deployment=deployment,
                estimate_provider=estimate_provider,**controller_options)
        elif given_route is not None:
            from .m2_given_contact_controller import make_expert
            if 'astra_supervisor' in config:
                from .astra_native import make_controller as make_astra
                controller=make_astra(env,kin,checker,route=given_route,
                    supervisor=config['astra_supervisor'],output=output,**controller_options)
            elif given_recovery:
                controller=recovery_adapter.make_controller(env,kin,checker,config=config,**controller_options)
            else:controller=make_expert(env,kin,checker,route=given_route,**controller_options)
            _write(snapshot/'given_reference_before_motion.json',controller.estimate_provider.snapshot())
        else:controller=FreshExpert(env,kin,checker,**controller_options)
        if formal_learned or formal_test is not None:
            from .m2_given_contact_evaluation import attach_native_audit
            attach_native_audit(controller,config,output)
        elif pilot_learned:
            from .m2_given_contact_pilot_native_audit import attach
            attach(controller,config,output,adapter=pilot_adapter)
        elif 'matched_scene_eval' in config:
            from .m2_rebuild_paired_eval import attach_analytic
            attach_analytic(controller,config,output)
        if _learned_lifecycle is not None:_learned_lifecycle['controller']=controller
        controller_wall_started=time.monotonic()
        outcome=controller.run(points)
    except BaseException as error:
        if raw_settings is None and not isinstance(error,Exception):raise
        import traceback
        errors.append(dict(stage='execution',error=repr(error)))
        try:(output/'fresh_expert_failure.txt').write_text(traceback.format_exc())
        except Exception as write_error:
            errors.append(dict(stage='failure_text_save',error=repr(write_error)))
    finally:
        if controller_wall_started is not None:controller_wall_ended=time.monotonic()
        if given_learned and controller is not None:
            try:controller.close_bridge()
            except Exception as error:errors.append(dict(stage='given_cpu_child_cleanup',error=repr(error)))
        if learned or given_learned:
            for row in [*([] if env is None else env.rows),
                        *getattr(controller,'control_rows',[])]:row.update(learned_adapter.LABELS)
        capture_error=('; '.join(item['error'] for item in errors) if errors else
                       None if outcome.get('success') else outcome.get('failure_reason','Expert execution incomplete'))
        video,cleanup_errors=finalize_probe(env,controller,recorder,output,
                                          capture=capture,error=capture_error)
        if _learned_lifecycle is not None:_learned_lifecycle['video']=video
        errors.extend(cleanup_errors)
    rows=[] if env is None else env.rows
    try:result=audit_physics(outcome,rows,retainer_witness_contract=native_retainer_contract)
    except Exception as error:
        errors.append(dict(stage='physics_audit',error=repr(error)))
        result=dict(success=False,physical_validation=False,training_accepted=False,points=[])
    if native_retainer_contract is not None:result['native_retainer_audit']=deepcopy(native_retainer_contract)
    if formal_learned or formal_test is not None:
        result['neutral_native_audit']=deepcopy(getattr(controller,'neutral_native_audit',None))
    if not result['physical_validation']:
        result.setdefault('failure_reason','actual per-point contact-interface/withdrawal qualification failed')
    result.update(geometry_id=GEOMETRY_ID,geometry_source_sha256=config['geometry_source_sha256'],
                  contact_surface=CONTACT_SURFACE,
                  geometry_artifacts=metadata['geometry_artifacts'],plan_bindings=metadata['plans'],
                  explicit_passive_preload_N=1.5,moving_mass_kg=mass_kg,
                  diagnostic_completed=True,physics_steps=len(rows),video=video,learned_policy=False,
                  training_eligible=False,raw_multimodal_training_capture=False,
                  scope='Actual new-geometry complete home/staging/contact/withdrawal expert mechanics; no model or generalization claim')
    if preload_settings is not None:
        result['passive_preload_model']=dict(metadata['passive_preload_model'],
            startup_usd_readback=preload_startup)
    if feedforward_settings is not None:
        result['gravity_feedforward_failure_receipts']=deepcopy(
            getattr(env,'gravity_feedforward_failure_receipts',[]))
        result['gravity_feedforward']=dict(feedforward_settings,
            schema='native_acceleration_drive_gravity_v2',required_drive_mode='acceleration',
            controller_type='model-based controller feedforward',
            native_api='get_generalized_gravity_forces',sign='positive native counter-gravity',
            position_drives_and_limits='unchanged; no setters',
            torque_screen='external native gravity independently within existing Nm caps; unchanged native maxForce applies to implicit drives; no combined total bound',
            pd_diagnostic='native-gain PD expression is rad/s^2; never added to gravity or compared with Nm',
            total_actuator_torque_validated=False,
            historical_units_correction='v1 PD-as-Nm and PD-plus-G-as-Nm fields/gate invalid for acceleration drives; removed',
            learned_policy=False,
            real_hardware_validated=False,training_eligible=False)
    if velocity_settings is not None:
        result['velocity_feedforward']=dict(velocity_settings,
            schema='native_joint_velocity_target_v1',
            source='adjacent executed position commands / unchanged physics dt',
            native_api='ArticulationAction joint_positions + joint_velocities; native applied-action target readback',
            first_command='explicit stationary zero',stationary_hold='zero; moving force corrections retain derivative',
            position_commands_gains_effort_caps_gravity='unchanged',
            joint_velocity_limit_rad_s=.175,learned_policy=False,
            real_hardware_validated=False,training_eligible=False)
        result['velocity_feedforward_failure_receipts']=deepcopy(
            getattr(env,'velocity_feedforward_failure_receipts',[]))
        result['velocity_feedforward_reset_receipt']=deepcopy(
            getattr(env,'velocity_feedforward_reset_receipt',None))
    if env is not None:
        if hasattr(env,'retainer_failure_evidence'):
            result['retainer_failure_evidence']=deepcopy(env.retainer_failure_evidence)
        if hasattr(env,'retainer_first_contact_evidence'):
            result['retainer_first_contact_evidence']=deepcopy(env.retainer_first_contact_evidence)
        try:
            from .m2_given_contact_pilot_native_audit import installed,trace_path
            trace=(trace_path(controller) if installed(controller) else
                capture.output/'native_audit.jsonl.gz' if is_compact_capture(capture) else env.out/'fresh_expert_trace.jsonl')
            result['trace_file']=str(trace)
            result['trace_sha256']=_sha(trace)
        except Exception as error:errors.append(dict(stage='trace_hash',error=repr(error)))
    from .m2_given_contact_pilot_native_audit import installed
    if installed(controller):
        result['neutral_native_audit']=deepcopy(getattr(controller,'neutral_native_audit',controller._pilot_light_native_audit.receipt))
        result['pilot_light_native_audit']=True
    # Post-run fingerprints are individually protected too: a missing trace or
    # replaced source file must not discard the other diagnostic errors.
    for binding in metadata['plans']:
        try:
            if _sha(binding['file'])!=binding['sha256']:
                raise ValueError('plan source changed during experiment: '+binding['file'])
        except Exception as error:errors.append(dict(stage='plan_fingerprint',error=repr(error)))
    result['runtime_sources_sha256']=source_hashes
    for path,expected in source_hashes.items():
        try:
            if _sha(path)!=expected:raise ValueError('runtime source changed during experiment: '+path)
        except Exception as error:errors.append(dict(stage='source_fingerprint',error=repr(error)))
    try:require_geometry_artifacts(output/'geometry',current)
    except Exception as error:errors.append(dict(stage='geometry_fingerprint',error=repr(error)))
    result['errors']=errors
    if capture is not None:
        result['raw_multimodal_capture']=capture.result(len(rows),
            execution_completed=bool(outcome.get('success')),errors=errors)
        result['raw_capture_complete']=result['raw_multimodal_capture']['raw_capture_complete']
        result['compact_capture_complete']=result['raw_multimodal_capture'].get('compact_capture_complete',False)
    elif raw_settings is not None:
        result['raw_capture_complete']=False
    if errors:
        result.update(success=False,physical_validation=False,
            failure_reason='; '.join(error['stage']+': '+error['error'] for error in errors))
    result['runtime_cost']=runtime_cost(physics_steps=len(rows),physics_dt_s=DT,
        probe_wall_s=time.monotonic()-probe_wall_started,
        controller_wall_s=(None if controller_wall_started is None else
                           controller_wall_ended-controller_wall_started),
        setup_wall_s=(None if controller_wall_started is None else
                      controller_wall_started-probe_wall_started),
        capture_profile=(raw_settings.get('profile','full_raw') if raw_settings is not None else 'diagnostic_video'))
    if learned:
        # Must follow every old expert result.update, including failure paths.
        # Keep physical audit success independent from the L2 handoff outcome.
        policy_receipt=getattr(controller,'policy_receipt',dict(schema='m2_act_l2_deployment_attempt_v1',
            checkpoint_sha256=config['m2_act_deployment']['checkpoint_sha256'],controller_constructed=False,
            configured_deployment=deepcopy(deployment)))
        handoffs=getattr(controller,'l2_handoffs',[]) or _learned_failure_handoffs(
            config,result.get('failure_reason','L2 did not reach handoff'),points[0][0].name)
        learned_adapter.tag_learned_result(result,policy_receipt,handoffs)
        result.update(m2_estimate_provider=estimate_provider.snapshot(),
            l2_queries=deepcopy(getattr(controller,'l2_queries',[])),
            l2_actions=deepcopy(getattr(controller,'l2_actions',[])),
            raw_capture_complete=False,compact_capture_complete=False)
    if formal_learned:
        formal_adapter.tag_result(result,deployment,controller)
    if pilot_learned:
        pilot_adapter.tag_result(result,deployment,controller)
    if given_route is not None:
        if 'handoff_policy_id' in given_route['metadata']:
            result['handoff_policy_id']=given_route['metadata']['handoff_policy_id']
        result.update(task_id=given_route['schema'],target_source='given_contact_pose',
            action_semantics_id=given_route['metadata']['action_semantics_id'],
            observation_contract=deepcopy(given_route['metadata']['observation_contract']),
            l2_handoffs=deepcopy(getattr(controller,'l2_handoffs',[])),
            given_reference=(controller.estimate_provider.snapshot() if controller is not None else given_ref.snapshot()),
            l3_initialization='m2_given_contact_l3_align_v1',
            scope='Prescribed full-pose state-only control baseline; no visual or sensor robustness claim')
    if formal_test is not None:
        from .m2_given_contact_evaluation import tag_arm_result
        tag_arm_result(result,config,formal_test)
    if 'astra_supervisor' in config:
        from .astra_native import tag_result
        tag_result(result,controller)
    try:_write(output/'fresh_expert_result.json',result)
    except Exception as error:
        errors.append(dict(stage='result_save',error=repr(error)))
        result.update(success=False,physical_validation=False,
            failure_reason='; '.join(item['stage']+': '+item['error'] for item in errors))
        if capture is not None:
            result['raw_multimodal_capture']=capture.result(len(rows),execution_completed=False,errors=errors)
            result['raw_capture_complete']=False
            result['compact_capture_complete']=False
        # The owner still receives a failed receipt even if this artifact cannot
        # be written. All release operations have already been attempted.
        print('FRESH_EXPERT_RESULT_SAVE_FAILED',repr(error),flush=True)
    return result
