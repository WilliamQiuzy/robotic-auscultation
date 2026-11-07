"""V6 checked route-B teacher with shared calibrated ADC/FK/RGB-D supervision.

This producer does not seal a training manifest. Independent contact/posture,
sensor pairing and action-label audits must qualify each completed rollout.
Teacher geometry is privileged; it never enters BodyTracker or encoded state.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import importlib

import numpy as np
from sim.auscultation_skill_v1.controller import ExpertController
from sim.auscultation_skill_v1.curve import FixedNormalCurve, force_derived_domain
from sim.auscultation_skill_v1.executor import NormalExecutor
from sim.auscultation_expert_layout_v1.controller import DT, CONTROL_STEPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from .observed_supervisor import ObservedContactSupervisorMixin, VERSION as SUPERVISOR_VERSION
from .supervision import contact_load_estimate
from .teacher_timing import measured_gentle_v1, VERSION as TIMING_VERSION

CLASS = 'sim.auscultation_pi05_v6.collect_demo.ObservedTeacher'


def collector_source_paths():
    """Bind all V6 modules actually consumed by this collector."""
    names=('collect_demo.py','collect_demo.sh','teacher_timing.py','observed_supervisor.py',
           'supervision.py','live_capture.py','render_warmup.py','hall_conditions.py',
           'dynamics_contract.py','runtime_dynamics.py')
    return {'pi05_v6/'+name:Path(__file__).with_name(name) for name in names}


def teacher_policy_description():
    return dict(enabled=True,recipe=TIMING_VERSION,supervisor=SUPERVISOR_VERSION,
        feedback='calibrated ADC contact-load estimate; measured-q FK and registered RGB-D task frame',
        inward_near_contact_limit_m_s=.0004,loaded_withdrawal_limit_m_s=.0002,
        near_contact_region_m=.003,loaded_withdrawal_region_m=.002,
        normal_executor='unchanged acceleration/jerk bounded NormalExecutor',hold_duration_s=3.,
        independent_safety='actual physics collision, posture, front-contact, force and stroke audits',
        supersedes_parent_policy_timing_description=True)


class ObservedTeacher(ObservedContactSupervisorMixin, ExpertController):
    """Analytic demonstration policy; shared measured supervision, checked curve.

    The initial approved route and contact curve remain privileged expert plan
    inputs. All subsequent feedback, phase and release decisions use measured
    observations. The parent still audits true physics at every executed tick.
    """
    def __init__(self,*args,speed_scale=.7,**kwargs):
        if speed_scale != .7:
            raise ValueError('V6 observed teacher requires speed_scale 0.7')
        if any(kwargs.get(name) is not None for name in ('action_provider','condition','sensor_condition')):
            raise ValueError('V6 observed teacher requires its measured analytic feedback')
        super().__init__(*args,speed_scale=speed_scale,**kwargs)

    def desired_normal_velocity(self,state):
        metric,hall,_=self._tracked_metrics(state)
        load=contact_load_estimate(hall)
        maximum=self.executor_config['normal_speed_limit_m_s']
        inward=self.executor_config['normal_inward_speed_limit_m_s']*self.speed_scale
        phase=self.manager.state.phase.value
        if phase in ('RETRACT','DONE'):
            desired=maximum
        elif load>.3 or hall.displacement_estimate_m>.00015:
            error=load-self.config.target_force_N
            desired=0. if abs(error)<=.02 else float(np.clip(.001*error,-inward,inward))
        else:
            desired=-inward
        center=self.contact_curves[-1]['nominal_contact_center_gap_m']
        return measured_gentle_v1(phase,desired,metric['gap_m'],center)

    def _observed_released(self,state):
        _,hall,_=self._tracked_metrics(state)
        return contact_load_estimate(hall)<.2 and hall.displacement_estimate_m<=.00015

    def observe_actor(self,state):
        _,hall,_=self._tracked_metrics(state)
        frame=self.sensor.rows[-1]
        return state,dict(observation_time_s=float(state['time_s']),
            hall_sample_time_s=hall.sample_time_s,geometry_source_frame=frame['index'],
            geometry_sample_time_s=frame['time_s'],
            observation_age_s=max(state['time_s']-frame['time_s'],state['time_s']-hall.sample_time_s))

    def _skill_observation(self,state,metadata):
        metric,hall,_=self._tracked_metrics(state)
        stamp=metadata['observation_time_s'];normal_velocity=0.
        if self._previous_actor_gap is not None:
            old_gap,old_stamp=self._previous_actor_gap
            if stamp>old_stamp:normal_velocity=(metric['gap_m']-old_gap)/(stamp-old_stamp)
        self._previous_actor_gap=metric['gap_m'],stamp
        frame=self.sensor.rows[-1]
        task=np.asarray(frame['body_reference_to_current_base'])@np.asarray(
            self.task_frames[self.target.name]['task_pose_in_body_reference'])
        return dict(gap_m=metric['gap_m'],hall_N=contact_load_estimate(hall),
            compression_m=hall.displacement_estimate_m,hall_voltage_V=hall.hall_voltage_v,
            spring_estimate_N=hall.spring_force_estimate_n,
            q=np.asarray(state['q']).tolist(),dq=np.asarray(state['dq']).tolist(),
            normal_base=(-task[:3,2]).tolist(),target_position_base_m=task[:3,3].tolist(),
            target_rotation_base=task[:3,:3].tolist(),normal_velocity_m_s=normal_velocity,
            previous_desired_normal_velocity_m_s=self._previous_desired,
            executor_gap_m=self.contact_executor.gap,executor_velocity_m_s=self.contact_executor.velocity,
            executor_acceleration_m_s2=self.contact_executor.acceleration,
            phase=self.manager.state.phase.value,speed_scale=self.speed_scale,
            geometry_source_frame=frame['index'],geometry_sample_time_s=frame['time_s'],
            observation_source='calibrated_ADC_measured_q_registered_RGBD_task')

    def _capture_observed_tick(self,state,subphase,active,recovery):
        if len(self.env.rows)%CONTROL_STEPS:return
        eligible=subphase not in {'settle','initial_home','return_bridge','return_home','home_settle','withdraw_settle'} and not recovery
        if active:
            phase=self.manager.state.phase.value
            if phase=='DONE':phase='RETRACT'
            if phase not in {'APPROACH','CONTACT','HOLD','RETRACT'}:
                raise EpisodeRejected('Invalid observed contact phase for collection')
        elif subphase in {'transfer','inter_point_transfer','align','settle','initial_home'}:
            phase='FREE'
        else:
            phase=self.manager.state.phase.value
            if phase not in {'FREE','APPROACH','CONTACT','HOLD','RETRACT'}:
                phase='FREE';eligible=False
        self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,
                            eligible=eligible,subphase=subphase)

    def _block(self,*args,**kwargs):
        # Keep the authoritative parent observation intact for the frozen
        # control/physics binder, but label it explicitly as audit-only truth.
        start=len(self.control_rows)
        try:
            return super()._block(*args,**kwargs)
        finally:
            for row in self.control_rows[start:]:
                row['observation_scope']='privileged simulator audit only; not actor feedback'
                row['skill_observation_scope']='calibrated ADC, measured q and registered RGB-D task'

    def run(self,points):
        result=super().run(points)
        result.update(executed_controller_class=CLASS,candidate_method=CLASS,
            controller_source='analytic_expert_shared_measured_supervisor',
            actual_contact_policy=teacher_policy_description(),
            normal_withdrawal_method='measured release and bounded normal executor; 0.2 mm/s within contact center + 2 mm, otherwise existing 4 mm/s ceiling; exact stationary prepose endpoint')
        return result

    def _contact_cycle(self):
        # Versioned copy of the frozen skill loop: executor, collision preflight,
        # deadlines, braking and endpoint checks are unchanged. Only observed
        # handoff/feedback/release and provenance replace simulator control input.
        self._initialize_contact_supervisor()
        self._normal_release_target = None
        self._previous_desired = 0.; self._previous_actor_gap = None
        self._contact_active = True
        loaded_gap = float((self.kin.tcp(np.asarray(self._point_plan['loaded_q']))[:3, 3]-self.target.position_base) @ self.normal_base)
        loaded_compression = float(self._point_plan['loaded_compression_m'])
        domain = force_derived_domain(loaded_gap, loaded_compression, self.normal,
                                      self.config.target_force_N, self.executor_config['nominal_compression_reserve_m'])
        center_gap = domain['nominal_contact_center_gap_m']
        def nominal_check(q, gap):
            compression = max(0., center_gap-gap)
            checked = self.checker.check(q, clearance=self.executor_config['contact_clearance_m'],
                                         allow_patient_contact=True, compression=compression)
            return dict(checked, nominal_compression_m=compression)
        report = dict(target=self.target.name, start_physics_row=len(self.env.rows), motion_complete=False,
                      preflight_passed=False, nominal_contact_center_gap_m=center_gap, nominal_contact_domain=domain)
        self.contact_curves.append(report)
        print('CONTACT_CURVE_START', self.target.name, 'physics_row', len(self.env.rows),
              'minimum_gap_m', domain['minimum_gap_m'], 'maximum_gap_m', self.pre_gap,
              'maximum_nominal_compression_m', domain['maximum_nominal_compression_m'], flush=True)
        try:
            curve = FixedNormalCurve.build(self.kin, self.env.base, self.posture_limits, self.q,
                self.target.position_base, self.normal_base, self.rotation_base, domain['minimum_gap_m'],
                spacing=self.executor_config['curve_spacing_m'],
                position_tolerance=self.executor_config['curve_position_tolerance_m'],
                angle_tolerance=self.executor_config['curve_angle_tolerance_rad'], collision_check=nominal_check)
        except (RuntimeError, ValueError) as error:
            report.update(failure_reason=str(error), preflight_failure=getattr(error, 'evidence', None),
                          end_physics_row_exclusive=len(self.env.rows))
            self._contact_active = False
            raise
        report.update(curve.report, preflight_passed=True)
        print('CONTACT_CURVE_READY', self.target.name, 'physics_row', len(self.env.rows),
              'knots', curve.report['knots'], 'checked_samples', curve.report['verification_samples'], flush=True)
        history = np.asarray([row['q_command'] for row in self.env.rows[-3:]], float)
        self.contact_executor = NormalExecutor(curve, self.executor_config, history, self.speed_scale)
        executor = self.contact_executor
        previous_phase = None; desired = 0.; actor_obs = metadata = None
        phase_ticks = 0; inward_braking = 0.; started_retract = False; final_hold_steps = 0
        try:
            for _ in range(int(np.ceil(80. / self.speed_scale / DT))):
                state = self.env.state(); phase = self.manager.state.phase
                if phase == Phase.ABORT:
                    raise EpisodeRejected('contact state machine aborted: '+str(self.manager.state.failure_reason))
                changed = phase != previous_phase
                if changed:
                    print('CONTACT_PHASE', self.target.name, phase.value, 'physics_row', len(self.env.rows),
                          'gap_m', executor.gap, 'measured_contact_load_N',
                          contact_load_estimate(self._tracked_metrics(state)[1]), flush=True)
                policy_update = changed or phase_ticks % CONTROL_STEPS == 0
                if policy_update:
                    conditioned, metadata = self.observe_actor(state)
                    if metadata.get('observation_age_s', 0.) > self.config.max_observation_age_s:
                        raise EpisodeRejected('actor observation is stale')
                    actor_obs = self._skill_observation(conditioned, metadata)
                    self.current_skill_observation = actor_obs
                    desired = self.desired_normal_velocity(conditioned)
                    self._previous_desired = desired
                    self._policy_group += 1
                    phase_ticks = 0
                previous_phase = phase
                if phase in (Phase.RETRACT, Phase.DONE):
                    started_retract = True
                released = self._observed_released(state)
                remaining = curve.maximum_gap-executor.gap
                if (started_retract and released and desired > 0 and remaining <= .002
                        and remaining > 0 and not executor.stop_pending):
                    executor.plan_stop(curve.maximum_gap)
                before_gap = executor.gap
                if executor.gap == curve.maximum_gap and started_retract and desired >= 0. and executor.velocity == executor.acceleration == 0.:
                    # This is the analytically reached endpoint, not a clipped step.
                    command = executor.hold_stationary()
                    final_hold_steps += 1
                else:
                    command = executor.step(desired)
                    final_hold_steps = 0
                if started_retract:
                    inward_braking += max(0., before_gap-executor.gap)
                    if inward_braking > self.executor_config['max_inward_braking_m'] + 1e-12:
                        raise EpisodeRejected('normal unload inward braking displacement limit')
                subphase = ('normal_release_confirm' if final_hold_steps else
                            'normal_unload' if started_retract and not released else
                            'normal_released_retreat' if started_retract else 'contact_cycle')
                row_index = len(self.control_rows)
                step_succeeded = False
                try:
                    self._block(np.asarray(command)[None], subphase, active=True, gap=executor.gap)
                    step_succeeded = True
                finally:
                    if len(self.control_rows) > row_index:
                        self.control_rows[-1].update(skill_observation=actor_obs, **metadata,
                            policy_update=policy_update, policy_group_id=self._policy_group,
                            desired_normal_velocity_m_s=float(desired), applied_normal_velocity_m_s=executor.velocity,
                            executor_gap_m=executor.gap, executor_velocity_m_s=executor.velocity,
                            executor_acceleration_m_s2=executor.acceleration,
                            executor_stop_mode=executor.mode,
                            endpoint_stop_active=executor.mode == 'endpoint_stop',
                            joint_command_derivative_peaks=executor.peaks,
                            hard_clamp_intervened=False, failure=not step_succeeded,
                            controller_source='analytic_expert_shared_measured_supervisor',
                            release_observation_source='calibrated_ADC_contact_load_and_travel')
                phase_ticks += 1
                if final_hold_steps >= int(np.ceil(.2/DT)) and self.manager.state.success:
                    break
            else:
                raise EpisodeRejected('normal contact execution budget exhausted')
            if not self.manager.state.success or self.best_audit_hold < self.config.hold_duration_s-2*DT:
                raise EpisodeRejected('continuous independently verified HOLD/release missing')
            self._normal_release_target = self.target.name
            report.update(motion_complete=True, inward_braking_distance_m=inward_braking)
        finally:
            report['end_physics_row_exclusive'] = len(self.env.rows)
            self._contact_active = False


def capture(app,errors,released,args,bundle,snapshot_sha,binding_sha):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot,digest
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from .live_capture import LiveCapture
    env=sensor=controller=None
    outcome=dict(success=False,training_accepted=False,points=[],failure_reason='construction incomplete')
    try:
        kin=OpenArmKinematics();base=np.array(bundle.plan['world_from_left_base'])
        with tempfile.TemporaryDirectory(prefix='wam_vla_geometry_') as temporary:
            folder=Path(temporary)
            for label,name in [('collision_metadata','collision_geometry.json'),('collision_arrays','collision_geometry.npz')]:
                (folder/name).write_bytes(bundle.inputs[label][1])
            checker=CertifiedCollisionChecker(kin,base,directory=folder)
        env=make_env(args.output,PhysicsProfile(args.position_iterations,1,args.armature))
        if not np.allclose(env.base,base,atol=1e-7,rtol=0):raise ValueError('live robot base differs from plan')
        from .runtime_dynamics import capture_runtime_contract
        dynamics=capture_runtime_contract(env)
        if args.sensors:
            sensor=LiveCapture(env,kin,args.source_project/'sim/camera_framing_20260923/recommended_camera.json',
                               args.annotation,args.output/'sensors',width=640)
            sensor.warm_up()
        controller=ObservedTeacher(env,kin,checker,sensor=sensor,
            generator_policy='revised_shortcut',speed_scale=args.speed_scale)
        outcome=controller.run([(bundle.target,bundle.row)])
        outcome['dynamics_recipe_sha256']=dynamics['recipe_sha256']
        verify_snapshot(args.output/'input_snapshot',verify_sources=True)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('LIVE_EXPERT_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=[args.point],input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding_sha,
            scope='V6 route-B expert with shared ADC/FK/RGB-D supervision; independent physics acceptance; audio absent',
            dataset_role='unqualified_live_demonstration' if sensor else 'physics_only_diagnostic_not_training_data',
            sensor_frames=0 if sensor is None else len(sensor.rows),
            body_source='existing_seated_patient_asset_v1',layout_family='approved_farther_lower_seat_camera7deg',
            motion_family='independent_target_plan_'+args.point,source_project=str(args.source_project))
        outcome['repair_contact_policy']=teacher_policy_description()
        outcome['collector_source_hashes']={name:digest(path.read_bytes())
            for name,path in collector_source_paths().items()}
        outcome['observed_supervisor_version']=SUPERVISOR_VERSION
        if env is not None:env.save('expert')
        if controller is not None:
            (args.output/'control_commands.json').write_text(json.dumps(controller.control_rows,allow_nan=False)+'\n')
        if sensor is not None:sensor.release(errors,released)
        (args.output/'outcome.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    return outcome


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--point',required=True,choices=[s+str(i) for i in range(1,6) for s in 'RL'])
    parser.add_argument('--speed-scale',type=float,default=.7)
    parser.add_argument('--sensors',action='store_true')
    parser.add_argument('--armature',type=float,default=.01)
    parser.add_argument('--position-iterations',type=int,default=32)
    args=parser.parse_args();args.source_project=args.source_project.resolve();args.output=args.output.resolve()
    if not args.sensors:raise ValueError('V6 measured supervisor requires --sensors')
    if args.speed_scale != .7:raise ValueError('V6 observed teacher requires --speed-scale 0.7')
    if args.output.exists():raise ValueError('new output required')
    reserve=3*2**30
    if shutil.disk_usage(args.output.parent).free<reserve:raise RuntimeError('insufficient collection reserve')
    sys.path.append(str(args.source_project))
    runtime=args.source_project/'sim/auscultation_expert_layout_v1'
    sys.path.insert(0,str(runtime/'python_deps'))
    from sim.auscultation_expert_layout_v1.controller import validate_plan,create_snapshot,verify_snapshot,digest
    from sim.auscultation_expert_layout_v1.run_sequence import source_paths
    from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
    plans=args.source_project/'sim/upper_pair_posture_20260924/results/candidate_b_plans'
    bundle=validate_plan(plans/('planned_paths_'+args.point+'.json'),plans,runtime)
    binding_sha=validate_geometry_binding(runtime)
    if bundle.plan.get('revalidation',{}).get('collision_regeneration_sha256')!=binding_sha:
        raise ValueError('plan geometry binding mismatch')
    args.output.mkdir();sources=source_paths(runtime)
    sources['collision_regeneration']=runtime/'results/collision_regeneration_verification.json'
    for name in ('collect_expert.py','live_capture.py','body_tracking.py','rigid_registration.py','tracking.py','observations.py','hall.py','multimodal.py'):
        sources['vla/'+name]=Path(__file__).parents[1]/'auscultation_vla_v2'/name
    for name in ('collect_repaired_expert.py','physics_profile.py','metrics.py','teardown.py','gentle_contact.py'):
        sources['repair/'+name]=Path(__file__).parents[1]/'auscultation_repair_v3'/name
    sources.update(collector_source_paths())
    for package in ('auscultation_skill_v1','curobo_expert_v1','expert_quality_20260924'):
        imported=importlib.import_module('sim.'+package)
        for directory in imported.__path__:
            for path in Path(directory).glob('*.py'):
                sources['runtime/'+package+'/'+path.name]=path
    calibration_module=importlib.import_module('auscultation.mechanical_sensor.force_feedback.force_sensor')
    sources['vla/force_sensor.py']=Path(calibration_module.__file__)
    sources['vla/hall_calibration.json']=Path(__file__).parents[1]/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
    sources['vla/annotation']=args.annotation
    snapshot=create_snapshot([bundle],args.output/'input_snapshot',sources);verify_snapshot(snapshot,verify_sources=True)
    snapshot_sha=digest((snapshot/'manifest.json').read_bytes())
    import fcl  # Fail before Kit if the explicit collision dependency path is absent.
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    status=1;report=dict(passed=False,errors=[])
    try:
        outcome,receipt=run_scoped_capture(app,capture,args,bundle,snapshot_sha,binding_sha)
        (args.output/'teardown.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if args.sensors:require_complete_teardown(receipt,annotators=2,products=1)
        else:
            from sim.auscultation_repair_v3.teardown import require_no_sensor_teardown
            require_no_sensor_teardown(receipt)
        from sim.auscultation_skill_v1.audits import audit_saved
        audit=audit_saved(args.output)
        from sim.auscultation_repair_v3.metrics import motion_metrics
        rows=[json.loads(line) for line in (args.output/'expert_trace.jsonl').open()]
        dynamics=motion_metrics([r['time_s'] for r in rows],[r['q'] for r in rows],[r['dq'] for r in rows])
        (args.output/'measured_dynamics.json').write_text(json.dumps(dynamics,indent=2)+'\n')
        report.update(passed=outcome['success'] and audit['accepted'],teardown=receipt,audit=audit,
                      sensor_frames=outcome['sensor_frames'],qualification='physics candidate; not sealed for training',
                      measured_dynamics=dynamics,
                      smoothness_target_met=dynamics['centered']['acceleration_rad_s2']['max']<=2 and dynamics['centered']['jerk_rad_s3']['max']<=20)
        report['mechanics_passed']=report['passed']
        report['passed']=report['passed'] and report['smoothness_target_met']
        status=0 if report['passed'] else 1
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    (args.output/'collection_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('LIVE_EXPERT_RESULT',json.dumps(report),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':main()
