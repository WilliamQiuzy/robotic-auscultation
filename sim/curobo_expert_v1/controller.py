"""Shared contact/withdrawal expert with two explicitly distinct free planners."""
import json
from pathlib import Path
import time

import numpy as np

from sim.auscultation_expert_layout_v1.controller import CONTROL_STEPS, EpisodeRejected, FRONT_GROUPS, FORBIDDEN_GROUPS, digest
from sim.expert_quality_20260924.candidate_controller import CandidateController
from sim.expert_quality_20260924.candidate_planner import checked_edge, joint_path, path_metrics
from .timing import smooth_commands,condition_path
from .planner_service import request_plan
from .withdrawal import NormalWithdrawalMixin
from .static_world import StaticWorldMonitor,load_static_world

CLASS='sim.curobo_expert_v1.controller.ExpertController'
FREE_BOUNDS={'joint_velocity_rad_s':.25,'joint_acceleration_rad_s2':1.,'joint_jerk_rad_s3':1.5}


class ExpertController(NormalWithdrawalMixin, CandidateController):
    def __init__(self,*args,generator_policy='revised_shortcut',bridge_dir=None,launcher=None,
                 static_world_bundle=None,static_geometry_dir=None,**kwargs):
        if generator_policy not in ('revised_shortcut','curobo'):
            raise ValueError('unknown independent expert policy')
        if generator_policy=='curobo' and (bridge_dir is None or launcher is None):
            raise ValueError('cuRobo requires a verified bridge and isolated launcher')
        if bridge_dir is not None and static_world_bundle is not None:
            raise ValueError('Bridge and explicit static-world bundle are mutually exclusive')
        self.generator_policy=generator_policy;self.bridge_dir=bridge_dir;self.launcher=launcher
        self.planner_calls=[];self.free_collision_samples=0
        super().__init__(*args,policy='checked_shortcut',**kwargs)
        self.static_world=StaticWorldMonitor(self.env,self.checker,
            load_static_world(bridge_dir) if bridge_dir is not None else static_world_bundle,
            geometry_dir=static_geometry_dir)
        if generator_policy=='curobo':
            from .robot_bridge import verify_bridge
            verify_bridge(bridge_dir)
            manifest=json.loads((Path(bridge_dir)/'bridge_manifest.json').read_text())
            if manifest.get('static_world_contract_sha256')!=self.static_world.bundle['contract_sha256']:
                raise ValueError('cuRobo bridge and runtime FCL static-world contract mismatch; export bridge_v2')

    def _tick(self,command,subphase,active=False,recovery=False):
        try:
            # Pre-check prevents even recovery from moving outside the verified
            # static-world envelope; post-check records the actual executed step.
            self.static_world.check()
            rows_before=len(self.env.rows)
            try:
                return super()._tick(command,subphase,active=active,recovery=recovery)
            finally:
                if len(self.env.rows)>rows_before:
                    self.static_world.after_step()
        except ValueError as error:
            raise EpisodeRejected(str(error)) from error

    def run(self,points):
        outcome=super().run(points)
        outcome.update(candidate_policy=self.generator_policy,candidate_method=CLASS,
            free_motion_command_bounds=dict(FREE_BOUNDS,scope='sampled free commands; actual dynamics separately evaluated'),
            candidate_geometry=('checked released-prepose direct shortcuts with explicit original fallback'
                if self.generator_policy=='revised_shortcut' else
                'cuRobo inter-point proposals; actual completed transfers are recorded in planner_calls; shared canonical initial and final routes; no planner fallback'),
            candidate_timing='same C2 spatial spline and quintic clock, verified at 240 Hz for both new candidates; jerk1.5 accel1 speed0.25',
            candidate_scope='simulation comparison; shared unchanged force gains/HOLD/targets plus shared gentle withdrawal',
            planner_calls=self.planner_calls,free_collision_samples=self.free_collision_samples,
            static_world_verification=self.static_world.summary())
        return outcome

    def _continuous_path(self,route,subphase):
        route=joint_path(route)
        if not np.allclose(route[0],self.q,atol=1e-9,rtol=0):
            raise EpisodeRejected('free route disconnected from retained command')
        commands,report=smooth_commands(route,speed_limit=min(self.joint_speed_limit,.25),
            acceleration_limit=1.,jerk_limit=1.5,maximum_seconds=300.,
            minimum_seconds=1./self.speed_scale,maximum_compute_seconds=15.,return_report=True)
        # The complete executed spline samples are checked before any motion.
        # This supplements <=.015rad edge sampling; it is not continuous-volume certification.
        started=time.monotonic();budget=max(180.,min(1200.,30.+.02*len(commands)));deadline=started+budget
        for index,command in enumerate(commands):
            if time.monotonic()>deadline:
                raise EpisodeRejected(f'command collision verification deadline: checked {index}/{len(commands)} within {budget}s')
            check=self.checker.check(command,clearance=.003,allow_patient_contact=False,compression=0.)
            distance=float(check.get('minimum_distance_m',float('nan')))
            if check.get('clear') is not True or not np.isfinite(distance) or distance<.003-1e-9:
                raise EpisodeRejected('free command failed mesh/posture check: '+str(check))
        self.free_collision_samples+=len(commands)
        report.update(subphase=subphase,target=self.target.name,start_physics_row=len(self.env.rows),
                      motion_complete=False,all_240hz_commands_mesh_checked=True,
                      collision_verification_samples=len(commands),collision_verification_seconds=time.monotonic()-started,
                      collision_verification_wall_budget_s=budget)
        self.free_motion_timing.append(report)
        for offset in range(0,len(commands),CONTROL_STEPS):
            self._block(commands[offset:offset+CONTROL_STEPS],subphase)
        report.update(end_physics_row_exclusive=len(self.env.rows),motion_complete=True)

    def _move_to_prepose(self,row):
        if self.sequence_index>0:
            compression=float(self.env.state().get('compression_m',float('nan')))
            if not np.isfinite(compression) or compression>.00015:
                raise EpisodeRejected('free transfer requires finite released compression <=0.00015m')
        if self.generator_policy=='revised_shortcut' or self.sequence_index==0:
            return super()._move_to_prepose(row)
        canonical,index=self._canonical(row)
        self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        state=self.env.state();release=self.released_prepose
        total=sum(self.env.last_contacts.get(name,0.) for name in FRONT_GROUPS+FORBIDDEN_GROUPS)
        if (release is None or release['physics_row_end_exclusive']!=len(self.env.rows)
                or abs(float(state['time_s'])-release['time_s'])>1e-9
                or not np.isfinite([state['hall_N'],total]).all() or state['hall_N']>=.2 or total>=.2):
            raise EpisodeRejected('cuRobo transfer requires actual verified release')
        observed=np.asarray(state['q'],float);goal=canonical[-1]
        relative='planner_requests/'+str(self.sequence_index).zfill(3)
        directory=self.env.out/relative
        call={'directory':relative,'from_target':release['target'],'to_target':self.target.name,
              'physics_row':len(self.env.rows),'simulation_time_s':float(state['time_s']),
              'measured_start_q':observed.tolist(),'retained_command_q':self.q.tolist(),
              'requested_goal_q':goal.tolist(),'proposal_accepted':False,'motion_complete':False}
        self.planner_calls.append(call)
        q,result,evidence=request_plan(observed,goal,directory,self.bridge_dir,self.launcher)
        call.update(backend_evidence=evidence,backend_metadata=result.get('metadata',{}),
                    original_proposal_metrics=path_metrics(q))
        # Keep generated approximate endpoints and explicitly check connectors.
        route=np.vstack([self.q,observed,q,goal])
        route=route[np.r_[True,np.max(np.abs(np.diff(route,axis=0)),axis=1)>1e-12]]
        # The dense native graph samples are a geometric proposal, not executed
        # commands. Check the conditioned proposal plus explicit endpoint
        # connectors, then independently check EVERY actual spline command.
        # Do not spend the entire budget checking duplicate micro-edges in the
        # 5,000-sample native resampling. No raw-proposal clearance is claimed.
        conditioned,conditioning=condition_path(route)
        proposal_edges=[(self.q,observed),(observed,q[0]),(q[-1],goal)]
        proposal_edges.extend(zip(conditioned,conditioned[1:]))
        checks=[];deadline=time.monotonic()+120.
        for a,b in proposal_edges:
            check=checked_edge(a,b,self.checker,deadline);checks.append(check)
            if check.get('clear') is not True:
                raise EpisodeRejected('cuRobo proposal failed independent mesh/posture gate: '+str(check))
        if len(self.env.rows)!=call['physics_row'] or self.env.state()['time_s']!=call['simulation_time_s']:
            raise EpisodeRejected('simulation advanced during external planning')
        record={'from_target':release['target'],'to_target':self.target.name,
                'start_physics_row':len(self.env.rows),'planning_simulation_time_s':call['simulation_time_s'],
                'measured_start_q':observed.tolist(),'retained_command_q':self.q.tolist(),
                'lower_staging_q':canonical[index].tolist(),'canonical_staging_index':index,
                'planning_accepted':True,'motion_complete':False,'edge_checks':checks,
                'edge_checks_scope':'conditioned geometric proposal and explicit measured/retained/native endpoint connectors; all executed 240Hz spline commands separately checked',
                'proposal_conditioning':conditioning,
                'free_clearance_m':.003,'joint_edge_sample_step_rad':.015,
                'candidate_policy':'curobo','planner':'curobo plan_single_js with independent full mesh/posture verification',
                'joint_waypoints_q':route.tolist(),'lower_staging_waypoint_index':None,
                'lower_staging_retained':False,'selected_metrics':path_metrics(route),
                'waypoints_sha256':digest(json.dumps(route.tolist(),separators=(',',':'),allow_nan=False).encode()),
                'waypoint_digest_format':'UTF-8 JSON joint_waypoints_q with separators comma and colon',
                'canonical_return_route_sha256':digest(json.dumps(canonical.tolist(),separators=(',',':'),allow_nan=False).encode()),
                'planning_wall_time_s':evidence['wall_time_s'],'planner_evidence_directory':relative}
        self.inter_point_routes.append(record);call['proposal_accepted']=True
        try:
            self._continuous_path(route,'inter_point_transfer');self._hold(1.,'align')
            record['motion_complete']=True;call['motion_complete']=True
        finally:record['end_physics_row_exclusive']=len(self.env.rows)
        return route
