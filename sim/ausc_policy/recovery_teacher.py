"""Actual stationary R4 recovery pilot with preparation kept outside labels."""
import hashlib
import json
import numpy as np

from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher, teacher_policy_description
from sim.auscultation_pi05_v6.supervision import contact_load_estimate
from sim.auscultation_expert_layout_v1.controller import CONTROL_STEPS, FRONT_GROUPS, FORBIDDEN_GROUPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
from .recovery_oracle import StationaryRecoveryOracle
from .contact_entry import check_retained_pose

CLASS = 'sim.ausc_policy.recovery_teacher.RecoveryTeacher'


class RecoveryTeacher(ObservedTeacher):
    def __init__(self, *args, perturbation_rad, **kwargs):
        self.perturbation = np.asarray(perturbation_rad,float)
        if (self.perturbation.shape != (7,) or not np.isfinite(self.perturbation).all()
                or np.max(abs(self.perturbation)) > .02):
            raise ValueError('Invalid bounded recovery perturbation')
        self.oracle_role = 'PREPARATION'
        self.oracle_receipt = dict(query_attempted=False,trajectory_executed=False,l3_started=False,
            training_dataset_eligible=False,perturbation_rad=self.perturbation.tolist())
        super().__init__(*args, **kwargs)

    def _capture_observed_tick(self, state, subphase, active, recovery):
        self.env.rows[-1]['oracle_role'] = self.oracle_role
        if len(self.env.rows) % CONTROL_STEPS:
            return
        phase = self.manager.state.phase.value if active else 'FREE'
        if phase == 'DONE': phase = 'RETRACT'
        # Raw RGB-D, q/dq and calibrated Hall are acquired normally. Independent
        # qualification and pair export have NOT happened, including after success.
        self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,
            eligible=False,subphase=subphase)

    def _move_to_prepose(self, row):
        if self.sequence_index != 0:
            raise EpisodeRejected('Bounded pilot permits one independently reset R4 only')
        canonical = super()._move_to_prepose(row)
        requested = self.q.copy()+self.perturbation
        self.oracle_receipt.update(preparation_perturbation_start=len(self.env.rows),
            requested_stationary_q=requested.tolist())
        self._joint_segment(requested,'initial_home')
        self._hold(1.,'settle')
        state = self.env.state()
        _, hall, _ = self._tracked_metrics(state)
        if not self._observed_released(state):
            raise EpisodeRejected('Perturbed start is not actually released')
        query = dict(q=np.asarray(state['q']).tolist(),dq=np.asarray(state['dq']).tolist(),
            command_history=[r['q_command'] for r in self.env.rows[-3:]],
            hall_load_N=float(contact_load_estimate(hall)),
            physical_contact_N=float(sum(self.env.last_contacts.get(k,0.) for k in FRONT_GROUPS+FORBIDDEN_GROUPS)),
            compression_m=float(state['compression_m']),time_s=float(state['time_s']),
            physics_row=len(self.env.rows)-1)
        context = dict(target=self.target.name,phase='APPROACH',
            revision=int(self.sensor.rows[-1]['target_estimate']['revision']),q_goal=canonical[-1].tolist())
        oracle = StationaryRecoveryOracle(self.checker,
            lambda start, goal, rng, deadline: checked_transfer(self.checker,self.posture_limits,
                start,goal,self.kin,self.env.base,rng,deadline),
            lower=self.posture_limits.joint_lower_rad,upper=self.posture_limits.joint_upper_rad)
        self.oracle_receipt.update(query_attempted=True,query_physics_row=query['physics_row'],
            query_time_s=query['time_s'],preparation_end_physics_row_exclusive=len(self.env.rows))
        proposal = oracle.query(query,context)
        data = (json.dumps(proposal,indent=2,allow_nan=False)+'\n').encode()
        (self.env.out/'oracle_query.json').write_bytes(data)
        self.oracle_receipt.update(query_sha256=hashlib.sha256(data).hexdigest(),kind=proposal['kind'],
            query_wall_time_s=proposal['wall_time_s'],abstention_reason=proposal.get('reason'))
        if proposal['kind'] != 'trajectory':
            raise EpisodeRejected('Recovery oracle abstained: '+str(proposal['reason']))
        if (len(self.env.rows)-1 != query['physics_row'] or self.env.state()['time_s'] != query['time_s']
                or not np.array_equal(self.env.state()['q'],query['q'])):
            raise EpisodeRejected('Physics advanced while the oracle was planning')
        self.oracle_role = 'EXPERT_RECOVERY'
        commands = np.asarray(proposal['commands_240hz'])
        for start in range(0,len(commands),CONTROL_STEPS):
            self._block(commands[start:start+CONTROL_STEPS],'lower_approach')
        self._hold(1.,'align')
        self.oracle_receipt.update(trajectory_executed=True,recovery_end_physics_row_exclusive=len(self.env.rows))
        return canonical

    def _contact_cycle(self):
        gate = check_retained_pose(self.kin.tcp(self.q),self.target.position_base,self.normal_base,
            self.rotation_base,position_tolerance_m=self.executor_config['curve_position_tolerance_m'],
            angle_tolerance_rad=self.executor_config['curve_angle_tolerance_rad'])
        self.oracle_receipt['curve_entry_geometry'] = gate
        if not gate['passed']:
            raise EpisodeRejected('Expert retained curve entry failed: '+str(gate['failures']))
        self.oracle_role = 'MEASURED_ANALYTIC_L3'
        self.oracle_receipt['l3_started'] = True
        super()._contact_cycle()
        self.oracle_receipt['l3_completed'] = True
        self.oracle_role = 'FINAL_RETURN'

    def run(self, points):
        if len(points) != 1 or points[0][0].name != 'R4':
            raise ValueError('R4 stationary recovery diagnostic only')
        result = super().run(points)
        for rows in (self.env.rows,self.control_rows,result['points']):
            for row in rows:
                row['training_accepted'] = False
        result.update(executed_controller_class=CLASS,controller_source='stationary recovery oracle + measured analytic L3',
            expert_demonstration=True,learned_policy=False,training_dataset_eligible=False,
            training_accepted=False,m0_passed=False,oracle_receipt=self.oracle_receipt,
            actual_contact_policy=teacher_policy_description(),
            scope='Single-body R4 stationary perturbation pilot; preparation excluded; no learned success or dataset qualification')
        return result
