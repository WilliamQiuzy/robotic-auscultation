"""Actual expert takeover after replaying a bound failed student's command prefix.

New physics and sensors run throughout; no pose reset or prerecorded observation
is supplied to the expert. This is not a new student-inference evaluation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from .recovery_teacher import RecoveryTeacher
from .recovery_oracle import StationaryRecoveryOracle
from .moving_brake import plan_moving_brake,check_prefix_chunk
from .moving_origin import selected_origin
from sim.auscultation_expert_layout_v1.controller import CONTROL_STEPS,EpisodeRejected,FRONT_GROUPS,FORBIDDEN_GROUPS
from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
from sim.auscultation_pi05_v6.supervision import contact_load_estimate


def load_source(config):
    source=Path(config['source_run'])
    index,phase,_=selected_origin(config)
    required={'result.json','output/case/outcome.json','output/case/expert_trace.jsonl','output/case/sensors/frames.jsonl'}
    if set(config['source_files'])!=required:
        raise ValueError('Only the predeclared bound ACT approach origin is supported')
    for name,digest in config['source_files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original failed ACT acquisition changed: '+name)
    owner=json.loads((source/'result.json').read_text());outcome=json.loads((source/'output/case/outcome.json').read_text())
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code']==0 and not outcome['success']
            and not outcome['expert_demonstration'] and outcome['expert_l2_fallbacks']==0):
        raise ValueError('Completed original failed student without expert fallback required')
    if phase=='FREE' and (owner['config'].get('variant')!='causal_history' or not owner['config'].get('causal_history_input')):
        raise ValueError('History FREE protocol requires the actual history student')
    physics=[json.loads(line) for line in (source/'output/case/expert_trace.jsonl').open()]
    action=outcome['l2_actions'][index];end=action['physics_row_start']
    if (not action['completed'] or action['phase']!=phase or action['target']!='R4'
            or not 13<=end<len(physics) or end%12 or physics[end-1]['time_s']!=action['time_s']
            or not np.allclose([r['time_s'] for r in physics[:end]],np.arange(1,end+1)/240,atol=1e-8,rtol=0)):
        raise ValueError('Source prefix/action clock is not consecutive')
    return outcome,physics




class MovingRecoveryTeacher(RecoveryTeacher):
    def __init__(self,*args,recovery_config,**kwargs):
        self.recovery_config=recovery_config
        self.source_index,self.source_phase,_=selected_origin(recovery_config)
        self.source_outcome,self.source_physics=load_source(recovery_config)
        self.moving_receipt=dict(source_prefix_replayed=True,student_inference_executed=False,
            source_action_index=self.source_index,source_phase=self.source_phase,source_run=recovery_config['source_run'],
            source_files=recovery_config['source_files'],prefix_completed=False,brake_completed=False,
            measured_prefix_tolerance_rad=1e-4,maximum_prefix_q_difference_rad=0.,
            training_dataset_eligible=False,physical_joint_resets=0)
        super().__init__(*args,**kwargs)

    def _measured_query(self):
        state=self.env.state();_,hall,_=self._tracked_metrics(state)
        return dict(q=np.asarray(state['q']).tolist(),dq=np.asarray(state['dq']).tolist(),
            command_history=[r['q_command'] for r in self.env.rows[-3:]],
            hall_load_N=float(contact_load_estimate(hall)),
            physical_contact_N=float(sum(self.env.last_contacts.get(k,0.) for k in FRONT_GROUPS+FORBIDDEN_GROUPS)),
            compression_m=float(state['compression_m']),time_s=float(state['time_s']),physics_row=len(self.env.rows)-1,
            recent_measurements=[dict(q=r['q'],q_command=r['q_command'],time_s=r['time_s'],physics_row=i)
                for i,r in enumerate(self.env.rows[-13:],start=len(self.env.rows)-13)])

    def _move_to_prepose(self,row):
        self._point_plan=row;canonical,index=self._canonical(row)
        self.canonical_transfer,self.canonical_stage_index=canonical.copy(),index
        if self.sequence_index!=0 or self.env.rows or self.env.state()['time_s']!=0.:
            raise EpisodeRejected('Moving recovery prefix requires the unchanged fresh scene origin')
        action=self.source_outcome['l2_actions'][self.source_index];end=action['physics_row_start']
        self.oracle_role='STUDENT_COMMAND_REPLAY'
        for start in range(0,end,CONTROL_STEPS):
            expected=self.source_physics[start:start+CONTROL_STEPS]
            # Match actual old command timing. Different source subphases are
            # grouped only where a full original block had that same subphase.
            subphases={r['subphase'] for r in expected}
            if len(subphases)!=1: raise EpisodeRejected('Replay block crosses original subphase')
            self._block(np.asarray([r['q_command'] for r in expected]),expected[0]['subphase'])
            difference=check_prefix_chunk(expected,self.env.rows[-CONTROL_STEPS:],self.q)
            self.moving_receipt['maximum_prefix_q_difference_rad']=max(
                self.moving_receipt['maximum_prefix_q_difference_rad'],difference)
        self.moving_receipt.update(prefix_completed=True,prefix_end_physics_row_exclusive=len(self.env.rows),
            prefix_end_time_s=self.env.state()['time_s'])
        query=self._measured_query();query.update(target='R4',phase=self.source_phase,
            command_state=action['posture_hint_preview']['source_command_state'])
        brake=plan_moving_brake(query,lambda q:self.checker.check(q,clearance=.003,
            allow_patient_contact=False,compression=0.),lower=self.posture_limits.joint_lower_rad,
            upper=self.posture_limits.joint_upper_rad)
        data=(json.dumps(brake,indent=2,allow_nan=False)+'\n').encode();(self.env.out/'brake_query.json').write_bytes(data)
        self.moving_receipt.update(brake_query_sha256=hashlib.sha256(data).hexdigest(),brake_kind=brake['kind'],
            brake_reason=brake.get('reason'),brake_query_physics_row=query['physics_row'])
        if brake['kind']!='braking_proposal': raise EpisodeRejected('Moving brake refused: '+str(brake['reason']))
        if not np.array_equal(self.env.state()['q'],query['q']) or self.env.state()['time_s']!=query['time_s']:
            raise EpisodeRejected('Physics changed during braking query')
        self.oracle_role='EXPERT_BRAKE'
        commands=np.asarray(brake['commands_240hz'])
        for start in range(0,len(commands),CONTROL_STEPS): self._block(commands[start:start+CONTROL_STEPS],'lower_approach')
        self._hold(1.,'settle')
        self.moving_receipt.update(brake_completed=True,brake_end_physics_row_exclusive=len(self.env.rows))
        # Only new measured history enters the stationary query. In particular,
        # no virtual Ruckig endpoint is used as a measured stopping position.
        query=self._measured_query()
        context=dict(target='R4',phase=self.source_phase,revision=int(self.sensor.rows[-1]['target_estimate']['revision']),
            q_goal=canonical[-1].tolist())
        oracle=StationaryRecoveryOracle(self.checker,lambda start,goal,rng,deadline:
            checked_transfer(self.checker,self.posture_limits,start,goal,self.kin,self.env.base,rng,deadline),
            lower=self.posture_limits.joint_lower_rad,upper=self.posture_limits.joint_upper_rad,
            timing_strategy=self.recovery_config.get('oracle_timing','global_cubic'))
        self.oracle_receipt.update(query_attempted=True,query_physics_row=query['physics_row'],query_time_s=query['time_s'],
            preparation_end_physics_row_exclusive=len(self.env.rows))
        proposal=oracle.query(query,context)
        data=(json.dumps(proposal,indent=2,allow_nan=False)+'\n').encode();(self.env.out/'oracle_query.json').write_bytes(data)
        self.oracle_receipt.update(query_sha256=hashlib.sha256(data).hexdigest(),kind=proposal['kind'],
            query_wall_time_s=proposal['wall_time_s'],abstention_reason=proposal.get('reason'))
        if proposal['kind']!='trajectory': raise EpisodeRejected('Post-brake oracle refused: '+str(proposal['reason']))
        if not np.array_equal(self.env.state()['q'],query['q']) or self.env.state()['time_s']!=query['time_s']:
            raise EpisodeRejected('Physics changed during stationary query')
        self.oracle_role='EXPERT_RECOVERY';commands=np.asarray(proposal['commands_240hz'])
        for start in range(0,len(commands),CONTROL_STEPS): self._block(commands[start:start+CONTROL_STEPS],'lower_approach')
        self._hold(1.,'align')
        self.oracle_receipt.update(trajectory_executed=True,recovery_end_physics_row_exclusive=len(self.env.rows))
        return canonical

    def run(self,points):
        result=super().run(points)
        result.update(executed_controller_class='sim.ausc_policy.moving_recovery_teacher.MovingRecoveryTeacher',
            controller_source='replayed student commands + native brake + stationary oracle + measured analytic L3',
            moving_recovery=self.moving_receipt,
            scope='One actual moving recovery experiment from a replayed failed ACT prefix; no new student inference or generalization')
        return result
