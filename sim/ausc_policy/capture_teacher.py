"""Physically prepare one declared entry node, then try unchanged measured L3.

No oracle correction to canonical pose after the perturbation. This is a
component test, not a learned-policy run or a certified dynamic capture region.
"""
import numpy as np
from .capture_probe import load_geometry_case
from .recovery_teacher import RecoveryTeacher
from .recovery_oracle import measured_stationarity
from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher
from sim.auscultation_expert_layout_v1.controller import EpisodeRejected


class CaptureTeacher(RecoveryTeacher):
    def __init__(self,*args,capture_config,**kwargs):
        self.geometry_case=load_geometry_case(capture_config)
        self.capture_receipt=dict(case_id=capture_config['case_id'],geometry_probe=capture_config['geometry_probe'],
            local_alignment_used=False,physical_joint_resets=0,preparation_completed=False,
            training_dataset_eligible=False,learned_policy=False)
        super().__init__(*args,**kwargs)

    def _move_to_prepose(self,row):
        if self.sequence_index!=0: raise EpisodeRejected('One fresh R4 capture component only')
        # Deliberately invoke approved preparation, not RecoveryTeacher's
        # perturb-and-correct oracle. Normal sensor/safety hooks remain active.
        canonical=ObservedTeacher._move_to_prepose(self,row)
        requested=np.asarray(self.geometry_case['q'])
        self.capture_receipt.update(perturbation_start_physics_row=len(self.env.rows),
            nominal_command=self.q.tolist(),requested_command=requested.tolist(),
            requested_pose=self.geometry_case['requested_pose'])
        self._joint_segment(requested,'lower_approach');self._hold(1.,'settle')
        state=self.env.state()
        if not np.array_equal(self.q,requested) or not self._observed_released(state):
            raise EpisodeRejected('Prepared command differs or actual patient contact is not released')
        query=dict(q=np.asarray(state['q']).tolist(),dq=np.asarray(state['dq']).tolist(),
            command_history=[r['q_command'] for r in self.env.rows[-3:]],time_s=state['time_s'],
            physics_row=len(self.env.rows)-1,
            recent_measurements=[dict(q=r['q'],q_command=r['q_command'],time_s=r['time_s'],physics_row=i)
                for i,r in enumerate(self.env.rows[-13:],start=len(self.env.rows)-13)])
        stationary=measured_stationarity(query)
        self.capture_receipt.update(preparation_completed=True,entry=query,stationarity=stationary,
            actual_command_pose=self.kin.tcp(self.q).tolist(),actual_measured_pose=self.kin.tcp(state['q']).tolist())
        if not stationary['passed']: raise EpisodeRejected('Actual capture preparation did not become stationary')
        # No alignment back to canonical: RecoveryTeacher's original curve
        # entry gate sees this exact retained perturbed command next.
        return canonical

    def run(self,points):
        result=super().run(points)
        result.update(executed_controller_class='sim.ausc_policy.capture_teacher.CaptureTeacher',
            controller_source='approved preparation + declared perturbed entry + measured analytic L3',
            capture_component=self.capture_receipt,
            scope='One direct-entry physical component test; no oracle correction, student inference or capture-region certification')
        return result
