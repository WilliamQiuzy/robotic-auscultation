import copy
import unittest
import numpy as np
from sim.auscultation_skill_v1.trace_binding import validate_control_trace, DT


def evidence(lengths):
    rows=[]; controls=[]; previous=np.zeros(7)
    for length in lengths:
        start=len(rows)
        observation=dict(q=previous.tolist(),dq=[0.]*7,hall_N=0.,compression_m=0.,
                         tcp_base_m=[0.]*3,tcp_rotation_base=np.eye(3).tolist())
        endpoint=previous.copy()
        for _ in range(length):
            endpoint=endpoint+1e-6
            rows.append(dict(time_s=(len(rows)+1)*DT,target='L4',phase='CONTACT',subphase='contact_cycle',
                q=endpoint.tolist(),dq=[0.]*7,hall_N=0.,compression_m=0.,tcp_base_m=[0.]*3,
                tcp_rotation_base=np.eye(3).tolist(),q_command=endpoint.tolist()))
        controls.append(dict(physics_row=start,physics_row_end_exclusive=len(rows),
            executed_physics_steps=length,command_block_complete=True,action_duration_s=length*DT,
            time_s=start*DT,target='L4',phase='CONTACT',subphase='contact_cycle',
            observation=observation,previous_q_command=previous.tolist(),q_command=endpoint.tolist(),
            action_joint_delta_rad=(endpoint-previous).tolist()))
        previous=endpoint
    return controls,rows


class TraceBindingTests(unittest.TestCase):
    def test_mixed_blocks_cover_true_steps(self):
        controls,rows=evidence([12,1,1,12])
        report=validate_control_trace(controls,rows)
        self.assertEqual(report['physics_rows_covered'],26)
        self.assertEqual(report['block_length_histogram'],{'12':2,'1':2})

    def test_partial_duration_cannot_claim_full_block(self):
        controls,rows=evidence([1,1]); controls[1]['action_duration_s']=.05
        with self.assertRaisesRegex(ValueError,'duration'): validate_control_trace(controls,rows)

    def test_conditioned_sensor_cannot_replace_truth(self):
        controls,rows=evidence([12,1]); controls[1]['observation']['hall_N']=.1
        with self.assertRaisesRegex(ValueError,'observation hall_N'): validate_control_trace(controls,rows)

    def test_intermediate_joint_jump_not_hidden_by_endpoint(self):
        controls,rows=evidence([12,1]); rows[5]['q_command'][0]=.1
        with self.assertRaisesRegex(ValueError,'speed'): validate_control_trace(controls,rows)

    def test_missing_sample_and_prior_discontinuity_rejected(self):
        controls,rows=evidence([12,1]); controls[1]['physics_row']=11
        with self.assertRaisesRegex(ValueError,'coverage'): validate_control_trace(controls,rows)


if __name__=='__main__': unittest.main()
