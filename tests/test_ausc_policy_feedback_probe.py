import unittest
import sim.ausc_policy.feedback_probe as module
import numpy as np
from scipy.spatial.transform import Rotation
from sim.ausc_policy.feedback_probe import relative_pose, shift_target, geometry


class TargetGeometryTests(unittest.TestCase):
    def state(self):
        r = Rotation.from_euler('xyz', [.3, -.2, .7]).as_matrix()
        return np.r_[np.arange(7)/10, [.03, -.02, -.025], r[:, :2].T.reshape(-1), .9, 1/3, 1].astype(np.float32)

    def test_shift_matches_rigid_target_translation_and_keeps_robot_inputs(self):
        state = self.state(); before = relative_pose(state)
        for axis in (0, 1):
            for delta in (-.005, -.001, .001, .005):
                changed = shift_target(state, axis, delta)
                transform = np.eye(4); transform[axis, 3] = delta
                np.testing.assert_allclose(relative_pose(changed), np.linalg.inv(transform) @ before, atol=3e-9)
                keep = np.ones(19, bool); keep[7+axis] = False
                np.testing.assert_array_equal(changed[keep], state[keep])
        np.testing.assert_array_equal(state, self.state())

    def test_geometry_distinguishes_axis_and_roll(self):
        pose = np.eye(4); pose[2, 3] = -.025
        pose[:3, :3] = Rotation.from_euler('z', 10, degrees=True).as_matrix()
        g = geometry(pose)
        self.assertEqual(g['gap_mm'], 25.)
        self.assertEqual(g['tangent_mm'], 0.)
        self.assertAlmostEqual(g['axis_deg'], 0.)
        self.assertAlmostEqual(g['so3_rad'], np.deg2rad(10))

    def test_invalid_interventions_fail(self):
        for axis, delta in ((2,.001), (0,.1), (0,float('nan'))):
            with self.assertRaises(ValueError):
                shift_target(self.state(), axis, delta)

    def test_robot_shift_recomputes_tcp_and_keeps_target_fixed(self):
        self.assertTrue(hasattr(module, 'shift_robot'), 'Consistent robot-state intervention is missing')
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin = OpenArmKinematics()
        state = self.state(); before = state.copy()
        task = kin.tcp(state[:7]) @ np.linalg.inv(relative_pose(state))
        for joint in range(7):
            changed = module.shift_robot(state, joint, .005, kin)
            expected = np.linalg.inv(task) @ kin.tcp(changed[:7])
            np.testing.assert_allclose(relative_pose(changed), expected, atol=2e-7)
            np.testing.assert_array_equal(changed[16:], state[16:])
            other = [i for i in range(7) if i != joint]
            np.testing.assert_array_equal(changed[other], state[other])
            self.assertAlmostEqual(float(changed[joint]-state[joint]), .005, places=7)
        np.testing.assert_array_equal(state, before)

    def test_robot_intervention_does_not_clip_joint_bound_or_invalid_axis(self):
        self.assertTrue(hasattr(module, 'shift_robot'), 'Consistent robot-state intervention is missing')
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin = OpenArmKinematics(); state = self.state()
        state[0] = kin.upper_limits[0]
        for joint, delta in ((0,.02),(7,.005),(0,.1)):
            with self.assertRaises(ValueError):
                module.shift_robot(state, joint, delta, kin)

    def test_response_measurement_distinguishes_copy_from_fixed_reference(self):
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin = OpenArmKinematics(); state = self.state()
        class Copy:
            def predict_state(self, x):
                return np.repeat(x[None,:7], 10, axis=0)
        original = Copy().predict_state(state)
        rows, skipped = module.robot_responses(state, original, 7, kin, Copy(), [-.02,-.005,.005,.02])
        self.assertEqual(len(rows)+len(skipped), 28)
        for row in rows:
            np.testing.assert_allclose(row['block_relative_h5_change_rad'], 0, atol=1e-9)
            self.assertAlmostEqual(row['changed_joint_response_gain'], 1, places=5)
            if row['tcp_perturbation_ratio'] is not None:
                self.assertAlmostEqual(row['tcp_perturbation_ratio'], 1, places=7)
        class Fixed:
            def predict_state(self, x):
                return original.copy()
        rows, _ = module.robot_responses(state, original, 7, kin, Fixed(), [-.02,-.005,.005,.02])
        grouped = module.robot_groups(rows)
        for group in grouped.values():
            self.assertEqual(group['median_joint_response_gain'], 0)
            self.assertEqual(group['median_tcp_perturbation_ratio'], 0)


if __name__ == '__main__':
    unittest.main()
