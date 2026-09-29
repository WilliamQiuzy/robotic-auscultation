import unittest
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


if __name__ == '__main__':
    unittest.main()
