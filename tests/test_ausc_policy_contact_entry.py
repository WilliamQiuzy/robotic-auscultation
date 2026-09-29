"""The fixed contact curve must accept its retained start, including roll."""
import importlib
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


class ContactEntryTests(unittest.TestCase):
    def check(self, tangent=0., roll=0.):
        try:
            module = importlib.import_module('sim.ausc_policy.contact_entry')
        except ModuleNotFoundError:
            self.fail('Contact entry must expose the retained curve geometry contract')
        pose = np.eye(4)
        pose[:3, 3] = [tangent, 0., .025]
        pose[:3, :3] = Rotation.from_rotvec([0., 0., roll]).as_matrix()
        return module.check_retained_pose(pose, np.zeros(3), np.array([0., 0., 1.]),
            np.eye(3), position_tolerance_m=2e-5, angle_tolerance_rad=2e-4)

    def test_inside_coarse_candidate_can_still_fail_retained_curve(self):
        result = self.check(tangent=.001)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['position_error_m'], .001)
        self.assertIn('retained_position', result['failures'])

    def test_roll_error_is_not_hidden_by_matching_normal_axis(self):
        result = self.check(roll=.01)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['full_rotation_error_rad'], .01)
        self.assertIn('retained_full_rotation', result['failures'])

    def test_exact_curve_start_passes_without_changing_pose(self):
        result = self.check(tangent=1e-5, roll=1e-4)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['gap_m'], .025)
        self.assertEqual(result['failures'], [])
        self.assertFalse(result['collision_or_contact_certified'])

    def test_nonfinite_pose_and_invalid_normal_are_rejected(self):
        self.check()
        module = importlib.import_module('sim.ausc_policy.contact_entry')
        for pose, normal in [(np.full((4,4), np.nan), [0,0,1]),
                             (np.eye(4), [0,0,2]), (np.eye(4), [0,0,0])]:
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                module.check_retained_pose(pose, np.zeros(3), normal, np.eye(3),
                    position_tolerance_m=2e-5, angle_tolerance_rad=2e-4)


if __name__ == '__main__':
    unittest.main()
