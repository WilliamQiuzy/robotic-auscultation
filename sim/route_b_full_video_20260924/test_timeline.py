import unittest
import numpy as np
from .timeline import full_schedule

class FullTimelineTests(unittest.TestCase):
    def test_full_source_and_fractional_tail(self):
        n=95484; times=np.arange(1,n+1)/240; poses=np.arange(7,n,8)
        rows,indices,ends=full_schedule(times,poses)
        self.assertEqual(len(rows),5968)
        self.assertEqual(rows[-1],n-1)
        self.assertEqual(poses[indices[-1]],95479)
        self.assertAlmostEqual(ends[-1],397.85)
        self.assertAlmostEqual(len(rows)/15-ends[-1],1/60)
        self.assertTrue(np.all(np.diff(rows)[:-1]==16))
        self.assertTrue(np.all(rows-poses[indices]<=7))
    def test_exact_multiple_does_not_add_frame(self):
        rows,_,ends=full_schedule(np.arange(1,241)/240,np.arange(7,240,8))
        self.assertEqual(len(rows),15);self.assertEqual(rows[-1],239);self.assertEqual(ends[-1],1.)
    def test_missing_trace_step_rejected(self):
        t=np.arange(1,241)/240;t[50:]+=1/240
        with self.assertRaisesRegex(ValueError,'time gap'):full_schedule(t,np.arange(7,240,8))
    def test_missing_pose_rejected(self):
        with self.assertRaisesRegex(ValueError,'time gap'):full_schedule(np.arange(1,241)/240,np.delete(np.arange(7,240,8),9))
    def test_missing_last_pose_rejected(self):
        with self.assertRaisesRegex(ValueError,'time gap'):full_schedule(np.arange(1,241)/240,np.arange(7,230,8))
    def test_missing_start_rejected(self):
        with self.assertRaisesRegex(ValueError,'bounds'):full_schedule(np.arange(1,241)/240,np.arange(15,240,8))

if __name__=='__main__':unittest.main()
