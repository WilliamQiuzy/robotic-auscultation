import unittest
import numpy as np
from sim.expert_quality_20260924.render_review_v2 import schedule

class ReviewScheduleTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(1,4801)/240.;self.p=np.arange(7,4800,8)
    def test_full_1x_interval(self):
        r,k,t=schedule(self.t,self.p,240,2640,15)
        self.assertEqual(len(r),150);np.testing.assert_allclose(np.diff(t),1/15)
        self.assertAlmostEqual(t[0],1+1/15);self.assertAlmostEqual(t[-1],11)
        self.assertTrue((self.p[k]<=r).all());self.assertTrue((self.t[r]-self.t[self.p[k]]<=1/30+1e-7).all())
    def test_gap_in_trace_rejected(self):
        t=self.t.copy();t[250:]+=.004
        with self.assertRaises(ValueError):schedule(t,self.p,0,2400)
    def test_missing_pose_rejected(self):
        with self.assertRaises(ValueError):schedule(self.t,np.delete(self.p,50),0,2400)
    def test_unsorted_pose_rejected(self):
        p=self.p.copy();p[5]=p[4]
        with self.assertRaises(ValueError):schedule(self.t,p,0,2400)
    def test_no_pose_yet_rejected(self):
        with self.assertRaises(ValueError):schedule(self.t,self.p+24,0,2400)
    def test_invalid_fps_and_bounds(self):
        for start,end,fps in [(0,2400,0),(-1,2400,15),(100,10,15),(0,9999,15)]:
            with self.assertRaises(ValueError):schedule(self.t,self.p,start,end,fps)

if __name__=='__main__':unittest.main()
