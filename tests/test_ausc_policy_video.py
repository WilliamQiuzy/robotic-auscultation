import unittest
import numpy as np
from sim.ausc_policy.video import frame_schedule


class VideoClockTests(unittest.TestCase):
    def test_uses_past_camera_not_future_or_duplicate_time_playback(self):
        rows=[dict(index=i,time_s=t) for i,t in enumerate([1.,1.049,1.049,1.101,1.15])]
        clock,indices=frame_schedule(rows)
        np.testing.assert_allclose(clock,[1,1.05,1.1,1.15])
        np.testing.assert_array_equal(indices,[0,2,2,4])

    def test_backward_nonfinite_and_nonconsecutive_ledger_rejected(self):
        for rows in [[],[dict(index=0,time_s=1),dict(index=1,time_s=.9)],
                     [dict(index=0,time_s=float('nan'))],[dict(index=2,time_s=1)]]:
            with self.assertRaises(ValueError):frame_schedule(rows)


if __name__=='__main__':unittest.main()
