import unittest
import numpy as np
from sim.ausc_policy.video import frame_schedule, video_identity


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

    def test_expert_and_policy_videos_cannot_exchange_ownership(self):
        owner = dict(completed=True,gpu_released=True,config=dict(experiment='recovery_run'))
        result = dict(execution_completed=True)
        outcome = dict(expert_demonstration=True,learned_policy=False,training_dataset_eligible=False)
        identity = video_identity(owner,'joint3-plus',result,outcome)
        self.assertTrue(identity['expert_demonstration'])
        self.assertFalse(identity['learned_policy'])
        with self.assertRaises(ValueError): video_identity(owner,'case',result,outcome)
        owner['config']['experiment'] = 'isaac_run'
        with self.assertRaises(ValueError): video_identity(owner,'joint3-plus',result,outcome)
        with self.assertRaises(ValueError): video_identity(owner,'case',result,outcome)

    def test_unfinished_owner_cannot_publish_video(self):
        with self.assertRaises(ValueError):
            video_identity(dict(completed=False,gpu_released=True),'case',{}, {})


if __name__=='__main__':unittest.main()
