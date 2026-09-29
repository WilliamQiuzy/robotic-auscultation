import copy
import json
from pathlib import Path
import tempfile
import unittest

from sim.expert_quality_20260924.visual_judge import (
    decision, pair_consistency, sample_indices, sha256, validate_timing, verify_trace_rows,
    checked_path, verify_weight_keys, phase_anchor_indices,
)


def clip():
    return {"frame_count": 3, "duration_s": .2, "frames": [
        {"frame_index": i, "video_time_s": i/15, "simulation_interval_end_s": (i+1)/15,
         "measurement_row": (i+1)*16-1, "measurement_time_s": (i+1)/15,
         "pose_row": (i+1)*16-8, "pose_time_s": ((i+1)*16-7)/240}
        for i in range(3)]}


class VisualJudgeTests(unittest.TestCase):
    def test_original_time(self):
        validate_timing(clip(), 15, 1/240, 1/30)

    def test_retimed_rejected(self):
        data=clip()
        for f in data["frames"]:
            f["simulation_interval_end_s"] *= 2
        with self.assertRaises(ValueError): validate_timing(data,15,1/240,1/30)

    def test_stale_pose_rejected(self):
        data=clip();data["frames"][1]["pose_time_s"]-=.1
        with self.assertRaises(ValueError): validate_timing(data,15,1/240,1/30)

    def test_nonfinite_time_rejected(self):
        data=clip();data["frames"][1]["measurement_time_s"]=float("nan")
        with self.assertRaises(ValueError): validate_timing(data,15,1/240,1/30)

    def test_missing_frame_rejected(self):
        data=clip();data["frames"][1]["frame_index"]=2
        with self.assertRaises(ValueError): validate_timing(data,15,1/240,1/30)

    def test_model_score_never_overrides_hard_fail(self):
        self.assertEqual(decision("fail",True,True,"pass")["status"],"fail")

    def test_missing_calibration_is_unknown(self):
        self.assertEqual(decision("pass",True,False,"pass")["status"],"unknown")

    def test_no_model_is_unknown(self):
        self.assertEqual(decision("pass",False,True,"pass")["status"],"unknown")

    def test_missing_hard_audit_is_unknown(self):
        self.assertEqual(decision("unknown",True,True,"pass")["status"],"unknown")

    def test_order_consistency(self):
        self.assertTrue(pair_consistency(.9,.1)["order_consistent"])
        self.assertFalse(pair_consistency(.9,.9)["order_consistent"])
        self.assertAlmostEqual(pair_consistency(.9,.1)["symmetrized_p_a"],.9)

    def test_nonfinite_preference_rejected(self):
        with self.assertRaises(ValueError): pair_consistency(float("nan"),.5)

    def test_sample_endpoints_and_count(self):
        indices=sample_indices(903,8)
        self.assertEqual(len(set(indices)),8)
        self.assertEqual((indices[0],indices[-1]),(0,902))

    def test_short_hold_is_not_skipped(self):
        frames=[{"phase":"APPROACH"} for _ in range(1000)]
        for i in range(830,835): frames[i]["phase"]="CONTACT"
        for i in range(835,880): frames[i]["phase"]="HOLD"
        for i in range(880,925): frames[i]["phase"]="RETRACT"
        indices=phase_anchor_indices(frames,8)
        self.assertEqual(len(indices),8)
        self.assertEqual({frames[i]["phase"] for i in indices},{"APPROACH","CONTACT","HOLD","RETRACT"})

    def test_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"source";p.write_text("source")
            expected=sha256(p);p.write_text("modified")
            with self.assertRaises(ValueError): checked_path(Path(directory),"source",expected)

    def test_unused_legacy_head_only(self):
        self.assertEqual(verify_weight_keys({"progress_head.0.weight", "similarity_head.0.weight"},
                         {"progress_head.0.weight"}),["similarity_head.0.weight"])
        with self.assertRaises(ValueError): verify_weight_keys(set(),{"progress_head.0.weight"})
        with self.assertRaises(ValueError): verify_weight_keys({"unknown.weight"},set())

    def test_time_mapping_binds_actual_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"expert_trace.jsonl"
            path.write_text("".join(json.dumps({"time_s":(i+1)/240})+"\n" for i in range(48)))
            verify_trace_rows(path,[clip()])
            data=copy.deepcopy(clip());data["frames"][1]["measurement_time_s"]+=.001
            with self.assertRaises(ValueError): verify_trace_rows(path,[data])


if __name__=="__main__": unittest.main()
