"""CPU metadata and synthetic-video tests; no simulation rendering."""
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.route_b_full_video_20260924.timeline import full_schedule


def schedule_fixture(targets=("R1", "L1", "R2"), lead=32):
    rows, points = [], []
    for name in targets:
        start = len(rows)
        states = [("APPROACH", "approach")] * lead + [("HOLD", "contact_cycle")] * 720
        states += [("RETRACT", "withdrawal")] * 32
        if name == targets[-1]:
            states += [("DONE", "home_settle")] * 120
        for phase, subphase in states:
            rows.append(dict(time_s=(len(rows)+1)/240, target=name, phase=phase, subphase=subphase))
        points.append(dict(target=name, start_physics_row=start, end_physics_row=len(rows), success=True))
    indices = np.arange(7, len(rows), 8)
    poses = np.zeros((len(indices), 1, 7)); poses[:, :, 3] = 1.
    rr, kk, tt = full_schedule([r["time_s"] for r in rows], indices, 15)
    frames = []
    for index, (r, k, t) in enumerate(zip(rr, kk, tt)):
        frames.append(dict(frame_index=index, video_time_s=index/15, simulation_interval_end_s=float(t),
                           measurement_row=int(r), measurement_time_s=rows[r]["time_s"], pose_index=int(k),
                           pose_row=int(indices[k]), pose_time_s=rows[indices[k]]["time_s"],
                           **{key: rows[r][key] for key in ("target", "phase", "subphase")}))
    data = dict(fps=15, playback_speed=1., resolution=[1920, 720], frame_count=len(rr), frames=frames,
                source_physics_rows=len(rows), source_duration_s=len(rows)/240, duration_s=len(rr)/15,
                terminal_video_padding_s=len(rr)/15-len(rows)/240, source_pose_dt_s=1/30,
                source_physics_dt_s=1/240, first_saved_pose_s=rows[indices[0]]["time_s"],
                last_saved_pose_s=rows[indices[-1]]["time_s"],
                point_intervals=[{k: p[k] for k in ("target", "start_physics_row", "end_physics_row")} for p in points])
    outcome = dict(requested_points=list(targets), points=points, physics_steps=len(rows), success=True,
                   phase_config={"hold_duration_s": 3.})
    return data, outcome, rows, indices, poses, np.asarray(["/World/Robot/link"])


class VideoTests(unittest.TestCase):
    def module(self):
        name = "sim.upper_pair_posture_20260924.verify_video"
        self.assertIsNotNone(importlib.util.find_spec(name), "independent media verifier must exist")
        return importlib.import_module(name)

    def test_three_and_ten_point_complete_schedules(self):
        mod = self.module()
        for targets in (("R1", "L1", "R2"), tuple(side+str(i) for i in range(1, 6) for side in ("R", "L"))):
            result = mod.validate_schedule(*schedule_fixture(targets))
            self.assertEqual(result["targets"], list(targets))
            self.assertTrue(result["final_home_included"])

    def test_missing_video_frame_or_wrong_pose_mapping_rejected(self):
        mod = self.module()
        fixture = list(schedule_fixture())
        fixture[0]["frames"].pop()
        with self.assertRaisesRegex(ValueError, "frame inventory"):
            mod.validate_schedule(*fixture)
        fixture = list(schedule_fixture())
        fixture[0]["frames"][4]["pose_row"] -= 8
        with self.assertRaisesRegex(ValueError, "pose_row"):
            mod.validate_schedule(*fixture)

    def test_shifted_clock_and_nonunit_replay_rejected(self):
        mod = self.module()
        fixture = list(schedule_fixture())
        for row in fixture[2]: row["time_s"] += 1.
        with self.assertRaisesRegex(ValueError, "clock"):
            mod.validate_schedule(*fixture)
        fixture = list(schedule_fixture()); fixture[4][0, 0, 3] = 2.
        with self.assertRaisesRegex(ValueError, "rotation"):
            mod.validate_schedule(*fixture)

    def test_missing_hold_and_final_home_rejected(self):
        mod = self.module()
        fixture = list(schedule_fixture()); fixture[2][32]["phase"] = "APPROACH"
        with self.assertRaisesRegex(ValueError, "HOLD"):
            mod.validate_schedule(*fixture)
        fixture = list(schedule_fixture()); fixture[2][-1]["subphase"] = "return_home"
        with self.assertRaisesRegex(ValueError, "final home"):
            mod.validate_schedule(*fixture)

    def test_exact_comparison_clip_count_and_frozen_padding(self):
        mod = self.module()
        manifests = [schedule_fixture()[0], schedule_fixture(lead=64)[0]]
        lengths = [min(len(m["frames"]), max(f["frame_index"] for f in m["frames"]
                    if f["target"] == "R2" and f["phase"] == "HOLD") + 2) for m in manifests]
        count = max(lengths)
        receipt = dict(fps=15, frames=count, duration_s=count/15, panel_frames=lengths,
                       panel_duration_s=[n/15 for n in lengths],
                       frozen_panel_padding_s=[(count-n)/15 for n in lengths], decoded_frames=count)
        self.assertEqual(mod.validate_comparison_metadata(receipt, manifests), lengths)
        receipt["frozen_panel_padding_s"][0] += 1/15
        with self.assertRaisesRegex(ValueError, "padding"):
            mod.validate_comparison_metadata(receipt, manifests)

    def test_synthetic_mp4_full_decode_and_pts(self):
        mod = self.module()
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15., (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(4): writer.write(np.full((48, 64, 3), i*45, np.uint8))
            writer.release()
            result = mod.decode_video(path, 4, resolution=(64, 48))
            self.assertEqual(result["decoded_frames"], 4)
            self.assertTrue(result["pts_verified"])
            for count in (3, 5):
                with self.assertRaisesRegex(ValueError, "frame count"):
                    mod.decode_video(path, count, resolution=(64, 48))

    def test_panel_check_rejects_wrong_source_frame(self):
        mod = self.module()
        import cv2
        source = np.full((720, 1920, 3), 90, np.uint8)
        frame = np.zeros((720, 1920, 3), np.uint8)
        frame[:540, :960] = cv2.resize(source[:720, :1280], (960, 540))
        frame[540:, :320] = cv2.resize(source[:360, 1280:], (320, 180))
        result = mod.check_panel_pixels(frame, source, 0)
        self.assertEqual(result["mean_absolute_error"], 0.)
        with self.assertRaisesRegex(ValueError, "panel source pixels"):
            mod.check_panel_pixels(frame, np.full_like(source, 150), 0)

    def test_bound_source_mutation_and_unsafe_path_rejected(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"source").write_bytes(b"original")
            hashes = {"source": mod.sha(root/"source")}
            mod.verify_sources(hashes, root)
            (root/"source").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source hash"):
                mod.verify_sources(hashes, root)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                mod.verify_sources({"../outside": "0"*64}, root)

    def test_receipt_destination_cannot_modify_sealed_package(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            with self.assertRaisesRegex(ValueError, "receipt"):
                mod.validate_receipt_path(project/"sim/curobo_expert_v1/new.json", project, [])
            safe = project/"sim/upper_pair_posture_20260924/media/verification.json"
            self.assertEqual(mod.validate_receipt_path(safe, project, []), safe.resolve())


if __name__ == "__main__":
    unittest.main()
