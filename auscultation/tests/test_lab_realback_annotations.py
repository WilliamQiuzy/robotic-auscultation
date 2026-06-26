import csv
import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lab"))


POINTS = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4")


class LabRealbackAnnotationTests(unittest.TestCase):
    def _write_image(self, td, name="case.jpg", size=(120, 160)):
        path = os.path.join(td, name)
        img = np.full((size[0], size[1], 3), 180, np.uint8)
        cv2.imwrite(path, img)
        return path

    def test_wide_csv_imports_landmark_manifest_and_overlay(self):
        import import_realback_annotations as lab

        with tempfile.TemporaryDirectory() as td:
            self._write_image(td)
            csv_path = os.path.join(td, "ann.csv")
            fieldnames = ["image", "subject_id", "frame_id"] + [f"{p}_{a}" for p in POINTS for a in ("x", "y")]
            row = {"image": "case.jpg", "subject_id": "s01", "frame_id": "f01"}
            for i, point in enumerate(POINTS):
                row[f"{point}_x"] = 30 + i
                row[f"{point}_y"] = 40 + i
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            out = os.path.join(td, "gold")
            entries = lab.parse_annotations(csv_path)
            manifest = lab.validate_and_write(entries, td, out)

            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["id"], "lab_s01_f01")
            self.assertEqual(manifest[0]["iuv"], None)
            self.assertEqual(set(manifest[0]["points"]), set(POINTS))
            self.assertTrue(os.path.exists(os.path.join(out, "manifest_landmarks.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(out, "overlays", "lab_s01_f01.jpg")))
            with open(os.path.join(out, "manifest_landmarks.jsonl")) as f:
                row = json.loads(f.readline())
            self.assertEqual(row["domain"], "lab_realback_gold")

    def test_long_csv_rejects_missing_required_point(self):
        import import_realback_annotations as lab

        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "ann.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "point", "x", "y"])
                writer.writeheader()
                for point in POINTS[:-1]:
                    writer.writerow({"image": "case.jpg", "point": point, "x": 10, "y": 20})

            with self.assertRaisesRegex(ValueError, "missing required back points"):
                lab.parse_annotations(csv_path)

    def test_import_rejects_visible_point_out_of_bounds(self):
        import import_realback_annotations as lab

        with tempfile.TemporaryDirectory() as td:
            self._write_image(td, size=(100, 100))
            csv_path = os.path.join(td, "ann.csv")
            fieldnames = ["image"] + [f"{p}_{a}" for p in POINTS for a in ("x", "y")]
            row = {"image": "case.jpg"}
            for point in POINTS:
                row[f"{point}_x"] = 50
                row[f"{point}_y"] = 50
            row["R4_x"] = 150
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "out of image bounds"):
                lab.validate_and_write(lab.parse_annotations(csv_path), td, os.path.join(td, "gold"))


if __name__ == "__main__":
    unittest.main()
