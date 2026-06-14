import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "qc"))


class RealbackReviewTests(unittest.TestCase):
    def test_subject_key_matches_clinical_variants(self):
        import realback_review
        import filter_review_manifest

        self.assertEqual(realback_review.subject_key("scol_case1_canon_clin0"), "scol_case1")
        self.assertEqual(filter_review_manifest.subject_key("scol_case1_aug4_clin2"), "scol_case1")

    def test_choose_representatives_prefers_canon(self):
        import realback_review

        rows = [
            {"id": "scol_case1_aug0", "points": {"L1": {"visible": True}}, "side_conf": 0.99},
            {"id": "scol_case1_canon", "points": {}, "side_conf": 0.50},
            {"id": "scol_case2_aug0", "points": {"L1": {"visible": True}, "R1": {"visible": True}}, "side_conf": 0.8},
        ]

        reps = dict(realback_review.choose_representatives(rows))

        self.assertEqual(reps["scol_case1"]["id"], "scol_case1_canon")
        self.assertEqual(reps["scol_case2"]["id"], "scol_case2_aug0")

    def test_auto_flags_marks_low_quality_rows(self):
        import realback_review

        flags = realback_review.auto_flags({"side_conf": 0.80}, width=120, height=140, visible_points=6)

        self.assertIn("tiny", flags)
        self.assertIn("few_visible", flags)
        self.assertIn("low_side_conf", flags)

    def test_filter_manifest_drops_reviewed_subject(self):
        import filter_review_manifest

        with tempfile.TemporaryDirectory() as td:
            manifest_in = os.path.join(td, "manifest.jsonl")
            review_csv = os.path.join(td, "review.csv")
            manifest_out = os.path.join(td, "filtered", "manifest.jsonl")
            rows = [
                {"id": "scol_case1_canon", "image": "a.jpg"},
                {"id": "scol_case1_aug0", "image": "b.jpg"},
                {"id": "scol_case2_canon", "image": "c.jpg"},
            ]
            with open(manifest_in, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with open(review_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["subject", "review_status"])
                writer.writeheader()
                writer.writerow({"subject": "scol_case1", "review_status": "bad_label"})

            kept, dropped, excluded = filter_review_manifest.filter_manifest(manifest_in, review_csv, manifest_out)

            self.assertEqual(kept, 1)
            self.assertEqual(dropped, 2)
            self.assertEqual(excluded, {"scol_case1"})
            with open(manifest_out) as f:
                out_rows = [json.loads(l) for l in f]
            self.assertEqual([r["id"] for r in out_rows], ["scol_case2_canon"])

    def test_filter_manifest_can_drop_auto_flagged_subject(self):
        import filter_review_manifest

        with tempfile.TemporaryDirectory() as td:
            manifest_in = os.path.join(td, "manifest.jsonl")
            review_csv = os.path.join(td, "review.csv")
            manifest_out = os.path.join(td, "filtered", "manifest.jsonl")
            with open(manifest_in, "w") as f:
                f.write(json.dumps({"id": "scol_case1_canon", "image": "a.jpg"}) + "\n")
                f.write(json.dumps({"id": "scol_case2_canon", "image": "b.jpg"}) + "\n")
            with open(review_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["subject", "review_status", "auto_flags"])
                writer.writeheader()
                writer.writerow({"subject": "scol_case1", "review_status": "needs_review", "auto_flags": "tiny"})

            kept, dropped, excluded = filter_review_manifest.filter_manifest(
                manifest_in,
                review_csv,
                manifest_out,
                exclude_auto_flags={"tiny"},
            )

            self.assertEqual(kept, 1)
            self.assertEqual(dropped, 1)
            self.assertEqual(excluded, {"scol_case1"})


if __name__ == "__main__":
    unittest.main()
