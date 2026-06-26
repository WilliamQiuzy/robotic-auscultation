import os
import sys
import types
import unittest
import tempfile
import json


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "datasets"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))


class PerceptionAugmentationTests(unittest.TestCase):
    def test_real_2d_framings_do_not_include_extreme_zoom(self):
        import common

        self.assertEqual(set(common.FRAMINGS_2D), {"F1_upper", "F2_torso", "F3_near"})
        self.assertNotIn("F4_zoom", common.FRAMINGS_2D)

    def test_synthetic_framings_match_deployment_three_scale_taxonomy(self):
        import synthetic

        self.assertEqual(set(synthetic.FRAMINGS), {"F1_upper", "F2_torso", "F3_near"})
        self.assertNotIn("F4_zoom", synthetic.FRAMINGS)

    def test_synthetic_variant_seed_offset_is_stable(self):
        import synthetic

        self.assertEqual(synthetic.variant_seed_offset("standing", "front", "F1_upper"), 991)
        self.assertEqual(synthetic.variant_seed_offset("standing", "front", "F1_upper"), 991)

    def test_scoliosis_generator_rejects_2d_only_labels(self):
        import scoliosis

        with self.assertRaisesRegex(ValueError, "older labels_hmr2.pkl"):
            scoliosis._require_refit_schema("case.jpg", {"verts2d": [], "wh": (100, 100)})


class TrainingRecordTests(unittest.TestCase):
    def test_subject_key_strips_generated_clinical_suffixes(self):
        import dataset

        self.assertEqual(dataset.subject_key("scol_case1_canon_clin0"), "scol_case1")
        self.assertEqual(dataset.subject_key("scol_case1_aug3_clin2"), "scol_case1")
        self.assertEqual(dataset.subject_key("p000438_a000040_back_f72_canon_clin1"), "p000438")

    def test_audit_config_uses_single_freeze_field(self):
        import train

        args = types.SimpleNamespace(
            root=None,
            roots="data/dataset/humman,data/dataset/syn2,data/dataset/scoliosis,data/dataset/scoliosis_clinical",
            unfreeze=1,
            enc_lr_mult=0.1,
            encoder="dinov2_vitb14",
            res=518,
            epochs=15,
            bs=8,
            lr=3e-4,
            workers=12,
            val_frac=0.08,
            out="outputs/perc_v1",
        )
        cfg = train.audit_config(args, frozen_encoder=False)

        self.assertNotIn("frozen", cfg)
        self.assertEqual(cfg["frozen_encoder"], False)
        self.assertEqual(cfg["trainable_encoder"], True)

    def test_lr_record_names_encoder_and_head_groups(self):
        import train

        opt = types.SimpleNamespace(
            param_groups=[
                {"name": "encoder", "lr": 3e-5},
                {"name": "head", "lr": 3e-4},
            ]
        )

        self.assertEqual(train.lr_record(opt), {"lr_encoder": 3e-5, "lr_head": 3e-4})

    def test_fixed_val_train_rows_exclude_val_subjects(self):
        import dataset

        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "ds")
            os.makedirs(root)
            rows = [
                {"id": "scol_case1_canon", "is_valid": True, "iuv": "iuv/a.png", "side_conf": 1.0},
                {"id": "scol_case1_aug0", "is_valid": True, "iuv": "iuv/b.png", "side_conf": 1.0},
                {"id": "scol_case2_canon", "is_valid": True, "iuv": "iuv/c.png", "side_conf": 1.0},
            ]
            with open(os.path.join(root, "manifest.jsonl"), "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            train_rows = dataset.train_rows_with_fixed_val([root], [{"id": "scol_case1_canon"}])

            self.assertEqual([r["id"] for r in train_rows], ["scol_case2_canon"])
            self.assertEqual(train_rows[0]["_root"], root)


if __name__ == "__main__":
    unittest.main()
