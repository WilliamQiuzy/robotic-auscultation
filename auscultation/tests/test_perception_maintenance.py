import os
import sys
import types
import unittest


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


class TrainingRecordTests(unittest.TestCase):
    def test_audit_config_uses_single_freeze_field(self):
        import train

        args = types.SimpleNamespace(
            root=None,
            roots="data/dataset/humman,data/dataset/synthetic",
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


if __name__ == "__main__":
    unittest.main()
