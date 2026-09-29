"""Coverage diagnostics expose missing feedback and bind their exact evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .test_learning import observation
from . import test_learning as learning_fixtures
from sim.auscultation_skill_v1.observations import encode, HISTORY_STEPS


def examples():
    records=[]
    for phase,hall,action,previous in (("HOLD",4.1,.0001,0.),("HOLD",3.9,-.0001,0.),
                                       ("RETRACT",4.1,.004,.004),("CONTACT",0.,-.002,-.002)):
        obs=observation(); obs.update(phase=phase,hall_N=hall,previous_desired_normal_velocity_m_s=previous)
        records.append(np.tile(encode(obs),(HISTORY_STEPS,1)))
    return {"observations":np.asarray(records),"actions":np.asarray([[.0001],[-.0001],[.004],[-.002]],np.float32),
            "split":np.asarray([0,0,1,2],np.uint8),"decision_steps":np.full(4,12,np.int16),
            "decision_duration_s":np.full(4,.05),"decision_reason":np.asarray(["full_period"]*4,dtype="U20"),
            "is_first_decision":np.asarray([True,False,True,True]),
            "physics_ranges":np.asarray([[0,12],[12,24],[0,12],[0,12]],np.int64),
            "sample_time_s":np.asarray([0,.05,0,0])},["R1","R1","L4","R1"]


class DiagnosticMetricsTests(unittest.TestCase):
    def test_force_corrections_do_not_count_retraction_and_test_is_withheld(self):
        from sim.auscultation_skill_v1.diagnostics import summarize_arrays
        arrays,points=examples(); result=summarize_arrays(arrays,points)
        hold=result["by_split_phase"]["train"]["HOLD"]
        self.assertEqual(hold["force_feedback"]["positive_error_outward_label"],1)
        self.assertEqual(hold["force_feedback"]["negative_error_inward_label"],1)
        self.assertEqual(result["by_split"]["validation"]["force_feedback"]["samples"],0)
        self.assertNotIn("test",result["by_split"])
        self.assertEqual(result["test_samples_withheld"],1)
        self.assertIsNone(hold["model_error"])
        self.assertAlmostEqual(hold["previous_action_error"]["rmse_m_s"],.0001,places=10)
        self.assertEqual(result["by_split_point_phase"]["train"]["L4"]["HOLD"]["samples"],0)

    def test_asymmetric_saturation_and_error_cells(self):
        from sim.auscultation_skill_v1.diagnostics import summarize_arrays
        arrays,points=examples()
        prediction=np.asarray([.00005,-.00015,.0036,-.0018])
        result=summarize_arrays(arrays,points,prediction,include_test=True)
        outward=result["by_split_point"]["validation"]["L4"]["near_bounds"]["outward"]
        inward=result["by_split"]["test"]["near_bounds"]["inward"]
        self.assertEqual(outward["samples"],1)
        self.assertAlmostEqual(outward["model_error"]["bias_m_s"],-.0004,places=8)
        self.assertAlmostEqual(outward["prediction_unit_quantiles"]["p50"],.9,places=6)
        self.assertAlmostEqual(inward["prediction_unit_quantiles"]["p50"],-.9,places=6)
        self.assertTrue(result["heldout_diagnostics_inspected"])
        prediction[-1]=-.003
        with self.assertRaisesRegex(ValueError,"bound"):
            summarize_arrays(arrays,points,prediction,include_test=True)

    def test_slow_family_saturation_uses_fixed_outward_and_scaled_inward(self):
        from sim.auscultation_skill_v1.diagnostics import summarize_arrays
        from sim.auscultation_skill_v1.observations import FRAME_FEATURES
        arrays,points=examples()
        arrays["observations"][:,:,FRAME_FEATURES.index("speed_scale")]=.75
        arrays["actions"][-1,0]=-.0015
        prediction=np.asarray([.00005,-.00015,.0036,-.00135])
        result=summarize_arrays(arrays,points,prediction,include_test=True)
        outward=result["by_split"]["validation"]["near_bounds"]["outward"]
        inward=result["by_split"]["test"]["near_bounds"]["inward"]
        self.assertEqual(outward["samples"],1); self.assertEqual(inward["samples"],1)
        self.assertAlmostEqual(outward["prediction_unit_quantiles"]["p50"],.9,places=6)
        self.assertAlmostEqual(inward["prediction_unit_quantiles"]["p50"],-.9,places=6)
        prediction[-1]=-.0018
        with self.assertRaisesRegex(ValueError,"bound"):
            summarize_arrays(arrays,points,prediction,include_test=True)

    def test_partial_and_first_action_diagnostics_preserve_actual_network_direction(self):
        from sim.auscultation_skill_v1.diagnostics import summarize_arrays
        arrays,points=examples()
        arrays["decision_steps"][0]=1; arrays["decision_duration_s"][0]=1/240
        arrays["physics_ranges"][0,1]=1; arrays["decision_reason"][0]="phase_boundary"
        result=summarize_arrays(arrays,points,np.asarray([.0002,-.0001,.004,-.002]))
        coverage=result["by_split"]["train"]["decision_coverage"]
        self.assertEqual(coverage["partial"],1); self.assertEqual(coverage["full_period"],1)
        first=result["by_split"]["train"]["first_target_decisions"]
        self.assertEqual(first["model_direction_counts"]["outward"],1)
        self.assertEqual(first["outward_start_risk_count"],1)
        self.assertEqual(result["by_split_decision_extent"]["train"]["partial"]["samples"],1)
        self.assertEqual(result["first_decision_checks"][0]["prediction_m_s"],.0002)
        self.assertEqual(result["first_decision_checks"][0]["decision_steps"],1)
        self.assertTrue(all(row["split"]!="test" for row in result["first_decision_checks"]))

    def test_formal_promotion_matches_proposal_and_preserves_archived_tests(self):
        from sim.auscultation_skill_v1.dataset import conditions_by_id, sha
        root=Path(__file__).parents[1]
        active=json.loads((root/"conditions.json").read_text())
        proposal=json.loads((root/"proposed_conditions_v3.json").read_text())
        baseline_bytes=(root/"results/control_revision_v3/before/conditions.json").read_bytes()
        baseline=json.loads(baseline_bytes)
        previous_bytes=(root/"results/control_revision_v4/before/conditions.json").read_bytes()
        previous=json.loads(previous_bytes)
        self.assertEqual(previous["previous_conditions_sha256"],sha(baseline_bytes))
        self.assertEqual(previous["experiment_revision"],3)
        self.assertEqual(active["previous_conditions_sha256"],sha(previous_bytes))
        self.assertEqual(active["experiment_revision"],4)
        self.assertEqual(active["action_scaling"],"inward_scaled_outward_fixed_v1")
        self.assertEqual(active["conditions"],previous["conditions"])
        self.assertEqual(active["conditions"],proposal["conditions"])
        self.assertEqual(len(conditions_by_id(active)),9)
        self.assertEqual({split:sum(r["split"]==split for r in active["conditions"])
                          for split in ("train","validation","test")},
                         {"train":3,"validation":2,"test":4})
        self.assertEqual(proposal["status"],"proposal_not_collected_or_qualified")
        self.assertEqual([r for r in active["conditions"] if r["split"]=="test"],
                         [r for r in baseline["conditions"] if r["split"]=="test"])
        rows=proposal["conditions"]
        self.assertEqual(len(rows),len({r["family_id"] for r in rows}))
        self.assertEqual(len(rows),len({r["seed"] for r in rows}))
        added=[r for r in rows if r["condition_id"] not in {r["condition_id"] for r in baseline["conditions"]}]
        self.assertEqual({(r["split"],r["speed_scale"],r["hall_noise_std_N"],r["hall_noise_clip_N"]) for r in added},
                         {("train",1.,.08,.24),("validation",.875,.06,.18)})
        with self.assertRaisesRegex(ValueError,"schema"): conditions_by_id(proposal)


class DiagnosticEvidenceTests(unittest.TestCase):
    def setUp(self):
        # Reuse portable numerical episodes; mock only physical qualification,
        # just as the existing dataset tests do. No fixture claims real physics.
        self.fixture=learning_fixtures.DatasetTests(methodName="test_train_only_scaler_and_duplicate_family_guard")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        from sim.auscultation_skill_v1.dataset import build_dataset
        self.entries=[self.fixture.episode(i) for i in range(3)]
        self.directory=self.fixture.root/"dataset"
        build_dataset(self.entries,self.directory,self.fixture.condition_path)

    def test_eligible_source_bound_dataset_and_report_hashes(self):
        from sim.auscultation_skill_v1.diagnostics import diagnose
        output=self.fixture.root/"diagnostics.json"
        result=diagnose(self.directory,output=output)
        self.assertTrue(result["dataset_evidence_verified"])
        self.assertEqual(result["by_split_point"]["train"]["R1"]["samples"],2)
        self.assertIn("manifest.json",result["dataset_sha256"])
        self.assertIn("diagnostics.py",result["diagnostic_source_sha256"])
        self.assertEqual(result["action_scaling"],"inward_scaled_outward_fixed_v1")
        self.assertIn("positive by fixed 0.004",result["definitions"]["unit_action"])
        self.assertIsNone(result["model"])
        self.assertEqual(json.loads(output.read_text()),result)

    def test_source_mutation_during_inference_never_publishes_report(self):
        from sim.auscultation_skill_v1 import diagnostics
        source=Path(self.entries[0]["episode_dir"])/"control_commands.json"
        original=diagnostics.summarize_arrays
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs)
            source.write_text(source.read_text()+" ")
            return result
        with patch.object(diagnostics,"summarize_arrays",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"source|changed"):
                diagnostics.diagnose(self.directory,output=self.fixture.root/"unpublished.json")
        self.assertFalse((self.fixture.root/"unpublished.json").exists())

    def test_sample_point_metadata_must_match_bound_decision(self):
        from sim.auscultation_skill_v1.diagnostics import bound_points
        from sim.auscultation_skill_v1.dataset import load_dataset
        arrays,manifest=load_dataset(self.directory)
        arrays["physics_ranges"][0,0]=1
        with self.assertRaisesRegex(ValueError,"decision|range"):
            bound_points(arrays,manifest)

    def test_bound_partial_histories_include_first_target_approach(self):
        from sim.auscultation_skill_v1.dataset import build_dataset
        from sim.auscultation_skill_v1.diagnostics import diagnose
        for entry in self.entries:
            self.fixture.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
        directory=self.fixture.root/"issued_decisions"
        build_dataset(self.entries,directory,self.fixture.condition_path)
        result=diagnose(directory)
        self.assertEqual(result["by_split_phase"]["train"]["APPROACH"]["samples"],1)
        self.assertEqual(result["by_split"]["train"]["decision_coverage"]["partial"],2)
        self.assertEqual(result["by_split_decision_extent"]["train"]["full_period"]["samples"],1)
        self.assertEqual(len(result["first_decision_checks"]),2)
        for first in result["first_decision_checks"]:
            self.assertEqual(first["phase"],"APPROACH"); self.assertEqual(first["decision_steps"],1)
            self.assertEqual(first["decision_reason"],"phase_boundary")
            self.assertIn(first["condition_id"],("a","b"))
            self.assertLess(first["label_m_s"],0.)
            self.assertIsNone(first["prediction_m_s"])

    def test_actual_frozen_model_inference_is_bound_and_withholds_test_metrics(self):
        from sim.auscultation_skill_v1.train import main as train
        from sim.auscultation_skill_v1.diagnostics import diagnose
        from sim.auscultation_skill_v1.dataset import sha
        model=self.fixture.root/"model"
        train(["--dataset",str(self.directory),"--output",str(model),"--device","cpu","--epochs","2"])
        result=diagnose(self.directory,model)
        self.assertEqual(result["model"]["checkpoint_sha256"],sha((model/"policy.pt").read_bytes()))
        self.assertIsNotNone(result["by_split"]["train"]["model_error"])
        self.assertEqual(result["test_samples_withheld"],2)
        self.assertFalse(result["closed_loop_success_established"])
        self.assertFalse(result["heldout_diagnostics_inspected"])
        self.assertNotIn("test",result["by_split"])
        (model/"policy.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"model|checkpoint"):
            diagnose(self.directory,model)


if __name__=="__main__": unittest.main()
