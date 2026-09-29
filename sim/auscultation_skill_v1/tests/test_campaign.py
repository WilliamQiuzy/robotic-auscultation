"""Campaign routing and accounting; runtime boundaries are mocked, never Isaac."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sim.auscultation_skill_v1 import campaign


class ValidationCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.model=self.root/"model"; self.model.mkdir()
        (self.model/"policy.pt").write_bytes(b"orchestration-only checkpoint fixture")
        (self.model/"run_manifest.json").write_text("{}")
        self.model_id=dict(checkpoint_sha256=campaign.sha256_file(self.model/"policy.pt"),
                           model_manifest_sha256=campaign.sha256_file(self.model/"run_manifest.json"))
        self.contract=patch.object(campaign,"verify_contract"); self.contract.start()
        self.addCleanup(self.contract.stop)
        self.conditions=json.loads((campaign.ROOT/"conditions.json").read_text())["conditions"]

    def receipt(self,directory,name,launcher,**kwargs):
        condition=kwargs["condition"]
        result=dict(condition_id=condition,learned_task_success=True,assisted_system_success=True,
                    complete_requested_sequence=True,requested_points=list(kwargs["points"]),**self.model_id)
        return dict(condition_id=condition,returncode=0,runner_accepted=True,worker_lifecycle_passed=True,
                    episode_simulation_gates_passed=True,evaluation=result,
                    episode=str(directory/"episodes"/name),quality_report=str(directory/"quality"/(name+".json")))

    def command(self,stage="validate",name="validation",extra=()):
        return [stage,"--output",str(self.root/name),"--model",str(self.model),*extra]

    def test_validation_defaults_all_validation_and_marks_development(self):
        with patch.object(campaign,"trial",side_effect=self.receipt) as trial:
            self.assertEqual(campaign.main(self.command()),0)
        expected=[r["condition_id"] for r in self.conditions if r["split"]=="validation"]
        self.assertEqual([c.kwargs["condition"] for c in trial.call_args_list],expected)
        self.assertTrue(all(c.kwargs["mode"]=="policy" for c in trial.call_args_list))
        summary=json.loads((self.root/"validation/summary.json").read_text())
        self.assertEqual(summary["requested_trials"],len(expected))
        self.assertEqual(summary["learned_successes"],len(expected))
        self.assertEqual(summary["required_points"],list(campaign.TARGETS))
        self.assertEqual(summary["pinned_model"],self.model_id)
        self.assertEqual(summary["evaluation_role"],"development_validation")
        self.assertEqual(summary["evaluation_split"],"validation")
        self.assertEqual(summary["stage"],"validate")
        self.assertFalse(summary["heldout_test_evaluation"])
        self.assertIn("development",summary["scope"])

    def test_validation_requires_model_and_rejects_train_or_test_before_launch(self):
        for condition in ("train_speed100","test_speed080"):
            with self.subTest(condition=condition), patch.object(campaign,"trial") as trial:
                with self.assertRaises(SystemExit) as error:
                    campaign.main(self.command(name=condition,extra=("--conditions",condition)))
                self.assertEqual(error.exception.code,2)
                trial.assert_not_called()
                self.assertFalse((self.root/condition).exists())
        with patch.object(campaign,"trial") as trial:
            with self.assertRaises(SystemExit) as error:
                campaign.main(["validate","--output",str(self.root/"missing-model")])
            self.assertEqual(error.exception.code,2)
            trial.assert_not_called()

    def test_validation_timeout_cannot_remove_requested_denominator(self):
        calls=[]
        def run(*args,**kwargs):
            result=self.receipt(*args,**kwargs); calls.append(result)
            if len(calls)==2: result["returncode"]=None
            return result
        with patch.object(campaign,"trial",side_effect=run):
            self.assertEqual(campaign.main(self.command()),1)
        summary=json.loads((self.root/"validation/summary.json").read_text())
        self.assertEqual(summary["requested_trials"],2)
        self.assertEqual(summary["learned_successes"],1)
        self.assertEqual(summary["learned_success_rate"],.5)
        self.assertEqual(summary["missing_conditions"],[calls[1]["condition_id"]])

    def test_validation_model_change_stops_trials_and_invalidates_summary(self):
        def run(*args,**kwargs):
            result=self.receipt(*args,**kwargs)
            (self.model/"policy.pt").write_bytes(b"changed after first trial")
            return result
        with patch.object(campaign,"trial",side_effect=run) as trial:
            self.assertEqual(campaign.main(self.command()),1)
        self.assertEqual(trial.call_count,1)
        summary=json.loads((self.root/"validation/summary.json").read_text())
        self.assertEqual(summary["requested_trials"],2)
        self.assertEqual(summary["learned_successes"],0)
        self.assertEqual(summary["assisted_successes"],0)
        self.assertFalse(summary["input_binding_passed"])
        self.assertEqual(summary["evaluation_role"],"development_validation")

    def test_validation_wrong_trial_checkpoint_stays_in_denominator(self):
        def run(*args,**kwargs):
            result=self.receipt(*args,**kwargs)
            result["evaluation"]["checkpoint_sha256"]="f"*64
            return result
        with patch.object(campaign,"trial",side_effect=run):
            self.assertEqual(campaign.main(self.command()),1)
        summary=json.loads((self.root/"validation/summary.json").read_text())
        self.assertEqual(summary["requested_trials"],2)
        self.assertEqual(summary["learned_successes"],0)
        self.assertEqual(set(summary["rejected_model_conditions"]),set(summary["condition_ids"]))
        self.assertEqual(set(summary["missing_conditions"]),set(summary["condition_ids"]))

    def test_evaluate_keeps_heldout_defaults_and_rejects_validation(self):
        with patch.object(campaign,"trial",side_effect=self.receipt) as trial:
            self.assertEqual(campaign.main(self.command("evaluate","heldout")),0)
        expected=[r["condition_id"] for r in self.conditions if r["split"]=="test"]
        self.assertEqual([c.kwargs["condition"] for c in trial.call_args_list],expected)
        summary=json.loads((self.root/"heldout/summary.json").read_text())
        self.assertEqual(summary["evaluation_split"],"test")
        self.assertEqual(summary["evaluation_role"],"heldout_test")
        self.assertTrue(summary["heldout_test_evaluation"])
        with patch.object(campaign,"trial") as trial:
            with self.assertRaises(SystemExit):
                campaign.main(self.command("evaluate","wrong-split",("--conditions","validation_speed0875")))
            trial.assert_not_called()

    def test_existing_qualification_collection_and_negative_modes_are_preserved(self):
        with patch.object(campaign,"trial",side_effect=self.receipt) as trial, \
                patch("sim.auscultation_skill_v1.quality.qualify_repetitions"):
            self.assertEqual(campaign.main(["qualify","--output",str(self.root/"qualification")]),0)
        self.assertEqual(trial.call_count,3)
        self.assertTrue(all(c.kwargs["mode"]=="expert" and c.kwargs["condition"] is None
                            for c in trial.call_args_list))
        with patch.object(campaign,"trial",side_effect=self.receipt) as trial, \
                patch("sim.auscultation_skill_v1.quality.verify_repetitions"), \
                patch("sim.auscultation_skill_v1.dataset.build_dataset"):
            self.assertEqual(campaign.main(["collect","--output",str(self.root/"collection"),
                                           "--repetition-report",str(self.root/"repetitions.json")]),0)
        expected=[r["condition_id"] for r in self.conditions if r["split"] in ("train","validation")]
        self.assertEqual([c.kwargs["condition"] for c in trial.call_args_list],expected)
        self.assertTrue(all(c.kwargs["mode"]=="expert" for c in trial.call_args_list))
        def negative(*args,**kwargs):
            result=self.receipt(*args,**kwargs)
            result.update(returncode=1,runner_accepted=False)
            result["evaluation"]=dict(negative_control_diagnostic_passed=True)
            return result
        with patch.object(campaign,"trial",side_effect=negative) as trial:
            self.assertEqual(campaign.main(self.command("negative","negative",("--mode","zero"))),0)
        self.assertEqual(trial.call_count,1)
        self.assertEqual(trial.call_args.kwargs["mode"],"zero")

    def test_validation_uses_run_then_quality_then_frozen_evaluation(self):
        events=[]
        condition="validation_speed0875"
        def launch(command,**kwargs):
            events.append("run")
            self.assertEqual(command[command.index("--mode")+1],"policy")
            self.assertEqual(command[command.index("--condition")+1],condition)
            episode=Path(command[command.index("--output")+1]); episode.mkdir()
            (episode/"runner_summary.json").write_text(json.dumps(dict(accepted=True,child_returncode=0,
                worker_lifecycle=dict(status="passed",returncode=0))))
            child=Mock(); child.wait.return_value=0
            return child
        def quality(*args,**kwargs):
            events.append("quality")
            return dict(episode_simulation_gates_passed=True,sim_control_imitation_eligible=False,
                        hard_failures=[],failures=[])
        def evaluate(episode,report,model,**kwargs):
            events.append("evaluate")
            self.assertEqual(kwargs["mode"],"policy")
            self.assertEqual(model,self.model)
            return dict(condition_id=condition,learned_task_success=True,assisted_system_success=True,
                        complete_requested_sequence=True,requested_points=list(campaign.TARGETS),**self.model_id)
        with patch.object(campaign.subprocess,"Popen",side_effect=launch), \
                patch("sim.auscultation_skill_v1.quality.audit_episode",side_effect=quality), \
                patch("sim.auscultation_skill_v1.evaluate.audit_episode",side_effect=evaluate):
            self.assertEqual(campaign.main(self.command(extra=("--conditions",condition))),0)
        self.assertEqual(events,["run","quality","evaluate"])


if __name__=="__main__": unittest.main()
