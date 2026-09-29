"""Frozen policy attribution, negative controls and evaluation denominators."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from .test_learning import observation


class StudentTests(unittest.TestCase):
    def test_student_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("sim.auscultation_skill_v1.student"))

    def test_negative_control_serialized_weights_and_zero_has_no_rescue(self):
        from sim.auscultation_skill_v1.student import prepare_negative_control,load_negative_control,StudentController
        from sim.auscultation_skill_v1.observations import History,ACTION_SCALING
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for mode in ("zero","untrained"):
                path=root/mode
                prepare_negative_control(path,mode,"a"*64,7)
                digest=hashlib.sha256((path/"run_manifest.json").read_bytes()).hexdigest()
                actor=load_negative_control(path,digest,"a"*64,mode)
                self.assertEqual(actor.metadata["action_scaling"],ACTION_SCALING)
                obs=observation(); obs["speed_scale"] = .75
                value=actor.predict(History().push(obs,"R1"))
                self.assertTrue(np.isfinite(value))
                if mode=="zero": self.assertEqual(value,0.)
                else:
                    self.assertGreaterEqual(value,-.0015); self.assertLessEqual(value,.004)
                with self.assertRaises(ValueError): load_negative_control(path,digest,"b"*64,mode)
                controller=StudentController.__new__(StudentController)
                controller.actor=actor; controller.mode=mode; controller.actor_history=History()
                controller.current_skill_observation=obs; controller.target=SimpleNamespace(name="R1")
                controller.speed_scale=.75; controller.policy_decisions=0
                controller.action_provider=controller._policy_action
                # Teacher would approach at this empty-force state. The action
                # hook must return the frozen model/zero and never invoke it.
                predicted=controller.desired_normal_velocity({"hall_N":0.,"compression_m":0.})
                self.assertEqual(predicted,value)
                self.assertFalse(controller.policy_record["expert_normal_rescue_used"])
                self.assertEqual(controller.policy_record["policy_action_scaling"],ACTION_SCALING)

    def test_student_guard_accepts_fixed_outward_endpoint_at_slow_speed(self):
        from sim.auscultation_skill_v1.student import StudentController
        from sim.auscultation_skill_v1.observations import History
        controller=StudentController.__new__(StudentController)
        controller.mode="policy"; controller.actor_history=History()
        controller.current_skill_observation=dict(observation(),speed_scale=.75)
        controller.target=SimpleNamespace(name="R1"); controller.speed_scale=.75; controller.policy_decisions=0
        for value in (.004,-.0015,0.,.00401,-.00151):
            controller.actor=SimpleNamespace(predict=lambda history:value,metadata={"checkpoint_sha256":"a"*64})
            if -.0015<=value<=.004:
                self.assertEqual(controller._policy_action(controller,{}),value)
            else:
                with self.assertRaisesRegex(ValueError,"desired normal velocity"):
                    controller._policy_action(controller,{})

    def test_student_rejects_old_complete_period_training_semantics(self):
        from sim.auscultation_skill_v1.student import StudentController
        from sim.auscultation_skill_v1.observations import ACTION_SCALING
        metadata=dict(schema="auscultation_scalar_training_v1",action_scaling=ACTION_SCALING,
                      action_limit_m_s=.004,action_inward_limit_m_s=.002)
        with self.assertRaisesRegex(ValueError,"sample semantics"):
            StudentController(mode="policy",actor=SimpleNamespace(metadata=metadata))

    def test_negative_control_rejects_missing_or_old_action_scaling(self):
        import torch
        from sim.auscultation_skill_v1.student import prepare_negative_control,load_negative_control,StudentController
        from sim.auscultation_skill_v1.dataset import sha
        from sim.auscultation_skill_v1.observations import ACTION_SCALING
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("zero","untrained"):
                path=Path(temporary)/mode; original=prepare_negative_control(path,mode,"a"*64,7)
                checkpoint=torch.load(path/"policy.pt",map_location="cpu",weights_only=True)
                self.assertEqual(checkpoint["action_scaling"],ACTION_SCALING)
                for location in ("manifest","checkpoint"):
                    for value in (None,"both_directions_scaled"):
                        manifest=copy.deepcopy(original); saved=copy.deepcopy(checkpoint)
                        target=manifest if location=="manifest" else saved
                        if value is None: target.pop("action_scaling")
                        else: target["action_scaling"]=value
                        torch.save(saved,path/"policy.pt")
                        manifest["checkpoint_sha256"]=sha((path/"policy.pt").read_bytes())
                        (path/"run_manifest.json").write_text(json.dumps(manifest))
                        with self.assertRaisesRegex(ValueError,"action scaling|interface"):
                            load_negative_control(path,sha((path/"run_manifest.json").read_bytes()),"a"*64,mode)
                missing=copy.deepcopy(original); missing.pop("action_scaling")
                with self.assertRaisesRegex(ValueError,"actor action scaling"):
                    StudentController(mode=mode,actor=SimpleNamespace(metadata=missing))

    def test_replay_detects_fallback_and_changed_action(self):
        from sim.auscultation_skill_v1.student import prepare_negative_control,load_negative_control
        from sim.auscultation_skill_v1.evaluate import verify_action_rows
        from sim.auscultation_skill_v1.observations import History
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"model"; prepare_negative_control(root,"zero","a"*64,7)
            digest=hashlib.sha256((root/"run_manifest.json").read_bytes()).hexdigest()
            actor=load_negative_control(root,digest,"a"*64,"zero")
            history=History().push(observation(),"R1")
            row=dict(policy_update=True,policy_group_id=1,target="R1",skill_observation=observation(),
                     policy_input_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
                     policy_checkpoint_sha256=actor.metadata["checkpoint_sha256"],
                     policy_mode="zero",network_prediction_m_s=None,policy_prediction_m_s=0.,
                     desired_normal_velocity_m_s=0.,hard_clamp_intervened=False,failure=False,recovery=False,
                     controller_source="learned_normal_policy")
            rows=[row,dict(row,policy_update=False)]
            self.assertTrue(verify_action_rows(rows,actor,"zero")["action_attribution_passed"])
            changed=copy.deepcopy(rows); changed[1]["desired_normal_velocity_m_s"]=-.001
            self.assertFalse(verify_action_rows(changed,actor,"zero")["action_attribution_passed"])
            changed=copy.deepcopy(rows); changed[1]["controller_source"]="analytic_skill_expert"
            self.assertFalse(verify_action_rows(changed,actor,"zero")["action_attribution_passed"])
            changed=copy.deepcopy(rows); changed[1]["hard_clamp_intervened"]=True
            self.assertFalse(verify_action_rows(changed,actor,"zero")["action_attribution_passed"])

    def test_missing_trial_is_a_failure_in_denominator(self):
        from sim.auscultation_skill_v1.evaluate import summarize_trials
        results=[dict(condition_id="a",learned_task_success=True,assisted_system_success=True,
                      complete_requested_sequence=True,requested_points=["R1"],
                      checkpoint_sha256="a"*64,model_manifest_sha256="b"*64)]
        summary=summarize_trials(results,["a","b"],required_points=["R1"])
        self.assertEqual(summary["requested_trials"],2)
        self.assertEqual(summary["learned_successes"],1)
        self.assertEqual(summary["learned_success_rate"],.5)
        self.assertEqual(summary["missing_conditions"],["b"])

    def test_mixed_or_missing_model_cannot_claim_heldout_success(self):
        from sim.auscultation_skill_v1.evaluate import summarize_trials
        first=dict(condition_id="a",learned_task_success=True,assisted_system_success=True,
                   complete_requested_sequence=True,requested_points=["R1"],
                   checkpoint_sha256="a"*64,model_manifest_sha256="b"*64)
        second=dict(first,condition_id="b",checkpoint_sha256="c"*64)
        with self.assertRaisesRegex(ValueError,"model"):
            summarize_trials([first,second],["a","b"],required_points=["R1"])
        missing=dict(first); del missing["model_manifest_sha256"]
        with self.assertRaisesRegex(ValueError,"model"):
            summarize_trials([missing],["a"],required_points=["R1"])
        with self.assertRaisesRegex(ValueError,"model"):
            summarize_trials([first],["a"],required_points=["R1"],
                             expected_model=dict(checkpoint_sha256="c"*64,model_manifest_sha256="b"*64))

    def test_saved_success_cannot_override_worker_failure(self):
        from sim.auscultation_skill_v1.evaluate import assess_evaluation
        from sim.auscultation_skill_v1.provenance import STUDENT_CLASS
        outcome=dict(success=True,requested_points=["R1"],points=[dict(target="R1",success=True)],
                     executed_controller_class=STUDENT_CLASS,candidate_method=STUDENT_CLASS,
                     dataset_role="student_evaluation",expert_demonstration=False,policy_mode="policy")
        quality=dict(hard_simulation_gates_passed=True,episode_simulation_gates_passed=True)
        attribution=dict(action_attribution_passed=True)
        lifecycle=dict(status="passed",verified_from_bound_files=True,runner_accepted=True,child_returncode=0)
        self.assertTrue(assess_evaluation(outcome,quality,[],attribution,"policy",lifecycle)["learned_task_success"])
        for change in ({"child_returncode":None},{"child_returncode":1},{"runner_accepted":False},
                       {"status":"failed"},{"verified_from_bound_files":False}):
            result=assess_evaluation(outcome,quality,[],attribution,"policy",dict(lifecycle,**change))
            self.assertFalse(result["learned_task_success"])
            self.assertFalse(result["assisted_system_success"])

    def test_negative_control_needs_normal_failed_worker(self):
        from sim.auscultation_skill_v1.evaluate import assess_evaluation
        from sim.auscultation_skill_v1.provenance import STUDENT_CLASS
        outcome=dict(success=False,requested_points=["R1"],points=[],policy_mode="zero",
                     executed_controller_class=STUDENT_CLASS,candidate_method=STUDENT_CLASS,
                     dataset_role="student_evaluation",expert_demonstration=False,failure_latched=True)
        lifecycle=dict(status="passed",verified_from_bound_files=True,runner_accepted=False,child_returncode=1)
        attribution=dict(action_attribution_passed=True)
        result=assess_evaluation(outcome,{},[],attribution,"zero",lifecycle)
        self.assertTrue(result["negative_control_diagnostic_passed"])
        self.assertFalse(result["learned_task_success"])
        self.assertFalse(assess_evaluation(outcome,{},[],attribution,"zero",dict(lifecycle,child_returncode=None))
                         ["negative_control_diagnostic_passed"])
        completed=dict(outcome,success=True,points=[dict(target="R1",success=True)])
        result=assess_evaluation(completed,{},[],attribution,"zero",lifecycle)
        self.assertFalse(result["negative_control_diagnostic_passed"])
        self.assertTrue(result["negative_control_task_completed"])

    def test_process_binding_rejects_changed_completion(self):
        from sim.auscultation_skill_v1.evaluate import verify_process_evidence
        from sim.auscultation_skill_v1.run import verify_worker_completion
        from sim.auscultation_skill_v1.quality import verify_runner_lifecycle
        def write(path,value): path.write_text(json.dumps(value))
        def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/"input_snapshot").mkdir()
            snapshot=root/"input_snapshot/manifest.json"; write(snapshot,{"files":{}})
            write(root/"outcome.json",dict(reset_id="fresh-reset",input_snapshot_manifest_sha256=digest(snapshot)))
            completion=dict(schema="auscultation_skill_worker_completion_v1",returncode=1,reset_id="fresh-reset",
                            outcome_sha256=digest(root/"outcome.json"),input_snapshot_manifest_sha256=digest(snapshot),
                            close_returned=True,completion_stage="post_app_close_pre_process_exit",
                            exit_policy="controlled_os_exit_after_close")
            write(root/"worker_completion.json",completion)
            actual=verify_worker_completion(root,1)
            write(root/"runner_summary.json",dict(accepted=False,child_returncode=1,
                worker_lifecycle=actual,evidence_sha256=actual["evidence_sha256"]))
            quality=dict(worker_lifecycle=verify_runner_lifecycle(root),artifact_sha256={
                name:digest(root/name) for name in ("runner_summary.json","worker_completion.json")})
            self.assertEqual(verify_process_evidence(root,quality)["child_returncode"],1)
            write(root/"worker_completion.json",dict(completion,returncode=0))
            with self.assertRaises(ValueError): verify_process_evidence(root,quality)


if __name__=="__main__": unittest.main()
