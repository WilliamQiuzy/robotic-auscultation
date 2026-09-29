"""Behavioral checks for the scalar contact learner and its evidence boundary."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def observation():
    return dict(gap_m=.03, hall_N=0., compression_m=0., normal_velocity_m_s=0.,
                previous_desired_normal_velocity_m_s=0., executor_gap_m=.03,
                executor_velocity_m_s=0., executor_acceleration_m_s2=0., speed_scale=1.,
                q=[0.]*7, dq=[0.]*7, target_position_base_m=[0., 0., .4],
                normal_base=[0., 0., 1.], target_rotation_base=np.eye(3).tolist(), phase="APPROACH")


class ObservationTests(unittest.TestCase):
    def test_observation_contract_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("sim.auscultation_skill_v1.observations"))

    def test_allowlist_history_and_invalid_inputs(self):
        from sim.auscultation_skill_v1.observations import encode, History, FRAME_FEATURES
        row = observation()
        clean = encode(row)
        row.update(time_s=999, expert_action=.002, diaphragm_N=50, audit_hold_s=3)
        np.testing.assert_array_equal(clean, encode(row))
        self.assertEqual(clean.shape, (len(FRAME_FEATURES),))
        history = History()
        np.testing.assert_array_equal(history.push(row, "R1"), np.tile(clean, (4, 1)))
        row["gap_m"] = .02
        self.assertNotEqual(history.push(row, "R1")[0, 0], encode(row)[0])
        np.testing.assert_array_equal(history.push(row, "L1"), np.tile(encode(row), (4, 1)))
        row["normal_base"] = [0., 0., 2.]
        with self.assertRaises(ValueError): encode(row)
        row = observation(); row.pop("hall_N")
        with self.assertRaises(ValueError): encode(row)

    def test_sensor_delay_noise_is_real_deterministic_and_preserves_truth(self):
        from sim.auscultation_skill_v1.observations import SensorCondition
        condition=dict(observation_delay_steps=1,hall_noise_std_N=.1,hall_noise_clip_N=.2,seed=4)
        a,b=SensorCondition(condition),SensorCondition(condition)
        for index in range(3):
            state=dict(time_s=index*.05,hall_N=float(index),q=np.zeros(7))
            x,meta=a.apply(state,"R1"); y,_=b.apply(state,"R1")
            self.assertEqual(x["hall_N"],y["hall_N"])
            self.assertLessEqual(abs(meta["hall_noise_N"]),.2)
            self.assertAlmostEqual(x["hall_N"],max(0,index-1)+meta["hall_noise_N"])
            self.assertEqual(state["hall_N"],float(index))
            self.assertEqual(meta["hall_sample_time_s"],max(0,index-1)*.05)
        x,meta=a.apply(dict(time_s=.2,hall_N=7.,q=np.zeros(7)),"L1")
        self.assertEqual(meta["observation_time_s"],.2)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Numerical fixtures have no robot scene; the independently tested
        # physics qualification reader is the sole mocked evidence boundary.
        self.qualification=patch("sim.auscultation_skill_v1.dataset.verify_eligibility",return_value=None)
        self.qualification.start(); self.addCleanup(self.qualification.stop)
        self.conditions = {"schema": "auscultation_skill_conditions_v1", "conditions": [
            dict(condition_id=name, family_id=name, split=split, speed_scale=1.,
                 hall_noise_std_N=0., hall_noise_clip_N=0., observation_delay_steps=0, seed=i)
            for i, (name, split) in enumerate((("a", "train"), ("b", "validation"), ("c", "test")))]}
        self.condition_path = self.root/"conditions.json"
        self.condition_path.write_text(json.dumps(self.conditions))

    def episode(self, index=0, eligible=True):
        condition = self.conditions["conditions"][index]
        directory = self.root/condition["condition_id"]; directory.mkdir()
        controls, trace = [], []
        # Two physically bound 20 Hz decisions, independent values per reset.
        for block in range(2):
            obs = observation(); obs["gap_m"] += .001*index; obs["speed_scale"] = condition["speed_scale"]
            for tick in range(12):
                row_index = block*12+tick
                row = dict(time_s=row_index/240, physics_row=row_index,
                           physics_row_end_exclusive=row_index+1, executed_physics_steps=1,
                           action_duration_s=1/240, command_block_complete=True,
                           target="R1", phase="APPROACH", subphase="contact_cycle",
                           policy_update=tick==0, policy_group_id=block,
                           desired_normal_velocity_m_s=-.001-index*.0001,
                           applied_normal_velocity_m_s=-.0001,
                           hard_clamp_intervened=False, failure=False, recovery=False,
                           controller_source="analytic_skill_expert", skill_observation=obs,
                           observation_time_s=row_index/240, hall_sample_time_s=row_index/240,
                           hall_noise_N=0.)
                controls.append(row)
                trace.append(dict(time_s=(row_index+1)/240, q=obs["q"], dq=obs["dq"],
                                  compression_m=obs["compression_m"], hall_N=0.,
                                  gap_m=obs["gap_m"], target="R1",phase="APPROACH"))
        outcome = dict(success=True, dataset_role="expert_demonstration", expert_demonstration=True,
                       condition_id=condition["condition_id"], family_id=condition["family_id"],
                       condition_parameters=condition, reset_id="reset-"+str(index),
                       points=[dict(target="R1",success=True,final_phase="DONE",contact_end_physics_row=24,end_physics_row=72)],
                       contact_curves=[dict(target="R1",motion_complete=True,start_physics_row=0,end_physics_row_exclusive=24)],
                       controller_type="sim.auscultation_skill_v1.controller.ExpertController")
        # Portable content-addressed snapshot: no source-machine paths required.
        snapshot=directory/"input_snapshot"; (snapshot/"objects").mkdir(parents=True)
        bindings={}; snapshot_hashes={}
        package=Path(__file__).parents[1]
        from sim.auscultation_skill_v1.dataset import EXECUTOR_CODE_FILES
        for name in EXECUTOR_CODE_FILES:
            contents=(package/name).read_bytes(); digest=sha(contents)
            (snapshot/"objects"/digest).write_bytes(contents)
            bindings["source/contact_skill/"+name]={"sha256":digest}
            snapshot_hashes["input_snapshot/objects/"+digest]=digest
        snapshot_bytes=json.dumps({"files":bindings}).encode()
        (snapshot/"manifest.json").write_bytes(snapshot_bytes)
        outcome.update(input_snapshot="input_snapshot/manifest.json",input_snapshot_manifest_sha256=sha(snapshot_bytes))
        snapshot_hashes["input_snapshot/manifest.json"]=sha(snapshot_bytes)
        documents = {"control_commands.json":json.dumps(controls).encode(),
                     "expert_trace.jsonl":("\n".join(json.dumps(x) for x in trace)+"\n").encode(),
                     "outcome.json":json.dumps(outcome).encode()}
        for name, data in documents.items(): (directory/name).write_bytes(data)
        quality = dict(schema="auscultation_skill_quality_v1", sim_control_imitation_eligible=eligible,
                       contract_sha256="a"*64, artifact_sha256={**snapshot_hashes,**{n:sha(d) for n,d in documents.items()}})
        report = self.root/(condition["condition_id"]+"_quality.json")
        report.write_text(json.dumps(quality))
        return dict(episode_dir=str(directory), quality_report=str(report))

    def mutate(self, entry, mutate):
        p=Path(entry["episode_dir"])/"control_commands.json"
        rows=json.loads(p.read_text()); mutate(rows); p.write_text(json.dumps(rows))
        q=Path(entry["quality_report"]); report=json.loads(q.read_text())
        report["artifact_sha256"][p.name]=sha(p.read_bytes()); q.write_text(json.dumps(report))

    def set_decisions(self,entry,specs):
        """Portable phase trace; each specification is (phase, steps, action)."""
        self.assertEqual(sum(s[1] for s in specs),24)
        def change(rows):
            cursor=0; previous=0.
            for group,(phase,steps,action) in enumerate(specs):
                for index in range(cursor,cursor+steps):
                    row=rows[index]; row["skill_observation"]=copy.deepcopy(row["skill_observation"])
                    row["skill_observation"].update(phase=phase,previous_desired_normal_velocity_m_s=previous)
                    row.update(phase=phase,policy_update=index==cursor,policy_group_id=group,
                               desired_normal_velocity_m_s=action)
                previous=action; cursor+=steps
        self.mutate(entry,change)
        root=Path(entry["episode_dir"]); rows=json.loads((root/"control_commands.json").read_text())
        trace=[json.loads(x) for x in (root/"expert_trace.jsonl").read_text().splitlines()]
        for row,measured in zip(rows,trace): measured["phase"]=row["phase"]
        path=root/"expert_trace.jsonl"; path.write_text("\n".join(json.dumps(x) for x in trace)+"\n")
        report_path=Path(entry["quality_report"]); report=json.loads(report_path.read_text())
        report["artifact_sha256"][path.name]=sha(path.read_bytes()); report_path.write_text(json.dumps(report))

    def test_unqualified_and_intervened_episodes_rejected(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode(eligible=False)
        with self.assertRaisesRegex(ValueError, "eligible"):
            load_episode(entry, self.conditions)
        q=Path(entry["quality_report"]); report=json.loads(q.read_text())
        report["sim_control_imitation_eligible"]=True; q.write_text(json.dumps(report))
        self.mutate(entry,lambda rows:rows[3].update(hard_clamp_intervened=True))
        with self.assertRaisesRegex(ValueError, "intervention"):
            load_episode(entry, self.conditions)

    def test_held_action_and_timestamps_are_bound(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        episode=load_episode(entry,self.conditions)
        self.assertEqual(episode["observations"].shape[0],2)
        self.mutate(entry,lambda rows:rows[4].update(desired_normal_velocity_m_s=.001))
        with self.assertRaisesRegex(ValueError, "held action"):
            load_episode(entry,self.conditions)

    def test_asymmetric_action_domain_accepts_outbound_four_not_inbound_three(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        self.mutate(entry,lambda rows:[row.update(desired_normal_velocity_m_s=.004,applied_normal_velocity_m_s=.003) for row in rows])
        result=load_episode(entry,self.conditions)
        self.assertTrue(np.allclose(result["actions"],.004))
        self.mutate(entry,lambda rows:[row.update(desired_normal_velocity_m_s=-.003) for row in rows])
        with self.assertRaisesRegex(ValueError,"action space"):
            load_episode(entry,self.conditions)

    def test_slow_family_keeps_fixed_outward_bound_and_scaled_inward_bound(self):
        from sim.auscultation_skill_v1.dataset import load_episode,build_dataset,load_dataset
        self.conditions["conditions"][0]["speed_scale"] = .75
        self.condition_path.write_text(json.dumps(self.conditions))
        entry=self.episode()
        self.mutate(entry,lambda rows:[row.update(desired_normal_velocity_m_s=.004,applied_normal_velocity_m_s=.004) for row in rows])
        self.assertTrue(np.allclose(load_episode(entry,self.conditions)["actions"],.004))
        output=self.root/"slow_data"
        build_dataset([entry,self.episode(1)],output,self.condition_path)
        arrays,_=load_dataset(output)
        self.assertTrue(np.allclose(arrays["actions"][arrays["split"]==0],.004))
        for action in (-.0015,.004):
            self.mutate(entry,lambda rows:[row.update(desired_normal_velocity_m_s=action,applied_normal_velocity_m_s=action) for row in rows])
            load_episode(entry,self.conditions)
        for action in (-.00151,.00401):
            self.mutate(entry,lambda rows:[row.update(desired_normal_velocity_m_s=action) for row in rows])
            with self.assertRaisesRegex(ValueError,"action space"):
                load_episode(entry,self.conditions)

    def test_train_only_scaler_and_duplicate_family_guard(self):
        from sim.auscultation_skill_v1.dataset import build_dataset, load_dataset
        entries=[self.episode(i) for i in range(3)]
        output=self.root/"data"
        manifest=build_dataset(entries,output,self.condition_path)
        arrays,loaded=load_dataset(output)
        self.assertEqual(set(arrays["split"].tolist()),{0,1,2})
        expected=arrays["observations"][arrays["split"]==0].reshape(2,-1).mean(0)
        np.testing.assert_allclose(manifest["scaler"]["mean"],expected)
        self.assertEqual(manifest["schema"],loaded["schema"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            build_dataset(entries+[entries[0]],self.root/"duplicate",self.condition_path)
        Path(entries[0]["quality_report"]).write_text("{}")
        with self.assertRaisesRegex(ValueError,"source"):
            load_dataset(output)

    def test_missing_or_old_action_scaling_is_not_a_v4_dataset(self):
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset
        from sim.auscultation_skill_v1.observations import ACTION_SCALING
        entries=[self.episode(i) for i in range(2)]; output=self.root/"data"
        manifest=build_dataset(entries,output,self.condition_path)
        self.assertEqual(manifest["action_scaling"],ACTION_SCALING)
        for value in (None,"both_directions_scaled"):
            changed=copy.deepcopy(manifest)
            if value is None: changed.pop("action_scaling")
            else: changed["action_scaling"]=value
            (output/"manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"schema"):
                load_dataset(output,verify_sources=False)
        changed=copy.deepcopy(manifest); changed["episodes"][0].pop("action_scaling")
        (output/"manifest.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"action scaling"):
            load_dataset(output,verify_sources=False)

    def test_future_observation_and_omitted_delay_rejected(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        self.mutate(entry,lambda rows:rows[12].update(observation_time_s=.1,hall_sample_time_s=.1))
        with self.assertRaisesRegex(ValueError,"delay"):
            load_episode(entry,self.conditions)

    def test_declared_delay_must_be_realized(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        self.conditions["conditions"][0]["observation_delay_steps"]=1
        entry=self.episode()
        with self.assertRaisesRegex(ValueError,"realized"):
            load_episode(entry,self.conditions)

    def test_first_approach_and_terminal_decisions_are_supervised_at_actual_duration(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        from sim.auscultation_skill_v1.observations import FRAME_FEATURES,encode
        entry=self.episode()
        self.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
        result=load_episode(entry,self.conditions)
        self.assertEqual(len(result["actions"]),3)
        np.testing.assert_array_equal(result["decision_steps"],[1,12,11])
        np.testing.assert_allclose(result["decision_duration_s"],np.asarray([1,12,11])/240)
        np.testing.assert_array_equal(result["physics_ranges"],[[0,1],[1,13],[13,24]])
        self.assertEqual(result["decision_reason"].tolist(),["phase_boundary","full_period","successful_terminal"])
        np.testing.assert_array_equal(result["is_first_decision"],[True,False,False])
        np.testing.assert_array_equal(result["observations"][0],np.tile(encode(observation()),(4,1)))
        phase=FRAME_FEATURES.index("phase_APPROACH")
        np.testing.assert_array_equal(result["observations"][:,0,phase],[1,1,1])
        self.assertAlmostEqual(float(result["actions"][0,0]),-.001,places=8)

    def test_unexplained_short_group_and_unproven_terminal_are_rejected(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        self.set_decisions(entry,[("APPROACH",1,-.001),("APPROACH",12,-.001),("DONE",11,.004)])
        with self.assertRaisesRegex(ValueError,"short decision"):
            load_episode(entry,self.conditions)
        self.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("CONTACT",11,-.001)])
        with self.assertRaisesRegex(ValueError,"terminal|short decision"):
            load_episode(entry,self.conditions)

    def test_issued_dataset_roundtrip_rejects_old_semantics_and_false_short_reason(self):
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset,SAMPLE_SEMANTICS,SAMPLE_WEIGHTING
        entries=[self.episode(i) for i in range(2)]
        for entry in entries:
            self.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
        output=self.root/"issued_data"; manifest=build_dataset(entries,output,self.condition_path)
        arrays,_=load_dataset(output)
        self.assertEqual(manifest["schema"],"auscultation_scalar_dataset_v2")
        self.assertEqual(manifest["sample_semantics"],SAMPLE_SEMANTICS)
        self.assertEqual(manifest["sample_weighting"],SAMPLE_WEIGHTING)
        self.assertEqual(manifest["decision_counts"],dict(issued=6,full_period=2,partial=4,first_per_target=2))
        for field,value in (("schema","auscultation_scalar_dataset_v1"),("sample_semantics",None),
                            ("sample_semantics","complete_12_steps_only"),("sample_weighting","duration_weighted")):
            changed=copy.deepcopy(manifest)
            if value is None: changed.pop(field)
            else: changed[field]=value
            (output/"manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"schema"):
                load_dataset(output,verify_sources=False)
        arrays["decision_reason"][0]="successful_terminal"
        np.savez_compressed(output/"dataset.npz",**arrays)
        manifest["arrays_sha256"]=sha((output/"dataset.npz").read_bytes())
        (output/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,"actual issued decision"):
            load_dataset(output)

    def test_rebinding_npz_split_cannot_move_heldout_labels_into_train(self):
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset
        entries=[self.episode(i) for i in range(3)]; output=self.root/"data"
        manifest=build_dataset(entries,output,self.condition_path)
        arrays,_=load_dataset(output)
        for key in ("episode_index","split"):
            old=arrays[key].copy()
            arrays[key][old==0]=2; arrays[key][old==2]=0
        train=arrays["observations"][arrays["split"]==0].reshape(2,-1)
        mean,std=train.mean(0,dtype=np.float64),train.std(0,dtype=np.float64)
        manifest["scaler"].update(mean=mean.tolist(),std=np.where(std<1e-6,1.,std).tolist())
        np.savez_compressed(output/"dataset.npz",**arrays)
        manifest["arrays_sha256"]=sha((output/"dataset.npz").read_bytes())
        (output/"manifest.json").write_text(json.dumps(manifest))
        for refresh in (True,False):
            with self.subTest(refresh=refresh),self.assertRaisesRegex(ValueError,"episode.*split.*binding"):
                load_dataset(output,verify_sources=refresh)

    def test_rebound_duplicate_episode_cannot_duplicate_issued_supervision(self):
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset,SAMPLE_ARRAYS,decision_counts
        entries=[self.episode(i) for i in range(2)]; output=self.root/"data"
        manifest=build_dataset(entries,output,self.condition_path); arrays,_=load_dataset(output)
        original={key:value.copy() for key,value in arrays.items()}; mask=original["episode_index"]==0
        manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
        for key in SAMPLE_ARRAYS:
            arrays[key]=np.concatenate([original[key],original[key][mask]])
        arrays["episode_index"]=np.r_[original["episode_index"],np.full(2,2,np.int32)]
        arrays["split"]=np.r_[original["split"],np.zeros(2,np.uint8)]
        train=arrays["observations"][arrays["split"]==0].reshape(4,-1)
        mean,std=train.mean(0,dtype=np.float64),train.std(0,dtype=np.float64)
        manifest["scaler"].update(mean=mean.tolist(),std=np.where(std<1e-6,1.,std).tolist())
        manifest.update(sample_count=len(arrays["actions"]),decision_counts=decision_counts(arrays))
        np.savez_compressed(output/"dataset.npz",**arrays)
        manifest["arrays_sha256"]=sha((output/"dataset.npz").read_bytes())
        (output/"manifest.json").write_text(json.dumps(manifest))
        for refresh in (True,False):
            with self.subTest(refresh=refresh),self.assertRaisesRegex(ValueError,"duplicate"):
                load_dataset(output,verify_sources=refresh)

    def test_phase_refresh_must_match_executed_physics_phase(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        self.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
        self.mutate(entry,lambda rows:rows[1].update(phase="APPROACH"))
        with self.assertRaisesRegex(ValueError,"phase trace"):
            load_episode(entry,self.conditions)
        self.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
        root=Path(entry["episode_dir"]); path=root/"outcome.json"; outcome=json.loads(path.read_text())
        outcome["points"][0]["contact_end_physics_row"]=72
        path.write_text(json.dumps(outcome)); q=Path(entry["quality_report"]); report=json.loads(q.read_text())
        report["artifact_sha256"][path.name]=sha(path.read_bytes()); q.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,"terminal|short decision"):
            load_episode(entry,self.conditions)

    def test_eligible_old_executor_rejected_even_with_unchanged_config_contract(self):
        from sim.auscultation_skill_v1.dataset import load_episode
        entry=self.episode()
        self.change_captured_source(entry,"executor.py")
        with self.assertRaisesRegex(ValueError,"executor source"):
            load_episode(entry,self.conditions)

    def test_run_and_checker_source_drift_cannot_reuse_qualified_data(self):
        from sim.auscultation_skill_v1.dataset import load_episode,EXECUTOR_CODE_FILES
        self.assertEqual(len(EXECUTOR_CODE_FILES),8)
        for index,name in enumerate(("run.py","certified_collision.py")):
            with self.subTest(source=name):
                self.assertIn(name,EXECUTOR_CODE_FILES)
                entry=self.episode(index)
                self.change_captured_source(entry,name)
                with self.assertRaisesRegex(ValueError,"executor source"):
                    load_episode(entry,self.conditions)

    def change_captured_source(self,entry,name):
        # Deliberately rebind every byte hash: only live-vs-qualified source
        # identity can reject this internally consistent old implementation.
        root=Path(entry["episode_dir"])
        manifest=root/"input_snapshot/manifest.json"; snapshot=json.loads(manifest.read_text())
        original=snapshot["files"]["source/contact_skill/"+name]["sha256"]
        changed=(root/"input_snapshot/objects"/original).read_bytes()+b"\n# alternate executor implementation\n"
        digest=sha(changed); (root/"input_snapshot/objects"/digest).write_bytes(changed)
        snapshot["files"]["source/contact_skill/"+name]["sha256"]=digest
        manifest.write_text(json.dumps(snapshot))
        outcome_path=root/"outcome.json"; outcome=json.loads(outcome_path.read_text())
        outcome["input_snapshot_manifest_sha256"]=sha(manifest.read_bytes()); outcome_path.write_text(json.dumps(outcome))
        quality_path=Path(entry["quality_report"]); quality=json.loads(quality_path.read_text())
        quality["artifact_sha256"].update({"input_snapshot/manifest.json":sha(manifest.read_bytes()),
                                         "input_snapshot/objects/"+digest:digest,"outcome.json":sha(outcome_path.read_bytes())})
        quality_path.write_text(json.dumps(quality))

    def test_dataset_load_checks_executor_even_without_external_source_refresh(self):
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset,current_executor_inventory
        entries=[self.episode(i) for i in range(3)]; output=self.root/"data"
        build_dataset(entries,output,self.condition_path)
        changed=current_executor_inventory(); changed["executor.py"]="b"*64
        with patch("sim.auscultation_skill_v1.dataset.current_executor_inventory",return_value=changed):
            with self.assertRaisesRegex(ValueError,"executor source"):
                load_dataset(output,verify_sources=False)


class PolicyTests(unittest.TestCase):
    def test_policy_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("sim.auscultation_skill_v1.policy"))

    def test_policy_output_scales_only_inward_and_is_continuous_at_zero(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch absent from this local interpreter")
        from sim.auscultation_skill_v1.policy import NormalVelocityPolicy
        from sim.auscultation_skill_v1.observations import encode
        obs=observation(); obs["speed_scale"]=.75
        x=torch.from_numpy(np.tile(encode(obs),(1,4,1)))
        model=NormalVelocityPolicy()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.network[-1].bias.fill_(-100.)
            self.assertAlmostEqual(float(model(x)[0,0]),-.002*.75,places=8)
            model.network[-1].bias.fill_(100.)
            self.assertAlmostEqual(float(model(x)[0,0]),.004,places=8)
            model.network[-1].bias.zero_()
            self.assertEqual(float(model(x)[0,0]),0.)

    def test_training_accepts_slow_outward_endpoint_and_rejects_excess_inward(self):
        from sim.auscultation_skill_v1.policy import NormalVelocityPolicy
        from sim.auscultation_skill_v1.train import fit
        from sim.auscultation_skill_v1.observations import encode
        obs=observation(); obs["speed_scale"] = .75
        x=np.tile(encode(obs),(2,4,1)); y=np.asarray([[-.0015],[.004]],np.float32)
        result=fit(NormalVelocityPolicy(hidden=(8,)),x,y,x,y,device="cpu",epochs=1)
        self.assertEqual(result["best_epoch"],1)
        for value in (None,"both_directions_scaled"):
            saved=copy.deepcopy(result["resume"])
            if value is None: saved.pop("action_scaling")
            else: saved["action_scaling"]=value
            with self.assertRaisesRegex(ValueError,"resume checkpoint action"):
                fit(NormalVelocityPolicy(hidden=(8,)),x,y,x,y,device="cpu",epochs=2,resume_state=saved)
        for field in ("sample_semantics","sample_weighting"):
            saved=copy.deepcopy(result["resume"]); saved.pop(field)
            with self.assertRaisesRegex(ValueError,"resume checkpoint sample semantics"):
                fit(NormalVelocityPolicy(hidden=(8,)),x,y,x,y,device="cpu",epochs=2,resume_state=saved)
        y[0,0]=-.002
        with self.assertRaisesRegex(ValueError,"supervised values"):
            fit(NormalVelocityPolicy(hidden=(8,)),x,y,x,y,device="cpu",epochs=1)

    def test_actual_optimization_bounded_output_and_no_cuda_fallback(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch absent from this local interpreter")
        from sim.auscultation_skill_v1.policy import NormalVelocityPolicy
        from sim.auscultation_skill_v1.train import fit, resolve_device
        from sim.auscultation_skill_v1.observations import encode, FRAME_FEATURES
        torch.set_num_threads(1); torch.manual_seed(7)
        base=encode(observation()); x=np.tile(base,(24,4,1))
        x[:12,:,0]=.04; x[12:,:,0]=.005
        y=np.r_[np.full((12,1),-.001),np.full((12,1),.001)].astype(np.float32)
        model=NormalVelocityPolicy(hidden=(16,16))
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=fit(model,x,y,x.copy(),y.copy(),device="cpu",epochs=50,batch_size=8,learning_rate=.02,seed=7)
        prediction=model(torch.from_numpy(x)).detach().numpy()
        self.assertLess(np.mean((prediction-y)**2),np.mean(y**2)*.4)
        self.assertGreaterEqual(np.min(prediction),-.002)
        self.assertLessEqual(np.max(prediction),.004)
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertGreater(result["best_epoch"],0)
        self.assertIn("optimizer_state_dict",result["resume"])
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError,"CUDA"):
                resolve_device("cuda")

    def test_training_cli_checkpoint_frozen_reload_and_resume(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch absent from this local interpreter")
        from sim.auscultation_skill_v1.dataset import build_dataset,load_dataset,EXECUTOR_CODE_FILES,SAMPLE_SEMANTICS,SAMPLE_WEIGHTING
        from sim.auscultation_skill_v1.train import main
        from sim.auscultation_skill_v1.policy import load_policy
        from sim.auscultation_skill_v1.observations import ACTION_SCALING
        fixture=DatasetTests(methodName="test_train_only_scaler_and_duplicate_family_guard")
        fixture.setUp()
        try:
            entries=[fixture.episode(i) for i in range(3)]
            for entry in entries:
                fixture.set_decisions(entry,[("APPROACH",1,-.001),("CONTACT",12,-.001),("DONE",11,.004)])
            dataset=fixture.root/"data"; output=fixture.root/"model"
            build_dataset(entries,dataset,fixture.condition_path)
            args=["--dataset",str(dataset),"--output",str(output),"--device","cpu","--epochs","3"]
            self.assertEqual(main(args),0)
            manifest_bytes=(output/"run_manifest.json").read_bytes(); run=json.loads(manifest_bytes)
            self.assertIs(run["test_evaluated"],False)
            self.assertEqual(run["action_scaling"],ACTION_SCALING)
            self.assertEqual(run["sample_semantics"],SAMPLE_SEMANTICS)
            self.assertEqual(run["sample_weighting"],SAMPLE_WEIGHTING)
            self.assertEqual(run["decision_counts_by_split"]["train"],
                             dict(issued=3,full_period=1,partial=2,first_per_target=1))
            for name in EXECUTOR_CODE_FILES:
                self.assertIn(name,run["code_sha256"])
                self.assertEqual(run["executor_code_sha256"][name],run["code_sha256"][name])
            actor=load_policy(output,sha(manifest_bytes),"a"*64)
            arrays,_=load_dataset(dataset)
            action=actor.predict(arrays["observations"][0])
            self.assertTrue(np.isfinite(action)); self.assertGreaterEqual(action,-.002); self.assertLessEqual(action,.004)
            self.assertFalse(any(p.requires_grad for p in actor.policy.parameters()))
            with self.assertRaisesRegex(ValueError,"contract"):
                load_policy(output,sha(manifest_bytes),"b"*64)
            original_read=Path.read_bytes
            actuator=Path(__file__).parents[1]/"executor.py"
            def changed_executor(path):
                data=original_read(path)
                return data+b"\n# runtime actuator changed\n" if path==actuator else data
            with patch.object(Path,"read_bytes",autospec=True,side_effect=changed_executor):
                with self.assertRaisesRegex(ValueError,"source hash changed: executor.py"):
                    load_policy(output,sha(manifest_bytes),"a"*64)
            resumed=fixture.root/"resumed"
            self.assertEqual(main(["--dataset",str(dataset),"--output",str(resumed),"--device","cpu",
                                   "--epochs","4","--resume",str(output)]),0)
            self.assertEqual(len(json.loads((resumed/"run_manifest.json").read_text())["history"]),4)
            saved_path=output/"policy.pt"; original_checkpoint=saved_path.read_bytes()
            for value in (None,"both_directions_scaled"):
                saved_path.write_bytes(original_checkpoint)
                changed=copy.deepcopy(run)
                if value is None: changed.pop("action_scaling")
                else: changed["action_scaling"]=value
                manifest_path=output/"run_manifest.json"; manifest_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"action scaling"):
                    load_policy(output,sha(manifest_path.read_bytes()),"a"*64)
                with self.assertRaisesRegex(ValueError,"resume changes"):
                    main(["--dataset",str(dataset),"--output",str(fixture.root/"invalid_resume"),"--device","cpu",
                          "--epochs","4","--resume",str(output)])
                checkpoint=torch.load(saved_path,map_location="cpu",weights_only=True)
                self.assertEqual(checkpoint["action_scaling"],ACTION_SCALING)
                if value is None: checkpoint.pop("action_scaling")
                else: checkpoint["action_scaling"]=value
                torch.save(checkpoint,saved_path)
                changed=copy.deepcopy(run); changed["checkpoint_sha256"]=sha(saved_path.read_bytes())
                manifest_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"checkpoint observation/action"):
                    load_policy(output,sha(manifest_path.read_bytes()),"a"*64)
            for field in ("sample_semantics","sample_weighting"):
                for value in (None,"complete_12_steps_only"):
                    saved_path.write_bytes(original_checkpoint)
                    changed=copy.deepcopy(run)
                    if value is None: changed.pop(field)
                    else: changed[field]=value
                    manifest_path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,"sample semantics"):
                        load_policy(output,sha(manifest_path.read_bytes()),"a"*64)
                    with self.assertRaisesRegex(ValueError,"resume changes"):
                        main(["--dataset",str(dataset),"--output",str(fixture.root/"invalid_resume"),"--device","cpu",
                              "--epochs","4","--resume",str(output)])
                    checkpoint=torch.load(saved_path,map_location="cpu",weights_only=True)
                    if value is None: checkpoint.pop(field)
                    else: checkpoint[field]=value
                    torch.save(checkpoint,saved_path)
                    changed=copy.deepcopy(run); changed["checkpoint_sha256"]=sha(saved_path.read_bytes())
                    manifest_path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,"checkpoint observation/action"):
                        load_policy(output,sha(manifest_path.read_bytes()),"a"*64)
            saved_path.write_bytes(original_checkpoint); (output/"run_manifest.json").write_bytes(manifest_bytes)
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
