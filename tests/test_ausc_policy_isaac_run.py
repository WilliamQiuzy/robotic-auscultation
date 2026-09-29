"""Lifecycle and source-selection tests; no Kit or hardware imports."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from sim.ausc_policy import isaac_run as owner
from sim.ausc_policy.isaac_case import approved_assets


class IsaacOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads(Path('configs/ausc_policy/m0_act_d0.json').read_text())
        self.config['reload_run'] = str(self.root)
        (self.root/'output').mkdir()
        self.receipt = dict(passed=True,max_reference_difference_rad=1.19e-7,max_reload_difference_rad=0.,
            receipt=dict(checkpoint_sha256=self.config['checkpoint_sha256']))
        owner.write(self.root/'result.json',dict(completed=True))
        owner.write(self.root/'output/COMPLETE.json',self.receipt)

    def test_actual_reload_required_and_physics_cannot_weaken(self):
        self.assertEqual(owner.validate_config(self.config),self.receipt['receipt'])
        for field,value in [('point','L4'),('checkpoint_step',512),('speed_scale',1.),
                            ('motion_velocity','unbounded'),
                            ('time_limit_seconds',7200),('physics',dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.01))]:
            candidate = copy.deepcopy(self.config);candidate[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):owner.validate_config(candidate)
        owner.write(self.root/'result.json',dict(completed=False))
        with self.assertRaises(ValueError):owner.validate_config(self.config)

    def test_reload_weight_change_and_difference_fail(self):
        for key,value in [('passed',False),('max_reference_difference_rad',2e-6),('max_reload_difference_rad',1e-9)]:
            altered=copy.deepcopy(self.receipt);altered[key]=value
            owner.write(self.root/'output/COMPLETE.json',altered)
            with self.assertRaises(ValueError):owner.validate_config(self.config)
        altered=copy.deepcopy(self.receipt);altered['receipt']['checkpoint_sha256']='0'*64
        owner.write(self.root/'output/COMPLETE.json',altered)
        with self.assertRaises(ValueError):owner.validate_config(self.config)

    def test_service_never_receives_labels_or_probe_arrays(self):
        self.config.update(inputs_npz='/private/input',reference_npz='/private/reference')
        result=owner.service_config(self.config)
        self.assertNotIn('inputs_npz',result);self.assertNotIn('reference_npz',result)
        self.assertEqual(result['mode'],'serve');self.assertEqual(result['port'],0)

    def test_quadratic_variant_requires_actual_bound_probe_with_original_limits(self):
        self.config.update(motion_velocity='quadratic5',motion_probe_run=str(self.root/'probe'))
        with self.assertRaises(FileNotFoundError):owner.validate_config(self.config)
        p=self.root/'probe';(p/'output').mkdir(parents=True)
        owner.write(p/'result.json',dict(completed=True,gpu_released=True))
        metrics=dict(velocity_estimator='quadratic5',branches={key:dict(completed=True) for key in
            ('zero_terminal_velocity','quadratic5_target_velocity')},schedule=dict(
            limits=dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999),
            horizon=10,execute_steps=5,period_s=.05,ticks_per_action=12))
        def save():
            owner.write(p/'output/metrics.json',metrics)
            owner.write(p/'output/COMPLETE.json',dict(comparison_completed=True,
                metrics_sha256=hashlib.sha256((p/'output/metrics.json').read_bytes()).hexdigest()))
        save();self.assertEqual(owner.validate_config(self.config),self.receipt['receipt'])
        metrics['schedule']['limits']['velocity']=.18;save()
        with self.assertRaisesRegex(ValueError,'native quadratic5'):owner.validate_config(self.config)
        metrics['schedule']['limits']['velocity']=.175;save()
        owner.write(p/'output/metrics.json',dict(metrics,velocity_estimator='adjacent'))
        with self.assertRaisesRegex(ValueError,'native quadratic5'):owner.validate_config(self.config)

    def test_posture_preview_requires_source_bound_native_reproduction(self):
        config=dict(posture_hint_preview=True,motion_velocity='quadratic5',motion_preview_probe_run=str(self.root/'posture'))
        with self.assertRaises(FileNotFoundError):owner.validate_posture_preview(config)
        p=self.root/'posture';(p/'output').mkdir(parents=True)
        owner.write(p/'result.json',dict(completed=True,gpu_released=True))
        metrics=dict(motion_preview_source_sha256=hashlib.sha256(Path('sim/ausc_policy/motion_preview.py').read_bytes()).hexdigest(),
            original_failure_reproduced=True,active_state_unchanged=True,preview_vs_generation_max_abs_rad=0.,
            limits=dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999),factors=[1.,.5,.25,.125,0.],ticks=12)
        def save():
            owner.write(p/'output/metrics.json',metrics)
            owner.write(p/'output/COMPLETE.json',dict(preview_passed=True,
                metrics_sha256=hashlib.sha256((p/'output/metrics.json').read_bytes()).hexdigest()))
        save();owner.validate_posture_preview(config)
        metrics['active_state_unchanged']=False;save()
        with self.assertRaisesRegex(ValueError,'native posture'):owner.validate_posture_preview(config)
        with self.assertRaises(ValueError):owner.validate_posture_preview(dict(posture_hint_preview='false'))
        metrics['active_state_unchanged']=True;config['posture_hint_selection']='nearest_reference';save()
        with self.assertRaisesRegex(ValueError,'native posture'):owner.validate_posture_preview(config)
        metrics['selection']='nearest_reference'
        metrics['reference_schedule']=dict(all_branches_completed=True,fixed_actual_ticks=16057,
            fixed_actual_action_intervals=1339,branches=dict(
                zero=dict(original_command_max_abs_difference_rad=0.,joint_mean_abs_error_rad=.002,tcp_mean_distance_mm=1.7),
                nearest_reference=dict(local_zero_comparisons=1339,local_zero_comparisons_all_not_worse=True,
                    joint_mean_abs_error_rad=.0019,tcp_mean_distance_mm=1.6)))
        save();owner.validate_posture_preview(config)
        metrics['reference_schedule']['branches']['nearest_reference']['tcp_mean_distance_mm']=1.8;save()
        with self.assertRaisesRegex(ValueError,'fixed-reference'):owner.validate_posture_preview(config)

    def test_asset_scope_restores_globals_even_on_error(self):
        from sim.auscultation_expert_layout_v1 import scene
        original=scene.ROOT,scene.PROJECT
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with approved_assets(self.root) as bound:
                self.assertEqual(bound.PROJECT,self.root)
                raise RuntimeError('injected')
        self.assertEqual((scene.ROOT,scene.PROJECT),original)

    def test_cleanup_does_not_signal_a_finished_child_and_escalates_only_timeout(self):
        done=Mock();done.poll.return_value=0;done.returncode=0
        self.assertEqual(owner.stop_child(done),0);done.terminate.assert_not_called()
        hung=Mock();hung.poll.return_value=None;hung.returncode=-9
        hung.wait.side_effect=[subprocess.TimeoutExpired('case',15),-9]
        self.assertEqual(owner.stop_child(hung),-9)
        hung.terminate.assert_called_once();hung.kill.assert_called_once()


if __name__=='__main__':unittest.main()
