import unittest
import json
from pathlib import Path
import tempfile
from sim.auscultation_skill_v1.run import bind_actual_timing, verify_worker_completion, sha256_file
from sim.auscultation_skill_v1.run import graceful_simulation_shutdown
from types import SimpleNamespace
from unittest.mock import Mock


class RunMetadataTests(unittest.TestCase):
    def test_entrypoint_timing_override_cannot_mislabel_actor(self):
        outcome={'variation':{'free_approach_far_speed_m_s':.012,'retraction_speed_m_s':.01,
                              'fault_retraction_speed_m_s':.005,'joint_command_speed_limit_rad_s':.25},
                 'actual_contact_policy':{'teacher_approach_speed_m_s':-.002,'teacher_unload_speed_m_s':.004}}
        bind_actual_timing(outcome)
        self.assertEqual(outcome['variation']['free_approach_far_speed_m_s'],.002)
        self.assertEqual(outcome['variation']['retraction_speed_m_s'],.004)
        self.assertEqual(outcome['variation']['joint_command_speed_limit_rad_s'],.25)
        self.assertNotIn('fault_retraction_speed_m_s',outcome['variation'])


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.root=Path(self.temporary.name)
        (self.root/'input_snapshot').mkdir()
        (self.root/'input_snapshot/manifest.json').write_text('{}')
        self.snapshot=sha256_file(self.root/'input_snapshot/manifest.json')
        (self.root/'outcome.json').write_text(json.dumps(dict(reset_id='test-reset',
            input_snapshot_manifest_sha256=self.snapshot)))

    def tearDown(self): self.temporary.cleanup()

    def receipt(self,code):
        (self.root/'worker_completion.json').write_text(json.dumps(dict(
            schema='auscultation_skill_worker_completion_v1',returncode=code,reset_id='test-reset',
            close_returned=True,completion_stage='post_app_close_pre_process_exit',
            exit_policy='controlled_os_exit_after_close',
            outcome_sha256=sha256_file(self.root/'outcome.json'),input_snapshot_manifest_sha256=self.snapshot)))

    def test_saved_outcome_without_worker_exit_is_insufficient(self):
        self.assertEqual(verify_worker_completion(self.root,0)['status'],'failed')

    def test_completed_rejected_negative_has_lifecycle_evidence(self):
        self.receipt(1)
        self.assertEqual(verify_worker_completion(self.root,1)['status'],'passed')
        self.assertEqual(verify_worker_completion(self.root,None)['status'],'failed')
        self.assertEqual(verify_worker_completion(self.root,0)['status'],'failed')

    def test_lifecycle_binds_exact_final_outcome(self):
        self.receipt(0)
        self.assertEqual(verify_worker_completion(self.root,0)['status'],'passed')
        (self.root/'outcome.json').write_text(json.dumps(dict(reset_id='test-reset',
            input_snapshot_manifest_sha256=self.snapshot,success=True)))
        self.assertEqual(verify_worker_completion(self.root,0)['status'],'failed')

class GracefulShutdownTests(unittest.TestCase):
    def test_forces_returning_close_without_mutating_callers_configuration(self):
        factory=Mock(); module=SimpleNamespace(SimulationApp=factory)
        configuration={'headless':True,'fast_shutdown':True}
        with graceful_simulation_shutdown(module) as lifecycle:
            module.SimulationApp(configuration,experience='test.kit')
        self.assertEqual(lifecycle,{'apps_created':1})
        factory.assert_called_once_with({'headless':True,'fast_shutdown':False},experience='test.kit')
        self.assertTrue(configuration['fast_shutdown'])
        self.assertIs(module.SimulationApp,factory)

    def test_restores_factory_after_worker_error(self):
        factory=Mock(); module=SimpleNamespace(SimulationApp=factory)
        with self.assertRaisesRegex(RuntimeError,'worker failed'):
            with graceful_simulation_shutdown(module):
                raise RuntimeError('worker failed')
        self.assertIs(module.SimulationApp,factory)
