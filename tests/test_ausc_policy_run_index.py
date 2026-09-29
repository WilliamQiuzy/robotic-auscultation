import json
from pathlib import Path
import tempfile
import unittest
from sim.ausc_policy.run_index import record


class RunIndexTests(unittest.TestCase):
    def recovery_record(self, passed):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'output').mkdir()
            (path/'result.json').write_text(json.dumps(dict(run_id='expert-recovery',
                config=dict(experiment='recovery_run'),completed=True,gpu_released=True,
                stage=dict(exit_code=0))))
            (path/'output/COMPLETE.json').write_text(json.dumps(dict(completed=True,
                all_cases_passed=passed,m0_passed=False,learned_policy=False,
                training_dataset_eligible=False,cases=[dict(case_id='history20',case_passed=passed,
                    failure_reason=None if passed else 'oracle abstained')])))
            return record(path)

    def test_expert_success_is_distinct_from_model_and_training_acceptance(self):
        row=self.recovery_record(True)
        self.assertTrue(row['case_passed'])
        self.assertEqual(row['expert_cases'],1)
        self.assertEqual(row['expert_cases_passed'],1)
        self.assertFalse(row['m0_passed'])
        self.assertFalse(row['learned_policy'])
        self.assertFalse(row['training_dataset_eligible'])
        self.assertIsNone(row['closed_loop_evaluated'])

    def test_completed_expert_failure_preserves_case_reason(self):
        row=self.recovery_record(False)
        self.assertTrue(row['completed'])
        self.assertFalse(row['case_passed'])
        self.assertEqual(row['expert_cases_passed'],0)
        self.assertEqual(row['failure'],'history20: oracle abstained')

    def test_clean_execution_cannot_hide_failed_contact(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'output').mkdir()
            (path/'result.json').write_text(json.dumps(dict(run_id='actual-case',config=dict(experiment='isaac_run'),
                completed=True,gpu_released=True,stage=dict(exit_code=0))))
            (path/'output/COMPLETE.json').write_text(json.dumps(dict(completed=True,case_passed=False,
                m0_passed=False,closed_loop_evaluated=True,failure_reason='handoff failed')))
            row=record(path)
            self.assertTrue(row['completed']);self.assertTrue(row['closed_loop_evaluated'])
            self.assertFalse(row['case_passed']);self.assertFalse(row['m0_passed'])
            self.assertEqual(row['failure'],'handoff failed')


if __name__=='__main__':unittest.main()
