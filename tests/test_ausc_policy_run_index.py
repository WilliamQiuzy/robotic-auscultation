import json
from pathlib import Path
import tempfile
import unittest
from sim.ausc_policy.run_index import record


class RunIndexTests(unittest.TestCase):
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
