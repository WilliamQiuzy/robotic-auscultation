import unittest
from sim.ausc_policy.recovery_run import selected_cases


class RecoverySelectionTests(unittest.TestCase):
    def test_finite_subset_preserves_predeclared_order(self):
        self.assertEqual(selected_cases({}),[('nominal',0.),('joint3-plus',.02),('joint3-minus',-.02)])
        self.assertEqual(selected_cases(dict(case_ids=['nominal'])),[('nominal',0.)])

    def test_empty_unknown_duplicates_and_reordering_rejected(self):
        for cases in ([],['R5'],['nominal','nominal'],['joint3-minus','nominal'],'nominal'):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                selected_cases(dict(case_ids=cases))


if __name__ == '__main__': unittest.main()
