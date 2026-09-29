import copy
import unittest
from sim.ausc_policy.evaluation_audit import interpret


class EvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.outcome=dict(success=True,expert_demonstration=False,training_dataset_eligible=False,
            training_accepted=False,m0_passed=False,points=[dict(target='R4',success=True,training_accepted=False)])
        self.audit=dict(accepted=False,audits=dict(mechanics=dict(accepted=False,reasons=['R4:controller_rejected']),
            posture=dict(accepted=False,reasons=['controller_did_not_accept'])))

    def test_only_eligibility_is_separated_without_changing_original(self):
        before=copy.deepcopy(self.audit)
        result=interpret(self.audit,self.outcome)
        self.assertTrue(result['accepted']);self.assertFalse(result['training_dataset_eligible'])
        self.assertEqual(before,self.audit);self.assertFalse(self.outcome['training_accepted'])

    def test_force_source_posture_and_trace_failures_cannot_be_excluded(self):
        for kind,reason in [('mechanics','physical_force_limit'),('mechanics','trace_or_source_binding:bad'),
                            ('mechanics','uncovered_trace_rows'),('posture','actual_replay_elbow_height')]:
            changed=copy.deepcopy(self.audit);changed['audits'][kind]['reasons'].append(reason)
            result=interpret(changed,self.outcome)
            self.assertFalse(result['accepted']);self.assertIn(reason,result['evaluation_audits'][kind]['reasons'])

    def test_incomplete_controller_and_unexplained_rejection_stay_rejected(self):
        self.outcome['success']=False
        self.assertFalse(interpret(self.audit,self.outcome)['accepted'])
        self.outcome['success']=True;self.audit['audits']['mechanics']['reasons']=[]
        self.assertFalse(interpret(self.audit,self.outcome)['accepted'])

    def test_other_points_rejection_cannot_be_ignored(self):
        self.audit['audits']['mechanics']['reasons'].append('L4:controller_rejected')
        self.assertFalse(interpret(self.audit,self.outcome)['accepted'])

    def test_explicit_expert_diagnostic_retains_all_physics_requirements(self):
        self.outcome['expert_demonstration'] = True
        with self.assertRaises(ValueError): interpret(self.audit,self.outcome)
        self.assertTrue(interpret(self.audit,self.outcome,expert_evaluation=True)['accepted'])
        self.audit['audits']['mechanics']['reasons'].append('bad_force')
        self.assertFalse(interpret(self.audit,self.outcome,expert_evaluation=True)['accepted'])


if __name__=='__main__':unittest.main()
