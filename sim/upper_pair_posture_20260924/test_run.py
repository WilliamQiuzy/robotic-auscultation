import unittest
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from . import run
from . import materialize


class RunContractTests(unittest.TestCase):
    def fixture(self):
        contract={'controller_policy':'revised_shortcut','speed_scale':1.,'baseline_phase_config':{'hold_s':3.}}
        outcome={'candidate_policy':'revised_shortcut','speed_scale':1.,'phase_config':{'hold_s':3.},
                 'requested_points':['R1','L1','R2'],'success':True,
                 'points':[{'target':n} for n in ['R1','L1','R2']]}
        return contract,outcome

    def test_exact_controller_and_requested_sequence_required(self):
        contract,outcome=self.fixture()
        run.validate_outcome_contract(outcome,contract,['R1','L1','R2'])
        for key,value in [('candidate_policy','curobo'),('speed_scale',.8),
                          ('phase_config',{'hold_s':2.}),('requested_points',['R1']),('success',False)]:
            changed=dict(outcome);changed[key]=value
            with self.assertRaises(ValueError,msg=key):
                run.validate_outcome_contract(changed,contract,['R1','L1','R2'])

    def test_partial_completion_rejected(self):
        contract,outcome=self.fixture();outcome['points']=outcome['points'][:2]
        with self.assertRaises(ValueError):run.validate_outcome_contract(outcome,contract,['R1','L1','R2'])

    def test_only_matched_screen_or_full_sequence_supported(self):
        self.assertEqual(run.validate_points(['R1','L1','R2']),['R1','L1','R2'])
        for points in (['L1','R1','R2'],['R1'],['R1','L1','R2','L2']):
            with self.assertRaises(ValueError):run.validate_points(points)

    def test_selected_candidate_requires_summary_hash_and_identity_membership(self):
        self.assertTrue(callable(getattr(materialize,'selected_source_record',None)), 'selection membership check required')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();source=root/'search/R1_candidate/planned_paths_R1.json';source.parent.mkdir(parents=True)
            document={'points':[{'name':'R1'}]};raw=json.dumps(document).encode();source.write_bytes(raw)
            summary_path=root/'search/search_summary.json'
            summary={'candidates':[{'id':'R1_candidate','name':'R1','plan_path':'R1_candidate/planned_paths_R1.json',
                                   'plan_sha256':hashlib.sha256(raw).hexdigest()}]}
            contract={'baseline_plans':{'R1':{'sha256':'a'*64}}}
            with patch.object(materialize,'PROJECT',root):
                selected=materialize.selected_source_record('R1',source,raw,summary_path,summary,contract)
                self.assertEqual(selected['source_kind'],'search_candidate')
                changed=copy.deepcopy(summary);changed['candidates'][0]['plan_sha256']='b'*64
                with self.assertRaises(ValueError):
                    materialize.selected_source_record('R1',source,raw,summary_path,changed,contract)
                changed=copy.deepcopy(summary);changed['candidates'][0]['plan_path']='other.json'
                with self.assertRaises(ValueError):
                    materialize.selected_source_record('R1',source,raw,summary_path,changed,contract)
                baseline={'baseline_plans':{'R1':{'sha256':hashlib.sha256(raw).hexdigest()}}}
                selected=materialize.selected_source_record('R1',source,raw,summary_path,{'candidates':[]},baseline)
                self.assertEqual(selected['source_kind'],'pinned_baseline')

    def test_semantic_selected_source_check_rejects_different_before_plan(self):
        self.assertTrue(callable(getattr(materialize,'selected_source_document',None)), 'source-to-before binding required')
        original={'points':[{'name':'R1','loaded_q':[0.]*7}],'revalidation':{'accepted':True}}
        expected={'points':[{'name':'R1','loaded_q':[0.]*7}]}
        self.assertEqual(materialize.selected_source_document(json.dumps(original).encode()),expected)
        # A reserialized source is fine; a different q cannot be represented as that source.
        changed=copy.deepcopy(expected);changed['points'][0]['loaded_q'][3]=.1
        self.assertNotEqual(materialize.selected_source_document(json.dumps(original).encode()),changed)

if __name__=='__main__':unittest.main()
