import copy
import hashlib
import json
import unittest
from . import contract


def fixture():
    fit={'normal_world':[0.,-1.,0.],'wrist_rotation_world':[[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]],'gap_m':.005}
    row={'name':'R1','success':True,'position_world':[0.,1.,1.5],'normal_world':[0.,-1.,0.],
         'posture_limits':{'joint_lower_rad':[-1]*7,'joint_upper_rad':[1]*7},
         'loaded_compression_m':.002,'free_approach_start_gap_m':.025,'free_approach_end_gap_m':.015,
         'geometric_clearance_m':.003,'loaded_noncontact_clearance_m':.0003,'joint_edge_sample_step_rad':.015,
         'selected_audit_candidate_index':0,'selected_fit':fit,'command_normal_world':[0.,-1.,0.],
         'command_rotation_world':fit['wrist_rotation_world'],'expected_contact_gap_m':.005,
         'expected_contact_center_world':[0.,.995,1.5]}
    plan={'points':[row],'world_from_left_base':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
          'targets_file':'contact_targets.json','audit_file':'audit.json','provenance':{'audit_file':'audit.json','audit_sha256':'a'*64}}
    alternative=copy.deepcopy(fit);alternative['wrist_rotation_world']=[[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]]
    return plan,{'name':'R1','compression_m':.002,'candidates':[fit,alternative]}


class ContractTests(unittest.TestCase):
    def validate(self,candidate,reference,audit):
        self.assertTrue(callable(getattr(contract,'validate_candidate_plan',None)), 'validator must exist')
        return contract.validate_candidate_plan(candidate,reference,audit)

    def test_same_normal_audited_roll_can_change(self):
        ref,audit=fixture();candidate=copy.deepcopy(ref);row=candidate['points'][0]
        row['selected_audit_candidate_index']=1;row['selected_fit']=audit['candidates'][1]
        row['command_rotation_world']=row['selected_fit']['wrist_rotation_world']
        self.assertIsNone(self.validate(candidate,ref,audit))

    def test_point_position_and_normal_are_fixed(self):
        for key in ('position_world','normal_world','command_normal_world'):
            ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0][key][0]+=.001
            with self.assertRaises(ValueError,msg=key):self.validate(candidate,ref,audit)

    def test_gap_and_compression_are_fixed(self):
        for key in ('loaded_compression_m','free_approach_start_gap_m','expected_contact_gap_m'):
            ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0][key]+=.001
            with self.assertRaises(ValueError,msg=key):self.validate(candidate,ref,audit)

    def test_weaker_envelope_and_clearance_rejected(self):
        for key in ('geometric_clearance_m','loaded_noncontact_clearance_m','joint_edge_sample_step_rad'):
            ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0][key]*=.5
            with self.assertRaises(ValueError,msg=key):self.validate(candidate,ref,audit)
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0]['posture_limits']['joint_lower_rad'][0]-=.1
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)

    def test_audit_candidate_must_be_exact(self):
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0]['selected_fit']['gap_m']=0.
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0]['selected_audit_candidate_index']=True
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)

    def test_base_provenance_and_unapproved_point_rejected(self):
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['world_from_left_base'][0][3]+=.05
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['provenance']['audit_sha256']='b'*64
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)
        ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0]['name']='R2'
        with self.assertRaises(ValueError):self.validate(candidate,ref,audit)

    def test_geometry_cannot_claim_physics_acceptance(self):
        for key in ('training_accepted','contact_success'):
            ref,audit=fixture();candidate=copy.deepcopy(ref);candidate['points'][0][key]=True
            with self.assertRaises(ValueError):self.validate(candidate,ref,audit)

    def receipt_fixture(self):
        self.assertTrue(callable(getattr(contract,'validate_geometry_receipt',None)),
                        'execution must consume the complete independent receipt')
        reference,_=fixture()
        for label in ('audit','targets','runtime','collision_metadata','collision_arrays','kinematics_urdf'):
            reference['provenance'][label+'_sha256']='a'*64
        fixed_contract={'fixed_files':{contract.COLLISION_BINDING:'b'*64}}
        before=b'{"fresh":"plan"}'
        receipt={'accepted':True,'independent_upper_pair_check':True,
            'schema':'upper_pair_independent_geometry_v1','point':'R1',
            'input_plan_sha256':hashlib.sha256(before).hexdigest(),
            'collision_regeneration_sha256':'b'*64,
            'collider_receipt':{'implementation':'sim.curobo_expert_v1.fast_collision.FastCollisionChecker',
                'urdf_planning_margin_deg':3.,'collider_count':2,'pair_count':1,'dynamic_collider_count':1,
                'contract_sha256':'c'*64,
                'inputs_sha256':{'plan':hashlib.sha256(before).hexdigest(),
                    **{label:'a'*64 for label in ('audit','targets','runtime','collision_metadata','collision_arrays','kinematics_urdf')}},
                'sources_sha256':{relative:contract.sha(contract.PROJECT/relative)
                                  for relative in contract.REVALIDATION_SOURCES}}}
        return receipt,reference,fixed_contract,before

    def test_receipt_binds_before_plan_contract_inputs_and_checker_sources(self):
        evidence,reference,fixed_contract,before=self.receipt_fixture()
        contract.validate_geometry_receipt(evidence,'R1',reference,fixed_contract,hashlib.sha256(before).hexdigest(),'c'*64)
        changes=[('schema','old_schema'),('point','L1'),('collision_regeneration_sha256','d'*64)]
        for key,value in changes:
            changed=copy.deepcopy(evidence);changed[key]=value
            with self.assertRaises(ValueError,msg=key):
                contract.validate_geometry_receipt(changed,'R1',reference,fixed_contract,hashlib.sha256(before).hexdigest(),'c'*64)
        for key in ('contract_sha256','inputs_sha256','sources_sha256'):
            changed=copy.deepcopy(evidence)
            changed['collider_receipt'][key]='d'*64 if key=='contract_sha256' else {}
            with self.assertRaises(ValueError,msg=key):
                contract.validate_geometry_receipt(changed,'R1',reference,fixed_contract,hashlib.sha256(before).hexdigest(),'c'*64)
        changed=copy.deepcopy(evidence)
        first=next(iter(changed['collider_receipt']['sources_sha256']))
        changed['collider_receipt']['sources_sha256'][first]='e'*64
        with self.assertRaisesRegex(ValueError,'source'):
            contract.validate_geometry_receipt(changed,'R1',reference,fixed_contract,hashlib.sha256(before).hexdigest(),'c'*64)

    def test_candidate_change_requires_motion_or_pose_difference(self):
        self.assertTrue(callable(getattr(contract,'candidate_motion_changed',None)), 'genuine change check required')
        ref,_=fixture();candidate=copy.deepcopy(ref)
        candidate['scope']='different words only'
        self.assertFalse(contract.candidate_motion_changed(candidate,ref))
        candidate['points'][0]['loaded_q']=[0.]*7
        ref['points'][0]['loaded_q']=[0.]*7
        candidate['points'][0]['loaded_q'][3]=1e-12
        self.assertFalse(contract.candidate_motion_changed(candidate,ref))
        candidate['points'][0]['loaded_q'][3]=.1
        self.assertTrue(contract.candidate_motion_changed(candidate,ref))

if __name__=='__main__':unittest.main()
