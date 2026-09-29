"""Preview orchestration sentinels; these fakes do not validate native Ruckig."""
import copy
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


class MotionPreviewTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('sim.ausc_policy.motion_preview')
        except ModuleNotFoundError:
            self.fail('The bounded motion preview helper is not implemented')

    def source(self):
        return SimpleNamespace(dt=1/240,input=SimpleNamespace(
            current_position=[.1]*7,current_velocity=[.03]*7,current_acceleration=[.04]*7,
            target_position=[.2]*7,target_velocity=[.05]*7,target_acceleration=[0.]*7,
            max_velocity=[.175]*7,max_acceleration=[.5]*7,max_jerk=[7.99999]*7),
            set_target=Mock(side_effect=AssertionError('live set_target')),step=Mock(side_effect=AssertionError('live step')))

    def factory(self, instances, *, defect=False, validation_error=False, nonfinite=False):
        class Fake:
            def __init__(self,q,**limits):
                self.dt=limits['dt'];self.limits=limits;self.ticks=0;self.first_state=None
                self.input=SimpleNamespace(current_position=np.array(q).tolist(),current_velocity=[0.]*7,
                    current_acceleration=[0.]*7,target_position=np.array(q).tolist(),target_velocity=[0.]*7,
                    target_acceleration=[0.]*7,max_velocity=[limits['velocity']]*7,
                    max_acceleration=[limits['acceleration']]*7,max_jerk=[limits['jerk']]*7)
                self.validations=[]
                def validate(*args,**kwargs):
                    self.validations.append(kwargs)
                    if validation_error and abs(self.input.target_velocity[0])>.021:
                        raise ValueError('native target validation error')
                    return True
                self.generator=SimpleNamespace(validate_input=validate);instances.append(self)
            def set_target(self,q):self.input.target_position=np.array(q).tolist()
            def step(self):
                if self.first_state is None:
                    self.first_state=copy.deepcopy(vars(self.input))
                self.ticks+=1;q=np.array(self.input.target_position)
                if defect and self.ticks==4 and abs(self.input.target_velocity[0])>.021:q[0]=-.01
                if nonfinite:q[0]=np.nan
                self.input.current_position=q.tolist()
                return q
        return Fake

    def test_fixed_order_first_clear_candidate_and_live_source_is_unchanged(self):
        m=self.module();source=self.source();before=copy.deepcopy(vars(source.input));clones=[]
        position=np.full(7,.001);velocity=np.array([.08,0,0,0,0,0,0]);original=(position.copy(),velocity.copy())
        chosen,receipt=m.choose_velocity_hint(source,position,velocity,
            lambda q:dict(clear=bool(q[0]>=0),reason='posture_joint_envelope'),
            trajectory_factory=self.factory(clones,defect=True))
        np.testing.assert_array_equal(chosen,velocity*.25)
        self.assertEqual(receipt['selected_factor'],.25)
        self.assertEqual([t['factor'] for t in receipt['trials']],[1.,.5,.25])
        self.assertEqual([len(t['preview_commands']) for t in receipt['trials']],[4,4,12])
        rejected=receipt['trials'][0]['first_rejection']
        self.assertEqual(rejected['step'],4);self.assertEqual(rejected['reason'],'posture_joint_envelope')
        self.assertEqual(rejected['command'][0],-.01)
        self.assertEqual(vars(source.input),before);source.set_target.assert_not_called();source.step.assert_not_called()
        np.testing.assert_array_equal(position,original[0]);np.testing.assert_array_equal(velocity,original[1])
        for clone in clones:
            self.assertEqual(clone.first_state['current_position'],before['current_position'])
            self.assertEqual(clone.first_state['current_velocity'],before['current_velocity'])
            self.assertEqual(clone.first_state['current_acceleration'],before['current_acceleration'])
            self.assertEqual(clone.input.target_acceleration,[0.]*7)
            self.assertEqual(clone.limits,dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999))
            self.assertEqual(clone.validations,[dict(check_current_state_within_limits=False,
                                                    check_target_state_within_limits=True)])
        json.dumps(receipt,allow_nan=False)
        self.assertFalse(receipt['collision_checked']);self.assertFalse(receipt['physics_executed'])

    def test_reject_at_twelfth_tick_prevents_early_acceptance(self):
        m=self.module();count=[0]
        def inspect(q):
            count[0]+=1
            return dict(clear=count[0]!=12,reason='late_posture_failure')
        chosen,receipt=m.choose_velocity_hint(self.source(),np.zeros(7),np.full(7,.08),inspect,
                                             trajectory_factory=self.factory([]))
        self.assertEqual(receipt['selected_factor'],.5);self.assertEqual(count[0],24)
        self.assertEqual(receipt['trials'][0]['first_rejection']['step'],12)
        np.testing.assert_array_equal(chosen,np.full(7,.04))

    def test_no_feasible_candidate_has_receipt_and_no_execution(self):
        m=self.module();source=self.source();clones=[]
        with self.assertRaises(m.MotionPreviewRejected) as caught:
            m.choose_velocity_hint(source,np.zeros(7),np.full(7,.05),lambda q:dict(clear=False,reason='urdf_joint_limits'),
                                   trajectory_factory=self.factory(clones))
        receipt=caught.exception.receipt
        self.assertEqual([t['factor'] for t in receipt['trials']],[1.,.5,.25,.125,0.])
        self.assertIsNone(receipt['selected_factor']);self.assertEqual(len(clones),5)
        self.assertTrue(all(t['first_rejection']['step']==1 for t in receipt['trials']))
        source.step.assert_not_called();json.dumps(receipt,allow_nan=False)

    def test_zero_factor_can_pass_without_resetting_initial_derivatives(self):
        m=self.module();clones=[]
        def inspect(q):return dict(clear=clones[-1].input.target_velocity==[0.]*7,reason='posture')
        chosen,receipt=m.choose_velocity_hint(self.source(),np.zeros(7),np.full(7,.08),inspect,
                                             trajectory_factory=self.factory(clones))
        self.assertEqual(receipt['selected_factor'],0.);np.testing.assert_array_equal(chosen,np.zeros(7))
        self.assertEqual(clones[-1].first_state['current_velocity'],[.03]*7)
        self.assertEqual(clones[-1].first_state['current_acceleration'],[.04]*7)

    def test_native_validation_failure_is_recorded_without_any_trial_ticks(self):
        m=self.module();clones=[]
        _,receipt=m.choose_velocity_hint(self.source(),np.zeros(7),np.array([.08,0,0,0,0,0,0]),
            lambda q:dict(clear=True),trajectory_factory=self.factory(clones,validation_error=True))
        self.assertEqual(receipt['selected_factor'],.25)
        self.assertEqual([c.ticks for c in clones],[0,0,12])
        self.assertEqual(receipt['trials'][0]['first_rejection']['step'],0)
        self.assertEqual(receipt['trials'][0]['first_rejection']['stage'],'native_input_validation')

    def test_invalid_inputs_and_modified_limits_fail_before_constructing_clone(self):
        m=self.module()
        for position,velocity in [(np.zeros(6),np.zeros(7)),(np.zeros(7),np.full(7,np.nan)),
                                  (np.zeros(7),np.full(7,.176))]:
            factory=Mock()
            with self.assertRaises(m.MotionPreviewRejected) as caught:
                m.choose_velocity_hint(self.source(),position,velocity,lambda q:dict(clear=True),trajectory_factory=factory)
            self.assertEqual(caught.exception.receipt['trials'],[]);factory.assert_not_called()
            json.dumps(caught.exception.receipt,allow_nan=False)
        source=self.source();source.input.max_jerk=[8.]*7
        with self.assertRaises(m.MotionPreviewRejected):
            m.choose_velocity_hint(source,np.zeros(7),np.zeros(7),lambda q:dict(clear=True),trajectory_factory=Mock())

    def test_nonfinite_preview_bad_inspection_and_factory_alias_are_fail_closed(self):
        m=self.module()
        for factory,inspector in [(self.factory([],nonfinite=True),lambda q:dict(clear=True)),
                                  (self.factory([]),lambda q:dict(clear=1))]:
            with self.assertRaises(m.MotionPreviewRejected) as caught:
                m.choose_velocity_hint(self.source(),np.zeros(7),np.zeros(7),inspector,trajectory_factory=factory)
            self.assertIsNone(caught.exception.receipt['selected_factor'])
            json.dumps(caught.exception.receipt,allow_nan=False)
        source=self.source();before=copy.deepcopy(vars(source.input))
        with self.assertRaises(m.MotionPreviewRejected):
            m.choose_velocity_hint(source,np.zeros(7),np.zeros(7),lambda q:dict(clear=True),
                                   trajectory_factory=lambda *args,**kwargs:source)
        self.assertEqual(vars(source.input),before);source.set_target.assert_not_called()


if __name__=='__main__':unittest.main()
