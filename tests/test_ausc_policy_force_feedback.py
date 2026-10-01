"""Hall remains an axial proxy; normal-force truth stays outside the actor."""
import importlib.util
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig, PhaseManager, ContactObservation, Phase


def module():
    assert importlib.util.find_spec('sim.ausc_policy.force_feedback') is not None, 'Force feedback correction missing'
    from sim.ausc_policy import force_feedback
    return force_feedback


SETTING = {'mode': 'axial_proxy_margin_v1'}


def test_margin_only_tightens_observable_hold_band():
    m = module()
    c = PhaseConfig(max_compression_m=.006)
    corrected = m.hold_config(c, SETTING)
    assert corrected.force_tolerance_N == .1
    assert corrected.target_force_N == 4.
    assert corrected.hold_duration_s == 3.
    assert corrected.hard_force_limit_N == 5.5
    assert corrected.max_compression_m == .006
    assert m.hold_config(c, None) is c


def test_r2_low_normal_load_proxy_cannot_start_hold():
    m = module()
    c = m.hold_config(PhaseConfig(), SETTING)
    manager = PhaseManager(c)
    manager.update(ContactObservation(0., 0., .04, 0., 0., 0.), 1/240)
    # v8 R2 starts its HOLD with Hall ~3.65 N while raw normal load is ~3.38 N.
    obs = ContactObservation(3.65, .0018, 0., 0., 0., 0.)
    for _ in range(4*240): manager.update(obs, 1/240)
    assert manager.state.phase == Phase.CONTACT
    assert manager.state.hold_elapsed == 0.


def test_three_seconds_still_required_and_transients_reset_hold():
    m = module()
    manager = PhaseManager(m.hold_config(PhaseConfig(), SETTING))
    manager.update(ContactObservation(0., 0., .04, 0., 0., 0.), 1/240)
    good = ContactObservation(4., .0022, 0., 0., 0., 0.)
    for _ in range(72+700): manager.update(good, 1/240)
    assert manager.state.phase == Phase.HOLD
    manager.update(ContactObservation(3.8, .002, 0., 0., 0., 0.), 1/240)
    assert manager.state.hold_elapsed == 0.
    for _ in range(72+719): manager.update(good, 1/240)
    assert manager.state.phase == Phase.HOLD
    manager.update(good, 1/240)
    assert manager.state.phase == Phase.RETRACT


@pytest.mark.parametrize('bad',[{},True,{'mode':'unknown'},{'mode':'axial_proxy_margin_v1','force_N':4.5}])
def test_unknown_configuration_fails_closed(bad):
    with pytest.raises(ValueError):module().hold_config(PhaseConfig(),bad)


def test_diagnostic_separates_normal_and_friction_and_binds_pair_orientation():
    m = module()
    h = NS(collider0='ring',collider1='patient',contact_data_offset=1,num_contact_data=1,
           friction_anchors_offset=1,num_friction_anchors_data=1)
    points = [None,NS(impulse=np.array([0.,0.,3.5])/240)]
    friction = [None,NS(impulse=np.array([.4,0.,0.])/240)]
    state=dict(axis_world=np.array([-.3,0.,-np.sqrt(.91)]),compression_m=.002,
               hall_raw_N=4.,hall_N=3.99)
    r=m.balance_receipt(([h],points,friction),lambda x:x,
        lambda a,b:a in ('ring','patient') and b in ('ring','patient'),
        lambda p:p=='ring',state,.0001,.067,1/240)
    assert r['normal_force_N']==pytest.approx(3.5)
    assert r['friction_force_vector_N']==pytest.approx([.4,0,0])
    assert r['normal_axial_N']==pytest.approx(3.5*np.sqrt(.91))
    assert r['friction_axial_N']==pytest.approx(.12)
    assert r['combined_axial_N']==pytest.approx(3.5*np.sqrt(.91)+.12)
    assert r['actor_input'] is False
    assert state['hall_N']==3.99
    h.collider0,h.collider1='patient','ring'
    points[1].impulse *= -1;friction[1].impulse *= -1
    reverse=m.balance_receipt(([h],points,friction),lambda x:x,
        lambda a,b:True,lambda p:p=='ring',state,.0001,.067,1/240)
    assert reverse['combined_axial_N']==pytest.approx(r['combined_axial_N'])


def test_incomplete_native_friction_report_is_not_silently_zero_filled():
    m=module()
    h=NS(collider0='ring',collider1='patient',contact_data_offset=0,num_contact_data=0,
         friction_anchors_offset=0,num_friction_anchors_data=1)
    with pytest.raises(ValueError):m.balance_receipt(([h],[],[]),lambda x:x,
        lambda a,b:True,lambda p:p=='ring',dict(axis_world=[0,0,-1],compression_m=0,hall_raw_N=0,hall_N=0),0,.067,1/240)


def test_native_balance_does_not_replace_gravity_verification_receipt(monkeypatch):
    import sys
    from sim.ausc_policy import fresh_expert as f
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',NS(ArticulationAction=NS))
    monkeypatch.setitem(sys.modules,'omni.physx',NS(get_physx_simulation_interface=lambda:NS(get_full_contact_report=lambda:([],[],[]))))
    monkeypatch.setitem(sys.modules,'pxr',NS(PhysicsSchemaTools=NS(intToSdfPath=lambda v:v)))
    env=f.FreshContactEnv.__new__(f.FreshContactEnv)
    env.rows=[];env.slider=1;env.passive_preload_N=1.5;env.moving_mass_kg=.067
    env.force_balance_enabled=True;env.video_recorder=None;env.contact_points=[];env.last_contacts={}
    env.bodies=[('/World/Stethoscope/Carriage',NS(get_world_pose=lambda:(np.zeros(3),np.array([1.,0.,0.,0.]))))]
    env.robot=NS(apply_action=lambda action:None,get_joint_velocities=lambda:np.zeros(2))
    gravity_receipt={'requested_efforts':[1.,-1.5]};verified=[]
    def verify(receipt):
        assert receipt is gravity_receipt
        verified.append(True)
    env.gravity_feedforward=NS(prepare=lambda q:(np.array([0,1]),np.array([1.,-1.5]),gravity_receipt),verify_applied=verify)
    state=dict(axis_world=np.array([0.,0.,-1.]),compression_m=0.,hall_raw_N=0.,hall_N=0.)
    def step(self,*args):self.rows.append({});return state
    monkeypatch.setattr(f.IsaacContactEnv,'step',step)
    assert env.step(np.zeros(7),'CONTACT','R2') is state
    assert verified==[True]
    assert env.rows[-1]['force_balance']['actor_input'] is False
