from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
from sim.curobo_expert_v1.controller import ExpertController


def actor(checker):
    controller=ExpertController.__new__(ExpertController)
    controller.q=np.zeros(7);controller.speed_scale=1.;controller.joint_speed_limit=.25
    controller.target=SimpleNamespace(name='R1');controller.env=SimpleNamespace(rows=[])
    controller.checker=checker;controller.free_collision_samples=0;controller.free_motion_timing=[]
    def block(commands,subphase):
        controller.env.rows.extend(commands.tolist());controller.q=commands[-1].copy()
    controller._block=block
    return controller


def test_free_collision_failure_prevents_entire_segment_motion():
    class Checker:
        def __init__(self):self.calls=0
        def check(self,q,**kw):
            self.calls+=1
            return {'clear':self.calls<3,'minimum_distance_m':0.}
    c=actor(Checker())
    with pytest.raises(EpisodeRejected,match='mesh/posture'):
        c._continuous_path(np.array([np.zeros(7),np.ones(7)*.03]),'transfer')
    assert c.env.rows==[] and c.free_motion_timing==[]


def test_every_executed_free_command_checked_and_stricter_bounds_recorded():
    class Checker:
        def __init__(self):self.calls=0
        def check(self,q,**kw):
            self.calls+=1
            assert kw=={'clearance':.003,'allow_patient_contact':False,'compression':0.}
            return {'clear':True,'minimum_distance_m':.003}
    c=actor(Checker());goal=np.ones(7)*.03
    c._continuous_path(np.array([np.zeros(7),goal]),'transfer')
    assert c.checker.calls==len(c.env.rows)==c.free_collision_samples
    np.testing.assert_array_equal(c.q,goal)
    r=c.free_motion_timing[0]
    assert r['motion_complete'] and r['peaks']['jerk_rad_s3']<=1.5
    assert r['acceleration_limit_rad_s2']==1.


def test_disconnected_route_does_not_move():
    c=actor(None)
    with pytest.raises(EpisodeRejected,match='disconnected'):
        c._continuous_path(np.array([np.ones(7)*.01,np.ones(7)*.03]),'transfer')
    assert c.env.rows==[]


@pytest.mark.parametrize('distance',[float('nan'),float('inf'),.00299])
def test_clear_boolean_cannot_override_missing_or_inadequate_distance(distance):
    class Checker:
        def check(self,q,**kw):return {'clear':True,'minimum_distance_m':distance}
    c=actor(Checker())
    with pytest.raises(EpisodeRejected,match='mesh/posture'):
        c._continuous_path(np.array([np.zeros(7),np.ones(7)*.03]),'transfer')
    assert c.env.rows==[]
