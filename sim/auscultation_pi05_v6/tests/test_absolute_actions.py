import numpy as np
import pytest
from sim.auscultation_pi05_v6.action_contract import compose_joints,LOWER,UPPER


def test_goal_does_not_accumulate_measured_servo_lag():
    action=np.array([.1,-.3,-1.,2.,.5,.1,.4])
    for measured in (action-.001,action+.002):
        np.testing.assert_array_equal(compose_joints(measured,action,'CONTACT',LOWER,UPPER),action)


def test_no_implicit_clipping_or_changed_urdf():
    q=np.zeros(7);a=q.copy();a[3]=-.001
    with pytest.raises(ValueError):compose_joints(q,a,'FREE',LOWER,UPPER)
    changed=np.array(UPPER);changed[0]+=.01
    with pytest.raises(ValueError):compose_joints(q,q,'FREE',LOWER,changed)
