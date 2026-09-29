import numpy as np
import pytest
from sim.ausc_policy.edge_timing import checked_edge_commands
from sim.expert_quality_20260924.candidate_planner import discrete_dynamic_peaks


def test_timed_corner_stays_on_original_edges_and_keeps_dynamic_seams():
    route=np.zeros((3,7));route[1,0]=.12;route[2,:2]=[.12,.08]
    commands,report=checked_edge_commands(route)
    first=(abs(commands[:,1])<1e-12)&(commands[:,0]>=0)&(commands[:,0]<=.12)
    second=(abs(commands[:,0]-.12)<1e-12)&(commands[:,1]>=0)&(commands[:,1]<=.08)
    assert np.all(first|second)
    np.testing.assert_array_equal(commands[-1],route[-1])
    peaks=discrete_dynamic_peaks(route,commands,1/240)
    assert peaks['velocity_rad_s']<=.175+1e-9
    assert peaks['acceleration_rad_s2']<=.5+1e-9
    assert peaks['jerk_rad_s3']<=1.5+1e-9
    assert len(commands)%12==0 and len(commands)>=120
    assert report['stationary_at_each_retained_waypoint']


def test_duplicate_knots_and_stationary_query_remain_finite_deterministic():
    p=np.full((3,7),.1);a,r=checked_edge_commands(p);b,_=checked_edge_commands(p)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,np.full(a.shape,.1))
    assert r['unique_knots']==1


def test_declared_duration_limit_is_not_silently_extended():
    with pytest.raises(ValueError):checked_edge_commands(np.array([[0.]*7,[.5]*7]),maximum_seconds=.05)
