from copy import deepcopy
import pytest
from sim.ausc_policy.moving_brake import check_prefix_chunk


def rows():
    return [dict(q=[.1]*7,q_command=[.101]*7,time_s=(i+1)/240) for i in range(12)]


def test_new_actual_observation_can_differ_within_declared_reproduction_budget():
    expected=rows();actual=deepcopy(expected);actual[2]['q'][0]+=.00001
    assert 0<check_prefix_chunk(expected,actual,[.101]*7)<1e-4
    assert actual[2]['q'][0]!=expected[2]['q'][0]


@pytest.mark.parametrize('fault',['measured_pose','command','clock','missing','nan'])
def test_replay_never_silently_fixes_a_changed_origin(fault):
    expected=rows();actual=deepcopy(expected)
    if fault=='measured_pose': actual[-1]['q'][0]+=.0002
    if fault=='command': actual[-1]['q_command'][0]+=.000001
    if fault=='clock': actual[-1]['time_s']+=.0001
    if fault=='missing': actual.pop()
    if fault=='nan': actual[0]['q'][0]=float('nan')
    with pytest.raises(ValueError): check_prefix_chunk(expected,actual,[.101]*7)
