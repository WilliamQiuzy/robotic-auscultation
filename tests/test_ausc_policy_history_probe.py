import numpy as np
import pytest
from sim.ausc_policy.history_probe import past_index, near_state


def test_past_index_selects_only_actual_exact_prior_sample():
    times=np.arange(20,dtype=float)*.05
    assert past_index(times,12,.25)==7
    assert past_index(times,12,.5)==2
    with pytest.raises(ValueError): past_index(times,1,.25)
    with pytest.raises(ValueError): past_index(times,12,-.05)
    broken=times.copy();broken[7]+=.001
    with pytest.raises(ValueError): past_index(broken,12,.25)


def test_observed_phase_difference_breaks_near_state_equivalence():
    a=np.zeros(19);b=a.copy();a[-1]=-1;b[-1]=1
    assert not near_state(a,b)
    b[-1]=-1;b[0]=.0004
    assert near_state(a,b)
    b[0]=.0006
    assert not near_state(a,b)
