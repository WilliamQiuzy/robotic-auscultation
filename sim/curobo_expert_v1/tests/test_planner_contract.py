import numpy as np
import pytest

from sim.curobo_expert_v1.planner_service import validate_proposal


def result(path):
    return dict(success=True, path=np.asarray(path).tolist(),
                timestamps=np.arange(len(path), dtype=float).tolist(),
                metadata={'planner': 'curobo'}, failure=None)


def test_accepts_only_matching_seven_joint_endpoints():
    a=np.zeros(7);b=np.ones(7)*.1
    route=validate_proposal(result([a,b]),a,b)
    np.testing.assert_array_equal(route,[a,b])


@pytest.mark.parametrize('change', ['failed','nan','wrong_start','wrong_end','one_row','bad_clock','no_clock'])
def test_bad_proposal_is_not_silently_repaired(change):
    a=np.zeros(7);b=np.ones(7)*.1;r=result([a,b])
    if change=='failed':r['success']=False
    if change=='nan':r['path'][1][0]=float('nan')
    if change=='wrong_start':r['path'][0][0]=.02
    if change=='wrong_end':r['path'][1][0]=.12
    if change=='one_row':r['path']=r['path'][:1]
    if change=='bad_clock':r['timestamps']=[1,0]
    if change=='no_clock':r.pop('timestamps')
    with pytest.raises(ValueError):validate_proposal(r,a,b)


def test_small_residual_preserved_for_separate_checked_connector():
    a=np.zeros(7);b=np.ones(7)*.1;r=result([a+.001,b-.001])
    q=validate_proposal(r,a,b)
    assert q[0,0]==.001 and q[-1,0]==.099
