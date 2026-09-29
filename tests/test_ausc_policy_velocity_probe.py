import numpy as np
import pytest

from sim.ausc_policy.velocity_probe import causal_velocity, proposals


def test_history_estimate_uses_no_future_samples():
    times = np.arange(5) * .05
    q = times[:, None] * np.arange(1, 8)[None] * .01
    expected = np.arange(1, 8) * .01
    np.testing.assert_allclose(causal_velocity(q, times, 2), expected)
    q[3:] = np.nan
    times[3:] = np.nan
    np.testing.assert_allclose(causal_velocity(q, times, 2), expected)


@pytest.mark.parametrize('origin,times', [(0,[0.,.05]), (1,[0.,.1]), (1,[0.,0.])])
def test_missing_or_gapped_history_is_not_zero_filled(origin, times):
    with pytest.raises(ValueError):
        causal_velocity(np.zeros((2,7)), np.array(times), origin)


def test_position_history_and_reported_velocity_remain_distinct():
    q = np.ones(7)
    result = proposals(q, np.full(7,.01), np.zeros(7))
    np.testing.assert_array_equal(result['hold'], result['position_history_cv'])
    np.testing.assert_allclose(result['api_dq_cv'][0], q+.0005)
    np.testing.assert_allclose(result['api_dq_cv'][-1], q+.005)
    np.testing.assert_array_equal(q,np.ones(7))


def test_history_rejects_nonfinite_measurements():
    q = np.zeros((2,7)); q[0,3] = np.inf
    with pytest.raises(ValueError):
        causal_velocity(q,np.array([0.,.05]),1)
