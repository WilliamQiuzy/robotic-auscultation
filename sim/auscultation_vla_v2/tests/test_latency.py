import importlib
import pytest


@pytest.fixture
def api():
    try: module=importlib.import_module('sim.auscultation_vla_v2.latency')
    except ModuleNotFoundError: module=None
    assert module is not None, 'latency feasibility arithmetic is not implemented'
    return module


def test_long_prediction_horizon_does_not_fix_execution_budget(api):
    result=api.assess([.12]*20,prediction_steps=50,execution_steps=1,action_period_s=.05,sensor_age_s=.02,transport_s=.01,max_observation_age_s=.15)
    assert result['prediction_horizon_s']==2.5
    assert result['can_replenish_prefix'] is False
    assert result['oldest_executed_observation_age_s']==pytest.approx(.15)


def test_observation_age_includes_tail_of_prefix(api):
    result=api.assess([.08]*20,prediction_steps=10,execution_steps=2,action_period_s=.05,sensor_age_s=.02,transport_s=.01,max_observation_age_s=.15)
    assert result['can_replenish_prefix'] is True
    assert result['oldest_executed_observation_age_s']==pytest.approx(.16)
    assert result['observation_age_passed'] is False
    assert result['closed_loop_ready'] is False


@pytest.mark.parametrize('values',[[],[-1.],[float('nan')],[float('inf')]])
def test_bad_measurements_rejected(api,values):
    with pytest.raises(ValueError):api.assess(values,prediction_steps=10,execution_steps=2,action_period_s=.05,sensor_age_s=.02,transport_s=.01,max_observation_age_s=.15)


def test_feasible_assumed_budget_is_not_closed_loop_evidence(api):
    result=api.assess([.03]*20,prediction_steps=10,execution_steps=2,action_period_s=.05,sensor_age_s=.02,transport_s=.01,max_observation_age_s=.15)
    assert result['budget_feasible_for_assumed_overheads'] is True
    assert result['closed_loop_ready'] is False
