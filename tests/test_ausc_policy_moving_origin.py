import pytest
from sim.ausc_policy.moving_origin import selected_origin


def test_legacy_origin_remains_explicit_approach609():
    assert selected_origin({'source_action_index':609})==(609,'APPROACH','act609')


def test_history_recovery_is_a_separate_bound_free_origin():
    assert selected_origin({'source_action_index':20,'moving_origin':'history_free20_v1'})==(20,'FREE','history20')


def test_history_case_cannot_be_misnamed_as_legacy_expert():
    from sim.ausc_policy.recovery_run import selected_cases
    c=dict(mode='moving_oracle_pilot',moving_origin='history_free20_v1',source_action_index=20,case_ids=['history20'])
    assert selected_cases(c)==[('history20',0.)]
    with pytest.raises(ValueError):selected_cases(dict(c,case_ids=['act609']))


@pytest.mark.parametrize('config',[
    {'source_action_index':20},
    {'source_action_index':609,'moving_origin':'history_free20_v1'},
    {'source_action_index':20,'moving_origin':'unknown'},
    {'source_action_index':True,'moving_origin':'history_free20_v1'},
])
def test_mismatched_or_implicit_new_source_cannot_run(config):
    with pytest.raises(ValueError):selected_origin(config)
