import pytest
from sim.ausc_policy.moving_origin import selected_origin, timing_validation_mode


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


@pytest.mark.parametrize('index',[120,220])
def test_later_origins_are_the_original_predeclared_three_point_schedule(index):
    c=dict(source_action_index=index,moving_origin=f'history_free{index}_v1',
           oracle_timing='checked_edges_v1',timing_validation='validated_algorithm_new_origin')
    assert selected_origin(c)==(index,'FREE',f'history{index}')
    assert timing_validation_mode(c)=='validated_algorithm_new_origin'
    with pytest.raises(ValueError):timing_validation_mode(dict(c,timing_validation='paired_retest'))


def test_paired_retest_cannot_skip_same_stopped_query():
    c=dict(source_action_index=20,moving_origin='history_free20_v1',oracle_timing='checked_edges_v1')
    assert timing_validation_mode(c)=='paired_retest'
    with pytest.raises(ValueError):timing_validation_mode(dict(c,timing_validation='validated_algorithm_new_origin'))
