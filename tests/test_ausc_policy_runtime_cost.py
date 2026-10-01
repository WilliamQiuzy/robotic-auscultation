import math
import pytest

from sim.ausc_policy.runtime_cost import runtime_cost
from sim.ausc_policy import runtime_cost as cost_module


def test_runtime_cost_separates_controller_from_probe_setup_and_finalize():
    result=runtime_cost(physics_steps=2400,physics_dt_s=1/240,
        probe_wall_s=31.,controller_wall_s=20.,setup_wall_s=8.,capture_profile='diagnostic_video')
    assert result['simulation_s']==10.
    assert result['controller_wall_s_per_sim_s']==2.
    assert result['probe_wall_s_per_sim_s']==3.1
    assert result['finalize_and_audit_wall_s']==3.
    assert result['isaac_startup_wall_s'] is None
    assert result['render_capture_encode_wall_s'] is None


def test_zero_physics_failure_has_no_success_or_infinite_throughput():
    result=runtime_cost(physics_steps=0,physics_dt_s=1/240,
        probe_wall_s=3.,controller_wall_s=None,setup_wall_s=None,capture_profile='diagnostic_video')
    assert result['probe_wall_s_per_sim_s'] is None
    assert result['controller_wall_s_per_sim_s'] is None
    assert result['physical_validation'] is False


@pytest.mark.parametrize('value',[float('nan'),-1,True])
def test_invalid_wall_measurements_rejected(value):
    with pytest.raises(ValueError):
        runtime_cost(physics_steps=240,physics_dt_s=1/240,probe_wall_s=value,
            controller_wall_s=None,setup_wall_s=None,capture_profile='diagnostic_video')


def test_overlapping_wall_totals_rejected():
    with pytest.raises(ValueError):
        runtime_cost(physics_steps=240,physics_dt_s=1/240,probe_wall_s=5,
            controller_wall_s=4,setup_wall_s=2,capture_profile='diagnostic_video')


def measured_run(**changes):
    """Synthetic timing fixture, not a measurement or throughput claim."""
    run = dict(run_id='run-1', context_id='scene-v2/runtime-v1', gpu='test GPU',
        scene_count=1, resolution=[640, 480], capture_tier='training',
        simulation_s=10., startup_wall_s=4., active_wall_s=20.,
        capture_wall_s=3., capture_scope='within_active',
        serialization_wall_s=2., serialization_scope='disjoint',
        teardown_wall_s=1., total_wall_s=27.)
    run.update(changes)
    return run


def test_run_ratios_keep_inclusive_capture_out_of_separate_overhead():
    result = cost_module.summarize_run(**measured_run())
    assert result['durations_s']['capture'] == 3.
    assert result['wall_s_per_sim_s']['active'] == 2.
    assert result['wall_s_per_sim_s']['total'] == 2.7
    assert result['separate_overhead_s']['capture'] is None
    assert result['separate_overhead_s']['serialization'] == 2.
    assert result['context']['capture_scope'] == 'within_active'


def test_disjoint_capture_is_counted_once_in_complete_total():
    result = cost_module.summarize_run(**measured_run(
        capture_scope='disjoint', total_wall_s=30.))
    assert result['separate_overhead_s']['capture'] == 3.
    assert result['wall_s_per_sim_s']['total'] == 3.


def test_overlapping_inclusive_clocks_are_not_added_together():
    result = cost_module.summarize_run(**measured_run(capture_wall_s=16.,
        serialization_wall_s=12., serialization_scope='within_active', total_wall_s=25.))
    assert result['wall_s_per_sim_s']['active'] == 2.
    assert result['separate_overhead_s'] == dict(capture=None, serialization=None)


def test_unknown_components_remain_unavailable():
    result = cost_module.summarize_run(**measured_run(startup_wall_s=None,
        teardown_wall_s=None, capture_wall_s=None, serialization_wall_s=None))
    for field in ('startup', 'teardown', 'capture', 'serialization'):
        assert result['durations_s'][field] is None
        assert result['wall_s_per_sim_s'][field] is None
    assert result['separate_overhead_s'] == dict(capture=None, serialization=None)


def test_single_run_aggregate_reports_one_sample_without_extrapolation():
    result = cost_module.aggregate_runs([measured_run(capture_wall_s=None)])
    assert result['sample_count'] == 1
    assert result['run_ids'] == ['run-1']
    assert result['wall_s_per_sim_s']['total'] == dict(n=1, p50=2.7, p90=2.7, p95=2.7)
    assert result['durations_s']['capture'] == dict(n=0, p50=None, p90=None, p95=None)
    assert result['context']['gpu'] == 'test GPU'


def test_quantiles_use_linear_interpolation_and_available_sample_counts():
    runs = [measured_run(run_id=f'run-{i}', active_wall_s=wall,
        total_wall_s=wall+7., capture_wall_s=None if i == 0 else 3.)
        for i, wall in enumerate((10., 20., 30., 40.))]
    result = cost_module.aggregate_runs(runs)
    ratios = result['wall_s_per_sim_s']['active']
    assert ratios['n'] == 4
    assert ratios['p50'] == 2.5
    assert ratios['p90'] == pytest.approx(3.7)
    assert ratios['p95'] == pytest.approx(3.85)
    assert result['durations_s']['capture']['n'] == 3
    assert result['durations_s']['capture']['p50'] == 3.


@pytest.mark.parametrize('changes', [
    dict(total_wall_s=26.),  # Less than measured disjoint parts.
    dict(total_wall_s=30.),  # Complete total counts inclusive capture twice.
    dict(capture_scope='disjoint', total_wall_s=27.),
    dict(capture_wall_s=21.),
    dict(serialization_wall_s=21., serialization_scope='within_active'),
    dict(capture_scope=None),
    dict(serialization_scope='unknown'),
])
def test_contradictory_or_ambiguous_timing_provenance_rejected(changes):
    with pytest.raises(ValueError):
        cost_module.summarize_run(**measured_run(**changes))


@pytest.mark.parametrize('field', ['simulation_s', 'startup_wall_s', 'active_wall_s',
    'capture_wall_s', 'serialization_wall_s', 'teardown_wall_s', 'total_wall_s'])
@pytest.mark.parametrize('value', [-1., math.inf, math.nan, True])
def test_invalid_run_duration_rejected(field, value):
    with pytest.raises(ValueError):
        cost_module.summarize_run(**measured_run(**{field: value}))


@pytest.mark.parametrize('value', [0., None])
def test_new_cost_summary_requires_positive_simulation_duration(value):
    with pytest.raises(ValueError):
        cost_module.summarize_run(**measured_run(simulation_s=value))


@pytest.mark.parametrize('changes', [dict(context_id='other-scene'), dict(gpu='other GPU'),
    dict(scene_count=2), dict(resolution=[1280, 720]), dict(capture_tier='tier_b'),
    dict(capture_scope='disjoint', total_wall_s=30.),
    dict(serialization_scope='within_active', total_wall_s=25.)])
def test_incompatible_measurement_contexts_cannot_be_aggregated(changes):
    with pytest.raises(ValueError, match='context'):
        cost_module.aggregate_runs([measured_run(), measured_run(run_id='run-2', **changes)])


@pytest.mark.parametrize('changes', [dict(run_id=''), dict(context_id=''), dict(gpu=None),
    dict(scene_count=0), dict(scene_count=True), dict(resolution=[0, 480]),
    dict(resolution=[640]), dict(resolution=[640, True]), dict(capture_tier='')])
def test_explicit_valid_metadata_required(changes):
    with pytest.raises(ValueError):
        cost_module.summarize_run(**measured_run(**changes))


def test_empty_or_duplicate_run_aggregate_rejected():
    with pytest.raises(ValueError):
        cost_module.aggregate_runs([])
    with pytest.raises(ValueError, match='run_id'):
        cost_module.aggregate_runs([measured_run(), measured_run()])
