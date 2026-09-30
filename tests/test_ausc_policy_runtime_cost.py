import math
import pytest

from sim.ausc_policy.runtime_cost import runtime_cost


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
