import pytest
from sim.auscultation_repair_v3.teardown import require_no_sensor_teardown


def receipt():
    return dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
        replicator_wait_completed=False,annotators_detached=0,render_products_destroyed=0,
        capture_scope_ended=True,gc_before_stage_close=True,pre_stage_close_updates=5,
        explicit_stage_close_checked=True,stage_absent_after_drain=True,
        post_stage_close_updates=5,gc_before_app_close=True,close_returned=True,errors=[])


def test_no_sensor_capture_has_no_replicator_wait():require_no_sensor_teardown(receipt())


@pytest.mark.parametrize('key,value',[('replicator_wait_completed',True),('close_returned',False),
    ('annotators_detached',1),('errors',['failure']),('pre_stage_close_updates',True)])
def test_cleanup_failures_cannot_pass(key,value):
    r=receipt();r[key]=value
    with pytest.raises(RuntimeError):require_no_sensor_teardown(r)
