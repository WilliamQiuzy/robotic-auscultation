"""No-sensor diagnostic cleanup is explicit; no fictitious Replicator wait."""


def require_no_sensor_teardown(receipt):
    expected=dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
        replicator_wait_completed=False,annotators_detached=0,render_products_destroyed=0,
        capture_scope_ended=True,gc_before_stage_close=True,pre_stage_close_updates=5,
        explicit_stage_close_checked=True,stage_absent_after_drain=True,
        post_stage_close_updates=5,gc_before_app_close=True,close_returned=True,errors=[])
    if (not isinstance(receipt,dict) or receipt!=expected or
            any(type(receipt.get(key)) is not type(value) for key,value in expected.items())):
        raise RuntimeError('No-sensor diagnostic cleanup did not complete')
