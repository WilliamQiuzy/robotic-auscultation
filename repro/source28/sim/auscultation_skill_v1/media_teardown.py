"""End capture-owned USD/Replicator scope before normal SimulationApp close.

This is the scoped ordering exercised by cleanup_probe.py, without Kit imports.
Capture functions register resources immediately and release all owned resources
in a finally block. Only plain payloads leave that frame. Exceptions are retained
as text, never traceback objects that would retain native capture locals. A
failed capture or cleanup still attempts normal close and cannot return success.
Actual child/guard exit zero remains a separate media-verification requirement.
"""
import gc
import json
import traceback


POLICY = 'capture_scope_end_then_stage_close_before_normal_close_v1'


def attempt(errors, label, function):
    """Attempt one operation without retaining its exception or native frame."""
    try:
        function()
        return True
    except BaseException:
        errors.append({'stage': label, 'traceback': traceback.format_exc()})
        return False


def release_owned_render_resources(replicator, annotators, products, errors):
    """Run within capture scope; attempt every detach before any destroy."""
    print('MEDIA_OWNED_TEARDOWN_BEGIN', flush=True)
    waited = False
    if replicator is not None:
        attempt(errors, 'capture_off', lambda: replicator.orchestrator.set_capture_on_play(False))
        attempt(errors, 'rep_stop', replicator.orchestrator.stop)
        waited = attempt(errors, 'rep_wait', replicator.orchestrator.wait_until_complete)
    detached = sum(attempt(errors, 'annotator_detach_'+str(i), annotator.detach)
                   for i, annotator in enumerate(annotators))
    destroyed = sum(attempt(errors, 'product_destroy_'+str(i), product.destroy)
                    for i, product in enumerate(products))
    return dict(replicator_wait_completed=waited, annotators_detached=detached,
                render_products_destroyed=destroyed)


def _drain_and_close(app, released, errors):
    """Called after capture returns/unwinds, matching the successful 5+5 probe."""
    print('MEDIA_CAPTURE_SCOPE_ENDED', flush=True)
    receipt = dict(policy=POLICY, **released, capture_scope_ended=True)
    receipt['gc_before_stage_close'] = attempt(errors, 'gc_before_stage_close', gc.collect)
    receipt['pre_stage_close_updates'] = sum(
        attempt(errors, 'preclose_update_'+str(i), app.update) for i in range(5))

    def close_stage():
        context = app.context
        if context.get_stage() is not None:
            if not context.can_close_stage() or context.close_stage() is not True:
                raise RuntimeError('explicit stage close rejected')
        if context.get_stage() is not None:
            raise RuntimeError('stage still present after explicit close')
    receipt['explicit_stage_close_checked'] = attempt(errors, 'explicit_stage_close', close_stage)
    receipt['post_stage_close_updates'] = sum(
        attempt(errors, 'postclose_update_'+str(i), app.update) for i in range(5))

    def assert_stage_remains_closed():
        if app.context.get_stage() is not None:
            raise RuntimeError('stage reopened during cleanup drain')
    receipt['stage_absent_after_drain'] = attempt(errors, 'stage_after_drain', assert_stage_remains_closed)
    receipt['gc_before_app_close'] = attempt(errors, 'gc_before_app_close', gc.collect)
    print('MEDIA_NORMAL_CLOSE_BEGIN', flush=True)
    receipt['close_returned'] = attempt(errors, 'normal_app_close',
        lambda: app.close(wait_for_replicator=True, skip_cleanup=False))
    receipt['errors'] = errors.copy()
    return receipt


def run_scoped_capture(app, capture, *args):
    """Capture returns plain data; close runs after even an error frame is gone.

    ``capture(app, errors, released, *args)`` owns all native scene/sensor locals.
    It must release its registered resources in finally and put only successful
    release counts in ``released``. Neither its arguments nor returned payload
    may retain native resources. Callers publish a manifest only after this
    function and their exact resource-count/source checks return successfully.
    """
    errors = []
    released = dict(replicator_wait_completed=False, annotators_detached=0, render_products_destroyed=0)
    payload = None
    try:
        payload = capture(app, errors, released, *args)
    except BaseException:
        errors.append({'stage': 'capture', 'traceback': traceback.format_exc()})
    # Leave the except block before gc/stage/app close so Python clears its
    # active exception and the capture traceback cannot retain USD/Rep locals.
    receipt = _drain_and_close(app, released, errors)
    if errors:
        raise RuntimeError('media capture/cleanup failed: '+json.dumps(errors)) from None
    print('MEDIA_SCOPED_TEARDOWN_COMPLETE', flush=True)
    return payload, receipt


def require_complete_teardown(receipt, *, annotators, products):
    expected = dict(policy=POLICY, replicator_wait_completed=True,
        annotators_detached=annotators, render_products_destroyed=products,
        capture_scope_ended=True, gc_before_stage_close=True, pre_stage_close_updates=5,
        explicit_stage_close_checked=True, stage_absent_after_drain=True,
        post_stage_close_updates=5, gc_before_app_close=True, close_returned=True, errors=[])
    if (not isinstance(receipt, dict) or receipt != expected or
            any(type(receipt.get(key)) is not type(value) for key, value in expected.items())):
        raise RuntimeError('media did not complete exact scoped-resource teardown')
