"""Scoped media cleanup without Kit; actual native exit remains a production gate."""
import ast
import gc
import importlib
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


def cleanup_module():
    return importlib.import_module('sim.auscultation_skill_v1.media_teardown')


def expected_receipt(products=2):
    return dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
        replicator_wait_completed=True, annotators_detached=2, render_products_destroyed=products,
        capture_scope_ended=True, gc_before_stage_close=True, pre_stage_close_updates=5,
        explicit_stage_close_checked=True, stage_absent_after_drain=True,
        post_stage_close_updates=5, gc_before_app_close=True, close_returned=True, errors=[])


def resources(events, failure=None):
    state = {'stage': object(), 'updates': 0}
    def event(name, *args, **kwargs):
        events.append((name, args, kwargs))
        if name == failure:
            raise RuntimeError(name+' failed')
    def update():
        state['updates'] += 1
        event('update_'+str(state['updates']))
    def close_stage():
        event('stage_close')
        if failure == 'stage_rejected': return False
        if failure != 'stage_remains': state['stage'] = None
        return True
    def get_stage():
        if failure == 'stage_reopened' and state['updates'] == 10:
            return object()
        return state['stage']
    context = SimpleNamespace(get_stage=get_stage, can_close_stage=lambda: failure != 'cannot_close', close_stage=close_stage)
    app = SimpleNamespace(context=context, update=update, close=lambda **kw: event('close', **kw))
    rep = SimpleNamespace(orchestrator=SimpleNamespace(
        set_capture_on_play=lambda value: event('capture', value),
        stop=lambda: event('stop'), wait_until_complete=lambda: event('wait')))
    annotators = [SimpleNamespace(detach=lambda name=name: event(name)) for name in ('rgb_a', 'rgb_b')]
    products = [SimpleNamespace(destroy=lambda name=name: event(name)) for name in ('product_a', 'product_b')]
    return app, rep, annotators, products


def invoke(events, failure=None, monkeypatch=None, rep_loaded=True, product_count=2, annotator_count=2):
    module = cleanup_module()
    app, rep, annotators, products = resources(events, failure)
    if monkeypatch is not None:
        original_collect = gc.collect
        def collect():
            events.append(('gc', (), {}))
            return original_collect()
        monkeypatch.setattr(module.gc, 'collect', collect)
    def capture_scope(app, errors, released):
        try:
            if failure == 'capture_body': raise RuntimeError('capture_body failed')
            return {'plain': 'manifest'}
        finally:
            released.update(module.release_owned_render_resources(rep if rep_loaded else None,
                annotators[:annotator_count], products[:product_count], errors))
    return module.run_scoped_capture(app, capture_scope)


def test_exact_success_order_and_receipt(monkeypatch):
    events = []
    payload, receipt = invoke(events, monkeypatch=monkeypatch)
    assert payload == {'plain': 'manifest'}
    assert [event[0] for event in events] == ['capture', 'stop', 'wait', 'rgb_a', 'rgb_b',
        'product_a', 'product_b', 'gc'] + ['update_'+str(i) for i in range(1,6)] + [
        'stage_close'] + ['update_'+str(i) for i in range(6,11)] + ['gc', 'close']
    assert events[0] == ('capture', (False,), {})
    assert events[-1] == ('close', (), {'wait_for_replicator': True, 'skip_cleanup': False})
    assert receipt == expected_receipt()


@pytest.mark.parametrize('failure', ['capture_body', 'capture', 'stop', 'wait', 'rgb_a', 'rgb_b',
    'product_a', 'product_b', 'update_1', 'stage_close', 'update_10', 'close'])
def test_every_failure_still_attempts_all_detaches_products_and_normal_close(failure):
    events = []
    with pytest.raises(RuntimeError, match=failure+' failed'):
        invoke(events, failure)
    names = [name for name, _, _ in events]
    assert names.count('close') == 1 and names[-1] == 'close'
    assert names.index('rgb_b') < names.index('product_a') < names.index('product_b')
    assert all('update_'+str(i) in names for i in range(1,11))


@pytest.mark.parametrize('failure', ['cannot_close','stage_rejected','stage_remains','stage_reopened'])
def test_stage_must_really_close_and_remain_closed(failure):
    events = []
    with pytest.raises(RuntimeError, match='stage'):
        invoke(events, failure)
    assert events[-1][0] == 'close'


@pytest.mark.parametrize('rep_loaded,product_count,annotator_count', [(False,0,0),(True,0,0),(True,1,0),(True,1,1)])
def test_partial_initialization_only_cleans_acquired_resources(rep_loaded, product_count, annotator_count):
    events = []
    _, receipt = invoke(events, rep_loaded=rep_loaded, product_count=product_count, annotator_count=annotator_count)
    assert receipt['annotators_detached'] == annotator_count
    assert receipt['render_products_destroyed'] == product_count
    assert receipt['replicator_wait_completed'] is rep_loaded
    assert sum(name.startswith('rgb_') for name, _, _ in events) == annotator_count
    assert sum(name.startswith('product_') for name, _, _ in events) == product_count
    assert events[-1][0] == 'close'


@pytest.mark.parametrize('capture_error', [None, RuntimeError, KeyboardInterrupt])
def test_native_scope_and_error_traceback_are_gone_before_stage_and_app_close(capture_error):
    module = cleanup_module()
    refs = []
    class NativeObject: pass
    app, _, _, _ = resources([])
    for name in ('close_stage',):
        original = getattr(app.context, name)
        def assert_dead(original=original):
            assert refs and refs[0]() is None, 'native reference retained through cleanup'
            return original()
        setattr(app.context, name, assert_dead)
    def close(**kwargs):
        assert refs[0]() is None
    app.close = close
    def capture(app, errors, released):
        native = NativeObject()
        refs.append(weakref.ref(native))
        # A cycle also requires the pre-stage gc, not merely Python refcounting.
        native.self = native
        if capture_error: raise capture_error('capture interrupted')
        return {'plain': 'data'}
    if capture_error:
        with pytest.raises(RuntimeError, match='capture interrupted') as caught:
            module.run_scoped_capture(app, capture)
        assert caught.value.__context__ is None
    else:
        assert module.run_scoped_capture(app, capture)[0] == {'plain':'data'}


def test_multiple_errors_preserved_as_text_after_all_cleanup():
    module = cleanup_module(); events=[]
    app, rep, anns, products = resources(events, 'wait')
    def fail_close(**kwargs):
        events.append(('close', (), kwargs))
        raise ValueError('close failed too')
    app.close = fail_close
    def capture(app, errors, released):
        try: raise RuntimeError('capture original')
        finally: released.update(module.release_owned_render_resources(rep, anns, products, errors))
    with pytest.raises(RuntimeError) as caught:
        module.run_scoped_capture(app, capture)
    assert all(text in str(caught.value) for text in ('wait failed', 'capture original', 'close failed too'))
    assert caught.value.__context__ is None
    assert events[-1][0] == 'close'


def test_private_copy_preserves_the_entire_pose_frame_and_encoder_loop():
    root = Path(__file__).resolve().parents[1]
    original = root.parent/'upper_pair_posture_20260924/render_episode.py'
    private = root/'private_renderer.py'
    assert private.is_file(), 'private renderer is missing'
    def frame_loop(path):
        tree = ast.parse(path.read_text())
        return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'enumerate')
    assert ast.dump(frame_loop(private), include_attributes=False) == ast.dump(frame_loop(original), include_attributes=False)
    original_text, private_text = original.read_text(), private.read_text()
    encoder_start = '        encoder = subprocess.Popen('
    encoder_end = '        manifest = '
    assert private_text.split(encoder_start)[1].split(encoder_end)[0] == original_text.split(encoder_start)[1].split(encoder_end)[0]


@pytest.mark.parametrize('name,expected', [
    ('rgbd.py','4951fc4cf0290cdf19163152dd0899c78fabaf584063436e8f065192ac29eb2c'),
    ('private_renderer.py','8ecb43e5e2b1af90aea614df46f0a55f4a89dbeff1015bb5bd1aab7f05a13c4a')])
def test_entire_native_capture_body_keeps_reviewed_rendering_logic(name, expected):
    # Fingerprints bind the reviewed bodies: archived before implementation,
    # with publication moved after close and eager RGBD resource registration.
    # All frame, pose, optics, encoder and depth computations stay unchanged.
    import hashlib
    tree = ast.parse((Path(__file__).resolve().parents[1]/name).read_text())
    scope = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_capture_scope')
    capture = next(node for node in scope.body if isinstance(node,ast.Try)
        and any(isinstance(item,ast.ImportFrom) and item.module=='pxr' for item in node.body))
    dumped = ast.dump(ast.Module(body=capture.body,type_ignores=[]),include_attributes=False)
    assert hashlib.sha256(dumped.encode()).hexdigest() == expected
    # Only the plain manifest escapes. Registered resources always release.
    assert isinstance(scope.body[-1],ast.Return) and scope.body[-1].value.id=='manifest'
    assert any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        and node.func.id=='release_owned_render_resources' for node in ast.walk(ast.Module(body=capture.finalbody,type_ignores=[])))
    outer = next(node for node in tree.body if isinstance(node,ast.FunctionDef)
        and node.name==('_capture' if name=='rgbd.py' else 'main'))
    assert any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        and node.func.id=='run_scoped_capture' for node in ast.walk(outer))
    assert not any(isinstance(node,ast.ImportFrom) and node.module=='pxr' for node in ast.walk(outer))


def test_rgbd_second_annotator_acquisition_failure_still_detaches_first():
    # Execute the actual acquisition statements with a fake registry: this
    # covers the partial-init window without importing or starting native Kit.
    module = cleanup_module(); events=[]
    _,rep,annotators,products = resources(events)
    def get_annotator(name):
        if name=='distance_to_image_plane': raise RuntimeError('depth acquisition failed')
        return annotators[0]
    rep.create = SimpleNamespace(render_product=lambda *args: products[0])
    rep.AnnotatorRegistry = SimpleNamespace(get_annotator=get_annotator)
    tree=ast.parse((Path(__file__).resolve().parents[1]/'rgbd.py').read_text())
    scope=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_capture_scope')
    body=next(node for node in scope.body if isinstance(node,ast.Try)).body
    start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='product')
    stop=next(i for i,node in enumerate(body[start:],start) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=='attach')
    acquisition=ast.Module(body=body[start:stop],type_ignores=[])
    owned_annotators=[];owned_products=[];errors=[]
    namespace=dict(rep=rep,camera={'camera_path':'fake-camera'},width=1,height=1,
                   owned_annotators=owned_annotators,owned_products=owned_products)
    with pytest.raises(RuntimeError,match='depth acquisition failed'):
        try: exec(compile(acquisition,'actual-rgbd-acquisition','exec'),namespace)
        finally: module.release_owned_render_resources(rep,owned_annotators,owned_products,errors)
    names=[name for name,_,_ in events]
    assert 'rgb_a' in names and names.index('rgb_a') < names.index('product_a')
    assert not errors
