"""CPU contract tests; fake rendering makes no Isaac or image-quality claim."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from sim.ausc_policy.m2_sensors import M2HallStream

ROOT = Path(__file__).resolve().parents[1]
CAMERA = (ROOT/'sim/camera_framing_20260923/recommended_camera.json').read_bytes()
CALIBRATION = (ROOT/'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json').read_bytes()
GEOMETRY = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                freshly_generated=True)
_OPEN_STREAMS = []


@pytest.fixture(autouse=True)
def close_test_streams():
    yield
    while _OPEN_STREAMS: _OPEN_STREAMS.pop().finalize()


def new_stream(path, **kwargs):
    options = dict(geometry_metadata=GEOMETRY, calibration_bytes=CALIBRATION, episode_id='m2-cpu-test')
    options.update(kwargs)
    stream = M2HallStream(path, **options)
    _OPEN_STREAMS.append(stream)
    return stream


class FakeEnv:
    def __init__(self):
        self.world = SimpleNamespace(current_time=7.)
        self.value = dict(time_s=0., q=np.arange(7)/10., dq=np.zeros(7), compression_m=.001)

    def state(self):
        return deepcopy(self.value)

    def step_to(self, row):
        self.value['time_s'] = (row+1)/240.
        self.world.current_time = 7.+self.value['time_s']
        return self.state()


class FakeBackend:
    def __init__(self, env):
        self.env = env
        camera = json.loads(CAMERA)
        self.camera = dict(camera_path='/World/RobotCam',
            world_from_camera_usd=camera['world_from_camera_usd'],
            intrinsics={key:camera['intrinsics'][key]/3. for key in ('fx','fy','cx','cy')},
            width=640, height=360, meters_per_unit=1., projection='perspective')
        self.rgb = np.full((360,640,4), 71, dtype=np.uint8)
        self.depth = np.full((360,640), .54321, dtype=np.float32)
        self.events = []
        self.on_render = self.on_setup = self.on_read = None
        self.fail_detach = False

    def snapshot_camera(self):
        self.events.append('camera')
        return deepcopy(self.camera)

    def setup(self):
        self.events.append('setup')
        if self.on_setup: self.on_setup()

    def render(self):
        self.events.append('render')
        if self.on_render: self.on_render()

    def read(self):
        self.events.append('read')
        if self.on_read: self.on_read()
        return self.rgb, self.depth

    def detach(self, name):
        self.events.append('detach:'+name)
        if self.fail_detach: raise RuntimeError('detach failed '+name)

    def destroy_product(self):
        self.events.append('destroy')


class PendingSimulator:
    def __init__(self, calibration, **parameters): pass
    def update(self, time_s, compression_m): pass
    def read(self, now_s): return None


def module():
    assert importlib.util.find_spec('sim.ausc_policy.m2_capture') is not None, 'M2 raw capture is missing'
    return importlib.import_module('sim.ausc_policy.m2_capture')


def make(tmp_path, *, env=None, backend=None, hall_stream=None, pending=False, **kwargs):
    env = env or FakeEnv()
    backend = backend or FakeBackend(env)
    hall_stream = hall_stream or new_stream(tmp_path/'hall',
        **(dict(simulator_factory=PendingSimulator) if pending else {}))
    capture = module().M2RGBDCapture(env, tmp_path/'camera', geometry_metadata=GEOMETRY,
        camera_bytes=CAMERA, camera_sha256=hashlib.sha256(CAMERA).hexdigest(),
        calibration_bytes=CALIBRATION, episode_id='m2-cpu-test', hall_stream=hall_stream,
        backend_factory=lambda current: backend, **kwargs)
    return capture, env, backend


def hall_result(capture, row=11, *, first_row=0, compression=.001):
    stream = capture._hall_stream
    result = None
    for i in range(first_row,row+1): result = stream.append(i, (i+1)/240., compression)
    return result


def take(capture, env, hall, *, row=11, **kwargs):
    return capture.capture(env.step_to(row), physics_row=row, phase='APPROACH',
                           target='L2', hall=hall, **kwargs)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_nominal_capture_preserves_raw_rgbd_and_same_step_hall(tmp_path):
    capture, env, backend = make(tmp_path)
    hall = hall_result(capture)
    row = take(capture, env, hall, context=dict(control_owner='L3', subphase='hold'))
    assert row['index'] == 0 and row['physics_row'] == 11 and row['time_s'] == .05
    assert row['phase'] == 'APPROACH' and row['target'] == 'L2'
    assert row['context'] == dict(control_owner='L3', subphase='hold')
    assert row['q'] == env.value['q'].tolist() and row['dq'] == env.value['dq'].tolist()
    assert row['hall'] == hall
    assert row['packet_id'] == hall['packet_id'] and row['acquisition_id'] == hall['acquisition_id']
    assert not hall['is_new_packet'] and not row['hall']['is_new_packet']
    raw = tmp_path/'camera'/row['raw_file']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == row['raw_sha256']
    with np.load(raw, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['rgb'], backend.rgb)
        np.testing.assert_array_equal(data['depth_m'], backend.depth)
        assert data['depth_valid'].all() and data['depth_valid'].dtype == bool
    backend.rgb[:] = 0
    with np.load(raw, allow_pickle=False) as data: assert data['rgb'][0,0,0] == 71
    assert records(tmp_path/'camera/frames.jsonl') == [row]
    metadata = capture.snapshot()
    assert metadata['camera_sha256'] == hashlib.sha256(CAMERA).hexdigest()
    assert metadata['calibration_sha256'] == hashlib.sha256(CALIBRATION).hexdigest()
    assert metadata['audio'] == 'absent' and metadata['depth_semantics'] == 'metric_optical_z'
    assert metadata['geometry'] == GEOMETRY and metadata['source_bindings']
    assert (tmp_path/'camera/approved_camera.json').read_bytes() == CAMERA
    assert (tmp_path/'camera/calibration.json').read_bytes() == CALIBRATION
    assert not metadata['training_eligible'] and not metadata['physical_qualification']
    receipt = capture.finalize()
    assert receipt['completed'] and receipt['frames'] == 1
    assert not receipt['training_eligible'] and not receipt['physical_qualification']
    assert backend.events[-3:] == ['detach:rgb','detach:distance_to_image_plane','destroy']
    assert capture.finalize() == receipt
    with pytest.raises(RuntimeError, match='finalized'): take(capture, env, hall)


def test_pending_hall_is_explicit_without_invented_packet(tmp_path):
    capture, env, _ = make(tmp_path, pending=True)
    hall = hall_result(capture)
    row = take(capture, env, hall)
    assert row['hall']['status'] == 'pending'
    assert row['hall']['packet'] is None and row['acquisition_id'] is None
    assert capture.finalize()['completed']


@pytest.mark.parametrize('mismatch',['episode','geometry','calibration'])
def test_constructor_rejects_a_different_hall_stream_identity(tmp_path,mismatch):
    options = {}
    if mismatch == 'episode': options['episode_id'] = 'other'
    if mismatch == 'geometry': options['geometry_metadata'] = dict(GEOMETRY, collision_geometry_sha256='a'*64)
    if mismatch == 'calibration': options['calibration_bytes'] = CALIBRATION+b'\n'
    stream = new_stream(tmp_path/'other-hall',**options)
    with pytest.raises(ValueError,match='Hall stream'):
        make(tmp_path,hall_stream=stream)
    assert not (tmp_path/'camera').exists()


@pytest.mark.parametrize('problem',['replacement','compression','stale','closed','failed'])
def test_frame_requires_bound_live_hall_and_measured_raw_compression(tmp_path,problem):
    capture, env, backend = make(tmp_path)
    hall = hall_result(capture)
    if problem == 'replacement':
        other = new_stream(tmp_path/'other-hall',episode_id='other')
        for i in range(12): hall = other.append(i,(i+1)/240.,.002)
    elif problem == 'compression': env.value['compression_m'] = .002
    elif problem == 'stale': capture._hall_stream.append(12,13/240.,.001)
    elif problem == 'closed': capture._hall_stream.finalize()
    else:
        with pytest.raises(ValueError): capture._hall_stream.append(13,14/240.,.001)
    with pytest.raises((ValueError,RuntimeError)): take(capture,env,hall)
    assert 'render' not in backend.events and not records(tmp_path/'camera/frames.jsonl')
    receipt = capture.finalize()
    assert not receipt['completed'] and backend.events[-1] == 'destroy'


def test_frame_records_hall_source_and_raw_compression_with_sensor_noise(tmp_path):
    stream = new_stream(tmp_path/'hall',condition=dict(noise_std_v=.0003,seed=19))
    capture,env,_ = make(tmp_path,hall_stream=stream)
    hall = hall_result(capture)
    row = take(capture,env,hall)
    evidence = stream.latest_evidence()
    assert row['hall_physics'] == evidence['physics']
    assert row['hall_stream_manifest_sha256'] == evidence['manifest_sha256']
    assert capture.snapshot()['hall_stream']['snapshot'] == stream.snapshot()
    assert row['hall']['decoded']['displacement_estimate_m'] != row['hall_physics']['compression_m']
    assert capture.finalize()['completed']


def test_second_frame_keeps_20hz_time_and_different_adc_identity(tmp_path):
    capture, env, _ = make(tmp_path)
    first = take(capture,env,hall_result(capture))
    second = take(capture,env,hall_result(capture,row=23,first_row=12),row=23)
    assert second['index'] == 1 and second['time_s'] == .1
    assert first['acquisition_id'] != second['acquisition_id']
    assert capture.finalize()['frames'] == 2


def test_depth_retains_nan_inf_negative_and_valid_pixels_exactly(tmp_path):
    capture, env, backend = make(tmp_path)
    backend.depth[0,:5] = [np.nan, np.inf, -np.inf, 0., -.1]
    expected = backend.depth.copy()
    row = take(capture, env, hall_result(capture))
    with np.load(tmp_path/'camera'/row['raw_file'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['depth_m'], expected)
        np.testing.assert_array_equal(data['depth_valid'], np.isfinite(expected)&(expected>0))
    capture.finalize()


@pytest.mark.parametrize('row', [0,10,12,23,True])
def test_first_frame_must_be_physics_row_eleven(tmp_path, row):
    capture, env, backend = make(tmp_path)
    with pytest.raises(ValueError, match='physics row'): take(capture, env, hall_result(capture), row=row)
    assert 'render' not in backend.events
    receipt = capture.finalize()
    assert not receipt['completed'] and receipt['failures']
    assert records(tmp_path/'camera/attempts.jsonl')[-1]['status'] == 'error'


@pytest.mark.parametrize('next_row', [11,12,35])
def test_repeated_or_missing_frame_stops_the_episode(tmp_path, next_row):
    capture, env, _ = make(tmp_path)
    hall = hall_result(capture)
    take(capture, env, hall)
    with pytest.raises(ValueError, match='physics row'): take(capture, env, hall, row=next_row)
    with pytest.raises(RuntimeError, match='failed'): take(capture, env, hall, row=23)
    assert not capture.finalize()['completed']
    assert len(records(tmp_path/'camera/frames.jsonl')) == 1


@pytest.mark.parametrize('mutation', ['time','q','dq','compression','hall_row','hall_time','calibration','decoded','target','phase'])
def test_caller_state_hall_and_context_must_match_actual_step(tmp_path, mutation):
    capture, env, backend = make(tmp_path)
    state = env.step_to(11)
    hall = hall_result(capture)
    phase, target = 'FREE', 'R5'
    if mutation == 'time': state['time_s'] = 0.
    if mutation == 'q': state['q'][0] += .1
    if mutation == 'dq': state['dq'][0] += .1
    if mutation == 'compression': env.value['compression_m'] += 1e-10
    if mutation == 'hall_row': hall['physics_row'] = 10
    if mutation == 'hall_time': hall['time_s'] -= 1/240.
    if mutation == 'calibration': hall['packet']['calibration_sha256'] = '0'*64
    if mutation == 'decoded': hall['decoded']['force_N'] = 777.
    if mutation == 'target': target = 'R6'
    if mutation == 'phase': phase = ''
    with pytest.raises(ValueError): capture.capture(state, physics_row=11, phase=phase, target=target, hall=hall)
    assert 'render' not in backend.events and not capture.finalize()['completed']


@pytest.mark.parametrize('mutation', ['world_time','time','q','dq'])
def test_render_must_not_advance_any_measured_physics(tmp_path, mutation):
    capture, env, backend = make(tmp_path)
    def move():
        if mutation == 'world_time': env.world.current_time += 1/240.
        elif mutation == 'time': env.value['time_s'] += 1/240.
        else: env.value[mutation][0] += .0001
    backend.on_render = move
    with pytest.raises(ValueError, match='render'): take(capture, env, hall_result(capture))
    assert not list((tmp_path/'camera').glob('*.npz'))
    receipt = capture.finalize()
    assert not receipt['completed'] and backend.events[-1] == 'destroy'


@pytest.mark.parametrize('mutation', ['world_time','time','q','dq','camera_pose','intrinsics','hall_step'])
def test_read_must_preserve_the_original_physics_and_camera_snapshot(tmp_path, mutation):
    capture, env, backend = make(tmp_path)
    def move():
        if mutation == 'world_time': env.world.current_time += 1/240.
        elif mutation == 'time': env.value['time_s'] += 1/240.
        elif mutation == 'camera_pose': backend.camera['world_from_camera_usd'][0][3] += .001
        elif mutation == 'intrinsics': backend.camera['intrinsics']['fx'] *= 1.001
        elif mutation == 'hall_step': capture._hall_stream.append(12,13/240.,.001)
        else: env.value[mutation][0] += .01
    backend.on_read = move
    with pytest.raises(ValueError): take(capture,env,hall_result(capture))
    assert not records(tmp_path/'camera/frames.jsonl')
    assert not list((tmp_path/'camera').glob('*.npz'))
    failed = records(tmp_path/'camera/attempts.jsonl')[-1]
    assert failed['status'] == 'error' and failed['failure']['stage'] == 'validate_after_read'
    receipt = capture.finalize()
    assert not receipt['completed'] and receipt['frames'] == 0
    assert not receipt['training_eligible'] and not receipt['physical_qualification']
    assert backend.events[-3:] == ['detach:rgb','detach:distance_to_image_plane','destroy']


def test_post_read_camera_check_runs_after_owned_rgb_and_depth_copies(tmp_path):
    capture, env, backend = make(tmp_path)
    original = backend.snapshot_camera
    def overwrite_borrowed_buffers():
        if 'read' in backend.events:
            backend.rgb[:] = 0
            backend.depth[:] = 0
        return original()
    backend.snapshot_camera = overwrite_borrowed_buffers
    row = take(capture,env,hall_result(capture))
    assert not backend.rgb.any() and not backend.depth.any()
    with np.load(tmp_path/'camera'/row['raw_file'],allow_pickle=False) as data:
        assert np.all(data['rgb'] == 71)
        assert np.all(data['depth_m'] == np.float32(.54321))
        assert data['depth_valid'].all()
    assert capture.finalize()['completed']


@pytest.mark.parametrize('when,what', [('setup','transform'),('frame','transform'),('frame','intrinsics')])
def test_camera_is_verified_and_never_repositioned(tmp_path, when, what):
    env = FakeEnv(); backend = FakeBackend(env)
    def move():
        if what == 'transform': backend.camera['world_from_camera_usd'][0][3] += .001
        else: backend.camera['intrinsics']['fx'] *= 1.001
    if when == 'setup':
        move()
        with pytest.raises(ValueError, match='camera'): make(tmp_path, env=env, backend=backend)
        assert 'setup' not in backend.events
        assert backend.events[-1] == 'destroy'
    else:
        capture, _, _ = make(tmp_path, env=env, backend=backend)
        move()
        with pytest.raises(ValueError, match='camera'): take(capture, env, hall_result(capture))
        assert 'render' not in backend.events and not capture.finalize()['completed']


@pytest.mark.parametrize('sensor', ['rgb_dtype','rgb_shape','depth_dtype','depth_shape'])
def test_invalid_render_data_is_rejected_with_error_evidence(tmp_path, sensor):
    capture, env, backend = make(tmp_path)
    if sensor == 'rgb_dtype': backend.rgb = backend.rgb.astype(float)
    if sensor == 'rgb_shape': backend.rgb = backend.rgb[:350]
    if sensor == 'depth_dtype': backend.depth = backend.depth.astype(np.uint16)
    if sensor == 'depth_shape': backend.depth = backend.depth[...,None]
    with pytest.raises(ValueError): take(capture, env, hall_result(capture))
    assert not capture.finalize()['completed']
    assert records(tmp_path/'camera/attempts.jsonl')[-1]['status'] == 'error'


def test_raw_write_failure_preserves_attempt_and_still_cleans_resources(tmp_path, monkeypatch):
    capture, env, backend = make(tmp_path)
    def fail(*args, **kwargs): raise OSError('disk failure')
    monkeypatch.setattr(module().np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='disk failure'): take(capture, env, hall_result(capture))
    assert not list((tmp_path/'camera').glob('*.npz'))
    assert not capture.finalize()['completed']
    assert backend.events[-3:] == ['detach:rgb','detach:distance_to_image_plane','destroy']


def test_frame_ledger_failure_keeps_complete_hashed_raw_file(tmp_path, monkeypatch):
    capture, env, _ = make(tmp_path)
    original = module()._append
    def fail_frame(stream, value):
        if Path(stream.name).name == 'frames.jsonl':
            raw = tmp_path/'camera'/value['raw_file']
            assert raw.is_file() and hashlib.sha256(raw.read_bytes()).hexdigest() == value['raw_sha256']
            raise OSError('frame ledger full')
        return original(stream,value)
    monkeypatch.setattr(module(),'_append',fail_frame)
    with pytest.raises(OSError,match='ledger full'): take(capture,env,hall_result(capture))
    receipt = capture.finalize()
    assert not receipt['completed'] and receipt['frames'] == 0
    error = records(tmp_path/'camera/attempts.jsonl')[-1]
    assert error['status'] == 'error' and error['raw_sha256']
    assert (tmp_path/'camera'/error['raw_file']).is_file()


def test_publish_failure_retains_partial_npz_without_a_false_frame(tmp_path, monkeypatch):
    capture, env, _ = make(tmp_path)
    def fail(*args): raise OSError('rename failed')
    monkeypatch.setattr(module().os,'replace',fail)
    with pytest.raises(OSError,match='rename failed'): take(capture,env,hall_result(capture))
    assert not capture.finalize()['completed']
    error = records(tmp_path/'camera/attempts.jsonl')[-1]
    assert (tmp_path/'camera'/error['partial_file']).is_file()
    assert not (tmp_path/'camera'/error['raw_file']).exists()


def test_finalize_hashes_ledgers_with_bounded_reads(tmp_path, monkeypatch):
    capture, env, _ = make(tmp_path)
    take(capture,env,hall_result(capture))
    def forbid(*args): raise AssertionError('unbounded read_bytes')
    monkeypatch.setattr(Path,'read_bytes',forbid)
    assert capture.finalize()['completed']


def test_time_varying_camera_is_refused_before_setup(tmp_path):
    env = FakeEnv(); backend = FakeBackend(env)
    backend.camera['time_varying'] = True
    with pytest.raises(ValueError,match='camera'):
        make(tmp_path,env=env,backend=backend)
    assert 'setup' not in backend.events


def test_cleanup_attempts_every_resource_even_when_detach_and_receipt_write_fail(tmp_path, monkeypatch):
    capture, env, backend = make(tmp_path)
    take(capture, env, hall_result(capture))
    backend.fail_detach = True
    original = module()._write_json
    def fail_receipt(path, value):
        if Path(path).name == 'receipt.json': raise OSError('receipt disk failure')
        return original(path, value)
    monkeypatch.setattr(module(), '_write_json', fail_receipt)
    receipt = capture.finalize(error='caller failure')
    assert not receipt['completed'] and len(receipt['failures']) >= 4
    assert backend.events[-3:] == ['detach:rgb','detach:distance_to_image_plane','destroy']
    assert receipt['receipt_written'] is False


def test_setup_failure_cleans_partial_resources(tmp_path):
    env = FakeEnv(); backend = FakeBackend(env)
    def fail(): raise RuntimeError('setup failed')
    backend.on_setup = fail
    with pytest.raises(RuntimeError, match='setup failed'): make(tmp_path, env=env, backend=backend)
    assert backend.events[-3:] == ['detach:rgb','detach:distance_to_image_plane','destroy']
    receipt = json.loads((tmp_path/'camera/receipt.json').read_text())
    assert not receipt['completed']


def test_camera_bytes_and_geometry_cannot_silently_change(tmp_path):
    cls = module().M2RGBDCapture
    for name, geometry, data, digest in [
        ('geometry', {**GEOMETRY,'geometry_id':'old'}, CAMERA, hashlib.sha256(CAMERA).hexdigest()),
        ('hash', GEOMETRY, CAMERA, '0'*64),
        ('camera', GEOMETRY, CAMERA+b' ', hashlib.sha256(CAMERA+b' ').hexdigest()),
    ]:
        with pytest.raises(ValueError): cls(FakeEnv(), tmp_path/name, geometry_metadata=geometry,
            camera_bytes=data, camera_sha256=digest, episode_id='bad',
            hall_stream=new_stream(tmp_path/(name+'-hall'),episode_id='bad'))
        assert not (tmp_path/name).exists()


def test_default_backend_refuses_unowned_or_mac_execution_before_native_import(tmp_path, monkeypatch):
    monkeypatch.delenv('AUSC_OWNED_RUN', raising=False)
    with pytest.raises(RuntimeError, match='owned Linux'):
        module().M2RGBDCapture(FakeEnv(), tmp_path/'native', geometry_metadata=GEOMETRY,
            camera_bytes=CAMERA, camera_sha256=hashlib.sha256(CAMERA).hexdigest(), episode_id='bad',
            hall_stream=new_stream(tmp_path/'hall',episode_id='bad'))


def test_native_setup_owns_new_product_when_video_uses_same_camera_and_resolution():
    class Product:
        def __init__(self): self.destroyed = False
        def destroy(self): self.destroyed = True
    class Annotator:
        def __init__(self): self.attached = None
        def attach(self, products): self.attached = products
        def detach(self): self.attached = None
    video_product = Product()
    calls, annotators = [], {}
    global_setting = {'capture_on_play':True}
    def set_capture_on_play(value): global_setting['capture_on_play'] = value
    def render_product(camera, resolution, **kwargs):
        calls.append((camera,resolution,kwargs))
        return Product() if kwargs.get('force_new',False) else video_product
    def get_annotator(name):
        annotators[name] = Annotator()
        return annotators[name]
    # Exercise the native ownership logic against a CPU Replicator stub,
    # without constructing USD/Kit or bypassing guards for a real renderer.
    backend = module()._NativeBackend.__new__(module()._NativeBackend)
    backend.product, backend.annotators = None, {}
    backend.rep = SimpleNamespace(create=SimpleNamespace(render_product=render_product),
        AnnotatorRegistry=SimpleNamespace(get_annotator=get_annotator),
        orchestrator=SimpleNamespace(set_capture_on_play=set_capture_on_play))
    backend.setup()
    owned_product = backend.product
    assert owned_product is not video_product
    assert calls == [('/World/RobotCam',(640,360),{'force_new':True})]
    assert all(annotator.attached == [owned_product] for annotator in annotators.values())
    for name in module().ANNOTATORS: backend.detach(name)
    backend.destroy_product()
    assert owned_product.destroyed and not video_product.destroyed
    assert all(annotator.attached is None for annotator in annotators.values())
    assert global_setting['capture_on_play'] is True


def test_capture_does_not_inherit_legacy_encoder_or_camera_writes():
    import ast
    current = module()
    tree = ast.parse(Path(current.__file__).read_text())
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(any(part in name for part in ('live_capture','observations','body_tracking','m2_contract')) for name in imports)
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not set(calls)&{'Set','ClearXformOpOrder','AddTransformOp','encode_state'}
    direct = [node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    assert not set(direct)&{'encode','encode_state'}
    assert current.M2RGBDCapture.__bases__ == (object,)
