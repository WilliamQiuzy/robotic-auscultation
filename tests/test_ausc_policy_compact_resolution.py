"""CPU image-size protocol tests, never native render/physics acceptance."""
from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from sim.ausc_policy import m2_capture as camera,training_capture as writer,expert_capture
from test_ausc_policy_m2_capture import FakeEnv,FakeBackend,CAMERA
from test_ausc_policy_training_capture import writer as make_writer,frame,rebind_artifacts
from test_ausc_policy_compact_expert_capture import compact_binding,tick
from test_ausc_policy_expert_capture import Env,GEOMETRY


class SmallBackend(FakeBackend):
    def __init__(self,env):
        super().__init__(env)
        self.camera.update(width=256,height=144)
        self.camera['intrinsics']={key:value*.4 for key,value in self.camera['intrinsics'].items()}
        self.rgb=np.full((144,256,4),71,np.uint8);self.depth=np.full((144,256),.54321,np.float32)


def test_sampler_defaults_to_native_policy_resolution_with_unchanged_view():
    env=FakeEnv();backend=SmallBackend(env)
    sampler=camera.M2CameraSampler(env,camera_bytes=CAMERA,backend_factory=lambda _:backend)
    rgb,depth,valid=sampler.capture(env.state())
    assert rgb.shape==(144,256,4) and depth.shape==(144,256) and valid.all()
    assert env.state()['time_s']==0 and env.world.current_time==7.
    receipt=sampler.finalize()
    assert receipt['native_image_size']==[256,144] and receipt['completed']
    assert receipt['actual_camera_at_setup']['world_from_camera_usd']==json.loads(CAMERA)['world_from_camera_usd']


def test_large_fixture_must_explicitly_declare_actual_sampler_size():
    env=FakeEnv()
    with pytest.raises(ValueError):camera.M2CameraSampler(env,camera_bytes=CAMERA,backend_factory=FakeBackend)
    sampler=camera.M2CameraSampler(env,camera_bytes=CAMERA,backend_factory=FakeBackend,native_image_size=(640,360))
    assert sampler.capture(env.state())[0].shape==(360,640,4)
    assert sampler.finalize()['native_image_size']==[640,360]


@pytest.mark.parametrize('mutation',['shape','optics','pose','clock'])
def test_small_sampler_preserves_camera_and_zero_step_guards(mutation):
    env=FakeEnv();backend=SmallBackend(env)
    sampler=camera.M2CameraSampler(env,camera_bytes=CAMERA,backend_factory=lambda _:backend)
    if mutation=='shape':backend.depth=np.ones((360,640),np.float32)
    elif mutation=='optics':backend.camera['intrinsics']['fx']*=1.1
    elif mutation=='pose':backend.camera['world_from_camera_usd'][0][3]+=.001
    else:backend.on_render=lambda:setattr(env.world,'current_time',8.)
    with pytest.raises(ValueError):sampler.capture(env.state())
    assert not sampler.finalize()['completed']


def test_native_render_product_uses_explicit_size_without_camera_mutation():
    backend=camera._NativeBackend.__new__(camera._NativeBackend)
    backend.native_image_size=(256,144);backend.annotators={};calls=[]
    product=object()
    def render_product(path,size,**kw):calls.append((path,size,kw));return product
    backend.rep=SimpleNamespace(create=SimpleNamespace(render_product=render_product),
        AnnotatorRegistry=SimpleNamespace(get_annotator=lambda _:SimpleNamespace(attach=lambda _:None)))
    backend.setup()
    assert calls==[(camera.CAMERA_PATH,(256,144),dict(force_new=True))]


def test_sampler_passes_policy_size_to_owned_native_backend(monkeypatch):
    calls=[]
    def native(env,*,native_image_size):calls.append(native_image_size);return SmallBackend(env)
    monkeypatch.setattr(camera,'_NativeBackend',native)
    env=FakeEnv();sampler=camera.M2CameraSampler(env,camera_bytes=CAMERA)
    assert calls==[(256,144)]
    assert sampler.capture(env.state())[1].shape==(144,256)
    sampler.finalize()


def test_direct_native_depth_keeps_metric_values_holes_and_intrinsics(tmp_path):
    w=make_writer(tmp_path,expected_frames=1,native_image_size=(256,144));row=frame()
    row['rgb']=np.full((144,256,3),33,np.uint8)
    depth=np.linspace(.2,12.,144*256,dtype=np.float32).reshape(144,256);depth[1,2]=np.nan
    row.update(depth_m=depth,depth_valid=np.isfinite(depth)&(depth>0))
    w.append(**row);w.close()
    manifest=json.loads((w.output/'manifest.json').read_text())
    assert manifest['native_image_size']==[256,144] and manifest['depth_resize']=='identity; direct native policy resolution'
    assert manifest['native_intrinsics']==manifest['policy_intrinsics']
    original=json.loads(CAMERA)['intrinsics']
    assert manifest['policy_intrinsics']['fx']==pytest.approx(original['fx']*256/original['width'])
    actual=list(writer.iter_training_records(w.output))[0]
    valid=actual['depth_valid'];assert not valid[1,2]
    np.testing.assert_array_equal(actual['depth_q'][valid],np.rint(depth[valid].astype(float)/.0001).astype(np.uint32))
    assert actual['depth_q'][1,2]==0


@pytest.mark.parametrize('size',[(512,288),(640,144),(True,144),(256.0,144)])
def test_only_explicit_supported_sizes_are_accepted(tmp_path,size):
    with pytest.raises(ValueError):make_writer(tmp_path,native_image_size=size)


def test_writer_rejects_large_arrays_labeled_as_direct_small_native(tmp_path):
    w=make_writer(tmp_path,native_image_size=(256,144))
    with pytest.raises(ValueError,match='RGB'):w.append(**frame())


@pytest.mark.parametrize('field,value',[('native_image_size',[640,360]),('native_intrinsics',{}),
    ('policy_intrinsics',{}),('depth_resize','bilinear'),('depth_semantics','normalized'),
    ('image_input_contract','unknown')])
def test_reader_rejects_false_resolution_or_depth_contract_after_rehash(tmp_path,field,value):
    w=make_writer(tmp_path,expected_frames=1,native_image_size=(256,144));row=frame()
    row.update(rgb=np.zeros((144,256,3),np.uint8),depth_m=np.ones((144,256)),depth_valid=np.ones((144,256),bool))
    w.append(**row);w.close()
    path=w.output/'manifest.json';manifest=json.loads(path.read_text());manifest[field]=value
    path.write_text(json.dumps(manifest));rebind_artifacts(w.output,['manifest.json'])
    with pytest.raises(ValueError,match='image contract'):list(writer.iter_training_records(w.output))


def test_existing_640_recordings_without_new_optional_intrinsics_still_read(tmp_path):
    w=make_writer(tmp_path,expected_frames=1);w.append(**frame());w.close()
    path=w.output/'manifest.json';manifest=json.loads(path.read_text())
    manifest.pop('image_input_contract');manifest.pop('native_intrinsics')
    path.write_text(json.dumps(manifest));rebind_artifacts(w.output,['manifest.json'])
    assert len(list(writer.iter_training_records(w.output)))==1


def test_compact_default_pipeline_direct_size_and_fullraw_fixture_unchanged(tmp_path):
    binding=compact_binding();binding['compact'].pop('native_image_size',None)
    env=Env();capture=expert_capture.create_expert_capture(tmp_path/'compact',binding=binding,geometry_metadata=GEOMETRY)
    capture.start(env,backend_factory=SmallBackend)
    for _ in range(27):tick(env,capture)
    capture.close();result=capture.result(27,execution_completed=True)
    assert result['compact_capture_complete'] and result['total_bytes']<=100_000_000
    outer=json.loads((capture.output/'manifest.json').read_text())
    inner=json.loads((capture.output/'compact/manifest.json').read_text())
    assert outer['native_image_size']==inner['native_image_size']==[256,144]
    assert len(list(writer.iter_training_records(capture.output/'compact')))==2
