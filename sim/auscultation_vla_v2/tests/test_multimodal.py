"""The real input path recomputes target geometry from depth, never caller truth."""
import importlib
import numpy as np
import pytest
from sim.auscultation_vla_v2.hall import HallSimulator,default_calibration


def api():
    try:module=importlib.import_module('sim.auscultation_vla_v2.multimodal')
    except ModuleNotFoundError:module=None
    assert module is not None, 'RGB-D/Hall/joint input assembly is missing'
    return module


def frame():
    rng=np.random.default_rng(42);rgb=rng.integers(30,230,(120,160,3),dtype=np.uint8)
    sensor=HallSimulator(default_calibration());sensor.update(1.,.001)
    return dict(rgb=rgb,depth_m=np.full((120,160),.8,np.float32),depth_valid=np.ones((120,160),bool),
        rgb_time_s=1.,depth_time_s=1.,now_s=1.,q=np.zeros(7),dq=np.zeros(7),
        robot_time_s=1.,hall=sensor.read(1.),phase='APPROACH',target_key='R1',seed_pixel_xy=np.array([80.,60.]))


def assembler(m):
    return m.MultimodalAssembler(np.array([[160.,0,80],[0,160.,60],[0,0,1.]]),np.eye(4))


def test_depth_changes_geometry_on_actual_assembly_path():
    m=api();a=frame();b=frame();b['depth_m']+=.02
    first=assembler(m).observe(**a);second=assembler(m).observe(**b)
    assert np.linalg.norm(second['state'][14:17]-first['state'][14:17])==pytest.approx(.02,abs=.001)
    assert np.array_equal(first['state'][:14],second['state'][:14])
    assert first['modality_receipt']['depth_used_for_geometry'] is True


def test_hall_voltage_and_joint_changes_survive_full_input_assembly():
    m=api();a=frame();b=frame();sensor=HallSimulator(default_calibration());sensor.update(1.,.002)
    b['hall']=sensor.read(1.);b['q'][0]=.05
    first=assembler(m).observe(**a);second=assembler(m).observe(**b)
    assert first['state'][20]!=second['state'][20]
    assert first['state'][21]!=second['state'][21]
    assert first['state'][0]!=second['state'][0]
    assert second['modality_receipt']['wrist_source']=='fk_measured_joints'


def test_seed_and_phase_are_explicit_and_privileged_pose_is_not_accepted():
    m=api();a=frame();a['target_world_m']=np.ones(3)
    with pytest.raises(TypeError):assembler(m).observe(**a)
    a=frame();a['seed_pixel_xy']=None
    with pytest.raises(ValueError,match='seed'):assembler(m).observe(**a)
    a=frame();a['phase']='invalid'
    with pytest.raises(ValueError,match='phase'):assembler(m).observe(**a)


def test_no_missing_depth_or_stale_hall_can_be_silently_imputed():
    m=api();a=frame();a['depth_valid'][:]=False
    with pytest.raises(ValueError):assembler(m).observe(**a)
    a=frame();a.update(now_s=1.1,rgb_time_s=1.1,depth_time_s=1.1,robot_time_s=1.1)
    with pytest.raises(ValueError,match='stale'):assembler(m).observe(**a)


def test_target_switch_requires_new_annotation_and_hold_is_in_prompt():
    m=api();pipe=assembler(m);a=frame();pipe.observe(**a)
    b=frame();b.update(now_s=1.02,rgb_time_s=1.02,depth_time_s=1.02,robot_time_s=1.02,target_key='L1',seed_pixel_xy=None)
    with pytest.raises(ValueError,match='seed'):pipe.observe(**b)
    c=frame();c['phase']='HOLD';encoded=assembler(m).observe(**c)
    assert 'hold' in encoded['prompt'].lower()
    assert encoded['modality_receipt']['audio']=='absent'


@pytest.mark.parametrize('channel',['robot','hall'])
def test_reordered_feedback_is_rejected_before_tracker_advances(channel):
    from dataclasses import replace
    m=api();pipe=assembler(m);pipe.observe(**frame())
    b=frame();b.update(now_s=1.02,rgb_time_s=1.02,depth_time_s=1.02,
                       robot_time_s=1.02,seed_pixel_xy=None)
    if channel=='robot':b['robot_time_s']=.99
    else:b['hall']=replace(b['hall'],sample_time_s=.99)
    with pytest.raises(ValueError,match='reversed'):pipe.observe(**b)
    # The bad packet did not consume this RGB-D frame; fresh feedback recovers.
    b['robot_time_s']=1.02;b['hall']=frame()['hall']
    pipe.observe(**b)
