from dataclasses import asdict
import hashlib
import numpy as np
import pytest
from sim.auscultation_vla_v2.body_tracking import BodyTracker
from sim.auscultation_vla_v2.tests.test_body_tracking import frame,K,POLY,SEEDS
from sim.auscultation_vla_v2.hall import HallSimulator,default_calibration
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_pi05_v4.live_capture import LiveCapture,CachedTracker


def test_same_acquired_frame_can_select_second_target_without_retracking(tmp_path):
    rgb,depth,valid=frame();capture=LiveCapture.__new__(LiveCapture)
    capture.K=K;capture.camera_to_base=np.eye(4);capture.out=tmp_path
    capture.tracker=CachedTracker(BodyTracker(K,np.eye(4)))
    capture.tracker.initialize(rgb,depth,valid,SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    capture.cal=default_calibration();capture.hall=HallSimulator(capture.cal);capture.kin=OpenArmKinematics()
    for i in range(1,241):capture.hall.update(i/240,0.)
    raw=tmp_path/'00000.npz';np.savez_compressed(raw,rgb=rgb,depth_m=depth,depth_valid=valid)
    packet=capture.hall.read(1.)
    capture.rows=[dict(index=0,time_s=1.,physics_row=239,target='R1',phase='FREE',q=[0.]*7,dq=[0.]*7,
        raw_file=raw.name,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),hall=asdict(packet))]
    state=dict(time_s=1.,q=np.zeros(7),dq=np.zeros(7))
    capture.capture(state,'FREE','L1',239,eligible=True,subphase='new_target_observation')
    assert capture.rows[-1]['target']=='L1'
    assert capture.rows[-1]['reused_camera_frame'] is True
    assert capture.tracker._stamp==1. and not capture.tracker.lost
    assert len(list(tmp_path.glob('*.npz')))==1
    assert len(capture.rows[-1]['state'])==32
    # Negative control: invoking the actual tracker twice does reject the time.
    with pytest.raises(ValueError,match='repeated'):
        capture.tracker.update(rgb,depth,valid,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
