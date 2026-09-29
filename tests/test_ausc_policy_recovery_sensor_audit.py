from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim.auscultation_vla_v2.hall import HallSimulator, default_calibration
from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
from sim.ausc_policy.recovery_sensor_audit import bind_sensor_frames, validate_pixels


def fixture():
    sensor=HallSimulator(default_calibration());physics=[];frames=[]
    for i in range(24):
        t=(i+1)/240
        sensor.update(t,0.)
        row=dict(time_s=t,q=[0.]*7,dq=[0.]*7,compression_m=0.,target='R4',
                 subphase='lower_approach',phase='APPROACH',oracle_role='EXPERT_RECOVERY')
        physics.append(row)
        if (i+1)%12==0:
            frames.append(dict(index=len(frames),physics_row=i,time_s=t,q=row['q'],dq=row['dq'],
                hall=asdict(sensor.read(t)),target='R4',subphase=row['subphase'],phase='FREE',
                prompt=INSTRUCTIONS['FREE'],eligible=False,render_physics_delta_s=0.))
    return physics,frames


def test_exact_240hz_hall_replay_keeps_capture_ineligible():
    physics,frames=fixture();old=deepcopy(frames)
    report=bind_sensor_frames(physics,frames)
    assert report['hall_packets_matched']==2
    assert frames==old
    assert report['training_dataset_eligible'] is False


@pytest.mark.parametrize('field,value', [('eligible',True),('time_s',.051),('q',[.01]*7),
                                       ('phase','HOLD'),('render_physics_delta_s',.001)])
def test_changed_observation_binding_rejected(field,value):
    physics,frames=fixture();frames[0][field]=value
    with pytest.raises(ValueError): bind_sensor_frames(physics,frames)


def test_changed_adc_or_missing_frame_rejected():
    physics,frames=fixture();frames[0]['hall']['raw']+=1
    with pytest.raises(ValueError): bind_sensor_frames(physics,frames)
    physics,frames=fixture()
    with pytest.raises(ValueError): bind_sensor_frames(physics,frames[1:])


def test_raw_pixels_reject_invalid_depth_mask_and_oracle_modalities():
    arrays=dict(rgb=np.zeros((2,3,3),np.uint8),depth_m=np.ones((2,3),np.float32),
                depth_valid=np.ones((2,3),bool))
    validate_pixels(arrays,(2,3))
    arrays['depth_valid'][0,0]=False
    with pytest.raises(ValueError): validate_pixels(arrays,(2,3))
    arrays['depth_valid'][0,0]=True;arrays['force_truth']=np.zeros(1)
    with pytest.raises(ValueError): validate_pixels(arrays,(2,3))


def test_remote_thin_package_can_start_before_calibration_path_is_bound(tmp_path):
    # Deployment deliberately contains sim code, with host calibration in the
    # original approved asset root. CLI startup cannot import it prematurely.
    import sim.ausc_policy.recovery_sensor_audit as module
    script=tmp_path/'replay.py'
    script.write_bytes(Path(module.__file__).read_bytes())
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    completed=subprocess.run([sys.executable,str(script),'--help'],cwd=tmp_path,
        env=env,capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
