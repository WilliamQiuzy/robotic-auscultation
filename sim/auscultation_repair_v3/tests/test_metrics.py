import numpy as np
import pytest
from sim.auscultation_repair_v3.metrics import motion_metrics


def test_cubic_derivatives_have_physical_units():
    t=np.arange(1201,dtype=float)/240
    q=np.tile((.01*t**3)[:,None],(1,7))
    report=motion_metrics(t,q,np.tile((.03*t**2)[:,None],(1,7)))
    assert report['forward']['jerk_rad_s3']['max']==pytest.approx(.06,abs=1e-7)
    assert report['centered']['jerk_rad_s3']['max']==pytest.approx(.06,abs=1e-7)
    assert report['stored_velocity']['acceleration_rad_s2']['max']==pytest.approx(.299875,abs=1e-7)
    assert report['samples']==1201


@pytest.mark.parametrize('bad', ['time_order','time_cadence','nonfinite','shape'])
def test_invalid_measurements_fail_closed(bad):
    t=np.arange(20,dtype=float)/240;q=np.zeros((20,7));dq=q.copy()
    if bad=='time_order':t[4]=t[3]
    if bad=='time_cadence':t[4]+=.001
    if bad=='nonfinite':q[3,2]=np.nan
    if bad=='shape':dq=dq[:-1]
    with pytest.raises(ValueError):motion_metrics(t,q,dq)


def test_peak_has_correct_joint_and_timestamp():
    t=np.arange(20,dtype=float)/240;q=np.zeros((20,7));q[10:,5]=.01
    result=motion_metrics(t,q,np.zeros_like(q))
    peak=result['forward']['velocity_rad_s']
    assert peak['joint_index']==5
    assert peak['time_s']==t[10]
    assert peak['max']==pytest.approx(2.4)
