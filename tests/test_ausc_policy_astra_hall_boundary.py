"""Actual observed residual regressions; CPU only, never new physics evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_ausc_policy_m2_capture import FakeEnv, FakeBackend, CAMERA, CALIBRATION
from sim.auscultation_vla_v2.hall import HallCalibration


def make(tmp_path):
    from sim.ausc_policy.astra_sensors import AstraSensors
    env=FakeEnv();env.rows=[];env.base=np.eye(4);env.value['compression_m']=0.
    sensor=AstraSensors(env,tmp_path/'sensors',camera_bytes=CAMERA,calibration_bytes=CALIBRATION,
                        backend_factory=FakeBackend)
    return sensor,env


@pytest.mark.parametrize('raw',[-1.5336825143208443e-08,-1.8642781469679903e-06,
                              -3.809856306074158e-06])
def test_recorded_negative_residual_and_exact_existing_bound_decode_as_zero_without_changing_state(tmp_path,raw):
    sensor,env=make(tmp_path)
    for row in range(3):
        state=env.step_to(row);env.value['compression_m']=raw;state=env.state();env.rows.append({})
        result=sensor.observe_hall(state,row+1)
        assert env.state()['compression_m']==state['compression_m']==raw
        assert result['source_compression_m']==raw
        assert result['hall_sensor_input']==dict(raw_compression_m=raw,sensor_compression_m=0.,zero_boundary_delta_m=-raw)
        assert result['packet']['raw']==31591 and result['packet']['source']=='simulated_adc'
        assert result['decoded']['spring_force_estimate_n']==pytest.approx(1.4975641978331033,abs=1e-12)
    receipt=sensor.close()
    assert receipt['hall_zero_conversion']['normalized_ticks']==3
    assert receipt['hall_zero_conversion']['max_delta_m']==-raw
    assert receipt['hall_zero_conversion']['lower_tolerance_m']==pytest.approx(3.809856306074158e-06,rel=1e-15)


def test_new_observer_rejects_beyond_existing_bound_and_upper_travel(tmp_path):
    sensor,env=make(tmp_path)
    for raw in [np.nextafter(-3.809856306074158e-06,-np.inf),.0072500001]:
        state=env.step_to(0);env.value['compression_m']=raw;env.rows=[{}]
        with pytest.raises(ValueError,match='travel'):sensor.observe_hall(env.state(),1)
        assert env.state()['compression_m']==raw
    assert sensor.close()['completed'] is False


def test_shared_lower_end_policy_is_identical_to_existing_capture_rule():
    from sim.ausc_policy.hall_adc_boundary import lower_endpoint_policy,sensor_input
    cal=HallCalibration(CALIBRATION);policy=lower_endpoint_policy(cal)
    assert policy['method']=='deterministic_adc_equivalent_lower_endpoint_v1'
    assert policy['lower_tolerance_m']==3.809856306074158e-06
    codes=np.rint(np.linspace(policy['zero_quantizer_input'],policy['lower_endpoint_linear_quantizer_input'],10001))
    np.testing.assert_array_equal(np.unique(codes),[1974])
    assert sensor_input(cal,-1.8642781469679903e-06)['sensor_compression_m']==0.
    for bad in (True,np.nan,np.inf,-.001,.008):
        with pytest.raises(ValueError):sensor_input(cal,bad)


def test_helper_source_sha_and_conversion_receipt_are_authoritative_sensor_artifacts(tmp_path):
    from sim.ausc_policy import hall_adc_boundary
    sensor,env=make(tmp_path)
    manifest=json.loads((sensor.output/'manifest.json').read_text())
    path=Path(hall_adc_boundary.__file__)
    assert manifest['hall_zero_conversion_helper']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['source_sha256'][str(path)]==manifest['hall_zero_conversion_helper']['sha256']
    assert manifest['hall_zero_conversion']['raw_physics_unchanged'] is True
    assert manifest['hall_zero_conversion']['physical_safety_gates_unchanged'] is True
    assert sensor.close()['hall_zero_conversion']['normalized_ticks']==0


def test_preload_and_operating_voltage_decode_keep_existing_semantics(tmp_path):
    sensor,env=make(tmp_path)
    zero=sensor.observe_hall(env.state(),0)
    assert zero['decoded']['hall_voltage_v']==pytest.approx(1.590757610437171)
    assert zero['decoded']['spring_force_estimate_n']==pytest.approx(1.4975641978331033)
    assert zero['bench_verified'] is False
    for row in range(3):
        env.step_to(row);env.value.update(compression_m=.0025,hall_N=999.);env.rows.append({})
        result=sensor.observe_hall(env.state(),row+1)
    assert result['decoded']['spring_force_estimate_n']==pytest.approx(4.,abs=.01)
    assert result['decoded']['hall_voltage_v']==pytest.approx(1.3261,abs=.002)
    sensor.close()
