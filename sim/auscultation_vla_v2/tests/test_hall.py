"""Sensor-domain regressions: the policy must see an ADC-derived observation."""
import importlib
from dataclasses import replace
import numpy as np
import pytest


def api():
    try:
        module = importlib.import_module('sim.auscultation_vla_v2.hall')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'Hall voltage/ADC simulation adapter missing'
    return module


def test_pressed_voltage_decreases_and_matches_existing_host_conversion():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    voltages=[]
    for i,compression in enumerate([0.,.0025,.004,.00725]):
        t=i*.02; sim.update(t,compression)
        packet=sim.read(t); obs=cal.decode(packet,now_s=t)
        voltages.append(obs.hall_voltage_v)
        assert abs(obs.displacement_estimate_m-compression)<.00002
        assert obs.spring_force_estimate_n==pytest.approx(1.5+1000*compression,abs=.015)
        assert obs.hall_voltage_v==pytest.approx(packet.raw/65535*3.3)
    assert np.all(np.diff(voltages)<0)
    assert voltages[0]==pytest.approx(1.5905,abs=.001)
    assert voltages[-1]==pytest.approx(.8236,abs=.001)
    assert cal.bench_verified is False  # File's calibrated:true is not a bench result.


def test_unloaded_preload_is_not_zero_patient_force():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    sim.update(0.,0.); obs=cal.decode(sim.read(0.),now_s=0.)
    assert obs.spring_force_estimate_n==pytest.approx(1.5,abs=.015)
    assert not hasattr(obs,'patient_contact_force_n')


def test_latency_does_not_refresh_original_sample_timestamp():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal,latency_s=.03)
    sim.update(1.,.001)
    assert sim.read(1.02) is None
    packet=sim.read(1.03)
    assert packet.sample_time_s==1. and packet.received_time_s==1.03
    with pytest.raises(ValueError,match='stale'):
        cal.decode(packet,now_s=1.10,max_age_s=.05)


def test_sampling_and_dropout_preserve_last_measurement_age():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    sim.update(0.,0.); first=sim.read(0.)
    sim.update(.005,.003)
    assert sim.read(.005)==first
    sim.update(.01,.003,drop=True)
    assert sim.read(.03)==first
    with pytest.raises(ValueError,match='stale'):
        cal.decode(sim.read(.08),now_s=.08,max_age_s=.05)


def test_same_raw_measurement_has_same_features_on_sim_and_real():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    sim.update(1.,.0025); p=sim.read(1.)
    a=cal.decode(p,now_s=1.); b=cal.decode(replace(p,source='real_adc'),now_s=1.)
    assert a.hall_voltage_v==b.hall_voltage_v and a.spring_force_estimate_n==b.spring_force_estimate_n


@pytest.mark.parametrize('change',[
    {'raw':-1},{'raw':65536},{'raw':True},{'raw':1.2},{'raw':0},{'raw':65535},
    {'sample_time_s':float('nan')},{'received_time_s':.5},{'sample_time_s':2.},
    {'calibration_sha256':'wrong'},{'source':'physx_force'},
])
def test_malformed_saturated_or_privileged_packets_rejected(change):
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    sim.update(1.,.0025); packet=replace(sim.read(1.),**change)
    with pytest.raises(ValueError): cal.decode(packet,now_s=1.)


def test_noise_is_applied_in_voltage_domain_before_adc_and_host_conversion():
    h=api(); cal=h.default_calibration()
    sim=h.HallSimulator(cal,voltage_bias_v=.01)
    sim.update(0.,.0025); obs=cal.decode(sim.read(0.),now_s=0.)
    assert obs.hall_voltage_v==pytest.approx(1.3361,abs=.001)
    assert obs.spring_force_estimate_n<3.93


def test_curve_error_changes_estimate_without_changing_decoder():
    h=api(); cal=h.default_calibration()
    a=h.HallSimulator(cal); b=h.HallSimulator(cal,curve_warp=.2)
    for model in [a,b]:model.update(0.,.0025)
    assert cal.decode(a.read(0.),now_s=0.).spring_force_estimate_n != cal.decode(b.read(0.),now_s=0.).spring_force_estimate_n


def test_rp2040_wire_value_is_scaled_12bit_not_true_16bit_adc():
    h=api(); cal=h.default_calibration(); sim=h.HallSimulator(cal)
    sim.update(0.,.0025); raw=sim.read(0.).raw
    possible={(n<<4)|(n>>8) for n in range(4096)}
    assert raw in possible


def test_reproducible_voltage_noise_and_no_time_reversal():
    h=api(); cal=h.default_calibration()
    a=h.HallSimulator(cal,noise_std_v=.003,seed=3); b=h.HallSimulator(cal,noise_std_v=.003,seed=3)
    for i in range(20):
        for s in [a,b]:s.update(i*.01,.0025)
        assert a.read(i*.01)==b.read(i*.01)
    with pytest.raises(ValueError):a.update(.1,.0025)


@pytest.mark.parametrize('kwargs',[{'noise_std_v':-1.},{'latency_s':-1.},{'sample_period_s':0.},{'voltage_bias_v':float('nan')},{'curve_warp':2.}])
def test_invalid_sensor_configuration_rejected(kwargs):
    h=api()
    with pytest.raises(ValueError):h.HallSimulator(h.default_calibration(),**kwargs)


@pytest.mark.parametrize('compression',[-.001,.008,float('nan')])
def test_impossible_slider_positions_fail_instead_of_clipping(compression):
    h=api();sim=h.HallSimulator(h.default_calibration())
    with pytest.raises(ValueError):sim.update(0.,compression)


def test_100hz_sensor_schedule_does_not_drift_to_80hz_on_240hz_physics():
    h=api();sim=h.HallSimulator(h.default_calibration());stamps=set()
    for i in range(480):
        t=i/240.;sim.update(t,.0025)
        stamps.add(sim.read(t).sample_time_s)
    assert len(stamps)==200
