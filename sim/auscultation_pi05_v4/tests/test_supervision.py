import pytest
from sim.auscultation_vla_v2.hall import HallSimulator,default_calibration
from sim.auscultation_pi05_v4.supervision import contact_load_estimate


def decode(compression):
    calibration=default_calibration();sensor=HallSimulator(calibration)
    for i in range(1,49):sensor.update(i/240,compression)
    return calibration.decode(sensor.read(.2),now_s=.2)


def test_unloaded_preload_is_not_patient_contact_force():
    unloaded=decode(0.)
    assert unloaded.spring_force_estimate_n>1.4
    assert contact_load_estimate(unloaded)==0
    # Keep raw state representation unchanged for the learned model.
    assert unloaded.spring_force_estimate_n>1.4


def test_loaded_force_keeps_preload_and_four_newton_target():
    loaded=decode(.0025)
    assert contact_load_estimate(loaded)==loaded.spring_force_estimate_n
    assert contact_load_estimate(loaded)==pytest.approx(4.,abs=.03)
