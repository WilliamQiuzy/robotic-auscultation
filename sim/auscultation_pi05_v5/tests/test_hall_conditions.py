import numpy as np
import pytest
from sim.auscultation_pi05_v5.hall_conditions import make_sensor,validate_condition
from sim.auscultation_vla_v2.hall import default_calibration


def packets(condition):
    s=make_sensor(default_calibration(),condition);raw=[]
    for i in range(100):
        t=i*.01;s.update(t,.002);raw.append(s.read(t).raw)
    return np.asarray(raw)


def test_nominal_is_unchanged_and_noise_acts_in_voltage_adc_domain():
    a=packets(None);assert np.all(a==a[0])
    b=packets({'noise_std_v':.001,'seed':1});c=packets({'noise_std_v':.001,'seed':2})
    np.testing.assert_array_equal(b,packets({'noise_std_v':.001,'seed':1}))
    assert np.sum(b!=c)>30
    assert np.sum(b!=a)>30


@pytest.mark.parametrize('bad',[{'noise_std_v':float('nan')},{'voltage_bias_v':.1},{'seed':True},{'force_noise_N':.1}])
def test_conditions_are_explicit_and_bounded(bad):
    with pytest.raises(ValueError):validate_condition(bad)
