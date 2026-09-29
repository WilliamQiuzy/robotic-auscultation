import pytest
from sim.auscultation_repair_v3.gentle_contact import limit_near_contact


def test_loaded_withdrawal_does_not_use_free_space_speed():
    assert limit_near_contact('RETRACT',.004,.003,.005)==.0005
    assert limit_near_contact('RETRACT',.004,.01,.005)==.004


def test_contact_approach_slows_before_nominal_touch():
    assert limit_near_contact('CONTACT',-.002,.007,.005)==-.001
    assert limit_near_contact('CONTACT',-.002,.02,.005)==-.002
    assert limit_near_contact('HOLD',0.,.003,.005)==0.


def test_never_increase_small_corrections_or_reverse_direction():
    assert limit_near_contact('RETRACT',.0001,.003,.005)==.0001
    assert limit_near_contact('HOLD',-.00003,.003,.005)==-.00003
    with pytest.raises(ValueError):limit_near_contact('RETRACT',-.001,.003,.005)
    with pytest.raises(ValueError):limit_near_contact('CONTACT',.001,float('nan'),.005)
