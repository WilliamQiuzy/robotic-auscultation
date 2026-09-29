import pytest
from sim.auscultation_pi05_v5.teacher_timing import gentle_v2


def test_approach_slows_before_touch_and_does_not_reverse_feedback():
    assert gentle_v2('CONTACT', -.002, .02, .005) == -.002
    assert gentle_v2('CONTACT', -.002, .007, .005) == -.0005
    assert gentle_v2('HOLD', .0001, .005, .005) == .0001
    assert gentle_v2('HOLD', -.0001, .005, .005) == -.0001
    assert gentle_v2('HOLD', 0., .005, .005) == 0.


def test_loaded_withdrawal_is_slower_and_outward_only():
    assert gentle_v2('RETRACT', .004, .005, .005) == .00025
    assert gentle_v2('DONE', .004, .008, .005) == .004
    with pytest.raises(ValueError):
        gentle_v2('RETRACT', -.001, .005, .005)


def test_invalid_requests_fail():
    with pytest.raises(ValueError):
        gentle_v2('FREE', 0., .01, .005)
    with pytest.raises(ValueError):
        gentle_v2('CONTACT', float('nan'), .01, .005)
