import pytest
from sim.auscultation_pi05_v6.teacher_timing import measured_gentle_v1


def test_exact_new_caps_and_boundary_inclusion():
    assert measured_gentle_v1('CONTACT', -.0014, .008, .005) == -.0004
    assert measured_gentle_v1('CONTACT', -.0014, .008001, .005) == -.0014
    assert measured_gentle_v1('HOLD', .0014, .005, .005) == .0004
    assert measured_gentle_v1('HOLD', -.0001, .005, .005) == -.0001
    assert measured_gentle_v1('HOLD', 0., .005, .005) == 0.
    assert measured_gentle_v1('RETRACT', .004, .007, .005) == .0002
    assert measured_gentle_v1('DONE', .004, .007001, .005) == .004


@pytest.mark.parametrize('phase,desired,gap,center',[
    ('FREE',0.,.005,.005), ('RETRACT',-.001,.005,.005),
    ('CONTACT',float('nan'),.005,.005), ('HOLD',0.,float('inf'),.005)])
def test_invalid_requests_fail_closed(phase,desired,gap,center):
    with pytest.raises(ValueError):measured_gentle_v1(phase,desired,gap,center)
