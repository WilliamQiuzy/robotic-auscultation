import pytest
from sim.auscultation_pi05_v6.train_openpi import phase_safe_starts


def test_last_origin_action_crossing_new_boundary_is_rejected():
    # All ten origin labels can be FREE while origin9's endpoint is APPROACH.
    assert phase_safe_starts([0,10],[9],20)==[10]


def test_no_boundary_metadata_is_not_assumed_safe():
    with pytest.raises(ValueError,match='indices'):
        phase_safe_starts([0],None,10)


def test_boundary_actions_stay_in_provenance_but_not_any_training_window():
    assert phase_safe_starts(list(range(21)),[10],30)==[0,*range(11,21)]
    with pytest.raises(ValueError,match='phase-safe'):
        phase_safe_starts([0],[0],10)
