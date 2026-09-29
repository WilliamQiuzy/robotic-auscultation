import numpy as np
import pytest
from sim.auscultation_training.render_expert import validate_replay_timing


def trace(n=256):
    return [{'time_s': (i+1)/240} for i in range(n)]


def test_missing_pose_samples_do_not_expand_allowed_age():
    with pytest.raises(ValueError, match='30 Hz'):
        validate_replay_timing(trace(), np.array([7,15,255]))


@pytest.mark.parametrize('bad', [float('nan'), 0., 2.])
def test_bad_measurement_time_rejected(bad):
    rows = trace()
    rows[100]['time_s'] = bad
    with pytest.raises(ValueError, match='timestamp'):
        validate_replay_timing(rows, np.arange(7,256,8))


def test_normal_recording_has_fixed_rate_bound():
    times, dt, period = validate_replay_timing(trace(), np.arange(7,256,8))
    assert dt == 1/240 and period == 1/30
