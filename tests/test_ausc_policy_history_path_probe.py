import numpy as np
import pytest
from sim.ausc_policy.act_baseline import TRAIN16
from sim.ausc_policy.history_path_probe import groups


def test_early_group_is_first_twenty_percent_per_episode_without_error_selection():
    rows=[dict(episode_id=e,phase='FREE' if s<6 else 'APPROACH') for e in TRAIN16 for s in range(11)]
    result=groups(rows)
    assert len(result['all'])==176
    for i,e in enumerate(TRAIN16):
        np.testing.assert_array_equal(result[e+'/early20pct'],np.arange(i*11,i*11+3))
    assert len(result['FREE'])==96 and len(result['APPROACH'])==80


def test_missing_training_episode_cannot_silently_shrink_denominator():
    with pytest.raises(ValueError):groups([dict(episode_id=TRAIN16[0],phase='FREE')])
