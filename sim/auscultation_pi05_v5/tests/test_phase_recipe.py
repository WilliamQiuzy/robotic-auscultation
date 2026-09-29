import numpy as np
import pytest
from sim.auscultation_pi05_v5.training_data import action_stats, phase_balanced_indices


def test_phase_scaling_does_not_mix_free_and_contact():
    norm={'actions_by_phase':{'FREE':{'low':[-.01]*7,'high':[.01]*7},
                             'HOLD':{'low':[-.0001]*7,'high':[.0001]*7}}}
    assert action_stats(norm,'HOLD')['high'][0]==.0001
    with pytest.raises(ValueError):action_stats(norm,'UNKNOWN')


def test_balanced_sampling_draws_real_indices_without_relabeling():
    phases=['FREE']*200+['HOLD']*3+['CONTACT']*7
    rng=np.random.default_rng(9)
    chosen=phase_balanced_indices(phases,6000,rng)
    assert set(chosen)<=set(range(len(phases)))
    counts={p:sum(phases[i]==p for i in chosen) for p in set(phases)}
    assert all(1700<count<2300 for count in counts.values())


