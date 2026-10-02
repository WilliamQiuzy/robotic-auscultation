import numpy as np
import pytest
from sim.ausc_policy.flatness_fit_probe import sampled_gaps


def test_planned_wrist_to_ring_frame_is_patientward():
    # Wrist z faces into patient; carriage z is opposite. Ring front is -47 mm.
    local=np.array([[.023,0,-.047],[-.023,0,-.047],[0,.023,-.047]])
    mesh=np.array([[[-1,-1,0],[1,-1,0],[1,1,0]],[[-1,-1,0],[1,1,0],[-1,1,0]]])
    gaps=sampled_gaps(local,mesh,[0,0,.003],np.diag([1.,-1.,-1.]),.0025)
    assert gaps==pytest.approx([.003]*3)
