import numpy as np
import pytest

from sim.auscultation_pi05_v6.state_encoding import StateCodec
from sim.auscultation_pi05_v6.openpi_adapter import state_prompt


def test_task_pose_channels_are_encoded_without_truncating_original_channels():
    codec=StateCodec(np.full(7,-2.),np.full(7,2.))
    state=(codec.low+codec.high)/2
    state[32:35]=[.001,-.002,.003];state[35:]=[1.,0.,0.,0.,1.,0.]
    encoded=codec.encode(state);indices=codec.discrete(state)
    decoded=codec.decode(indices)
    assert encoded.shape==(41,) and indices.shape==(41,)
    np.testing.assert_allclose(decoded[32:35],state[32:35],atol=.000025,rtol=0)
    prompt=state_prompt('Contact the supplied goal',encoded)
    assert len(prompt.split('State: ')[1].split(';')[0].split())==41
    with pytest.raises(ValueError):codec.encode(state[:32])


def test_task_position_outside_declared_domain_is_rejected():
    codec=StateCodec(np.full(7,-2.),np.full(7,2.));state=(codec.low+codec.high)/2
    state[32]=1.501
    with pytest.raises(ValueError,match='outside'):codec.encode(state)
