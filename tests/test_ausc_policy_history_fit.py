import numpy as np
import pytest
from sim.ausc_policy.history_fit import history_deltas, history_stats, make_packer, frozen_parameter_prediction


@pytest.mark.parametrize('fail',[False,True])
def test_inference_parameter_flags_restore_trainable_and_frozen_masks_even_on_failure(fail):
    class Parameter:
        def __init__(self,flag):self.requires_grad=flag
        def requires_grad_(self,flag):self.requires_grad=flag
    class Policy:
        def __init__(self):self.values=[Parameter(True),Parameter(False),Parameter(True)]
        def parameters(self):return iter(self.values)
        def predict_action_chunk(self,batch):
            assert all(not p.requires_grad for p in self.values)
            if fail:raise RuntimeError('inference failure')
            return batch
    policy=Policy();batch=object()
    if fail:
        with pytest.raises(RuntimeError):frozen_parameter_prediction(policy,batch)
    else:assert frozen_parameter_prediction(policy,batch) is batch
    assert [p.requires_grad for p in policy.values]==[True,False,True]


def test_history_is_measured_past_minus_current_and_control_is_repeated_current():
    history=np.zeros((2,5,19),np.float32)
    history[:,:,0]=np.arange(5)
    expected=np.array([-4,-3,-2,-1])
    np.testing.assert_array_equal(history_deltas(history,'causal_history')[:,::7],np.stack([expected,expected]))
    np.testing.assert_array_equal(history_deltas(history,'repeat_current'),np.zeros((2,28)))
    changed=history.copy();changed[:,:,7:]=100
    np.testing.assert_array_equal(history_deltas(changed,'causal_history'),history_deltas(history,'causal_history'))


def test_history_shape_nonfinite_and_unknown_condition_are_rejected():
    with pytest.raises(ValueError):history_deltas(np.zeros((2,4,19)),'causal_history')
    with pytest.raises(ValueError):history_deltas(np.full((2,5,19),np.nan),'causal_history')
    with pytest.raises(ValueError):history_deltas(np.zeros((2,5,19)),'future')


def test_history_statistics_have_frozen_floor_and_no_clipping():
    stats=history_stats(np.zeros((20,28)))
    np.testing.assert_allclose(np.array(stats['high'])-stats['low'],.002)
    assert stats['actual_clamp'] is False


def test_paired_inputs_preserve_current_state_labels_and_block_origin():
    row=dict(episode_id='R4',start=3,state=np.zeros(19,np.float32),actions=np.ones((10,7),np.float32),measured_q=np.zeros(7,np.float32))
    history=np.zeros((5,19),np.float32);history[:4,:7]=.01
    source={('R4',3):history};aux=history_stats(history_deltas(history[None],'causal_history'))
    stats={k:dict(low=[-1.]*n,high=[1.]*n) for k,n in [('state',19),('action',7)]}
    control=make_packer(source,aux,'repeat_current')([row],stats,training=True)
    causal=make_packer(source,aux,'causal_history')([row],stats,training=True)
    assert causal['observation.environment_state'].shape==(1,40)
    for key in ('observation.state','action','action_is_pad'):
        np.testing.assert_array_equal(control[key],causal[key])
    np.testing.assert_array_equal(control['observation.environment_state'][:,:12],causal['observation.environment_state'][:,:12])
    assert not np.array_equal(control['observation.environment_state'][:,12:],causal['observation.environment_state'][:,12:])
    row['state'][0]=.2
    with pytest.raises(ValueError):make_packer(source,aux,'causal_history')([row],stats,training=False)
