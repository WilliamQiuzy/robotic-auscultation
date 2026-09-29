import numpy as np
import pytest
from sim.auscultation_repair_v3.pi05.openpi_adapter import state_prompt
from sim.auscultation_repair_v3.pi05.training_data import fit_normalization,normalize,physical_codec


def test_recipe_rejects_outside_instead_of_clipping():
    x=np.zeros(32);x[0]=1.001
    with pytest.raises(ValueError):state_prompt('Approach',x)


def test_recipe_uses_fine_tokens():
    x=np.zeros(32);y=x.copy();y[0]=.001
    assert state_prompt('Approach',x)!=state_prompt('Approach',y)
    assert '2048' in state_prompt('Approach',x)


def test_normalization_is_not_fitted_to_demonstration_state():
    a=np.zeros((20,32));a[:,20]=1.59;a[:,21]=1.5;a[:,22]=1
    c=physical_codec();a[:,:7]=(c.low[:7]+c.high[:7])/2
    def fit(x):return fit_normalization([dict(split='train',episode_id='a',state=x,actions=np.zeros((20,7)))])
    b=a.copy();b[:,:7]+=.2
    one,two=fit(a),fit(b)
    assert one['state']==two['state']
    assert one['state']['schema']=='wam_physical_asinh_state_4096_v1'
    assert normalize(a,one['state']).shape==(20,32)
