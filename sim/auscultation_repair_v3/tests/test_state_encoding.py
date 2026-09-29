import numpy as np
import pytest
from sim.auscultation_repair_v3.state_encoding import StateCodec


def codec():return StateCodec(np.array([-3.14]*7),np.array([3.14]*7))
def sample():
    s=np.zeros(32);s[20]=1.59;s[21]=1.5;s[22]=1
    return s


def test_different_valid_unseen_joint_angles_do_not_collapse():
    c=codec();a=sample();b=a.copy();a[4]=1.8;b[4]=2.1
    assert c.discrete(a)[4]!=c.discrete(b)[4]
    assert abs(c.decode(c.discrete(a))[4]-a[4])<.0008


def test_contact_target_and_hall_resolution():
    c=codec();a=sample();b=a.copy();b[14]+=.0001;b[20]-=3.3/4095
    assert c.discrete(a)[14]!=c.discrete(b)[14]
    assert c.discrete(a)[20]!=c.discrete(b)[20]
    assert abs(c.decode(c.discrete(b))[14]-b[14])<.00003


def test_all_features_are_preserved_and_domain_is_enforced():
    c=codec();a=sample();encoded=c.encode(a)
    assert encoded.shape==(32,)
    for index,value in [(0,3.2),(7,2.1),(14,1.6),(17,1.1),(20,3.4),(21,17),(22,1.1),(23,.6),(26,1.1)]:
        b=a.copy();b[index]=value
        with pytest.raises(ValueError,match='domain'):c.encode(b)
    a[0]=np.nan
    with pytest.raises(ValueError):c.encode(a)


def test_batch_roundtrip_bounds_and_schema():
    c=codec();x=np.stack([sample(),sample()]);x[1,:7]=2
    y=c.decode(c.discrete(x))
    assert y.shape==x.shape and np.max(abs(x[:,:7]-y[:,:7]))<.0008
    assert c.spec()['bins']==4096
    assert c.spec()['fitted_on']=='none'
    with pytest.raises(ValueError):StateCodec(np.zeros(7),np.zeros(7))
