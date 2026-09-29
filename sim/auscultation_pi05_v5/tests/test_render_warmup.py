import numpy as np
import pytest
from sim.auscultation_pi05_v5.render_warmup import stable_pixels


def test_texture_transition_in_torso_cannot_hide_in_static_background():
    a=np.zeros((100,100,3),np.uint8);b=a.copy()
    mask=np.zeros((100,100),bool);mask[30:40,30:40]=True
    b[mask]=80
    assert not stable_pixels(a,b,mask)[0]
    assert stable_pixels(b,b,mask)[0]


def test_invalid_support_is_rejected():
    a=np.zeros((10,10,3),np.uint8)
    with pytest.raises(ValueError):stable_pixels(a,a,np.zeros((10,10),bool))


def test_low_level_temporal_render_noise_is_distinct_from_texture_change():
    a=np.full((10,10,3),100,np.uint8);b=a.copy();b[::2]+=1
    mask=np.ones((10,10),bool)
    assert stable_pixels(a,b,mask)[0]
