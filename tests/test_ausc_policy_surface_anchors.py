import numpy as np
import pytest


def test_anchor_follows_shape_and_uses_real_surface_winding():
    from sim.ausc_policy.surface_anchors import evaluate_anchors
    vertices=np.array([[0.,0,0],[2,0,0],[0,2,0]])
    anchor={'L1':dict(smpl_vertex_indices=[2,0,1],barycentric_weights=[.25,.5,.25])}
    result=evaluate_anchors(vertices,np.array([[0,1,2]]),anchor)
    np.testing.assert_allclose(result['L1']['position_m'],[.5,.5,0])
    np.testing.assert_allclose(result['L1']['normal'],[0,0,1])
    moved=evaluate_anchors(vertices+[1,2,3],np.array([[0,1,2]]),anchor)
    np.testing.assert_allclose(moved['L1']['position_m'],[1.5,2.5,3])


def test_non_surface_triangle_is_rejected():
    from sim.ausc_policy.surface_anchors import evaluate_anchors
    with pytest.raises(ValueError):
        evaluate_anchors(np.eye(3),np.array([[0,1,2]]),{'bad':dict(smpl_vertex_indices=[0,0,2],barycentric_weights=[.5,.25,.25])})
