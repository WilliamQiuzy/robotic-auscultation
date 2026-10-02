import numpy as np
import pytest
from sim.ausc_policy import contact_surface_gap as G


def plane(z=0.):
    return np.array([[[-1,-1,z],[1,-1,z],[1,1,z]],[[-1,-1,z],[1,1,z],[-1,1,z]]],float)


def test_signed_offsets_hit_plane_on_both_sides():
    result=G.ray_offsets([[0,0,.003],[.1,.1,-.002]],[0,0,-1],plane())
    assert result==pytest.approx([.003,-.002])


def test_parallel_and_missing_rays_remain_unknown():
    assert np.isnan(G.ray_offsets([[0,0,.003]],[1,0,0],plane())[0])
    assert np.isnan(G.ray_offsets([[2,2,.003]],[0,0,-1],plane())[0])


def test_first_outer_surface_not_deeper_layer():
    triangles=np.concatenate([plane(-.01),plane()])
    assert G.ray_offsets([[0,0,.003]],[0,0,-1],triangles)[0]==pytest.approx(.003)


def test_search_interval_is_bounded():
    assert np.isnan(G.ray_offsets([[0,0,.2]],[0,0,-1],plane())[0])


def test_front_samples_preserve_float32_exported_geometry():
    a=np.array([0.,np.pi/16])
    face=np.array([[r*np.cos(t),r*np.sin(t),-.047] for r in (.021,.023) for t in a],np.float32)
    vertices=np.concatenate([face,face+[0,0,.0015]])
    samples=G.front_samples(vertices)
    assert samples.shape==(3,3)
    assert np.all(samples[:,2]==float(np.float32(-.047)))


def test_wrong_front_geometry_rejected():
    with pytest.raises(ValueError):G.front_samples(np.array([[0,0,-.048]]*8))


def test_geometry_must_match_original_native_binding(tmp_path):
    (tmp_path/'collision_geometry.json').write_text('{}')
    (tmp_path/'collision_geometry.npz').write_bytes(b'changed geometry')
    native={'geometry_artifacts':{'collision_geometry_sha256':G.sha(tmp_path/'collision_geometry.json'),
                                  'collision_arrays_sha256':'0'*64}}
    with pytest.raises(ValueError,match='Original geometry'):
        G.verify_geometry_binding(tmp_path,native)


@pytest.mark.parametrize('names',[[],['R1'],['R1','L1','L2','R2','L3','R3','L4','R4','L5','R5']])
def test_missing_or_reordered_hold_points_rejected(names):
    native={'plan_bindings':[{'name':x} for x in ['L1','R1','L2','R2','L3','R3','L4','R4','L5','R5']]}
    with pytest.raises(ValueError,match='point order'):
        G.verify_point_coverage(dict.fromkeys(names),native)


def test_circle_tilt_height_matches_exact_infinite_plane_geometry():
    a=np.linspace(0,2*np.pi,32,endpoint=False);theta=np.deg2rad(17.)
    points=np.array([.023*np.cos(a)*np.cos(theta),.023*np.sin(a),.023*np.cos(a)*np.sin(theta)]).T
    offsets=G.ray_offsets(points,[0,0,-1],plane())
    assert np.ptp(offsets)==pytest.approx(.046*np.sin(theta))


@pytest.mark.parametrize('direction',[[0,0,0],[0,0,2],[0,0,float('nan')]])
def test_invalid_direction_fails(direction):
    with pytest.raises(ValueError):G.ray_offsets([[0,0,.003]],direction,plane())
