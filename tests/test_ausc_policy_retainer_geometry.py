"""Contract tests for the independently versioned recessed membrane geometry."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def geometry_module():
    # Keep the first RED signal an assertion rather than an import error.
    assert importlib.util.find_spec("sim.ausc_policy.retainer_geometry") is not None, "Retainer revision is not implemented"
    from sim.ausc_policy import retainer_geometry
    return retainer_geometry


def test_front_ring_and_recessed_membrane_preserve_tool_tcp():
    g = geometry_module()
    dimensions = g.geometry_metrics()
    assert dimensions["ring_front_z_m"] == pytest.approx(-.047)
    assert dimensions["membrane_front_z_m"] == pytest.approx(-.046)
    assert dimensions["membrane_recess_m"] == pytest.approx(.001)
    assert dimensions["nominal_protrusion_m"] == pytest.approx(.005)
    assert dimensions["contact_wrist_offset_m"] == pytest.approx(.132)
    assert dimensions["membrane_wrist_offset_m"] == pytest.approx(.131)
    assert dimensions["physical_validation"] is False
    assert dimensions["training_eligible"] is False
    assert "user" in dimensions["dimension_provenance"]["membrane_recess_m"].lower()
    assert "legacy" in dimensions["dimension_provenance"]["radii_m"].lower()
    assert "discretization" in dimensions["dimension_provenance"]["ring_axial_partition_m"].lower()
    json.dumps(dimensions)


def test_each_ring_sector_is_closed_convex_and_leaves_center_open():
    g = geometry_module()
    sectors = g.ring_sector_meshes()
    assert len(sectors) == 32
    for i, (vertices, faces) in enumerate(sectors):
        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)
        np.testing.assert_allclose(np.unique(np.round(vertices[:, 2], 8)), [-.047, -.0455])
        np.testing.assert_allclose(np.sort(np.linalg.norm(vertices[:, :2], axis=1)), [.021]*4 + [.023]*4)
        # Every triangular surface points outward; all vertices lie inside its half space.
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        assert np.all(np.einsum("fvi,fi->fv", vertices[None, :, :] - vertices[faces[:, 0], None, :], normals) <= 1e-14)
        radial = np.array([np.cos((i+.5)*2*np.pi/32), np.sin((i+.5)*2*np.pi/32)])
        assert min(vertices[:, :2] @ radial) == pytest.approx(.021*np.cos(np.pi/32))
        assert min(vertices[:, :2] @ radial) > .0208
        edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()
        next_vertices = sectors[(i+1) % 32][0]
        np.testing.assert_allclose(vertices[[3, 2, 7, 6]], next_vertices[[0, 1, 4, 5]], atol=1e-15)


def test_visual_front_is_an_annulus_with_recessed_disk_and_original_shell():
    g = geometry_module()
    source = np.load(Path(__file__).resolve().parents[1]/"sim/real2sim_20260922/assets/components/thinklabs_black_edge.npz")
    old_v = source["vertices"] + [0, 0, -.0058]
    old_f = source["faces"]
    vertices, faces = g.recessed_black_edge_mesh(old_v, old_f)
    untouched = old_f[~np.all(np.isclose(old_v[old_f, 2], -.047, atol=1e-8, rtol=0), axis=1)]
    np.testing.assert_array_equal(vertices[:len(old_v)], old_v)
    np.testing.assert_array_equal(faces[:len(untouched)], untouched)
    front = faces[np.all(np.isclose(vertices[faces, 2], -.047, atol=1e-8, rtol=0), axis=1)]
    assert len(front) > 0
    assert np.linalg.norm(vertices[front, :2].mean(axis=1), axis=1).min() > .0209
    center_faces = faces[np.any(np.linalg.norm(vertices[faces, :2], axis=2) < 1e-9, axis=1)]
    assert len(center_faces) > 0
    np.testing.assert_allclose(vertices[center_faces, 2], -.046)
    np.testing.assert_allclose(vertices.min(axis=0), old_v.min(axis=0))
    np.testing.assert_allclose(vertices.max(axis=0), old_v.max(axis=0))


def test_visual_rejects_wrong_front_or_invalid_topology():
    g = geometry_module()
    with pytest.raises(ValueError):
        g.recessed_black_edge_mesh([[0, 0, -.040]], [[0, 0, 0]])
    with pytest.raises(ValueError):
        g.recessed_black_edge_mesh([[0, 0, float("nan")]], [[0, 0, 0]])


def test_usd_revision_preserves_protected_scene_and_inherits_contact_settings():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    g = geometry_module()
    source = Path(__file__).resolve().parents[1]/"results/ausc_policy/tool_protrusion_review_5mm_v1/review/candidate.usdc"
    original = Usd.Stage.Open(str(source))
    stage = Usd.Stage.Open(original.Flatten())
    before = g.protected_scene_snapshot(stage)
    edge = stage.GetPrimAtPath(g.BACKING_PATH)
    margins = {name: edge.GetAttribute(name).Get() for name in ("physxCollision:contactOffset", "physxCollision:restOffset")}
    report = g.apply_to_stage(stage)
    assert g.protected_scene_snapshot(stage) == before
    assert report["protected_scene_unchanged"] is True
    assert report["membrane_recess_m"] == pytest.approx(.001)
    parent = stage.GetPrimAtPath(g.RING_ROOT)
    assert not parent.HasAPI(UsdPhysics.CollisionAPI)
    assert len(parent.GetChildren()) == 32
    assert not parent.HasAPI(UsdPhysics.MeshCollisionAPI)
    cache = UsdGeom.XformCache()
    body_inv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(g.MOVING)).GetInverse()
    for path, front, back in [(g.MEMBRANE_PATH, -.046, -.0455), (g.BACKING_PATH, -.0455, -.041)]:
        cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath(path))
        transform = cache.GetLocalToWorldTransform(cylinder.GetPrim())*body_inv
        h = cylinder.GetHeightAttr().Get()
        assert transform.Transform(Gf.Vec3d(0, 0, -h/2))[2] == pytest.approx(front)
        assert transform.Transform(Gf.Vec3d(0, 0, h/2))[2] == pytest.approx(back)
    for prim in parent.GetChildren():
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
        assert {name: prim.GetAttribute(name).Get() for name in margins} == margins
        assert prim.GetAttribute("physxCollision:restOffset").Get() <= 0
    with pytest.raises(ValueError, match="already"):
        g.apply_to_stage(stage)
    # Caller gets a disposable flattened stage; the source still has the old solid edge.
    assert not original.GetPrimAtPath(g.RING_ROOT)


def test_usd_positive_rest_offset_fails_before_any_edit():
    pytest.importorskip("pxr")
    from pxr import Usd
    g = geometry_module()
    source = Path(__file__).resolve().parents[1]/"results/ausc_policy/tool_protrusion_review_5mm_v1/review/candidate.usdc"
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
    stage.GetPrimAtPath(g.BACKING_PATH).GetAttribute("physxCollision:restOffset").Set(.001)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="restOffset"):
        g.apply_to_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before


def test_review_rejects_unbound_source_without_creating_candidate(tmp_path):
    assert importlib.util.find_spec('sim.ausc_policy.retainer_geometry_review') is not None, 'Review worker is missing'
    from sim.ausc_policy.retainer_geometry_review import build_candidate
    source = tmp_path/'source.usdc'
    source.write_bytes(b'not approved geometry')
    output = tmp_path/'review'
    with pytest.raises(ValueError, match='source'):
        build_candidate({'source': str(source), 'source_sha256': '0'*64}, output)
    assert not output.exists()


def test_review_exports_bound_candidate_and_collision_geometry(tmp_path):
    pytest.importorskip('pxr')
    pytest.importorskip('trimesh')
    assert importlib.util.find_spec('sim.ausc_policy.retainer_geometry_review') is not None, 'Review worker is missing'
    from sim.ausc_policy.retainer_geometry_review import build_candidate, SOURCE_SHA256
    from sim.ausc_policy import retainer_geometry as g
    source = Path(__file__).resolve().parents[1]/'results/ausc_policy/tool_protrusion_review_5mm_v1/review/candidate.usdc'
    stage, report = build_candidate({'source': str(source), 'source_sha256': SOURCE_SHA256}, tmp_path/'review')
    assert report['source_layers_unchanged'] is True
    assert report['actual_physics_executed'] is False
    assert report['candidate_sha256'] != SOURCE_SHA256
    assert report['geometry_id'] == g.GEOMETRY_VERSION
    assert (tmp_path/'review/approved_camera_restoration.json').is_file()
    metadata = json.loads((tmp_path/'review/geometry/collision_geometry.json').read_text())
    ring = [entry for entry in metadata['colliders'] if entry['path'].startswith(g.RING_ROOT+'/')]
    assert len(ring) == 32
    assert all(entry['approximation'] == 'convexHull' for entry in ring)
    assert not (tmp_path/'review/COMPLETE.json').exists(), 'CPU export is not completed Isaac rendering'
