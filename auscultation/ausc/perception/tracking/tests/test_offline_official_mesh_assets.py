"""Deterministic official-mesh conversion for the Mac-only occlusion study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from pxr import Usd, UsdGeom


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, os.fspath(WORKSPACE_ROOT / "auscultation"))

from ausc.perception.tracking import offline_official_mesh_assets as converter  # noqa: E402
from ausc.perception.tracking.official_openarm_experiment import POINT_NAMES  # noqa: E402
from scripts.perception import prepare_offline_official_assets as asset_builder  # noqa: E402


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_fixed_obj_and_reflection_transform_are_deterministic() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    transformed, corrected = converter.apply_urdf_mesh_transform(
        vertices,
        faces,
        scale=np.asarray([0.001, -0.001, 0.001]),
        origin_xyz=np.asarray([0.1, 0.2, 0.3]),
        origin_rpy=np.asarray([0.0, 0.0, np.pi / 2.0]),
    )
    np.testing.assert_allclose(
        transformed,
        [[0.1, 0.2, 0.3], [0.1, 1.2, 0.3], [1.1, 0.2, 0.3]],
        atol=1.0e-12,
    )
    assert corrected.tolist() == [[0, 2, 1]]
    raw = converter.fixed_obj_bytes(transformed, corrected)
    assert raw == (
        b"# WAM deterministic triangle mesh v1\n"
        b"v 0.100000000 0.200000000 0.300000000\n"
        b"v 0.100000000 1.200000000 0.300000000\n"
        b"v 1.100000000 0.200000000 0.300000000\n"
        b"f 1 3 2\n"
    )
    assert converter.fixed_obj_bytes(transformed, corrected) == raw


def test_real_official_dae_scene_load_is_finite_and_stable() -> None:
    source = (
        asset_builder.RELEASE_PATH
        / "openarm_description/assets/robot/openarm_v1.0/mesh/arm/visual/link7.dae"
    )
    first_vertices, first_faces = converter.load_triangle_source(source)
    second_vertices, second_faces = converter.load_triangle_source(source)
    assert first_vertices.ndim == 2 and first_vertices.shape[1] == 3
    assert first_faces.ndim == 2 and first_faces.shape[1] == 3
    assert len(first_vertices) > 0 and len(first_faces) > 0
    assert np.isfinite(first_vertices).all()
    assert np.array_equal(first_vertices, second_vertices)
    assert np.array_equal(first_faces, second_faces)


def test_reflective_nonidentity_dae_scene_corrects_winding(tmp_path: Path) -> None:
    path = tmp_path / "reflective.dae"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="triangle"><mesh>
    <source id="positions"><float_array id="positions-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
      <technique_common><accessor source="#positions-array" count="3" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="1"><input semantic="VERTEX" source="#vertices" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="Scene"><node id="reflect">
    <matrix>-1 0 0 2  0 1 0 3  0 0 1 4  0 0 0 1</matrix>
    <instance_geometry url="#triangle"/>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
""",
        encoding="utf-8",
    )
    vertices, faces = converter.load_triangle_source(path)
    np.testing.assert_allclose(
        vertices,
        [[2.0, 3.0, 4.0], [1.0, 3.0, 4.0], [2.0, 4.0, 4.0]],
        atol=1.0e-12,
    )
    assert faces.tolist() == [[0, 2, 1]]


def test_unbound_usd_mesh_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unbound.usda"
    stage = Usd.Stage.CreateNew(os.fspath(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    with pytest.raises(converter.MeshConversionError, match="SkelRoot|binding|skinned"):
        converter.bake_official_human(path)


def test_usd_skinning_applies_nonidentity_skeleton_world_transform_once(
    tmp_path: Path,
) -> None:
    source = asset_builder.RELEASE_PATH / "human/male_adult_medical_01.usd"
    shifted_path = tmp_path / "shifted-human.usd"
    shutil.copy2(source, shifted_path)
    stage = Usd.Stage.Open(os.fspath(shifted_path))
    root = UsdGeom.Xformable(stage.GetPrimAtPath("/Root"))
    translation = np.asarray([0.125, -0.25, 0.375], dtype=np.float64)
    translate_op = next(
        op for op in root.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"
    )
    translate_op.Set(tuple(float(value) for value in translation))
    stage.GetRootLayer().Save()

    baseline = converter.bake_official_human(source)
    shifted = converter.bake_official_human(shifted_path)
    baseline_labcoat = next(
        row for row in baseline["meshes"] if row["name"] == "opaque__fabric__labcoat"
    )
    shifted_labcoat = next(
        row for row in shifted["meshes"] if row["name"] == "opaque__fabric__labcoat"
    )
    np.testing.assert_allclose(
        shifted_labcoat["_vertices"],
        baseline_labcoat["_vertices"] + translation,
        atol=2.0e-7,
    )


def test_full_official_conversion_is_complete_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    release = converter.convert_official_assets(
        asset_builder.RELEASE_PATH,
        build_root=tmp_path / "conversion-build",
    )
    assert release.is_dir() and not release.is_symlink()
    assert not os.path.lexists(tmp_path / "conversion-build/current")
    assert list((tmp_path / "conversion-build/.staging").iterdir()) == []
    manifest_raw = (release / "conversion_manifest.json").read_bytes()
    manifest = converter.verify_conversion_release(release)
    assert manifest_raw == converter.canonical_json_bytes(manifest)
    assert release.name == f"official_meshes-{_sha(manifest_raw)[:20]}"

    robot = manifest["robot"]
    assert robot["link_count"] == 24
    assert robot["mesh_reference_count"] == 42
    assert robot["visual_reference_count"] == 21
    assert robot["collision_reference_count"] == 21
    assert len(robot["links"]) == 24
    assert len(robot["meshes"]) == 42
    assert {row["role"] for row in robot["meshes"]} == {"visual", "collision"}
    assert all(row["primitive_fallback"] is False for row in robot["meshes"])
    assert all((release / row["output_path"]).is_file() for row in robot["meshes"])
    assert any(row["reflection_winding_corrected"] for row in robot["meshes"])

    human = manifest["human"]
    assert human["visible_mesh_count"] == 7
    assert human["vertex_count"] == 8893
    assert human["triangle_count"] == 16506
    assert human["skeleton_count"] == 1
    assert human["joint_count"] == 101
    assert human["binding_count"] == 1
    assert human["pose"] == "default_equals_rest"
    assert human["surface_representation"] == "official_authored_catmullclark_control_cage"
    assert human["isaac_surface_equivalence"] is False
    assert {row["subdivision_scheme"] for row in human["meshes"]} == {"catmullClark"}
    assert all((release / row["output_path"]).is_file() for row in human["meshes"])

    labcoat = human["labcoat"]
    assert labcoat["prim_path"].endswith("/opaque__fabric__labcoat")
    assert labcoat["vertex_count"] == 1765
    assert labcoat["triangle_count"] == 3256
    assert labcoat["baked_float32_sha256"] == (
        "c2533573ce4b9a08225c97b0e468f957bc6217f17b84a19f154512d96a7ddcc7"
    )
    anchors = labcoat["anchors"]
    assert tuple(row["name"] for row in anchors) == POINT_NAMES
    assert len({row["triangle_id"] for row in anchors}) == 10
    assert all(abs(sum(row["barycentric"]) - 1.0) <= 1.0e-12 for row in anchors)
    assert all(len(row["position_m"]) == len(row["normal"]) == 3 for row in anchors)

    second = converter.convert_official_assets(
        asset_builder.RELEASE_PATH,
        build_root=tmp_path / "conversion-build",
    )
    assert second == release
    assert (second / "conversion_manifest.json").read_bytes() == manifest_raw
    for row in [*robot["meshes"], *human["meshes"]]:
        raw = (release / row["output_path"]).read_bytes()
        assert _sha(raw) == row["output_sha256"]
        assert len(raw) == row["output_size"]
        vertices, faces = converter.parse_fixed_obj_bytes(raw)
        assert len(vertices) == row["vertex_count"]
        assert len(faces) == row["triangle_count"]
        assert row["output_aabb"] == [
            [float(value) for value in vertices.min(axis=0)],
            [float(value) for value in vertices.max(axis=0)],
        ]
        assert row["serialization_tolerance_m"] == 5.1e-10
        assert row["max_serialization_error_m"] <= row["serialization_tolerance_m"]


def test_manifest_rewrite_and_release_rename_cannot_forge_provenance(
    tmp_path: Path,
) -> None:
    release = converter.convert_official_assets(
        asset_builder.RELEASE_PATH,
        build_root=tmp_path / "conversion-build",
    )
    manifest = json.loads((release / "conversion_manifest.json").read_text())
    row = manifest["robot"]["meshes"][0]
    victim = release / row["output_path"]
    forged_raw = victim.read_bytes() + b"# forged but self-consistent manifest\n"
    victim.write_bytes(forged_raw)
    row["output_sha256"] = _sha(forged_raw)
    row["output_size"] = len(forged_raw)
    manifest_raw = converter.canonical_json_bytes(manifest)
    (release / "conversion_manifest.json").write_bytes(manifest_raw)
    (release / "CONVERSION_COMPLETE.json").write_bytes(
        converter._completion_bytes(manifest_raw)
    )
    forged_release = release.with_name(f"official_meshes-{_sha(manifest_raw)[:20]}")
    release.rename(forged_release)
    with pytest.raises(
        converter.MeshConversionError,
        match="pinned|provenance|canonical identity|OBJ",
    ):
        converter.verify_conversion_release(forged_release)


def test_raw_manifest_migration_preserves_versioned_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "raw/conversion_manifest.json"
    destination.parent.mkdir(parents=True)
    old = {
        "schema": converter.SCHEMA,
        "asset_manifest_sha256": converter.EXPECTED_ASSET_MANIFEST_SHA256,
        "production_activation": False,
        "isaac_equivalence": False,
        "scope": "offline_method_simulation_only",
        "revision": "old",
    }
    new = {**old, "revision": "new"}
    old_raw = converter.canonical_json_bytes(old)
    new_raw = converter.canonical_json_bytes(new)
    destination.write_bytes(old_raw)
    monkeypatch.setattr(converter, "DEFAULT_RAW_MANIFEST", destination)
    monkeypatch.setattr(converter, "DEFAULT_BUILD_ROOT", tmp_path / "conversion-build")

    converter._emit_raw_manifest(new_raw)

    assert destination.read_bytes() == new_raw
    archive = (
        destination.parent
        / "history"
        / f"conversion_manifest-{_sha(old_raw)}.json"
    )
    assert archive.read_bytes() == old_raw
