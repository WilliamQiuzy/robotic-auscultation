"""Deterministic official-mesh assets for the Mac-only occlusion study.

This converter preserves the full official OpenArm v1 URDF mesh inventory and
bakes the pinned NVIDIA medical human at the USD default/rest pose.  It does
not create an active pointer and it deliberately labels the human OBJ files as
the authored Catmull-Clark *control cage*, not an Isaac-render-equivalent
subdivision surface.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from . import official_asset_contract
from .official_openarm_experiment import (
    ExperimentContractError,
    anchor_ten_points_to_back_mesh,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUILD_ROOT = WORKSPACE_ROOT / ".wam-build/offline-official-openarm-meshes"
DEFAULT_RAW_MANIFEST = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw/conversion_manifest.json"
)
SCHEMA = "wam_offline_official_mesh_conversion_v1"
COMPLETION_SCHEMA = "wam_offline_official_mesh_completion_v1"
OBJ_HEADER = b"# WAM deterministic triangle mesh v1\n"
SERIALIZATION_TOLERANCE_M = 5.1e-10
PACKAGE_PREFIX = "package://openarm_description/"
EXPECTED_ASSET_MANIFEST_SHA256 = (
    "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
)
# Produced by two byte-identical locked-runtime builds after the conversion
# contract was finalized.  The verifier rejects any self-consistent rewrite.
EXPECTED_CONVERSION_MANIFEST_SHA256 = (
    "b5e826af2539bceafea1d638503362a2e8811038ad238e29bb9c263a652c3202"
)
EXPECTED_HUMAN_MESH_NAMES = {
    "opaque__fabric__labcoat",
    "opaque__fabric__scrubspants",
    "opaque__metal__stethoscope",
    "opaque__plastic__idbadge",
    "opaque__leather__sneakers",
    "opaque__fabric__scrubsshirt",
    "trans__organic__malecaucasianskin",
}
_HEX64 = frozenset("0123456789abcdef")


class MeshConversionError(RuntimeError):
    """The offline derived asset would not preserve the pinned official inputs."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MeshConversionError("conversion value is not canonical-JSON encodable") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise MeshConversionError(f"could not hash source/output file: {path}") from exc
    return digest.hexdigest(), size


def _safe_relative(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise MeshConversionError("path must be a safe relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise MeshConversionError("path must be a safe relative POSIX path")
    return value


def _require_real_directory(path: Path) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MeshConversionError(f"directory is unavailable: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise MeshConversionError(f"directory must be real: {path}")
    return path


def _require_regular_file(path: Path) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MeshConversionError(f"file is unavailable: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise MeshConversionError(f"path must be a regular file: {path}")
    return path


def _vector(element: ET.Element | None, name: str, default: str) -> np.ndarray:
    raw = default if element is None else element.get(name, default)
    try:
        value = np.asarray([float(item) for item in raw.split()], dtype=np.float64)
    except (AttributeError, ValueError) as exc:
        raise MeshConversionError(f"URDF {name} vector is invalid") from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise MeshConversionError(f"URDF {name} vector is invalid")
    return value


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _validated_mesh(vertices: object, faces: object) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise MeshConversionError("mesh vertices must be finite with shape (N>=3, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
        raise MeshConversionError("mesh faces must be nonempty triangles")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise MeshConversionError("mesh face index is out of range")
    return np.ascontiguousarray(points), np.ascontiguousarray(triangles)


def _load_triangle_source_details(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source = _require_regular_file(Path(path))
    if source.suffix.lower() not in {".dae", ".stl"}:
        raise MeshConversionError("only official DAE/STL sources are accepted")
    try:
        scene = trimesh.load_scene(source, process=False)
    except Exception as exc:
        raise MeshConversionError(f"official mesh could not be decoded: {source}") from exc
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    node_determinants: list[float] = []
    reflection_count = 0
    offset = 0
    try:
        nodes = sorted(str(node) for node in scene.graph.nodes_geometry)
        if not nodes:
            raise MeshConversionError("official mesh scene has no triangle geometry")
        for node in nodes:
            transform, geometry_name = scene.graph[node]
            geometry = scene.geometry[geometry_name]
            points, triangles = _validated_mesh(geometry.vertices, geometry.faces)
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise MeshConversionError("DAE/STL scene transform is invalid")
            determinant = float(np.linalg.det(matrix[:3, :3]))
            if not math.isfinite(determinant) or abs(determinant) <= 1.0e-15:
                raise MeshConversionError("DAE/STL scene transform is singular")
            homogeneous = np.column_stack((points, np.ones(len(points))))
            transformed = (homogeneous @ matrix.T)[:, :3]
            if not np.isfinite(transformed).all():
                raise MeshConversionError("DAE/STL scene transform produced nonfinite vertices")
            vertices_parts.append(transformed)
            transformed_faces = triangles.copy()
            if determinant < 0.0:
                transformed_faces[:, [1, 2]] = transformed_faces[:, [2, 1]]
                reflection_count += 1
            faces_parts.append(transformed_faces + offset)
            node_determinants.append(determinant)
            offset += len(points)
    except MeshConversionError:
        raise
    except Exception as exc:
        raise MeshConversionError("official triangle scene inventory is invalid") from exc
    vertices, faces = _validated_mesh(np.vstack(vertices_parts), np.vstack(faces_parts))
    return vertices, faces, {
        "scene_node_count": len(node_determinants),
        "scene_node_determinants": node_determinants,
        "scene_reflection_count": reflection_count,
    }


def load_triangle_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a DAE/STL scene and correct winding for authored reflections."""

    vertices, faces, _metadata = _load_triangle_source_details(path)
    return vertices, faces


def apply_urdf_mesh_transform(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scale: np.ndarray,
    origin_xyz: np.ndarray,
    origin_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply authored scale and URDF origin exactly once, correcting reflection winding."""

    points, triangles = _validated_mesh(vertices, faces)
    scale_value = np.asarray(scale, dtype=np.float64)
    translation = np.asarray(origin_xyz, dtype=np.float64)
    rpy = np.asarray(origin_rpy, dtype=np.float64)
    if (
        scale_value.shape != (3,)
        or translation.shape != (3,)
        or rpy.shape != (3,)
        or not np.isfinite(scale_value).all()
        or not np.isfinite(translation).all()
        or not np.isfinite(rpy).all()
        or np.any(scale_value == 0.0)
    ):
        raise MeshConversionError("URDF mesh scale/origin is invalid")
    rotation = _rotation_from_rpy(rpy)
    linear = rotation @ np.diag(scale_value)
    transformed = (linear @ points.T).T + translation
    corrected = triangles.copy()
    if float(np.linalg.det(linear)) < 0.0:
        corrected[:, [1, 2]] = corrected[:, [2, 1]]
    return _validated_mesh(transformed, corrected)


def fixed_obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Encode one indexed triangle mesh with a fixed locale-independent format."""

    points, triangles = _validated_mesh(vertices, faces)

    def fixed(value: float) -> str:
        if abs(float(value)) < 0.5e-9:
            value = 0.0
        return f"{float(value):.9f}"

    lines = [OBJ_HEADER.decode("ascii").rstrip("\n")]
    lines.extend(f"v {fixed(x)} {fixed(y)} {fixed(z)}" for x, y, z in points)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_fixed_obj_bytes(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Strictly decode only the deterministic OBJ grammar emitted above."""

    if type(raw) is not bytes:
        raise MeshConversionError("OBJ must be supplied as immutable bytes")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise MeshConversionError("OBJ is not ASCII") from exc
    if not lines or lines[0] != OBJ_HEADER.decode("ascii").rstrip("\n"):
        raise MeshConversionError("OBJ deterministic header is invalid")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_phase = False
    try:
        for line in lines[1:]:
            fields = line.split(" ")
            if len(fields) != 4 or any(field == "" for field in fields):
                raise MeshConversionError("OBJ line is outside the fixed grammar")
            if fields[0] == "v" and not face_phase:
                vertices.append([float(value) for value in fields[1:]])
            elif fields[0] == "f":
                face_phase = True
                faces.append([int(value) - 1 for value in fields[1:]])
            else:
                raise MeshConversionError("OBJ line is outside the fixed grammar")
    except ValueError as exc:
        raise MeshConversionError("OBJ numeric field is invalid") from exc
    return _validated_mesh(vertices, faces)


def _serialized_geometry(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[bytes, np.ndarray, np.ndarray, float]:
    raw = fixed_obj_bytes(vertices, faces)
    decoded_vertices, decoded_faces = parse_fixed_obj_bytes(raw)
    if not np.array_equal(decoded_faces, faces):
        raise MeshConversionError("OBJ serialization changed triangle topology")
    error = float(np.max(np.abs(decoded_vertices - vertices)))
    if not math.isfinite(error) or error > SERIALIZATION_TOLERANCE_M:
        raise MeshConversionError("OBJ serialization exceeded its declared tolerance")
    return raw, decoded_vertices, decoded_faces, error


def _aabb(vertices: np.ndarray) -> list[list[float]]:
    return [
        [float(value) for value in np.min(vertices, axis=0)],
        [float(value) for value in np.max(vertices, axis=0)],
    ]


def _samples(vertices: np.ndarray) -> list[dict[str, object]]:
    indices = sorted({0, len(vertices) // 2, len(vertices) - 1})
    return [
        {"index": int(index), "position": [float(value) for value in vertices[index]]}
        for index in indices
    ]


def _write_exclusive(path: Path, raw: bytes) -> None:
    from scripts.perception import prepare_offline_official_assets as asset_builder

    try:
        asset_builder._write_exclusive(path, raw)
    except asset_builder.PreparationError as exc:
        raise MeshConversionError(str(exc)) from exc


def _robot_rows(
    asset_root: Path, staging: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    urdf_path = asset_root / official_asset_contract.OFFICIAL_URDF_PATH
    urdf_raw = _require_regular_file(urdf_path).read_bytes()
    if _sha256(urdf_raw) != official_asset_contract.OFFICIAL_URDF_SHA256:
        raise MeshConversionError("official URDF hash changed")
    try:
        root = ET.fromstring(urdf_raw)
    except ET.ParseError as exc:
        raise MeshConversionError("official URDF is not well formed") from exc
    links = root.findall("link")
    if len(links) != 24 or len(root.findall("joint")) != 23:
        raise MeshConversionError("official OpenArm topology changed")
    link_names = [link.get("name") for link in links]
    if any(type(name) is not str or not name for name in link_names):
        raise MeshConversionError("official OpenArm link name is invalid")
    cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    role_counts = {"visual": 0, "collision": 0}
    for link in links:
        link_name = str(link.get("name"))
        for role in ("visual", "collision"):
            for role_index, item in enumerate(link.findall(role)):
                mesh = item.find("./geometry/mesh")
                if mesh is None:
                    raise MeshConversionError("primitive or missing URDF geometry is forbidden")
                filename = mesh.get("filename")
                if type(filename) is not str or not filename.startswith(PACKAGE_PREFIX):
                    raise MeshConversionError("URDF mesh does not use the official package")
                source_relative = filename[len(PACKAGE_PREFIX) :]
                _safe_relative(source_relative)
                source = asset_root / official_asset_contract.CHECKOUT_PATH / source_relative
                source_resolved = _require_regular_file(source).resolve(strict=True)
                checkout_resolved = (
                    asset_root / official_asset_contract.CHECKOUT_PATH
                ).resolve(strict=True)
                try:
                    source_resolved.relative_to(checkout_resolved)
                except ValueError as exc:
                    raise MeshConversionError("URDF mesh escaped the official checkout") from exc
                source_key = source_relative
                if source_key not in cache:
                    cache[source_key] = _load_triangle_source_details(source)
                source_vertices, source_faces, scene_metadata = cache[source_key]
                scale = _vector(mesh, "scale", "1 1 1")
                origin = item.find("origin")
                origin_xyz = _vector(origin, "xyz", "0 0 0")
                origin_rpy = _vector(origin, "rpy", "0 0 0")
                output_vertices, output_faces = apply_urdf_mesh_transform(
                    source_vertices,
                    source_faces,
                    scale=scale,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                )
                output_relative = (
                    f"robot/{role}/{role_counts[role]:02d}_{link_name}.obj"
                )
                (
                    output_raw,
                    serialized_vertices,
                    serialized_faces,
                    serialization_error,
                ) = _serialized_geometry(output_vertices, output_faces)
                _write_exclusive(staging / output_relative, output_raw)
                source_sha, source_size = _sha256_file(source)
                determinant = float(np.linalg.det(_rotation_from_rpy(origin_rpy) @ np.diag(scale)))
                rows.append(
                    {
                        "link": link_name,
                        "role": role,
                        "role_index": int(role_index),
                        "source_path": (
                            f"{official_asset_contract.CHECKOUT_PATH}/{source_relative}"
                        ),
                        "source_sha256": source_sha,
                        "source_size": source_size,
                        "source_vertex_count": int(len(source_vertices)),
                        "source_triangle_count": int(len(source_faces)),
                        "source_face_sha256": _sha256(
                            np.ascontiguousarray(source_faces, dtype="<i8").tobytes()
                        ),
                        "source_aabb": _aabb(source_vertices),
                        "source_vertex_samples": _samples(source_vertices),
                        **scene_metadata,
                        "scale": [float(value) for value in scale],
                        "origin_xyz": [float(value) for value in origin_xyz],
                        "origin_rpy": [float(value) for value in origin_rpy],
                        "linear_determinant": determinant,
                        "urdf_reflection_winding_corrected": determinant < 0.0,
                        "reflection_winding_corrected": (
                            int(scene_metadata["scene_reflection_count"]) > 0
                            or determinant < 0.0
                        ),
                        "output_path": output_relative,
                        "output_sha256": _sha256(output_raw),
                        "output_size": len(output_raw),
                        "vertex_count": int(len(serialized_vertices)),
                        "triangle_count": int(len(serialized_faces)),
                        "source_derived_output_aabb": _aabb(output_vertices),
                        "output_aabb": _aabb(serialized_vertices),
                        "output_vertex_samples": _samples(serialized_vertices),
                        "output_face_sha256": _sha256(
                            np.ascontiguousarray(serialized_faces, dtype="<i8").tobytes()
                        ),
                        "serialization_tolerance_m": SERIALIZATION_TOLERANCE_M,
                        "max_serialization_error_m": serialization_error,
                        "vertex_correspondence": "source_index_preserved",
                        "primitive_fallback": False,
                    }
                )
                role_counts[role] += 1
    if role_counts != {"visual": 21, "collision": 21} or len(rows) != 42:
        raise MeshConversionError("official OpenArm 42-reference inventory changed")
    return (
        {
            "urdf_path": official_asset_contract.OFFICIAL_URDF_PATH,
            "urdf_sha256": _sha256(urdf_raw),
            "link_count": 24,
            "joint_count": 23,
            "links": link_names,
            "mesh_reference_count": 42,
            "visual_reference_count": 21,
            "collision_reference_count": 21,
            "unique_source_mesh_count": len(cache),
            "meshes": rows,
            "root_scale": 1,
        },
        rows,
    )


def _matrix_array(values: Sequence[object]) -> np.ndarray:
    result = np.asarray([np.asarray(value, dtype=np.float64) for value in values])
    if result.ndim != 3 or result.shape[1:] != (4, 4) or not np.isfinite(result).all():
        raise MeshConversionError("USD skeleton transform array is invalid")
    return result


def _mesh_topology(mesh: UsdGeom.Mesh, time: Usd.TimeCode) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(time), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(time), dtype=np.int64)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts != 3):
        raise MeshConversionError("official human control cage must remain authored triangles")
    if indices.ndim != 1 or len(indices) != 3 * len(counts):
        raise MeshConversionError("official human topology is malformed")
    return indices.reshape(-1, 3)


def bake_official_human(path: Path) -> dict[str, object]:
    """Bake the exact official skinning binding at the declared default/rest pose."""

    usd_path = _require_regular_file(Path(path))
    try:
        stage = Usd.Stage.Open(os.fspath(usd_path.resolve(strict=True)))
    except Exception as exc:
        raise MeshConversionError("official human USD could not be opened") from exc
    if not stage:
        raise MeshConversionError("official human USD stage is invalid")
    time = Usd.TimeCode.Default()
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise MeshConversionError("official human stage up axis changed")
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12:
        raise MeshConversionError("official human stage metre scale changed")
    roots = [UsdSkel.Root(prim) for prim in stage.Traverse() if prim.IsA(UsdSkel.Root)]
    skeletons = [
        UsdSkel.Skeleton(prim) for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)
    ]
    visible_meshes = [
        UsdGeom.Mesh(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and prim.IsActive()
        and prim.IsLoaded()
        and UsdGeom.Imageable(prim).ComputeVisibility(time) != UsdGeom.Tokens.invisible
    ]
    if len(roots) != 1 or len(skeletons) != 1:
        raise MeshConversionError("official human must have exactly one SkelRoot and Skeleton")
    if len(visible_meshes) != 7 or {mesh.GetPrim().GetName() for mesh in visible_meshes} != EXPECTED_HUMAN_MESH_NAMES:
        raise MeshConversionError("official human visible mesh inventory changed")
    cache = UsdSkel.Cache()
    if not cache.Populate(roots[0], Usd.PrimDefaultPredicate):
        raise MeshConversionError("UsdSkel cache population failed")
    bindings = cache.ComputeSkelBindings(roots[0], Usd.PrimDefaultPredicate)
    if len(bindings) != 1:
        raise MeshConversionError("official human must have exactly one skeleton binding")
    binding = bindings[0]
    skeleton = binding.GetSkeleton()
    if not skeleton or skeleton.GetPath() != skeletons[0].GetPath():
        raise MeshConversionError("official human skeleton binding changed")
    query = cache.GetSkelQuery(skeleton)
    if not query:
        raise MeshConversionError("official human skeleton query is invalid")
    if query.GetAnimQuery():
        raise MeshConversionError("animated human pose is forbidden in the neutral conversion")
    joints = skeleton.GetJointsAttr().Get(time)
    if joints is None or len(joints) != 101:
        raise MeshConversionError("official human joint inventory changed")
    default_local = _matrix_array(query.ComputeJointLocalTransforms(time, False))
    rest_local = _matrix_array(query.ComputeJointLocalTransforms(time, True))
    if not np.array_equal(default_local, rest_local):
        raise MeshConversionError("official human default pose no longer equals rest pose")
    skinning_transforms = query.ComputeSkinningTransforms(time)
    if len(skinning_transforms) != len(joints):
        raise MeshConversionError("official human skinning transform inventory changed")
    skeleton_to_world = UsdGeom.XformCache(time).GetLocalToWorldTransform(
        skeleton.GetPrim()
    )
    targets = list(binding.GetSkinningTargets())
    target_paths = {str(target.GetPrim().GetPath()) for target in targets}
    visible_paths = {str(mesh.GetPrim().GetPath()) for mesh in visible_meshes}
    if target_paths != visible_paths:
        raise MeshConversionError("every visible human mesh must be one bound skinned target")

    rows: list[dict[str, object]] = []
    baked: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for skin_query in sorted(targets, key=lambda item: str(item.GetPrim().GetPath())):
        prim = skin_query.GetPrim()
        mesh = UsdGeom.Mesh(prim)
        if not skin_query.HasJointInfluences() or skin_query.HasBlendShapes():
            raise MeshConversionError("human mesh must have joints and no blend shapes")
        if str(skin_query.GetSkinningMethod()) != "classicLinear":
            raise MeshConversionError("unsupported official human skinning method")
        if skin_query.GetJointOrder() is not None or skin_query.GetJointMapper() is not None:
            raise MeshConversionError("unimplemented human joint remapping is forbidden")
        influence_count = int(skin_query.GetNumInfluencesPerComponent())
        if influence_count <= 0 or str(skin_query.GetInterpolation()) != "vertex":
            raise MeshConversionError("human joint influence interpolation changed")
        raw_points = mesh.GetPointsAttr().Get(time)
        if raw_points is None:
            raise MeshConversionError("human mesh points are unavailable")
        source_points = np.asarray(raw_points, dtype=np.float64)
        faces = _mesh_topology(mesh, time)
        source_points, faces = _validated_mesh(source_points, faces)
        indices = np.asarray(
            skin_query.GetJointIndicesPrimvar().Get(time), dtype=np.int64
        )
        weights = np.asarray(
            skin_query.GetJointWeightsPrimvar().Get(time), dtype=np.float64
        )
        if (
            indices.size != len(source_points) * influence_count
            or weights.size != indices.size
        ):
            raise MeshConversionError("human joint influence array shape changed")
        indices = indices.reshape(len(source_points), influence_count)
        weights = weights.reshape(len(source_points), influence_count)
        if (
            np.any(indices < 0)
            or np.any(indices >= len(joints))
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=2.0e-6, rtol=0.0)
        ):
            raise MeshConversionError("human joint influence values are invalid")
        mutable = Vt.Vec3fArray(raw_points)
        if not skin_query.ComputeSkinnedPoints(skinning_transforms, mutable, time):
            raise MeshConversionError("UsdSkel failed to bake human points")
        points = np.asarray(
            [
                skeleton_to_world.Transform(Gf.Vec3d(*map(float, point)))
                for point in mutable
            ],
            dtype=np.float64,
        )
        points, faces = _validated_mesh(points, faces)
        orientation = mesh.GetOrientationAttr().Get(time)
        if orientation not in {UsdGeom.Tokens.rightHanded, UsdGeom.Tokens.leftHanded}:
            raise MeshConversionError("human mesh orientation is unsupported")
        if orientation == UsdGeom.Tokens.leftHanded:
            faces = faces.copy()
            faces[:, [1, 2]] = faces[:, [2, 1]]
        subdivision = mesh.GetSubdivisionSchemeAttr().Get(time)
        if subdivision != UsdGeom.Tokens.catmullClark:
            raise MeshConversionError("official human Catmull-Clark scheme changed")
        name = prim.GetName()
        baked[name] = (points, faces)
        rows.append(
            {
                "name": name,
                "prim_path": str(prim.GetPath()),
                "subdivision_scheme": str(subdivision),
                "orientation": str(orientation),
                "skinning_method": "classicLinear",
                "influences_per_vertex": influence_count,
                "source_vertex_count": int(len(source_points)),
                "vertex_count": int(len(points)),
                "triangle_count": int(len(faces)),
                "source_face_sha256": _sha256(
                    np.ascontiguousarray(faces, dtype="<i8").tobytes()
                ),
                "source_aabb": _aabb(source_points),
                "output_aabb": _aabb(points),
                "source_vertex_samples": _samples(source_points),
                "output_vertex_samples": _samples(points),
                "vertex_correspondence": "identity_index_after_classic_linear_skinning",
                "baked_float32_sha256": _sha256(
                    np.ascontiguousarray(points, dtype=np.float32).tobytes()
                ),
                "_vertices": points,
                "_faces": faces,
            }
        )
    return {
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "skel_root_count": 1,
        "skeleton_count": 1,
        "joint_count": len(joints),
        "binding_count": 1,
        "pose": "default_equals_rest",
        "visible_mesh_count": len(rows),
        "vertex_count": sum(int(row["vertex_count"]) for row in rows),
        "triangle_count": sum(int(row["triangle_count"]) for row in rows),
        "surface_representation": "official_authored_catmullclark_control_cage",
        "isaac_surface_equivalence": False,
        "meshes": rows,
    }


def _human_rows(asset_root: Path, staging: Path) -> dict[str, object]:
    usd_path = asset_root / official_asset_contract.HUMAN_USD_PATH
    usd_sha, usd_size = _sha256_file(usd_path)
    if (
        usd_sha != official_asset_contract.HUMAN_USD_SHA256
        or usd_size != official_asset_contract.HUMAN_USD_SIZE
    ):
        raise MeshConversionError("official human USD hash/size changed")
    human = bake_official_human(usd_path)
    public_rows: list[dict[str, object]] = []
    labcoat_geometry: tuple[np.ndarray, np.ndarray] | None = None
    labcoat_row: dict[str, object] | None = None
    for index, row in enumerate(human["meshes"]):
        value = dict(row)
        vertices = value.pop("_vertices")
        faces = value.pop("_faces")
        output_relative = f"human/mesh/{index:02d}_{value['name']}.obj"
        (
            output_raw,
            serialized_vertices,
            serialized_faces,
            serialization_error,
        ) = _serialized_geometry(vertices, faces)
        _write_exclusive(staging / output_relative, output_raw)
        value.update(
            {
                "output_path": output_relative,
                "output_sha256": _sha256(output_raw),
                "output_size": len(output_raw),
                "source_derived_output_aabb": _aabb(vertices),
                "output_aabb": _aabb(serialized_vertices),
                "output_vertex_samples": _samples(serialized_vertices),
                "output_face_sha256": _sha256(
                    np.ascontiguousarray(serialized_faces, dtype="<i8").tobytes()
                ),
                "serialization_tolerance_m": SERIALIZATION_TOLERANCE_M,
                "max_serialization_error_m": serialization_error,
            }
        )
        public_rows.append(value)
        if value["name"] == "opaque__fabric__labcoat":
            labcoat_geometry = (vertices, faces)
            labcoat_row = value
    if labcoat_geometry is None or labcoat_row is None:
        raise MeshConversionError("exactly one official labcoat mesh is required")
    try:
        anchors = anchor_ten_points_to_back_mesh(
            labcoat_geometry[0], labcoat_geometry[1], back_axis=1, back_sign=1
        )
    except ExperimentContractError as exc:
        raise MeshConversionError("ten-point labcoat anchoring failed") from exc
    labcoat = {
        "prim_path": labcoat_row["prim_path"],
        "output_path": labcoat_row["output_path"],
        "vertex_count": labcoat_row["vertex_count"],
        "triangle_count": labcoat_row["triangle_count"],
        "baked_float32_sha256": labcoat_row["baked_float32_sha256"],
        "source_output_vertex_correspondence": (
            "identity_index_after_classic_linear_skinning"
        ),
        "anchors": [
            {
                "name": anchor.name,
                "triangle_id": int(anchor.triangle_id),
                "barycentric": [float(value) for value in anchor.barycentric],
                "position_m": [float(value) for value in anchor.position_m],
                "normal": [float(value) for value in anchor.surface_normal],
                "source_vertex_indices": [
                    int(value) for value in labcoat_geometry[1][anchor.triangle_id]
                ],
            }
            for anchor in anchors
        ],
    }
    human["meshes"] = public_rows
    human["usd_path"] = official_asset_contract.HUMAN_USD_PATH
    human["usd_sha256"] = usd_sha
    human["usd_size"] = usd_size
    human["labcoat"] = labcoat
    return human


def _asset_builder():
    from scripts.perception import prepare_offline_official_assets as asset_builder

    return asset_builder


def _verified_asset_manifest(asset_release: Path) -> dict[str, object]:
    builder = _asset_builder()
    try:
        manifest = builder.verify_release(Path(asset_release))
    except builder.PreparationError as exc:
        raise MeshConversionError(f"official asset release verification failed: {exc}") from exc
    raw = _require_regular_file(Path(asset_release) / "asset_manifest.json").read_bytes()
    if _sha256(raw) != EXPECTED_ASSET_MANIFEST_SHA256:
        raise MeshConversionError("official asset manifest identity changed")
    return manifest


def _build_manifest(asset_release: Path, staging: Path) -> dict[str, object]:
    asset_manifest = _verified_asset_manifest(asset_release)
    robot, _rows = _robot_rows(asset_release, staging)
    human = _human_rows(asset_release, staging)
    return {
        "schema": SCHEMA,
        "asset_release_id": Path(asset_release).name,
        "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "asset_manifest_inventory_sha256": _sha256(
            official_asset_contract.canonical_json_bytes(asset_manifest["files"])
        ),
        "robot": robot,
        "human": human,
        "primitive_fallback": False,
        "production_activation": False,
        "isaac_equivalence": False,
        "scope": "offline_method_simulation_only",
    }


def _completion_bytes(manifest_raw: bytes) -> bytes:
    manifest_sha = _sha256(manifest_raw)
    release_id = f"official_meshes-{manifest_sha[:20]}"
    return canonical_json_bytes(
        {
            "schema": COMPLETION_SCHEMA,
            "status": "complete",
            "conversion_manifest_sha256": manifest_sha,
            "release_id": release_id,
        }
    )


def _decode_canonical(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshConversionError("conversion manifest is not valid UTF-8 JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise MeshConversionError("conversion manifest is not canonical")
    return value


def _finite_vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise MeshConversionError(f"{label} must be a finite {length}-vector")
    return result


def _finite_aabb(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (2, 3)
        or not np.isfinite(result).all()
        or np.any(result[0] > result[1])
    ):
        raise MeshConversionError(f"{label} must be a finite ordered AABB")
    return result


def _expected_urdf_reference_rows(asset_root: Path) -> tuple[list[str], list[dict[str, object]]]:
    raw = _require_regular_file(
        asset_root / official_asset_contract.OFFICIAL_URDF_PATH
    ).read_bytes()
    if _sha256(raw) != official_asset_contract.OFFICIAL_URDF_SHA256:
        raise MeshConversionError("official URDF hash changed during conversion verification")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise MeshConversionError("official URDF is malformed during verification") from exc
    links = root.findall("link")
    names = [str(link.get("name")) for link in links]
    rows: list[dict[str, object]] = []
    for link in links:
        for role in ("visual", "collision"):
            for role_index, item in enumerate(link.findall(role)):
                mesh = item.find("./geometry/mesh")
                if mesh is None:
                    raise MeshConversionError("official URDF geometry inventory changed")
                filename = mesh.get("filename")
                if type(filename) is not str or not filename.startswith(PACKAGE_PREFIX):
                    raise MeshConversionError("official URDF package path changed")
                origin = item.find("origin")
                rows.append(
                    {
                        "link": str(link.get("name")),
                        "role": role,
                        "role_index": role_index,
                        "source_path": (
                            f"{official_asset_contract.CHECKOUT_PATH}/"
                            f"{filename[len(PACKAGE_PREFIX):]}"
                        ),
                        "scale": [float(value) for value in _vector(mesh, "scale", "1 1 1")],
                        "origin_xyz": [
                            float(value) for value in _vector(origin, "xyz", "0 0 0")
                        ],
                        "origin_rpy": [
                            float(value) for value in _vector(origin, "rpy", "0 0 0")
                        ],
                    }
                )
    if len(names) != 24 or len(rows) != 42:
        raise MeshConversionError("official URDF topology changed during verification")
    return names, rows


def _validate_conversion_manifest_contract(
    manifest: dict[str, object], *, asset_root: Path, asset_manifest: dict[str, object]
) -> list[dict[str, object]]:
    expected_top = {
        "schema",
        "asset_release_id",
        "asset_manifest_sha256",
        "asset_manifest_inventory_sha256",
        "robot",
        "human",
        "primitive_fallback",
        "production_activation",
        "isaac_equivalence",
        "scope",
    }
    if set(manifest) != expected_top or manifest.get("schema") != SCHEMA:
        raise MeshConversionError("conversion manifest top-level schema is invalid")
    inventory_sha = _sha256(
        official_asset_contract.canonical_json_bytes(asset_manifest["files"])
    )
    if (
        manifest.get("asset_release_id") != asset_root.name
        or manifest.get("asset_manifest_sha256") != EXPECTED_ASSET_MANIFEST_SHA256
        or manifest.get("asset_manifest_inventory_sha256") != inventory_sha
        or manifest.get("primitive_fallback") is not False
        or manifest.get("production_activation") is not False
        or manifest.get("isaac_equivalence") is not False
        or manifest.get("scope") != "offline_method_simulation_only"
    ):
        raise MeshConversionError("conversion manifest scope/source provenance changed")
    robot = manifest.get("robot")
    robot_keys = {
        "urdf_path",
        "urdf_sha256",
        "link_count",
        "joint_count",
        "links",
        "mesh_reference_count",
        "visual_reference_count",
        "collision_reference_count",
        "unique_source_mesh_count",
        "meshes",
        "root_scale",
    }
    if type(robot) is not dict or set(robot) != robot_keys:
        raise MeshConversionError("robot conversion schema is invalid")
    link_names, expected_references = _expected_urdf_reference_rows(asset_root)
    rows = robot.get("meshes")
    if (
        robot.get("urdf_path") != official_asset_contract.OFFICIAL_URDF_PATH
        or robot.get("urdf_sha256") != official_asset_contract.OFFICIAL_URDF_SHA256
        or robot.get("link_count") != 24
        or robot.get("joint_count") != 23
        or robot.get("links") != link_names
        or robot.get("mesh_reference_count") != 42
        or robot.get("visual_reference_count") != 21
        or robot.get("collision_reference_count") != 21
        or robot.get("unique_source_mesh_count") != 20
        or robot.get("root_scale") != 1
        or type(rows) is not list
        or len(rows) != 42
    ):
        raise MeshConversionError("robot conversion inventory changed")
    file_inventory = {row["path"]: row for row in asset_manifest["files"]}
    robot_row_keys = {
        "link",
        "role",
        "role_index",
        "source_path",
        "source_sha256",
        "source_size",
        "source_vertex_count",
        "source_triangle_count",
        "source_face_sha256",
        "source_aabb",
        "source_vertex_samples",
        "scene_node_count",
        "scene_node_determinants",
        "scene_reflection_count",
        "scale",
        "origin_xyz",
        "origin_rpy",
        "linear_determinant",
        "urdf_reflection_winding_corrected",
        "reflection_winding_corrected",
        "output_path",
        "output_sha256",
        "output_size",
        "vertex_count",
        "triangle_count",
        "source_derived_output_aabb",
        "output_aabb",
        "output_vertex_samples",
        "output_face_sha256",
        "serialization_tolerance_m",
        "max_serialization_error_m",
        "vertex_correspondence",
        "primitive_fallback",
    }
    for row, expected in zip(rows, expected_references):
        if type(row) is not dict or set(row) != robot_row_keys:
            raise MeshConversionError("robot mesh conversion row schema is invalid")
        for key in ("link", "role", "role_index", "source_path", "scale", "origin_xyz", "origin_rpy"):
            if row.get(key) != expected[key]:
                raise MeshConversionError(f"robot mesh {key} is not bound to the URDF")
        source = file_inventory.get(row["source_path"])
        if (
            type(source) is not dict
            or row.get("source_sha256") != source.get("sha256")
            or row.get("source_size") != source.get("size")
            or row.get("primitive_fallback") is not False
            or row.get("vertex_correspondence") != "source_index_preserved"
        ):
            raise MeshConversionError("robot mesh source provenance changed")
        if (
            type(row.get("scene_node_count")) is not int
            or row["scene_node_count"] <= 0
            or type(row.get("scene_reflection_count")) is not int
            or not 0 <= row["scene_reflection_count"] <= row["scene_node_count"]
            or type(row.get("scene_node_determinants")) is not list
            or len(row["scene_node_determinants"]) != row["scene_node_count"]
        ):
            raise MeshConversionError("robot DAE/STL scene transform evidence is invalid")
        determinants = np.asarray(row["scene_node_determinants"], dtype=np.float64)
        if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1.0e-15):
            raise MeshConversionError("robot scene determinant evidence is invalid")
        expected_urdf_reflection = float(row.get("linear_determinant")) < 0.0
        if (
            row.get("urdf_reflection_winding_corrected") is not expected_urdf_reflection
            or row.get("reflection_winding_corrected")
            is not (row["scene_reflection_count"] > 0 or expected_urdf_reflection)
        ):
            raise MeshConversionError("robot reflection winding evidence is inconsistent")
        _finite_aabb(row.get("source_aabb"), "robot source_aabb")
        _finite_aabb(
            row.get("source_derived_output_aabb"), "robot source-derived output AABB"
        )
    human = manifest.get("human")
    human_keys = {
        "up_axis",
        "meters_per_unit",
        "skel_root_count",
        "skeleton_count",
        "joint_count",
        "binding_count",
        "pose",
        "visible_mesh_count",
        "vertex_count",
        "triangle_count",
        "surface_representation",
        "isaac_surface_equivalence",
        "meshes",
        "usd_path",
        "usd_sha256",
        "usd_size",
        "labcoat",
    }
    if type(human) is not dict or set(human) != human_keys:
        raise MeshConversionError("human conversion schema is invalid")
    human_rows = human.get("meshes")
    if (
        human.get("up_axis") != "Z"
        or human.get("meters_per_unit") != 1.0
        or human.get("skel_root_count") != 1
        or human.get("skeleton_count") != 1
        or human.get("joint_count") != 101
        or human.get("binding_count") != 1
        or human.get("pose") != "default_equals_rest"
        or human.get("visible_mesh_count") != 7
        or human.get("vertex_count") != 8893
        or human.get("triangle_count") != 16506
        or human.get("surface_representation")
        != "official_authored_catmullclark_control_cage"
        or human.get("isaac_surface_equivalence") is not False
        or human.get("usd_path") != official_asset_contract.HUMAN_USD_PATH
        or human.get("usd_sha256") != official_asset_contract.HUMAN_USD_SHA256
        or human.get("usd_size") != official_asset_contract.HUMAN_USD_SIZE
        or type(human_rows) is not list
        or len(human_rows) != 7
    ):
        raise MeshConversionError("human conversion inventory changed")
    human_row_keys = {
        "name",
        "prim_path",
        "subdivision_scheme",
        "orientation",
        "skinning_method",
        "influences_per_vertex",
        "source_vertex_count",
        "vertex_count",
        "triangle_count",
        "source_face_sha256",
        "source_aabb",
        "output_aabb",
        "source_vertex_samples",
        "output_vertex_samples",
        "vertex_correspondence",
        "baked_float32_sha256",
        "output_path",
        "output_sha256",
        "output_size",
        "source_derived_output_aabb",
        "output_face_sha256",
        "serialization_tolerance_m",
        "max_serialization_error_m",
    }
    if {row.get("name") for row in human_rows if type(row) is dict} != EXPECTED_HUMAN_MESH_NAMES:
        raise MeshConversionError("human mesh identity changed")
    for row in human_rows:
        if type(row) is not dict or set(row) != human_row_keys:
            raise MeshConversionError("human mesh conversion row schema is invalid")
        if (
            row.get("subdivision_scheme") != "catmullClark"
            or row.get("skinning_method") != "classicLinear"
            or row.get("vertex_correspondence")
            != "identity_index_after_classic_linear_skinning"
        ):
            raise MeshConversionError("human mesh skin/control-cage provenance changed")
        _finite_aabb(row.get("source_aabb"), "human source AABB")
        _finite_aabb(
            row.get("source_derived_output_aabb"), "human source-derived output AABB"
        )
    labcoat = human.get("labcoat")
    labcoat_keys = {
        "prim_path",
        "output_path",
        "vertex_count",
        "triangle_count",
        "baked_float32_sha256",
        "source_output_vertex_correspondence",
        "anchors",
    }
    if (
        type(labcoat) is not dict
        or set(labcoat) != labcoat_keys
        or not str(labcoat.get("prim_path", "")).endswith("/opaque__fabric__labcoat")
        or labcoat.get("vertex_count") != 1765
        or labcoat.get("triangle_count") != 3256
        or labcoat.get("source_output_vertex_correspondence")
        != "identity_index_after_classic_linear_skinning"
        or type(labcoat.get("anchors")) is not list
        or len(labcoat["anchors"]) != 10
    ):
        raise MeshConversionError("human labcoat anchor schema is invalid")
    anchor_keys = {
        "name",
        "triangle_id",
        "barycentric",
        "position_m",
        "normal",
        "source_vertex_indices",
    }
    if tuple(anchor.get("name") for anchor in labcoat["anchors"]) != (
        "R1",
        "L1",
        "R2",
        "L2",
        "R3",
        "L3",
        "R4",
        "L4",
        "R5",
        "L5",
    ):
        raise MeshConversionError("human labcoat point identity changed")
    for anchor in labcoat["anchors"]:
        if type(anchor) is not dict or set(anchor) != anchor_keys:
            raise MeshConversionError("human labcoat anchor row schema is invalid")
        barycentric = _finite_vector(anchor.get("barycentric"), 3, "barycentric")
        _finite_vector(anchor.get("position_m"), 3, "anchor position")
        _finite_vector(anchor.get("normal"), 3, "anchor normal")
        indices = np.asarray(anchor.get("source_vertex_indices"), dtype=np.int64)
        if (
            abs(float(barycentric.sum()) - 1.0) > 1.0e-12
            or np.any(barycentric < 0.0)
            or type(anchor.get("triangle_id")) is not int
            or not 0 <= anchor["triangle_id"] < 3256
            or indices.shape != (3,)
            or np.any(indices < 0)
            or np.any(indices >= 1765)
        ):
            raise MeshConversionError("human labcoat anchor values are invalid")
    return [*rows, *human_rows]


def verify_conversion_release(release: Path) -> dict[str, object]:
    release = _require_real_directory(Path(release))
    builder = _asset_builder()
    try:
        builder._assert_tree_no_symlinks(release)
    except builder.PreparationError as exc:
        raise MeshConversionError(str(exc)) from exc
    manifest_raw = _require_regular_file(release / "conversion_manifest.json").read_bytes()
    manifest_sha = _sha256(manifest_raw)
    if len(manifest_sha) != 64 or not set(manifest_sha) <= _HEX64:
        raise MeshConversionError("conversion manifest hash is invalid")
    if manifest_sha != EXPECTED_CONVERSION_MANIFEST_SHA256:
        raise MeshConversionError("conversion manifest is not the pinned canonical identity")
    if release.name != f"official_meshes-{manifest_sha[:20]}":
        raise MeshConversionError("conversion release name does not match manifest hash")
    manifest = _decode_canonical(manifest_raw)
    asset_root = _asset_builder().RELEASE_PATH
    asset_manifest = _verified_asset_manifest(asset_root)
    rows = _validate_conversion_manifest_contract(
        manifest, asset_root=asset_root, asset_manifest=asset_manifest
    )
    expected_files = {"conversion_manifest.json", "CONVERSION_COMPLETE.json"}
    for row in rows:
        relative = _safe_relative(row.get("output_path"))
        if relative in expected_files:
            raise MeshConversionError("duplicate conversion output path")
        expected_files.add(relative)
        path = _require_regular_file(release.joinpath(*PurePosixPath(relative).parts))
        digest, size = _sha256_file(path)
        if digest != row.get("output_sha256") or size != row.get("output_size"):
            raise MeshConversionError(f"conversion output hash/size changed: {relative}")
        raw = path.read_bytes()
        vertices, faces = parse_fixed_obj_bytes(raw)
        if (
            type(row.get("vertex_count")) is not int
            or row["vertex_count"] != len(vertices)
            or type(row.get("triangle_count")) is not int
            or row["triangle_count"] != len(faces)
        ):
            raise MeshConversionError("conversion OBJ counts do not match manifest")
        actual_aabb = np.asarray(_aabb(vertices), dtype=np.float64)
        declared_aabb = _finite_aabb(row.get("output_aabb"), "output AABB")
        if not np.array_equal(actual_aabb, declared_aabb):
            raise MeshConversionError("conversion OBJ AABB does not match manifest")
        if row.get("output_vertex_samples") != _samples(vertices):
            raise MeshConversionError("conversion OBJ vertex samples do not match manifest")
        face_sha = _sha256(np.ascontiguousarray(faces, dtype="<i8").tobytes())
        if row.get("output_face_sha256") != face_sha:
            raise MeshConversionError("conversion OBJ topology hash does not match manifest")
        tolerance = row.get("serialization_tolerance_m")
        error = row.get("max_serialization_error_m")
        if (
            tolerance != SERIALIZATION_TOLERANCE_M
            or type(error) not in {int, float}
            or not math.isfinite(float(error))
            or not 0.0 <= float(error) <= SERIALIZATION_TOLERANCE_M
        ):
            raise MeshConversionError("conversion serialization tolerance evidence is invalid")
        derived_aabb = _finite_aabb(
            row.get("source_derived_output_aabb"), "source-derived output AABB"
        )
        if float(np.max(np.abs(actual_aabb - derived_aabb))) > SERIALIZATION_TOLERANCE_M:
            raise MeshConversionError("serialized OBJ exceeds source-derived AABB tolerance")
    marker_raw = _require_regular_file(release / "CONVERSION_COMPLETE.json").read_bytes()
    if marker_raw != _completion_bytes(manifest_raw):
        raise MeshConversionError("conversion completion marker is invalid")
    actual_files = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise MeshConversionError("conversion release has missing or unexpected files")
    return manifest


def _emit_raw_manifest(raw: bytes) -> None:
    builder = _asset_builder()
    destination = DEFAULT_RAW_MANIFEST
    lock_path = DEFAULT_BUILD_ROOT / "raw-evidence.lock"
    try:
        new_value = _decode_canonical(raw)
        for value, label in ((new_value, "new"),):
            if (
                value.get("schema") != SCHEMA
                or value.get("asset_manifest_sha256")
                != EXPECTED_ASSET_MANIFEST_SHA256
                or value.get("production_activation") is not False
                or value.get("isaac_equivalence") is not False
                or value.get("scope") != "offline_method_simulation_only"
            ):
                raise MeshConversionError(f"{label} raw conversion evidence is invalid")
        builder._mkdir_real(lock_path.parent)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if not os.path.lexists(os.fspath(destination)):
                builder._write_result_copy(destination, raw)
                return
            existing = builder._require_regular_file(
                destination, "existing conversion result evidence"
            ).read_bytes()
            if existing == raw:
                return
            old_value = _decode_canonical(existing)
            if (
                old_value.get("schema") != SCHEMA
                or old_value.get("asset_manifest_sha256")
                != EXPECTED_ASSET_MANIFEST_SHA256
                or old_value.get("production_activation") is not False
                or old_value.get("isaac_equivalence") is not False
                or old_value.get("scope") != "offline_method_simulation_only"
            ):
                raise MeshConversionError(
                    "existing raw conversion evidence is not a migratable predecessor"
                )
            old_sha = _sha256(existing)
            history = destination.parent / "history" / f"conversion_manifest-{old_sha}.json"
            builder._write_result_copy(history, existing)
            if destination.read_bytes() != existing:
                raise MeshConversionError("raw conversion evidence changed during migration")
            temporary = destination.parent / (
                f".{destination.name}.migration-{secrets.token_hex(12)}"
            )
            try:
                builder._write_exclusive(temporary, raw)
                os.replace(temporary, destination)
                builder._fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
            if destination.read_bytes() != raw or history.read_bytes() != existing:
                raise MeshConversionError("raw conversion evidence migration did not persist")
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except builder.PreparationError as exc:
        raise MeshConversionError(str(exc)) from exc


def convert_official_assets(
    asset_release: Path,
    *,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> Path:
    """Build or verify one content-addressed conversion release without activation."""

    asset_release = Path(asset_release)
    build_root = Path(build_root)
    _verified_asset_manifest(asset_release)
    build_root.mkdir(parents=True, exist_ok=True)
    _require_real_directory(build_root)
    if os.path.lexists(os.fspath(build_root / "current")):
        raise MeshConversionError("conversion current pointer is forbidden")
    staging_root = build_root / ".staging"
    releases_root = build_root / "releases"
    staging_root.mkdir(exist_ok=True)
    releases_root.mkdir(exist_ok=True)
    _require_real_directory(staging_root)
    _require_real_directory(releases_root)
    expected_release = releases_root / (
        f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
    )
    if os.path.lexists(os.fspath(expected_release)):
        verified = verify_conversion_release(expected_release)
        manifest_raw = canonical_json_bytes(verified)
        if build_root.resolve() == DEFAULT_BUILD_ROOT.resolve():
            _emit_raw_manifest(manifest_raw)
        return expected_release
    staging = staging_root / f"build-{secrets.token_hex(16)}"
    staging.mkdir(mode=0o700)
    try:
        manifest = _build_manifest(asset_release, staging)
        manifest_raw = canonical_json_bytes(manifest)
        if _sha256(manifest_raw) != EXPECTED_CONVERSION_MANIFEST_SHA256:
            raise MeshConversionError(
                "conversion output is not the pinned canonical manifest identity"
            )
        _write_exclusive(staging / "conversion_manifest.json", manifest_raw)
        _write_exclusive(staging / "CONVERSION_COMPLETE.json", _completion_bytes(manifest_raw))
        release = expected_release
        if os.path.lexists(os.fspath(release)):
            verified = verify_conversion_release(release)
            if canonical_json_bytes(verified) != manifest_raw:
                raise MeshConversionError("existing conversion release conflicts")
            shutil.rmtree(staging)
        else:
            builder = _asset_builder()
            try:
                builder._fsync_tree(staging)
                identity = builder._release_directory_identity(staging)
                builder._rename_no_replace(staging, release)
            except builder.PreparationError as exc:
                raise MeshConversionError(str(exc)) from exc
            try:
                verify_conversion_release(release)
            except Exception:
                try:
                    builder._rollback_owned_release(release, identity)
                except builder.PreparationError as rollback_error:
                    raise MeshConversionError(str(rollback_error)) from rollback_error
                raise
        if build_root.resolve() == DEFAULT_BUILD_ROOT.resolve():
            _emit_raw_manifest(manifest_raw)
        if os.path.lexists(os.fspath(build_root / "current")):
            raise MeshConversionError("conversion current pointer appeared unexpectedly")
        return release
    finally:
        if os.path.lexists(os.fspath(staging)):
            try:
                if staging.is_symlink() or not staging.is_dir():
                    staging.unlink()
                else:
                    shutil.rmtree(staging)
            except OSError as exc:
                raise MeshConversionError("could not clean conversion staging") from exc


__all__ = [
    "DEFAULT_BUILD_ROOT",
    "MeshConversionError",
    "apply_urdf_mesh_transform",
    "bake_official_human",
    "canonical_json_bytes",
    "convert_official_assets",
    "fixed_obj_bytes",
    "load_triangle_source",
    "verify_conversion_release",
]
