"""Official-mesh MuJoCo kinematics, camera, and rendered-product contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields, replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, os.fspath(WORKSPACE_ROOT / "auscultation"))

from ausc.perception.tracking import offline_mesh_simulation as simulation  # noqa: E402
from ausc.perception.tracking.official_openarm_experiment import (  # noqa: E402
    POINT_NAMES,
    left_joint_limits,
    left_tip_forward_kinematics,
    measure_rendered_visibility,
)
from ausc.perception.tracking.experimental_visibility_geometry import (  # noqa: E402
    DecisionSurface,
    evaluate_visibility_geometry,
)


ASSET_RELEASE = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-assets/releases/"
    "official_assets-05e66c94cd8b36ff79be"
)
CONVERSION_RELEASE = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-meshes/releases/"
    "official_meshes-b5e826af2539bceafea1"
)
URDF_PATH = ASSET_RELEASE / "openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
CONVERSION_MANIFEST = json.loads(
    (CONVERSION_RELEASE / "conversion_manifest.json").read_text(encoding="utf-8")
)
OFFICIAL_LINK_NAMES = tuple(CONVERSION_MANIFEST["robot"]["links"])


POLICY_STREAM_SHA256 = "2" * 64


def _synthetic_oracle_frame_evidence() -> simulation.OfflineFrameEvidence:
    """Build structurally valid but deliberately unattested oracle evidence."""

    patient = np.eye(4, dtype=np.float64)
    world_from_cv = simulation.camera_pose_transform("primary")
    anchor_rows = CONVERSION_MANIFEST["human"]["labcoat"]["anchors"]
    anchor_local = np.asarray(
        [row["position_m"] for row in anchor_rows], dtype=np.float64
    )
    anchor_normal = np.asarray(
        [row["normal"] for row in anchor_rows], dtype=np.float64
    )
    anchor_world = anchor_local.copy()
    anchor_camera = (
        world_from_cv[:3, :3].T
        @ (anchor_world - world_from_cv[:3, 3]).T
    ).T
    projected = simulation.project_cv_points(anchor_camera)
    rgb = np.zeros((simulation.HEIGHT, simulation.WIDTH, 3), dtype=np.uint8)
    depth = np.ones((simulation.HEIGHT, simulation.WIDTH), dtype=np.float32)
    segmentation = np.full(
        (simulation.HEIGHT, simulation.WIDTH, 2), -1, dtype=np.int32
    )
    robot_mask = np.zeros((simulation.HEIGHT, simulation.WIDTH), dtype=np.bool_)
    human_mask = np.zeros((simulation.HEIGHT, simulation.WIDTH), dtype=np.bool_)
    products = simulation._immutable_render_products(  # noqa: SLF001
        simulation.RenderProducts(
            rgb=rgb,
            depth_m=depth,
            segmentation=segmentation,
            robot_mask=robot_mask,
            human_mask=human_mask,
        )
    )
    visibility = measure_rendered_visibility(
        point_names=POINT_NAMES,
        projected_pixels=projected,
        point_depth_m=anchor_camera[:, 2],
        robot_mask=products.robot_mask,
        human_mask=products.human_mask,
        depth_m=products.depth_for_visibility_m,
        support_radius_px=6,
        robot_mask_dilation_px=1,
        front_depth_margin_m=0.003,
    )
    provenance = {
        int(row["geom_id"]): {
            **{key: value for key, value in row.items() if key != "geom_id"},
            "compiled_vertex_max_error_m": 0.0,
            "compiled_aabb_max_error_m": 0.0,
            "compiled_face_indices_exact": True,
        }
        for row in simulation._pinned_geom_provenance_rows()  # noqa: SLF001
    }
    return simulation.OfflineFrameEvidence(
        timestamp_s=0.0,
        conversion_release_id=(
            f"official_meshes-{simulation.EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
        ),
        conversion_manifest_sha256=simulation.EXPECTED_CONVERSION_MANIFEST_SHA256,
        asset_release_id=(
            f"official_assets-{simulation.EXPECTED_ASSET_MANIFEST_SHA256[:20]}"
        ),
        asset_manifest_sha256=simulation.EXPECTED_ASSET_MANIFEST_SHA256,
        runtime_manifest_sha256=simulation.EXPECTED_RUNTIME_MANIFEST_SHA256,
        geom_provenance=provenance,
        left_joint_positions=simulation._immutable_array(  # noqa: SLF001
            np.zeros(7, dtype=np.float64)
        ),
        world_from_patient=simulation._immutable_array(patient),  # noqa: SLF001
        world_from_cv_camera=simulation._immutable_array(world_from_cv),  # noqa: SLF001
        anchor_names=POINT_NAMES,
        anchor_local_position_m=simulation._immutable_array(anchor_local),  # noqa: SLF001
        anchor_world_position_m=simulation._immutable_array(anchor_world),  # noqa: SLF001
        anchor_world_normal=simulation._immutable_array(anchor_normal),  # noqa: SLF001
        anchor_camera_position_m=simulation._immutable_array(anchor_camera),  # noqa: SLF001
        projected_pixels=simulation._immutable_array(projected),  # noqa: SLF001
        visibility=visibility,
        products=products,
        support_radius_px=6,
        robot_mask_dilation_px=1,
        front_depth_margin_m=0.003,
    )


def test_oracle_evidence_rejects_unattested_public_construction_and_validates_bindings() -> None:
    evidence = _synthetic_oracle_frame_evidence()

    with pytest.raises(simulation.OfflineSimulationError, match="scene-owned"):
        simulation.validate_oracle_frame_evidence(evidence)
    malformed_public_construction = replace(evidence, products=object())
    with pytest.raises(simulation.OfflineSimulationError, match="scene-owned"):
        simulation.validate_oracle_frame_evidence(malformed_public_construction)
    simulation.validate_oracle_frame_evidence(
        evidence, allow_test_only_synthetic=True
    )

    bad_shape = replace(
        evidence,
        products=replace(evidence.products, rgb=evidence.products.rgb[:-1]),
    )
    with pytest.raises(simulation.OfflineSimulationError, match="RGB shape"):
        simulation.validate_oracle_frame_evidence(
            bad_shape, allow_test_only_synthetic=True
        )

    bad_camera = replace(
        evidence,
        world_from_cv_camera=simulation._immutable_array(np.eye(4)),  # noqa: SLF001
    )
    with pytest.raises(simulation.OfflineSimulationError, match="primary camera"):
        simulation.validate_oracle_frame_evidence(
            bad_camera, allow_test_only_synthetic=True
        )

    shifted_world = evidence.anchor_world_position_m.copy()
    shifted_world[0, 0] += 0.01
    bad_anchor = replace(
        evidence,
        anchor_world_position_m=simulation._immutable_array(shifted_world),  # noqa: SLF001
    )
    with pytest.raises(simulation.OfflineSimulationError, match="anchor world"):
        simulation.validate_oracle_frame_evidence(
            bad_anchor, allow_test_only_synthetic=True
        )

    forged_robot_mask = evidence.products.robot_mask.copy()
    forged_robot_mask[0, 0] = True
    bad_mask = replace(
        evidence,
        products=replace(
            evidence.products,
            robot_mask=simulation._immutable_array(forged_robot_mask),  # noqa: SLF001
        ),
    )
    with pytest.raises(simulation.OfflineSimulationError, match="segmentation.*mask"):
        simulation.validate_oracle_frame_evidence(
            bad_mask, allow_test_only_synthetic=True
        )

    provenance = {key: dict(value) for key, value in evidence.geom_provenance.items()}
    provenance[0]["identity"] = "forged_robot"
    bad_provenance = replace(evidence, geom_provenance=provenance)
    with pytest.raises(simulation.OfflineSimulationError, match="provenance"):
        simulation.validate_oracle_frame_evidence(
            bad_provenance, allow_test_only_synthetic=True
        )


def test_raw_render_hash_excludes_event_time_but_binds_render_inputs() -> None:
    evidence = _synthetic_oracle_frame_evidence()
    later = replace(evidence, timestamp_s=7.25)

    assert simulation.raw_render_sha256(evidence) == simulation.raw_render_sha256(
        later
    )
    assert simulation.oracle_frame_evidence_sha256(
        evidence
    ) != simulation.oracle_frame_evidence_sha256(later)

    changed_rgb = evidence.products.rgb.copy()
    changed_rgb[0, 0, 0] = 1
    changed = replace(
        evidence,
        products=replace(
            evidence.products,
            rgb=simulation._immutable_array(changed_rgb),  # noqa: SLF001
        ),
    )
    assert simulation.raw_render_sha256(evidence) != simulation.raw_render_sha256(
        changed
    )


def _policy_observation_stream(
    *, visible_point_count_report_only: int = 6
) -> tuple[object, ...]:
    rows = []
    for frame_index in range(14):
        timestamp_s = frame_index * 0.05
        accepted = frame_index in {0, 13}
        rows.append(
            simulation.ObservationFrame(
                scenario_id="rigid_x_20mmps",
                target_name="R3",
                wrist_index=0,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                geometry_accepted=accepted,
                accepted_estimate_world_m=(
                    np.asarray([0.020 * timestamp_s, 0.0, 1.0])
                    if accepted
                    else None
                ),
                accepted_error_p95_m=0.004 if accepted else None,
                patient_motion_bound_m_s=0.020,
                visible_point_count_report_only=visible_point_count_report_only,
                observation_stream_sha256=POLICY_STREAM_SHA256,
            )
        )
    return tuple(rows)


def _policy_ground_truth_stream() -> tuple[object, ...]:
    return tuple(
        simulation.GroundTruthFrame(
            scenario_id="rigid_x_20mmps",
            target_name="R3",
            wrist_index=0,
            frame_index=frame_index,
            timestamp_s=frame_index * 0.05,
            observation_stream_sha256=POLICY_STREAM_SHA256,
            target_world_m=np.asarray(
                [round(0.020 * frame_index * 0.05, 12), 0.0, 1.0]
            ),
        )
        for frame_index in range(14)
    )


def _rpy_rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _body_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    joint_id = int(model.body_jntadr[body_id])
    address = int(model.jnt_qposadr[joint_id])
    return data.qpos[address : address + 7].copy()


def _independent_full_tree_fk(raw: bytes, left_q: np.ndarray) -> dict[str, np.ndarray]:
    """Test oracle using SciPy rotations, not the production FK math."""

    root = ET.fromstring(raw)
    links = tuple(str(link.get("name")) for link in root.findall("link"))
    q_by_name = {
        f"openarm_left_joint{index}": float(value)
        for index, value in enumerate(left_q, start=1)
    }
    result = {"world": np.eye(4)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in remaining.copy():
            parent = str(joint.find("parent").get("link"))
            child = str(joint.find("child").get("link"))
            if parent not in result:
                continue
            origin = joint.find("origin")
            xyz = np.asarray(
                [float(value) for value in ("0 0 0" if origin is None else origin.get("xyz", "0 0 0")).split()]
            )
            rpy = np.asarray(
                [float(value) for value in ("0 0 0" if origin is None else origin.get("rpy", "0 0 0")).split()]
            )
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            local[:3, 3] = xyz
            if joint.get("type") == "revolute":
                axis = np.asarray(
                    [float(value) for value in joint.find("axis").get("xyz").split()]
                )
                angle = q_by_name.get(str(joint.get("name")), 0.0)
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    axis / np.linalg.norm(axis) * angle
                ).as_matrix()
                local = local @ motion
            result[child] = result[parent] @ local
            remaining.remove(joint)
            progressed = True
        assert progressed, "independent URDF traversal stalled"
    assert set(result) == set(links)
    return {name: result[name] for name in links}


def test_full_tree_fk_is_exact_and_only_left_chain_moves() -> None:
    raw = URDF_PATH.read_bytes()
    limits = left_joint_limits(raw)
    first_q = limits.mean(axis=1)
    second_q = first_q.copy()
    second_q[[0, 2, 4, 6]] += np.asarray([0.2, -0.2, 0.15, -0.15])
    second_q = np.clip(second_q, limits[:, 0] + 0.02, limits[:, 1] - 0.02)

    first = simulation.full_openarm_link_fk(raw, first_q)
    second = simulation.full_openarm_link_fk(raw, second_q)
    first_independent = _independent_full_tree_fk(raw, first_q)
    second_independent = _independent_full_tree_fk(raw, second_q)
    assert len(first) == len(second) == 24
    assert tuple(first) == tuple(second) == OFFICIAL_LINK_NAMES
    urdf_root = ET.fromstring(raw)
    assert len(urdf_root.findall("joint")) == 23
    assert sum(joint.get("type") == "revolute" for joint in urdf_root.findall("joint")) == 14
    assert first["world"].shape == (4, 4)
    np.testing.assert_allclose(first["world"], np.eye(4), atol=0.0)
    for transform in (*first.values(), *second.values()):
        assert transform.shape == (4, 4) and np.isfinite(transform).all()
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=0.0)
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1.0e-12
        )
        assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0, abs=1.0e-12)
    for name in OFFICIAL_LINK_NAMES:
        np.testing.assert_allclose(first[name], first_independent[name], atol=1.0e-12)
        np.testing.assert_allclose(second[name], second_independent[name], atol=1.0e-12)
    np.testing.assert_allclose(
        first["openarm_left_link7"],
        left_tip_forward_kinematics(raw, first_q),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        second["openarm_left_link7"],
        left_tip_forward_kinematics(raw, second_q),
        atol=1.0e-12,
    )
    assert not np.array_equal(
        first["openarm_left_link7"], second["openarm_left_link7"]
    )
    for name in first:
        if name.startswith("openarm_right_"):
            assert np.array_equal(first[name], second[name]), name

    invalid = first_q.copy()
    invalid[0] = limits[0, 1] + 1.0e-3
    with pytest.raises(simulation.OfflineSimulationError, match="limit"):
        simulation.full_openarm_link_fk(raw, invalid)


def test_camera_basis_is_right_handed_and_projection_probe_is_subpixel() -> None:
    assert simulation.SOURCE_INTRINSICS == {
        "width": 1920,
        "height": 1080,
        "fx": 1123.173095703125,
        "fy": 1122.9962158203125,
        "cx": 955.7200927734375,
        "cy": 533.1757202148438,
    }
    assert simulation.INTRINSICS == {
        "width": 640,
        "height": 360,
        "fx": 374.3910319010417,
        "fy": 374.3320719401042,
        "cx": 318.5733642578125,
        "cy": 177.7252400716146,
    }
    assert 640 / 2 - simulation.INTRINSICS["cx"] == 1.4266357421875
    assert 360 / 2 - simulation.INTRINSICS["cy"] == pytest.approx(
        2.2747599283854, abs=1.0e-14
    )
    world_from_cv = simulation.world_from_cv_camera(
        simulation.CAMERA_POSITION_M,
        simulation.CAMERA_LOOK_AT_M,
        np.asarray([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(world_from_cv[:3, :3].T @ world_from_cv[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(world_from_cv[:3, :3]) > 0.999999999
    forward = simulation.CAMERA_LOOK_AT_M - simulation.CAMERA_POSITION_M
    forward /= np.linalg.norm(forward)
    np.testing.assert_allclose(world_from_cv[:3, 2], forward, atol=1e-12)

    rows = simulation.camera_projection_probe()
    assert {row["name"] for row in rows} == {
        "center",
        "x_negative",
        "x_positive",
        "y_negative",
        "y_positive",
    }
    assert max(row["pixel_error"] for row in rows) <= 0.5
    assert max(row["depth_error_m"] for row in rows) <= 0.001
    for row in rows:
        point = np.asarray(row["point_cv_m"], dtype=np.float64)
        expected = np.asarray(
            [
                simulation.INTRINSICS["fx"] * point[0] / point[2]
                + simulation.INTRINSICS["cx"],
                simulation.INTRINSICS["fy"] * point[1] / point[2]
                + simulation.INTRINSICS["cy"],
            ]
        )
        np.testing.assert_allclose(row["analytic_pixel"], expected, atol=1.0e-12)
        assert np.linalg.norm(np.asarray(row["rendered_pixel"]) - expected) == pytest.approx(
            row["pixel_error"], abs=1.0e-12
        )
        assert 1.98 < row["rendered_front_depth_m"] < 1.99
    by_name = {row["name"]: row for row in rows}
    assert by_name["x_negative"]["rendered_pixel"][0] < by_name["center"]["rendered_pixel"][0]
    assert by_name["x_positive"]["rendered_pixel"][0] > by_name["center"]["rendered_pixel"][0]
    assert by_name["y_negative"]["rendered_pixel"][1] < by_name["center"]["rendered_pixel"][1]
    assert by_name["y_positive"]["rendered_pixel"][1] > by_name["center"]["rendered_pixel"][1]


def test_camera_pose_name_is_a_strict_frozen_enum() -> None:
    assert simulation.CAMERA_POSE_NAMES == ("primary", "oblique")
    assert simulation._camera_xml() == simulation._camera_xml("primary")
    primary = simulation.camera_pose_transform("primary")
    np.testing.assert_array_equal(
        primary,
        simulation.world_from_cv_camera(
            simulation.CAMERA_POSITION_M,
            simulation.CAMERA_LOOK_AT_M,
            np.asarray([0.0, 0.0, 1.0]),
        ),
    )
    oblique = simulation.camera_pose_transform("oblique")
    assert oblique.shape == (4, 4)
    np.testing.assert_allclose(oblique[:3, :3].T @ oblique[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(oblique[:3, :3]) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(oblique[:3, 3] - primary[:3, 3]) > 0.25
    for invalid in ("side", "PRIMARY", "", 1, None):
        with pytest.raises(simulation.OfflineSimulationError, match="camera pose"):
            simulation.camera_pose_transform(invalid)  # type: ignore[arg-type]
        with pytest.raises(simulation.OfflineSimulationError, match="camera pose"):
            simulation.OfflineMeshScene.from_verified_conversion(
                CONVERSION_RELEASE,
                camera_pose_name=invalid,  # type: ignore[arg-type]
            )


def test_exported_scene_geometry_contracts_are_truly_immutable() -> None:
    arrays = (
        simulation.CAMERA_POSITION_M,
        simulation.CAMERA_LOOK_AT_M,
        simulation.OBLIQUE_CAMERA_POSITION_M,
        simulation.OBLIQUE_CAMERA_LOOK_AT_M,
        simulation.ROBOT_WORLD_FROM_ROOT,
    )
    for value in arrays:
        before = value.copy()
        try:
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = float(value.flat[0]) + 0.01
        finally:
            if value.flags.writeable:
                value[...] = before
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="writeable flag|cannot set"):
            value.flags.writeable = True

    for mapping, key in (
        (simulation.SOURCE_INTRINSICS, "fx"),
        (simulation.INTRINSICS, "fx"),
    ):
        before = mapping[key]
        try:
            with pytest.raises(TypeError):
                mapping[key] = float(before) + 1.0  # type: ignore[index]
        finally:
            if mapping[key] != before:
                mapping[key] = before  # type: ignore[index]


def test_module_rebinding_cannot_redefine_the_pinned_scene_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert simulation.EXPECTED_SCENE_CONTRACT_SHA256 == (
        "ec571eb51deb21738145c63bbf5db9a6a89974bca8ca53a02131b23bcddc499c"
    )
    assert (
        simulation.scene_contract_sha256()
        == simulation.EXPECTED_SCENE_CONTRACT_SHA256
    )
    canonical_intrinsics = simulation.INTRINSICS
    canonical_camera_position = simulation.CAMERA_POSITION_M
    drifted_intrinsics = dict(simulation.INTRINSICS)
    drifted_intrinsics["fx"] += 1.0
    monkeypatch.setattr(simulation, "INTRINSICS", drifted_intrinsics)
    with pytest.raises(simulation.OfflineSimulationError, match="scene contract drifted"):
        simulation.camera_pose_transform("primary")
    monkeypatch.setattr(
        simulation,
        "INTRINSICS",
        canonical_intrinsics,
    )
    monkeypatch.setattr(
        simulation,
        "CAMERA_POSITION_M",
        canonical_camera_position.copy(),
    )
    with pytest.raises(simulation.OfflineSimulationError, match="was rebound"):
        simulation.camera_pose_transform("primary")


def test_segmentation_decode_requires_geom_object_type() -> None:
    segmentation = np.asarray(
        [
            [[3, int(mujoco.mjtObj.mjOBJ_GEOM)], [7, int(mujoco.mjtObj.mjOBJ_GEOM)], [-1, -1]],
            [[3, int(mujoco.mjtObj.mjOBJ_BODY)], [99, int(mujoco.mjtObj.mjOBJ_GEOM)], [7, int(mujoco.mjtObj.mjOBJ_BODY)]],
        ],
        dtype=np.int32,
    )
    robot, human = simulation.decode_instance_masks(segmentation, {3}, {7})
    np.testing.assert_array_equal(
        robot, [[True, False, False], [False, False, False]]
    )
    np.testing.assert_array_equal(
        human, [[False, True, False], [False, False, False]]
    )
    with pytest.raises(simulation.OfflineSimulationError, match="disjoint"):
        simulation.decode_instance_masks(segmentation, {3}, {3})


def test_camera_products_switch_explicitly_from_rgb_to_depth_to_segmentation() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.mode = "rgb"

        def disable_depth_rendering(self) -> None:
            self.events.append("disable_depth")
            if self.mode == "depth":
                self.mode = "rgb"

        def disable_segmentation_rendering(self) -> None:
            self.events.append("disable_segmentation")
            if self.mode == "segmentation":
                self.mode = "rgb"

        def enable_depth_rendering(self) -> None:
            self.events.append("enable_depth")
            self.mode = "depth"

        def enable_segmentation_rendering(self) -> None:
            self.events.append("enable_segmentation")
            self.mode = "segmentation"

        def update_scene(self, _data: object, *, camera: str) -> None:
            assert camera == "offline_camera"
            self.events.append(f"update_{self.mode}")

        def render(self) -> np.ndarray:
            self.events.append(f"render_{self.mode}")
            if self.mode == "rgb":
                return np.full((2, 3, 3), 11, dtype=np.uint8)
            if self.mode == "depth":
                return np.full((2, 3), 1.25, dtype=np.float32)
            return np.full((2, 3, 2), 5, dtype=np.int32)

    recorder = Recorder()
    rgb, depth, segmentation = simulation.render_sequential_camera_products(
        recorder, object()
    )
    assert recorder.events == [
        "disable_depth",
        "disable_segmentation",
        "update_rgb",
        "render_rgb",
        "disable_segmentation",
        "enable_depth",
        "update_depth",
        "render_depth",
        "disable_depth",
        "enable_segmentation",
        "update_segmentation",
        "render_segmentation",
        "disable_depth",
        "disable_segmentation",
    ]
    assert rgb.dtype == np.uint8 and np.all(rgb == 11)
    assert depth.dtype == np.float32 and np.all(depth == 1.25)
    assert segmentation.dtype == np.int32 and np.all(segmentation == 5)

    failing = Recorder()
    original_render = failing.render

    def fail_on_depth() -> np.ndarray:
        if failing.mode == "depth":
            failing.events.append("render_depth_failed")
            raise RuntimeError("synthetic depth failure")
        return original_render()

    failing.render = fail_on_depth  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synthetic"):
        simulation.render_sequential_camera_products(failing, object())
    assert failing.events[-2:] == ["disable_depth", "disable_segmentation"]


def test_flat_official_scene_and_rendered_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scene = simulation.OfflineMeshScene.from_verified_conversion(CONVERSION_RELEASE)
    try:
        model = scene.model
        assert model.nbody == 32  # world + 24 official robot links + 7 human meshes
        assert scene.conversion_release_id == "official_meshes-b5e826af2539bceafea1"
        assert scene.conversion_manifest_sha256 == (
            "b5e826af2539bceafea1d638503362a2e8811038ad238e29bb9c263a652c3202"
        )
        assert scene.asset_release_id == "official_assets-05e66c94cd8b36ff79be"
        assert scene.asset_manifest_sha256 == (
            "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
        )
        assert scene.runtime_manifest_sha256 == (
            "7e3307a163e8b9e6c3c32cb038f442171d0d23794182cffb1aa1c1cb983ef346"
        )
        assert scene.compiled_from_owned_asset_bytes is True
        assert model.njnt == 31
        assert model.nq == 31 * 7
        assert np.all(model.body_parentid[1:] == 0)
        assert np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        assert model.ngeom == 28
        assert model.nmesh == 28
        assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        assert np.all(model.geom_contype == 0)
        assert np.all(model.geom_conaffinity == 0)
        assert len(scene.robot_geom_ids) == 21
        assert len(scene.human_geom_ids) == 7
        assert scene.robot_geom_ids.isdisjoint(scene.human_geom_ids)
        assert set(scene.geom_provenance) == set(range(28))
        assert {row["identity"] for row in scene.geom_provenance.values()} == {
            *(row["link"] for row in CONVERSION_MANIFEST["robot"]["meshes"] if row["role"] == "visual"),
            *(row["name"] for row in CONVERSION_MANIFEST["human"]["meshes"]),
        }
        expected_by_path = {
            row["output_path"]: row
            for row in (
                *(
                    row
                    for row in CONVERSION_MANIFEST["robot"]["meshes"]
                    if row["role"] == "visual"
                ),
                *CONVERSION_MANIFEST["human"]["meshes"],
            )
        }
        for geom_id, provenance in scene.geom_provenance.items():
            row = expected_by_path[provenance["output_path"]]
            assert provenance["output_sha256"] == row["output_sha256"]
            assert provenance["compiled_vertex_max_error_m"] <= 5.0e-7
            assert provenance["compiled_face_indices_exact"] is True
            mesh_id = int(model.geom_dataid[geom_id])
            assert int(model.mesh_vertnum[mesh_id]) == row["vertex_count"]
            assert int(model.mesh_facenum[mesh_id]) == row["triangle_count"]

        first_q = np.asarray(
            [-2.088194, -0.247218, 0.685394, 1.067373, -0.355934, -0.137615, 0.432899],
            dtype=np.float64,
        )
        second_q = np.asarray(
            [0.198075, -2.963995, -0.529316, 1.067420, 0.648251, -0.457586, 1.366376],
            dtype=np.float64,
        )
        patient = np.eye(4)
        patient[:3, :3] = _rpy_rotation((0.07, -0.11, 0.16))
        patient[:3, 3] = [0.012, -0.018, 0.021]
        frames = scene.set_frame(first_q, patient)
        assert len(frames) == 24
        first_robot_poses = {}
        for name in OFFICIAL_LINK_NAMES:
            expected = simulation.ROBOT_WORLD_FROM_ROOT @ frames[name]
            body_name = f"robot__{name}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            np.testing.assert_allclose(scene.data.xpos[body_id], expected[:3, 3], atol=1e-10)
            np.testing.assert_allclose(
                scene.data.xmat[body_id].reshape(3, 3), expected[:3, :3], atol=1e-10
            )
            qpos = _body_qpos(model, scene.data, body_name)
            np.testing.assert_allclose(qpos[:3], expected[:3, 3], atol=1e-12)
            np.testing.assert_allclose(qpos[3:], scene.data.xquat[body_id], atol=1e-12)
            assert np.linalg.norm(qpos[3:]) == pytest.approx(1.0, abs=1e-12)
            first_robot_poses[name] = expected.copy()
        assert abs(
            _body_qpos(model, scene.data, "robot__openarm_left_link7")[3]
        ) > 0.1  # exact freejoint order is xyz + wxyz
        for row in CONVERSION_MANIFEST["human"]["meshes"]:
            body_name = f"human__{row['name']}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            np.testing.assert_allclose(scene.data.xpos[body_id], patient[:3, 3], atol=1e-12)
            np.testing.assert_allclose(
                scene.data.xmat[body_id].reshape(3, 3), patient[:3, :3], atol=1e-12
            )
            qpos = _body_qpos(model, scene.data, body_name)
            np.testing.assert_allclose(qpos[:3], patient[:3, 3], atol=1e-12)
            np.testing.assert_allclose(qpos[3:], scene.data.xquat[body_id], atol=1e-12)
        assert scene.data.ncon == 0 and np.array_equal(scene.data.qvel, np.zeros(model.nv))

        invalid_patient = patient.copy()
        invalid_patient[0, 0] *= 1.1
        with pytest.raises(simulation.OfflineSimulationError, match="proper"):
            scene.set_frame(first_q, invalid_patient)

        first = scene.render()
        assert first.rgb.shape == (360, 640, 3) and first.rgb.dtype == np.uint8
        assert first.depth_m.shape == (360, 640) and np.issubdtype(first.depth_m.dtype, np.floating)
        assert first.segmentation.shape == (360, 640, 2) and first.segmentation.dtype == np.int32
        assert first.robot_mask.dtype == first.human_mask.dtype == np.bool_
        assert np.count_nonzero(first.robot_mask) > 0
        assert np.count_nonzero(first.human_mask) > 0
        assert not np.any(first.robot_mask & first.human_mask)
        robot_from_raw, human_from_raw = simulation.decode_instance_masks(
            first.segmentation, scene.robot_geom_ids, scene.human_geom_ids
        )
        np.testing.assert_array_equal(first.robot_mask, robot_from_raw)
        np.testing.assert_array_equal(first.human_mask, human_from_raw)
        assert set(np.unique(first.segmentation[..., 0][first.robot_mask])) <= scene.robot_geom_ids
        assert set(np.unique(first.segmentation[..., 0][first.human_mask])) <= scene.human_geom_ids
        assert np.all(first.segmentation[..., 1][first.robot_mask] == int(mujoco.mjtObj.mjOBJ_GEOM))
        assert np.all(first.segmentation[..., 1][first.human_mask] == int(mujoco.mjtObj.mjOBJ_GEOM))
        assert np.all(np.isfinite(first.depth_m[first.robot_mask | first.human_mask]))
        assert 0.3 < float(np.min(first.depth_m[first.human_mask])) < 3.0
        background = ~(first.robot_mask | first.human_mask)
        assert np.isfinite(first.depth_m[background]).any()
        assert np.isnan(first.depth_for_visibility_m[background]).all()
        np.testing.assert_array_equal(
            first.depth_for_visibility_m[first.human_mask], first.depth_m[first.human_mask]
        )

        second_frames = scene.set_frame(second_q, patient)
        second = scene.render()
        assert np.count_nonzero(first.robot_mask ^ second.robot_mask) > 100
        for name in OFFICIAL_LINK_NAMES:
            expected = simulation.ROBOT_WORLD_FROM_ROOT @ second_frames[name]
            if name == "world" or name.startswith("openarm_right_") or name == "openarm_body_link0":
                np.testing.assert_array_equal(expected, first_robot_poses[name])
        for row in CONVERSION_MANIFEST["human"]["meshes"]:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"human__{row['name']}"
            )
            np.testing.assert_allclose(scene.data.xpos[body_id], patient[:3, 3], atol=1e-12)

        evidence = scene.capture_frame_evidence(0.05)
        simulation.validate_oracle_frame_evidence(evidence)
        copied_guard_tamper = replace(evidence, timestamp_s=0.051)
        object.__setattr__(
            copied_guard_tamper,
            "_scene_capture_guard_hmac",
            evidence._scene_capture_guard_hmac,  # noqa: SLF001
        )
        with pytest.raises(simulation.OfflineSimulationError, match="scene-owned"):
            simulation.validate_oracle_frame_evidence(copied_guard_tamper)
        with pytest.raises(ValueError, match="read-only"):
            evidence.anchor_local_position_m[0, 0] += 0.25
        with pytest.raises(ValueError, match="read-only"):
            evidence.products.segmentation[0, 0, 0] = 123
        with pytest.raises(ValueError, match="read-only"):
            scene.model.geom_pos[0, 0] += 0.5
        with pytest.raises(ValueError, match="read-only"):
            scene.model.vis.rgba.haze[0] += 0.1
        assert evidence.conversion_release_id == scene.conversion_release_id
        assert evidence.conversion_manifest_sha256 == scene.conversion_manifest_sha256
        assert evidence.asset_release_id == scene.asset_release_id
        assert evidence.asset_manifest_sha256 == scene.asset_manifest_sha256
        assert evidence.runtime_manifest_sha256 == scene.runtime_manifest_sha256
        assert evidence.geom_provenance == scene.geom_provenance
        assert evidence.anchor_names == POINT_NAMES
        assert evidence.anchor_local_position_m.shape == (10, 3)
        assert evidence.anchor_world_position_m.shape == (10, 3)
        np.testing.assert_allclose(
            evidence.anchor_local_position_m,
            [row["position_m"] for row in CONVERSION_MANIFEST["human"]["labcoat"]["anchors"]],
            atol=1e-9,
        )
        expected_world = (
            patient[:3, :3] @ evidence.anchor_local_position_m.T
            + patient[:3, 3, None]
        ).T
        np.testing.assert_allclose(evidence.anchor_world_position_m, expected_world, atol=1e-12)
        world_from_cv = evidence.world_from_cv_camera
        expected_camera = (
            world_from_cv[:3, :3].T
            @ (expected_world - world_from_cv[:3, 3]).T
        ).T
        np.testing.assert_allclose(evidence.anchor_camera_position_m, expected_camera, atol=1e-12)
        expected_pixels = np.column_stack(
            (
                simulation.INTRINSICS["fx"] * expected_camera[:, 0] / expected_camera[:, 2]
                + simulation.INTRINSICS["cx"],
                simulation.INTRINSICS["fy"] * expected_camera[:, 1] / expected_camera[:, 2]
                + simulation.INTRINSICS["cy"],
            )
        )
        np.testing.assert_allclose(evidence.projected_pixels, expected_pixels, atol=1e-12)
        independent_visibility = measure_rendered_visibility(
            point_names=evidence.anchor_names,
            projected_pixels=evidence.projected_pixels,
            point_depth_m=evidence.anchor_camera_position_m[:, 2],
            robot_mask=evidence.products.robot_mask,
            human_mask=evidence.products.human_mask,
            depth_m=evidence.products.depth_for_visibility_m,
            support_radius_px=evidence.support_radius_px,
            robot_mask_dilation_px=evidence.robot_mask_dilation_px,
            front_depth_margin_m=evidence.front_depth_margin_m,
        )
        assert evidence.visibility == independent_visibility
        assert all(
            row.visibility_source == "rendered_instance_mask_and_depth"
            for row in evidence.visibility
        )
        assert max(row.robot_front_fraction for row in evidence.visibility) > 0.5
        without_robot = measure_rendered_visibility(
            point_names=evidence.anchor_names,
            projected_pixels=evidence.projected_pixels,
            point_depth_m=evidence.anchor_camera_position_m[:, 2],
            robot_mask=np.zeros_like(evidence.products.robot_mask),
            human_mask=evidence.products.human_mask,
            depth_m=evidence.products.depth_for_visibility_m,
            support_radius_px=evidence.support_radius_px,
            robot_mask_dilation_px=evidence.robot_mask_dilation_px,
            front_depth_margin_m=evidence.front_depth_margin_m,
        )
        assert max(row.robot_front_fraction for row in without_robot) == 0.0

        output = tmp_path / "frame_000001"
        persisted = simulation.persist_frame_evidence(evidence, output)
        assert persisted["schema"] == (
            "wam_oracle_landmark_noise_simulation_frame_evidence_v1"
        )
        assert persisted["status"] == "oracle_landmark_noise_simulation"
        assert persisted["scope"] == "oracle_landmark_noise_simulation"
        assert persisted["production_activation"] is False
        assert persisted["model_validation"] is False
        assert "scene_capture_guard_hmac" not in persisted
        assert persisted["isaac_equivalence"] is False
        assert np.array_equal(
            np.load(output / "depth_m.npy", allow_pickle=False), evidence.products.depth_m
        )
        assert np.array_equal(
            np.load(output / "segmentation.npy", allow_pickle=False),
            evidence.products.segmentation,
        )
        for row in persisted["files"]:
            assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        assert json.loads((output / "frame_manifest.json").read_text()) == persisted
        with pytest.raises(simulation.OfflineSimulationError, match="new"):
            simulation.persist_frame_evidence(evidence, output)
        assert simulation.verify_frame_evidence_release(output) == persisted
        assert (output / "FRAME_COMPLETE.json").is_file()

        canonical_intrinsics = simulation.INTRINSICS
        drifted_intrinsics = dict(canonical_intrinsics)
        drifted_intrinsics["cx"] += 0.25
        monkeypatch.setattr(simulation, "INTRINSICS", drifted_intrinsics)
        with pytest.raises(
            simulation.OfflineSimulationError,
            match="scene contract drifted",
        ):
            scene.render()
        with pytest.raises(
            simulation.OfflineSimulationError,
            match="scene contract drifted",
        ):
            simulation.verify_frame_evidence_release(output)
        monkeypatch.setattr(simulation, "INTRINSICS", canonical_intrinsics)
        assert simulation.verify_frame_evidence_release(output) == persisted

        public_anchor_copy = scene.anchor_local_position_m
        public_anchor_copy[:] = 999.0
        np.testing.assert_allclose(
            scene.anchor_local_position_m,
            evidence.anchor_local_position_m,
            atol=0.0,
        )
        saved_qpos = scene.data.qpos.copy()
        scene.data.qpos[0] += 0.01
        with pytest.raises(simulation.OfflineSimulationError, match="pose drift"):
            scene.capture_frame_evidence(0.075)
        scene.data.qpos[:] = saved_qpos
        original_znear = float(scene.model.vis.map.znear)
        scene.model.vis.map.znear = original_znear * 1.2
        with pytest.raises(simulation.OfflineSimulationError, match="model|MjModel"):
            scene.capture_frame_evidence(0.08)
        scene.model.vis.map.znear = original_znear

        repeated = scene.capture_frame_evidence(0.10)
        np.testing.assert_array_equal(evidence.products.rgb, repeated.products.rgb)
        np.testing.assert_array_equal(evidence.products.depth_m, repeated.products.depth_m)
        np.testing.assert_array_equal(
            evidence.products.segmentation, repeated.products.segmentation
        )
        np.testing.assert_array_equal(evidence.left_joint_positions, repeated.left_joint_positions)
        np.testing.assert_array_equal(evidence.world_from_patient, repeated.world_from_patient)

        original_write = simulation._write_evidence_file
        writes = 0

        def fail_during_write(path: Path, raw: bytes) -> None:
            nonlocal writes
            writes += 1
            if writes == 3:
                raise OSError("synthetic evidence write failure")
            original_write(path, raw)

        monkeypatch.setattr(simulation, "_write_evidence_file", fail_during_write)
        failed_output = tmp_path / "frame_failed"
        with pytest.raises(simulation.OfflineSimulationError, match="persistence"):
            simulation.persist_frame_evidence(evidence, failed_output)
        assert not failed_output.exists()
        assert not tuple(tmp_path.glob(".frame_failed.staging-*"))
        monkeypatch.setattr(simulation, "_write_evidence_file", original_write)
    finally:
        scene.close()


def test_real_oblique_camera_reuses_exact_official_scene_and_geometry() -> None:
    left_q = np.asarray(
        [-2.088194, -0.247218, 0.685394, 1.067373, -0.355934, -0.137615, 0.432899],
        dtype=np.float64,
    )
    patient = np.eye(4, dtype=np.float64)
    patient[:3, :3] = _rpy_rotation((0.07, -0.11, 0.16))
    patient[:3, 3] = [0.012, -0.018, 0.021]

    with simulation.OfflineMeshScene.from_verified_conversion(
        CONVERSION_RELEASE,
        camera_pose_name="primary",
    ) as primary, simulation.OfflineMeshScene.from_verified_conversion(
        CONVERSION_RELEASE,
        camera_pose_name="oblique",
    ) as oblique:
        assert primary.camera_pose_name == "primary"
        assert oblique.camera_pose_name == "oblique"
        np.testing.assert_array_equal(
            primary.world_from_cv_camera,
            simulation.camera_pose_transform("primary"),
        )
        np.testing.assert_array_equal(
            oblique.world_from_cv_camera,
            simulation.camera_pose_transform("oblique"),
        )
        assert primary.model.ngeom == oblique.model.ngeom == 28
        assert primary.model.nmesh == oblique.model.nmesh == 28
        assert primary.robot_geom_ids == oblique.robot_geom_ids
        assert primary.human_geom_ids == oblique.human_geom_ids
        assert primary.geom_provenance == oblique.geom_provenance
        np.testing.assert_array_equal(primary.model.mesh_scale, oblique.model.mesh_scale)
        np.testing.assert_array_equal(primary.model.mesh_vert, oblique.model.mesh_vert)
        np.testing.assert_array_equal(primary.model.mesh_face, oblique.model.mesh_face)

        primary.set_frame(left_q, patient)
        oblique.set_frame(left_q, patient)
        np.testing.assert_array_equal(primary.data.qpos, oblique.data.qpos)
        np.testing.assert_array_equal(primary.data.xpos, oblique.data.xpos)
        np.testing.assert_array_equal(primary.data.xmat, oblique.data.xmat)

        primary_products = primary.render(camera_pose_name="primary")
        oblique_products = oblique.render(camera_pose_name="oblique")
        for products in (primary_products, oblique_products):
            assert np.count_nonzero(products.robot_mask) > 0
            assert np.count_nonzero(products.human_mask) > 0
            assert not np.any(products.robot_mask & products.human_mask)
        assert np.count_nonzero(
            primary_products.robot_mask ^ oblique_products.robot_mask
        ) > 1000
        assert np.count_nonzero(
            primary_products.human_mask ^ oblique_products.human_mask
        ) > 1000

        primary_projection = primary.project_current_anchors()
        oblique_projection = oblique.project_current_anchors()
        np.testing.assert_array_equal(
            primary_projection.anchor_world_position_m,
            oblique_projection.anchor_world_position_m,
        )
        assert np.mean(
            np.linalg.norm(
                primary_projection.projected_pixels
                - oblique_projection.projected_pixels,
                axis=1,
            )
        ) > 10.0
        np.testing.assert_allclose(
            oblique_projection.projected_pixels,
            simulation.project_cv_points(
                oblique_projection.anchor_camera_position_m
            ),
            atol=0.0,
        )
        assert np.all(oblique_projection.anchor_camera_position_m[:, 2] > 0.0)
        visibility = measure_rendered_visibility(
            point_names=oblique.anchor_names,
            projected_pixels=oblique_projection.projected_pixels,
            point_depth_m=oblique_projection.anchor_camera_position_m[:, 2],
            robot_mask=oblique_products.robot_mask,
            human_mask=oblique_products.human_mask,
            depth_m=oblique_products.depth_for_visibility_m,
            support_radius_px=6,
            robot_mask_dilation_px=1,
            front_depth_margin_m=0.003,
        )
        assert len(visibility) == 10
        assert all(
            row.visibility_source == "rendered_instance_mask_and_depth"
            for row in visibility
        )
        assert sum(row.valid_depth_fraction > 0.0 for row in visibility) >= 8

        with pytest.raises(simulation.OfflineSimulationError, match="bound camera pose"):
            oblique.render(camera_pose_name="primary")
        with pytest.raises(simulation.OfflineSimulationError, match="camera pose"):
            primary.render(camera_pose_name="side")  # type: ignore[arg-type]
        with pytest.raises(simulation.OfflineSimulationError, match="primary camera"):
            oblique.capture_frame_evidence(0.0)


def test_policy_gate_uses_geometry_not_visible_point_count() -> None:
    surface = DecisionSurface(
        min_relative_geometry_weight=0.05,
        max_depth_consistency_sigma=3.0,
        min_second_spatial_eigenvalue_m2=2.0e-4,
        min_hull_area_m2=1.0e-3,
        min_spread_m=8.0e-2,
        max_condition_number=40.0,
        max_residual_m=4.0e-3,
        max_target_leverage=4.0,
        max_target_error_p95_m=7.0e-3,
    )
    clustered_six = np.asarray(
        [
            [-0.025, -0.0010, 1.0],
            [-0.015, 0.0007, 1.0],
            [-0.005, -0.0004, 1.0],
            [0.005, 0.0005, 1.0],
            [0.015, -0.0008, 1.0],
            [0.025, 0.0010, 1.0],
        ],
        dtype=np.float64,
    )
    broad_four = np.asarray(
        [
            [-0.12, -0.15, 1.0],
            [0.12, -0.15, 1.0],
            [-0.12, 0.15, 1.0],
            [0.12, 0.15, 1.0],
        ],
        dtype=np.float64,
    )

    def evaluate(points: np.ndarray):
        return evaluate_visibility_geometry(
            points,
            points.copy(),
            np.asarray([0.0, 0.14, 1.0]),
            depth_m=points[:, 2].copy(),
            mask_confidence=np.full(len(points), 0.95),
            lateral_noise_std_m=np.full(len(points), 1.0e-3),
            axial_noise_std_m=np.full(len(points), 2.0e-3),
            robust_delta_m=3.0e-3,
            decision_surface=surface,
        )

    assert not evaluate(clustered_six).accepted
    assert evaluate(broad_four).accepted

    six_trace = simulation.evaluate_occlusion_policies(
        _policy_observation_stream(visible_point_count_report_only=6)
    )
    four_trace = simulation.evaluate_occlusion_policies(
        _policy_observation_stream(visible_point_count_report_only=4)
    )
    for policy_name in six_trace:
        six_rows = six_trace[policy_name]
        four_rows = four_trace[policy_name]
        assert len(six_rows) == len(four_rows)
        for six, four in zip(six_rows, four_rows, strict=True):
            assert six.visible_point_count_report_only == 6
            assert four.visible_point_count_report_only == 4
            assert (
                six.policy_name,
                six.state,
                six.valid,
                six.episode_id,
                six.last_valid_age_s,
                six.predicted_error_p95_m,
                six.reacquired,
            ) == (
                four.policy_name,
                four.state,
                four.valid,
                four.episode_id,
                four.last_valid_age_s,
                four.predicted_error_p95_m,
                four.reacquired,
            )
            if six.estimate_world_m is None:
                assert four.estimate_world_m is None
            else:
                np.testing.assert_array_equal(six.estimate_world_m, four.estimate_world_m)


def test_three_policies_are_gt_free_bounded_and_reacquire_new_episode() -> None:
    observation_field_names = {field.name for field in fields(simulation.ObservationFrame)}
    output_field_names = {field.name for field in fields(simulation.PolicyFrameOutput)}
    assert not any("ground_truth" in name or "actual_error" in name for name in observation_field_names)
    assert not any("ground_truth" in name or "actual_error" in name for name in output_field_names)
    assert "ground_truth" not in " ".join(
        inspect.signature(simulation.evaluate_occlusion_policies).parameters
    )

    observations = _policy_observation_stream()
    outputs = simulation.evaluate_occlusion_policies(observations)
    assert tuple(outputs) == (
        "direct_geometry_only",
        "indefinite_world_lock_baseline",
        "bounded_freeze_then_lost",
    )
    bounded = outputs["bounded_freeze_then_lost"]
    assert tuple(row.state for row in bounded) == (
        "DIRECT_GEOMETRY_UPDATE",
        *("BOUNDED_PATIENT_FRAME_FREEZE" for _ in range(6)),
        *("LOST" for _ in range(6)),
        "DIRECT_GEOMETRY_UPDATE",
    )
    assert [row.predicted_error_p95_m for row in bounded[:8]] == pytest.approx(
        [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
    )
    for row in bounded:
        if row.state == "BOUNDED_PATIENT_FRAME_FREEZE":
            assert row.valid
            assert row.last_valid_age_s <= 0.40
            assert row.predicted_error_p95_m <= 0.010
        if row.state == "LOST":
            assert not row.valid
            assert row.estimate_world_m is None
            assert row.episode_id is None
    assert bounded[0].episode_id == "E0001"
    assert bounded[13].episode_id == "E0002"
    assert bounded[13].reacquired is True

    indefinite = outputs["indefinite_world_lock_baseline"]
    assert indefinite[11].valid
    np.testing.assert_array_equal(indefinite[11].estimate_world_m, [0.0, 0.0, 1.0])
    assert indefinite[11].state == "INDEFINITE_WORLD_LOCK_UNSAFE"
    assert indefinite[11].deliberately_unsafe is True
    for policy_name in outputs:
        assert outputs[policy_name][13].episode_id == "E0002"

    metrics = simulation.score_occlusion_policy_metrics(
        outputs, _policy_ground_truth_stream()
    )
    unsafe = metrics["indefinite_world_lock_baseline"]
    assert unsafe["target_error_m"]["p99"] > 0.010
    assert unsafe["false_valid_above_10mm"]["numerator"] == 2
    assert unsafe["false_valid_above_10mm"]["denominator"] == 14
    assert unsafe["state_counts"] == {
        "DIRECT_GEOMETRY_UPDATE": 2,
        "INDEFINITE_WORLD_LOCK_UNSAFE": 12,
    }
    bounded_metrics = metrics["bounded_freeze_then_lost"]
    assert bounded_metrics["false_valid_above_10mm"]["numerator"] == 0
    assert bounded_metrics["time_to_lost_s"]["p50"] == pytest.approx(0.30)
    assert bounded_metrics["reacquisition_time_s"]["p50"] == pytest.approx(0.0)
    assert bounded_metrics["rejection_run_duration_s"]["p50"] == pytest.approx(0.60)


def test_policy_scoring_fails_closed_and_preserves_undefined_metrics() -> None:
    observations = tuple(
        simulation.ObservationFrame(
            scenario_id="never_seen",
            target_name="L4",
            wrist_index=1,
            frame_index=index,
            timestamp_s=index * 0.05,
            geometry_accepted=False,
            accepted_estimate_world_m=None,
            accepted_error_p95_m=None,
            patient_motion_bound_m_s=0.01,
            visible_point_count_report_only=0,
            observation_stream_sha256="3" * 64,
        )
        for index in range(2)
    )
    ground_truth = tuple(
        simulation.GroundTruthFrame(
            scenario_id="never_seen",
            target_name="L4",
                wrist_index=1,
                frame_index=index,
                timestamp_s=index * 0.05,
                observation_stream_sha256="3" * 64,
                target_world_m=np.asarray([0.0, 0.0, 1.0]),
        )
        for index in range(2)
    )
    outputs = simulation.evaluate_occlusion_policies(observations)
    metrics = simulation.score_occlusion_policy_metrics(outputs, ground_truth)
    for policy_metrics in metrics.values():
        assert policy_metrics["target_error_m"] == {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
        assert policy_metrics["jitter_m"] == {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
        assert policy_metrics["coverage"] == {
            "numerator": 0,
            "denominator": 2,
            "rate": 0.0,
        }
        assert policy_metrics["false_valid_above_5mm"]["denominator"] == 0
        assert policy_metrics["false_valid_above_5mm"]["rate"] is None
        assert policy_metrics["time_to_lost_s"] == {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    with pytest.raises(simulation.OfflineSimulationError, match="duplicate|one-to-one"):
        simulation.score_occlusion_policy_metrics(
            outputs, ground_truth + (ground_truth[0],)
        )
    missing = {name: rows[:-1] for name, rows in outputs.items()}
    with pytest.raises(simulation.OfflineSimulationError, match="missing|one-to-one"):
        simulation.score_occlusion_policy_metrics(missing, ground_truth)

    accepted = _policy_observation_stream()[0]
    with pytest.raises(ValueError):
        accepted.accepted_estimate_world_m[0] = 1.0


def test_scorer_rejects_mixed_observation_traces_and_forged_geometry() -> None:
    outputs = simulation.evaluate_occlusion_policies(_policy_observation_stream())
    truth = _policy_ground_truth_stream()

    mixed_hash = dict(outputs)
    mixed_hash["bounded_freeze_then_lost"] = (
        replace(
            outputs["bounded_freeze_then_lost"][0],
            observation_stream_sha256="4" * 64,
        ),
        *outputs["bounded_freeze_then_lost"][1:],
    )
    with pytest.raises(simulation.OfflineSimulationError, match="observation trace|stream"):
        simulation.score_occlusion_policy_metrics(mixed_hash, truth)

    forged_geometry = dict(outputs)
    forged_geometry["direct_geometry_only"] = (
        replace(outputs["direct_geometry_only"][0], geometry_accepted=False),
        *outputs["direct_geometry_only"][1:],
    )
    with pytest.raises(simulation.OfflineSimulationError, match="observation trace|geometry"):
        simulation.score_occlusion_policy_metrics(forged_geometry, truth)

    for policy_name in (
        "indefinite_world_lock_baseline",
        "bounded_freeze_then_lost",
    ):
        forged_hold = dict(outputs)
        forged_hold[policy_name] = (
            outputs[policy_name][0],
            replace(
                outputs[policy_name][1],
                estimate_world_m=np.asarray([9.0, 9.0, 9.0]),
            ),
            *outputs[policy_name][2:],
        )
        with pytest.raises(
            simulation.OfflineSimulationError,
            match="held estimate|last accepted|observation trace",
        ):
            simulation.score_occlusion_policy_metrics(forged_hold, truth)

    forged_age = dict(outputs)
    forged_age["indefinite_world_lock_baseline"] = (
        outputs["indefinite_world_lock_baseline"][0],
        replace(
            outputs["indefinite_world_lock_baseline"][1],
            last_valid_age_s=0.123,
        ),
        *outputs["indefinite_world_lock_baseline"][2:],
    )
    with pytest.raises(
        simulation.OfflineSimulationError,
        match="age|last accepted|observation trace",
    ):
        simulation.score_occlusion_policy_metrics(forged_age, truth)

    forged_bounded_early_lost = dict(outputs)
    forged_bounded_early_lost["bounded_freeze_then_lost"] = (
        outputs["bounded_freeze_then_lost"][0],
        replace(
            outputs["bounded_freeze_then_lost"][1],
            state="LOST",
            valid=False,
            estimate_world_m=None,
            episode_id=None,
        ),
        *outputs["bounded_freeze_then_lost"][2:],
    )
    with pytest.raises(
        simulation.OfflineSimulationError,
        match="bounded decision|FREEZE|observation trace",
    ):
        simulation.score_occlusion_policy_metrics(
            forged_bounded_early_lost, truth
        )


def test_false_valid_threshold_is_mathematically_strict() -> None:
    observation = simulation.ObservationFrame(
        scenario_id="strict_threshold",
        target_name="R1",
        wrist_index=0,
        frame_index=0,
        timestamp_s=0.0,
        geometry_accepted=True,
        accepted_estimate_world_m=np.asarray([0.0, 0.0, 1.0]),
        accepted_error_p95_m=0.001,
        patient_motion_bound_m_s=0.0,
        visible_point_count_report_only=10,
        observation_stream_sha256="5" * 64,
    )
    outputs = simulation.evaluate_occlusion_policies((observation,))

    exact = simulation.GroundTruthFrame(
        scenario_id="strict_threshold",
        target_name="R1",
        wrist_index=0,
        frame_index=0,
        timestamp_s=0.0,
        observation_stream_sha256="5" * 64,
        target_world_m=np.asarray([0.010, 0.0, 1.0]),
    )
    just_above = replace(
        exact, target_world_m=np.asarray([0.0100000000005, 0.0, 1.0])
    )
    just_below = replace(
        exact, target_world_m=np.asarray([0.0099999999995, 0.0, 1.0])
    )
    assert simulation.score_occlusion_policy_metrics(
        outputs, (exact,)
    )["direct_geometry_only"]["false_valid_above_10mm"]["numerator"] == 0
    assert simulation.score_occlusion_policy_metrics(
        outputs, (just_above,)
    )["direct_geometry_only"]["false_valid_above_10mm"]["numerator"] == 1
    assert simulation.score_occlusion_policy_metrics(
        outputs, (just_below,)
    )["direct_geometry_only"]["false_valid_above_10mm"]["numerator"] == 0


def test_stratified_metrics_are_equal_weighted_and_keep_failed_ik_denominator() -> None:
    observations = []
    truth = []
    for scenario_id, digest, frame_count in (
        ("short_stratum", "a" * 64, 2),
        ("long_stratum", "b" * 64, 4),
    ):
        for frame_index in range(frame_count):
            accepted = scenario_id == "long_stratum" or frame_index == 0
            observations.append(
                simulation.ObservationFrame(
                    scenario_id=scenario_id,
                    target_name="L2",
                    wrist_index=0,
                    frame_index=frame_index,
                    timestamp_s=frame_index * 0.05,
                    geometry_accepted=accepted,
                    accepted_estimate_world_m=(
                        np.asarray([0.0, 0.0, 1.0]) if accepted else None
                    ),
                    accepted_error_p95_m=0.001 if accepted else None,
                    patient_motion_bound_m_s=0.0,
                    visible_point_count_report_only=4 if accepted else 0,
                    observation_stream_sha256=digest,
                )
            )
            truth.append(
                simulation.GroundTruthFrame(
                    scenario_id=scenario_id,
                    target_name="L2",
                    wrist_index=0,
                    frame_index=frame_index,
                    timestamp_s=frame_index * 0.05,
                    observation_stream_sha256=digest,
                    target_world_m=np.asarray([0.0, 0.0, 1.0]),
                )
            )

    required = (
        simulation.RequiredStratum(
            scenario_id="short_stratum",
            target_name="L2",
            wrist_index=0,
            observation_stream_sha256="a" * 64,
            observation_seed=101,
            patient_motion_seed=201,
            execution_status="COMPLETE",
            declared_frame_count=2,
        ),
        simulation.RequiredStratum(
            scenario_id="long_stratum",
            target_name="L2",
            wrist_index=0,
            observation_stream_sha256="b" * 64,
            observation_seed=102,
            patient_motion_seed=202,
            execution_status="COMPLETE",
            declared_frame_count=4,
        ),
        simulation.RequiredStratum(
            scenario_id="failed_ik_stratum",
            target_name="L2",
            wrist_index=1,
            observation_stream_sha256="c" * 64,
            observation_seed=103,
            patient_motion_seed=203,
            execution_status="IK_UNREACHABLE",
            declared_frame_count=0,
        ),
    )
    metrics = simulation.score_occlusion_policy_metrics(
        simulation.evaluate_occlusion_policies(tuple(observations)),
        tuple(truth),
        required_strata=required,
    )["direct_geometry_only"]

    assert metrics["pooled_frame"]["coverage"]["rate"] == pytest.approx(5 / 6)
    assert metrics["equal_weight_per_stratum"]["coverage"] == {
        "contributing_strata_count": 2,
        "mean": pytest.approx(0.75),
    }
    assert metrics["denominators"] == {
        "required_strata_count": 3,
        "completed_strata_count": 2,
        "failed_strata_count": 1,
        "metric_eligible_strata_count": 2,
        "metric_ineligible_strata_count": 1,
        "contributing_strata_count": 2,
        "eligible_paired_frames": 6,
        "completion_rate": pytest.approx(2 / 3),
    }
    assert len(metrics["per_stratum"]) == 3
    failed = next(
        row for row in metrics["per_stratum"] if row["scenario_id"] == "failed_ik_stratum"
    )
    assert failed == {
        "scenario_id": "failed_ik_stratum",
        "target_name": "L2",
        "wrist_index": 1,
        "observation_stream_sha256": "c" * 64,
        "observation_seed": 103,
        "patient_motion_seed": 203,
        "execution_status": "IK_UNREACHABLE",
        "declared_frame_count": 0,
        "metric_eligible": False,
        "metric_ineligibility_reason": "execution_status_ik_unreachable",
        "metrics_available": False,
        "metrics": None,
    }
