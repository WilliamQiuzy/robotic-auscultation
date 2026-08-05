from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.production_runtime_activation import (
    ProductionRuntimeActivationError,
    _build_identity,
    _build_handeye,
    _load_artifact,
    _validate_handeye_runtime_bound,
    load_completed_handeye_release,
    load_production_runtime_assembly,
)
from ausc.perception.robot.openarm_pair_activation import (
    COMPLETE_MARKER,
    CONTRACT_FILE,
    LEFT_JOINT_NAMES,
    build_activation_envelope,
    build_completion_marker,
    canonical_json_bytes,
)
from ausc.perception.robot.robot_capture_coordinator import (
    ProductionRobotCaptureCoordinator,
)
from ausc.perception.robot.robot_mask_provider import (
    ActivatedHandEye,
    CameraProjectionProfile,
    KinematicCollisionLink,
    OpenArmCollisionGeometry,
    OpenArmToolGeometry,
    ProvisionalRobotMaskHandEye,
    ProductionRobotMaskProvider,
    RobotMaskUncertaintyProfile,
)
from ausc.perception.tracking.depth_noise_profile import (
    ActivatedDepthNoiseProfile,
    depth_noise_profile_semantic_sha256,
)
from ausc.perception.tracking.occlusion_runtime import (
    ActivatedCameraProjection,
    ActivatedOcclusionIdentity,
    ActivatedOcclusionRuntimeProfile,
    OcclusionRuntime,
    OcclusionRuntimeConfig,
)
from ausc.perception.tracking.patient_pose_provider import (
    MetricPatientPoseProvider,
    PatientPoseProviderConfig,
)
from ausc.perception.tracking.posterior_back_mask_provider import (
    IndependentPosteriorBackMaskProvider,
    PosteriorBackMaskConfig,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
POINT_NAMES = tuple(f"P{index}" for index in range(1, 11))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_artifact(
    root: Path,
    name: str,
    payload: dict,
    *,
    raw: bytes | None = None,
) -> tuple[dict[str, str], Path]:
    artifact = root / "artifacts" / f"{name}.json"
    complete = root / "complete" / f"{name}.COMPLETE"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    complete.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical(payload) if raw is None else raw
    artifact.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    marker = {
        "artifact_sha256": digest,
        "schema": "wam_production_runtime_artifact_completion_v1",
        "status": "COMPLETE",
    }
    marker_raw = _canonical(marker) + b"\n"
    complete.write_bytes(marker_raw)
    return {
        "complete_path": str(complete),
        "complete_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "path": str(artifact),
        "sha256": digest,
    }, complete


def _provisional_handeye_payload() -> dict:
    follower_base_from_camera = np.eye(4, dtype=np.float64)
    follower_base_from_camera[:3, 3] = (0.30, -0.20, 0.70)
    camera_from_follower_base = np.linalg.inv(follower_base_from_camera)
    covariance = np.diag((1.0e-6, 1.1e-6, 1.2e-6, 4.0e-6, 5.0e-6, 6.0e-6))
    return {
        "acceptance_note": "William accepted approximately 5 mm for bootstrap use.",
        "authorized_scopes": [
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ],
        "autonomous_targeting_authorized": False,
        "bootstrap_reproducibility": {
            "missing": ["bootstrap_seed", "bootstrap_sample_count", "resample_policy"],
            "status": "INCOMPLETE",
        },
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "camera_id": "CL8326300A0",
        "code_release": "handeye_calibration_code-" + "a" * 20,
        "covariance_scope": "random_only",
        "covariance_se3": {
            "bound_transform": "transform_follower_base_from_camera",
            "definition": (
                "empirical bootstrap cov of delta=[Rodrigues(R_ref^T R_boot), "
                "t_boot - t_ref]; random precision only, excludes systematic bias"
            ),
            "matrix": covariance.tolist(),
            "order": ["rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m"],
        },
        "failed_quality_gate_m": 0.003,
        "provisional": True,
        "quality": {
            "result_precision_mm_1sigma_max": 6.05,
            "result_precision_mm_1sigma_median": 4.96,
        },
        "quality_gate_passed": False,
        "schema": "wam_openarm_handeye_activation_v2",
        "side": "left",
        "source_evidence": {"sample_count": 23},
        "source_frame": "orbbec_color_optical_frame",
        "source_run_id": "handeye_left_20260718T195334Z_70cf253b",
        "systematic_bias_bound_m": None,
        "systematic_bias_status": "NOT_CHARACTERIZED",
        "target_frame": "openarm_left_link0",
        "tip_offset_frame": "openarm_left_hand_tcp",
        "tip_offset_tcp_m": [0.002, 0.028, 0.180],
        "transform_camera_from_follower_base": camera_from_follower_base.tolist(),
        "transform_follower_base_from_camera": follower_base_from_camera.tolist(),
        "workspace_1sigma_m": {"max": 0.00605, "median": 0.00496},
    }


def _write_provisional_handeye_release(root: Path) -> tuple[dict[str, str], Path]:
    payload = _provisional_handeye_payload()
    artifact_raw = _canonical(payload) + b"\n"
    artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()
    release = root / ("handeye_activation-" + artifact_sha256[:20])
    release.mkdir(parents=True)
    artifact = release / "handeye.json"
    complete = release / "handeye.COMPLETE"
    release_complete = release / "COMPLETE"
    artifact.write_bytes(artifact_raw)
    marker = {
        "artifact_sha256": artifact_sha256,
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "provisional": True,
        "schema": "wam_openarm_handeye_artifact_completion_v2",
        "status": "COMPLETE",
    }
    marker_raw = _canonical(marker) + b"\n"
    complete.write_bytes(marker_raw)
    inventory = {
        "handeye.COMPLETE": {
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size_bytes": len(marker_raw),
        },
        "handeye.json": {
            "sha256": artifact_sha256,
            "size_bytes": len(artifact_raw),
        },
    }
    release_marker = {
        "artifact_sha256": artifact_sha256,
        "inventory": inventory,
        "release_id": release.name,
        "schema": "wam_openarm_handeye_release_completion_v2",
        "status": "COMPLETE",
    }
    release_complete.write_bytes(_canonical(release_marker) + b"\n")
    for path in release.iterdir():
        path.chmod(0o444)
    release.chmod(0o555)
    return {
        "complete_path": str(complete),
        "complete_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "path": str(artifact),
        "sha256": artifact_sha256,
    }, release


def _make_provisional_release_writable(release: Path) -> None:
    release.chmod(0o755)
    for path in release.iterdir():
        path.chmod(0o644)


def _box(center=(0.03, 0.0, 0.0), half=(0.01, 0.01, 0.01)) -> list[list[float]]:
    centre = np.asarray(center, dtype=np.float64)
    radius = np.asarray(half, dtype=np.float64)
    return [
        (centre + np.asarray((sx, sy, sz)) * radius).tolist()
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def _transform(x=0.0, y=0.0, z=0.0) -> list[list[float]]:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value.tolist()


def _pair_profile(*, camera_profile_sha256: str, timing_sha256: str) -> dict:
    source_manifest = [
        {"path": "config/follower.yaml", "sha256": "1" * 64},
        {
            "path": "control/openarm_unilateral_control.cpp",
            "sha256": "2" * 64,
        },
    ]
    sdk_manifest = [
        {
            "kind": "header",
            "path": "include/openarm/openarm.hpp",
            "sha256": "3" * 64,
        },
        {
            "kind": "library",
            "path": "lib/libopenarm.so",
            "sha256": "4" * 64,
        },
    ]
    motor_ids = {
        "arm_receive_can_ids": list(range(17, 24)),
        "arm_send_can_ids": list(range(1, 8)),
        "gripper_receive_can_id": 24,
        "gripper_send_can_id": 8,
    }
    return {
        "schema": "wam_openarm_pair_contract_v2",
        "side": "left",
        "controller_mode": "unilateral",
        "leader": {
            "role": "leader",
            "can_interface": "can1",
            "adapter": {
                "id_path": "pci-0000:00:14.0-usb-0:1.2:1.0",
                "serial": "LEFT-LEADER",
                "channel": "0",
            },
            "expected_motor_ids": motor_ids,
        },
        "follower": {
            "role": "follower",
            "can_interface": "can3",
            "adapter": {
                "id_path": "pci-0000:00:14.0-usb-0:1.4:1.0",
                "serial": "LEFT-FOLLOWER",
                "channel": "0",
            },
            "expected_motor_ids": motor_ids,
        },
        "joint_names": list(LEFT_JOINT_NAMES),
        "urdf_sha256": SHA_A,
        "joint_mapper_header_sha256": SHA_B,
        "controller_source_manifest": source_manifest,
        "controller_source_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(source_manifest)
        ).hexdigest(),
        "follower_config_sha256": "1" * 64,
        "openarm_sdk_manifest": sdk_manifest,
        "openarm_sdk_sha256": hashlib.sha256(
            canonical_json_bytes(sdk_manifest)
        ).hexdigest(),
        "candidate_binary_sha256": SHA_C,
        "telemetry_socket": (
            f"/run/user/{os.geteuid()}/wam/openarm-left-telemetry.sock"
        ),
        "camera_profile_sha256": camera_profile_sha256,
        "timing_artifact_sha256": timing_sha256,
    }


def _write_pair_release(
    root: Path, *, camera_profile_sha256: str, timing_sha256: str,
) -> tuple[Path, str]:
    envelope = build_activation_envelope(
        _pair_profile(
            camera_profile_sha256=camera_profile_sha256,
            timing_sha256=timing_sha256,
        )
    )
    release = root / str(envelope["release_id"])
    release.mkdir(parents=True)
    raw = canonical_json_bytes(envelope)
    (release / CONTRACT_FILE).write_bytes(raw)
    (release / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes(build_completion_marker(envelope, raw))
    )
    return release, str(envelope["contract_sha256"])


def _activation_fixture(tmp_path: Path) -> dict:
    alignment, _ = _write_artifact(
        tmp_path,
        "alignment",
        {
            "align_mode": "software_depth_to_color",
            "camera_id": "CL8326300A0",
            "schema": "wam_rgbd_alignment_activation_v1",
        },
    )
    capture_mode = {
        "align_mode": "software_depth_to_color",
        "backend": "orbbec",
        "camera_id": "CL8326300A0",
        "color_format": "MJPG",
        "depth_mode": {
            "format": "Y16",
            "fps": 30.0,
            "height": 576,
            "width": 640,
        },
        "fps": 30.0,
        "frame_aggregate_mode": "full_frame_require",
        "height": 1080,
        "time_source": "orbbec_device_and_host_arrival",
        "width": 1920,
    }
    camera_profile_payload = {
        "alignment_sha256": alignment["sha256"],
        "camera_id": "CL8326300A0",
        "capture_mode": capture_mode,
        "intrinsics": {
            "cx": 959.5,
            "cy": 539.5,
            "depth_scale_mm": 1.0,
            "fx": 1200.0,
            "fy": 1201.0,
            "height": 1080,
            "width": 1920,
        },
        "min_projection_depth_m": 0.05,
        "schema": "wam_rgbd_camera_profile_activation_v1",
    }
    camera_profile, _ = _write_artifact(
        tmp_path, "camera_profile", camera_profile_payload,
    )
    latency_payload = {
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure",
        "latency_ms": 18.0,
        "latency_std_ms": 2.0,
        "n_samples": 40,
        "measurement_method": "hardware_led_flash",
        "capture_mode": capture_mode,
    }
    latency_raw = (json.dumps(latency_payload, indent=2) + "\n").encode("utf-8")
    latency, latency_complete = _write_artifact(
        tmp_path, "camera_latency", latency_payload, raw=latency_raw,
    )

    links = []
    for index in range(7):
        links.append({
            "joint_axis": [0.0, 0.0, 1.0],
            "joint_index": index,
            "name": f"openarm_left_link{index + 1}",
            "parent_index": index - 1,
            "transform_parent_from_joint_zero": (
                _transform(z=1.0) if index == 0 else _transform(x=0.07)
            ),
            "vertices_link_m": _box(),
        })
    geometry_payload = {
        "joint_names": list(LEFT_JOINT_NAMES),
        "links": links,
        "schema": "wam_openarm_collision_geometry_source_v1",
        "side": "left",
        "source": "official_openarm_v10_urdf_collision_meshes",
        "urdf_sha256": SHA_A,
    }
    geometry_entry, _ = _write_artifact(
        tmp_path, "collision_geometry", geometry_payload,
    )
    geometry = OpenArmCollisionGeometry.create(
        source=geometry_payload["source"],
        side="left",
        joint_names=LEFT_JOINT_NAMES,
        urdf_sha256=SHA_A,
        links=tuple(
            KinematicCollisionLink(
                name=row["name"],
                parent_index=row["parent_index"],
                joint_index=row["joint_index"],
                transform_parent_from_joint_zero=np.asarray(
                    row["transform_parent_from_joint_zero"]
                ),
                joint_axis=np.asarray(row["joint_axis"]),
                vertices_link_m=np.asarray(row["vertices_link_m"]),
            )
            for row in links
        ),
    )
    tool_payload = {
        "parent_link_index": 6,
        "schema": "wam_openarm_tool_geometry_source_v1",
        "source": "commissioned_openarm_stethoscope_tool",
        "transform_parent_from_tool": _transform(x=0.07),
        "vertices_tool_m": _box((0.05, 0.0, 0.0), (0.02, 0.025, 0.02)),
    }
    tool_entry, _ = _write_artifact(tmp_path, "tool_geometry", tool_payload)
    tool = OpenArmToolGeometry.create(
        source=tool_payload["source"],
        parent_link_index=6,
        transform_parent_from_tool=np.asarray(
            tool_payload["transform_parent_from_tool"]
        ),
        vertices_tool_m=np.asarray(tool_payload["vertices_tool_m"]),
    )
    handeye_payload = {
        "camera_id": "CL8326300A0",
        "covariance_se3": (np.eye(6) * 1.0e-8).tolist(),
        "schema": "wam_openarm_handeye_activation_v1",
        "side": "left",
        "transform_camera_from_follower_base": _transform(z=0.1),
    }
    handeye_entry, handeye_complete = _write_artifact(
        tmp_path, "handeye", handeye_payload,
    )
    handeye = ActivatedHandEye.create(
        camera_id="CL8326300A0",
        side="left",
        transform_camera_from_follower_base=np.asarray(
            handeye_payload["transform_camera_from_follower_base"]
        ),
        covariance_se3=np.asarray(handeye_payload["covariance_se3"]),
        status="COMPLETE",
    )
    uncertainty_payload = {
        "base_dilation_px": 1.0,
        "link_geometry_error_m": 0.0005,
        "projection_error_px": 0.25,
        "schema": "wam_robot_mask_uncertainty_activation_v1",
        "sigma": 3.0,
        "timing_uncertainty_s": 0.002,
        "tool_geometry_error_m": 0.0005,
    }
    uncertainty_entry, _ = _write_artifact(
        tmp_path, "robot_mask_uncertainty", uncertainty_payload,
    )
    uncertainty = RobotMaskUncertaintyProfile.create(
        **{key: value for key, value in uncertainty_payload.items() if key != "schema"}
    )
    back_mask_config = PosteriorBackMaskConfig.create()
    back_mask_payload = {
        "schema": "wam_posterior_back_mask_activation_v1",
        "config": {
            name: getattr(back_mask_config, name)
            for name in back_mask_config.__dataclass_fields__
            if name not in {"semantic_sha256", "release_id"}
        },
    }
    back_mask_entry, _ = _write_artifact(
        tmp_path, "back_mask", back_mask_payload,
    )

    depth_values = {
        "camera_id": "CL8326300A0",
        "camera_profile_sha256": camera_profile["sha256"],
        "alignment_sha256": alignment["sha256"],
        "calibration_artifact_sha256": "5" * 64,
        "calibration_complete_sha256": "6" * 64,
        "width": 1920,
        "height": 1080,
        "fx": 1200.0,
        "fy": 1201.0,
        "cx": 959.5,
        "cy": 539.5,
        "depth_scale_mm": 1.0,
        "calibrated_min_depth_m": 0.20,
        "calibrated_max_depth_m": 3.0,
        "axial_sigma_floor_m": 0.0005,
        "axial_sigma_quadratic_m_inv": 0.0005,
        "pixel_quantization_sigma_px": 0.30,
        "alignment_sigma_u_px": 0.25,
        "alignment_sigma_v_px": 0.20,
        "alignment_sigma_depth_m": 0.0004,
        "fit_residual_scale": 1.0,
        "uncertainty_expansion_factor": 1.0,
        "calibration_quantile": 0.95,
        "calibration_frame_count": 200,
        "calibration_sample_count": 20_000,
        "calibration_depth_bin_count": 10,
    }
    depth_values["semantic_sha256"] = depth_noise_profile_semantic_sha256(
        **depth_values
    )
    depth_payload = {
        "schema": "wam_activated_aligned_depth_noise_profile_v1",
        **depth_values,
    }
    depth_entry, depth_complete = _write_artifact(
        tmp_path, "depth_noise", depth_payload,
    )
    depth_profile = ActivatedDepthNoiseProfile(**depth_values)
    patient_pose_config = PatientPoseProviderConfig()
    patient_pose_payload = {
        "schema": "wam_metric_patient_pose_activation_v1",
        "config": {
            name: getattr(patient_pose_config, name)
            for name in patient_pose_config.__dataclass_fields__
        },
    }
    patient_pose_entry, _ = _write_artifact(
        tmp_path, "patient_pose", patient_pose_payload,
    )
    patient_pose_release_id = "patient_pose-" + patient_pose_entry["sha256"][:20]
    occlusion_config = OcclusionRuntimeConfig()
    occlusion_payload = {
        "schema": "wam_occlusion_runtime_activation_v1",
        "config": {
            name: getattr(occlusion_config, name)
            for name in occlusion_config.__dataclass_fields__
        },
    }
    occlusion_entry, _ = _write_artifact(
        tmp_path, "occlusion_runtime", occlusion_payload,
    )
    projection = CameraProjectionProfile.create(
        camera_id="CL8326300A0",
        camera_profile_sha256=camera_profile["sha256"],
        width=1920,
        height=1080,
        fx=1200.0,
        fy=1201.0,
        cx=959.5,
        cy=539.5,
        min_projection_depth_m=0.05,
    )
    runtime_projection = ActivatedCameraProjection.create(
        camera_id="CL8326300A0",
        camera_profile_sha256=camera_profile["sha256"],
        width=1920,
        height=1080,
        fx=1200.0,
        fy=1201.0,
        cx=959.5,
        cy=539.5,
        min_projection_depth_m=0.05,
    )
    assert projection.camera_projection_sha256 == (
        runtime_projection.camera_projection_sha256
    )
    runtime_profile = ActivatedOcclusionRuntimeProfile.create(
        config=occlusion_config,
        patient_motion_artifact_sha256=patient_pose_entry["sha256"],
        handeye_artifact_sha256=handeye.handeye_sha256,
        camera_latency_artifact_sha256=latency["sha256"],
        camera_projection_artifact_sha256=(
            projection.camera_projection_sha256
        ),
        tool_geometry_artifact_sha256=tool.tool_sha256,
    )
    pair_release, pair_contract_sha256 = _write_pair_release(
        tmp_path / "pairs",
        camera_profile_sha256=camera_profile["sha256"],
        timing_sha256=latency["sha256"],
    )
    identity_values = {
        "camera_id": "CL8326300A0",
        "camera_profile_sha256": camera_profile["sha256"],
        "geometry_release_id": geometry.geometry_release_id,
        "geometry_sha256": geometry.geometry_sha256,
        "tool_release_id": tool.tool_release_id,
        "tool_sha256": tool.tool_sha256,
        "handeye_release_id": handeye.handeye_release_id,
        "handeye_sha256": handeye.handeye_sha256,
        "handeye_capability_schema": handeye.capability_schema,
        "handeye_calibration_tier": handeye.calibration_tier,
        "handeye_policy_sha256": handeye.policy_sha256,
        "camera_latency_sha256": latency["sha256"],
        "camera_projection_sha256": projection.camera_projection_sha256,
        "occlusion_profile_sha256": uncertainty.occlusion_profile_sha256,
        "occlusion_runtime_profile_sha256": (
            runtime_profile.occlusion_runtime_profile_sha256
        ),
        "alignment_sha256": alignment["sha256"],
        "joint_mapper_sha256": SHA_B,
        "pair_contract_sha256": pair_contract_sha256,
        "controller_binary_sha256": SHA_C,
        "patient_pose_release_id": patient_pose_release_id,
        "patient_pose_sha256": patient_pose_entry["sha256"],
        "depth_noise_model_sha256": depth_profile.semantic_sha256,
        "back_mask_release_id": back_mask_config.release_id,
        "back_mask_sha256": back_mask_config.semantic_sha256,
        "max_camera_to_exposure_age_ns": 24_000_000,
        "max_exposure_interval_ns": 40_000_000,
    }
    ActivatedOcclusionIdentity(**identity_values)
    identity_payload = {
        "schema": "wam_activated_occlusion_identity_v2",
        **identity_values,
    }
    identity_entry, identity_complete = _write_artifact(
        tmp_path, "occlusion_identity", identity_payload,
    )
    artifacts = {
        "alignment": alignment,
        "back_mask": back_mask_entry,
        "camera_latency": latency,
        "camera_profile": camera_profile,
        "collision_geometry": geometry_entry,
        "depth_noise": depth_entry,
        "handeye": handeye_entry,
        "occlusion_identity": identity_entry,
        "occlusion_runtime": occlusion_entry,
        "patient_pose": patient_pose_entry,
        "robot_mask_uncertainty": uncertainty_entry,
        "tool_geometry": tool_entry,
    }
    manifest = {
        "artifacts": artifacts,
        "follower_can_interface": "can3",
        "leader_can_interface": "can1",
        "openarm_pair": {
            "contract_sha256": pair_contract_sha256,
            "release_directory": str(pair_release),
        },
        "robot_state_bounds": {
            "buffer_duration_ns": 2_000_000_000,
            "gripper_motor_position_limits_rad": [-0.20, 0.20],
            "joint_position_limits_rad": [[-3.0, 3.0]] * 7,
            "max_abs_joint_velocity_rad_s": 8.0,
            "max_exposure_interval_ns": 40_000_000,
            "max_gap_ns": 20_000_000,
            "max_packets": 256,
            "max_staleness_ns": 50_000_000,
        },
        "schema": "wam_production_runtime_activation_v1",
        "side": "left",
        "status": "COMPLETE",
    }
    activation_dir = tmp_path / "activation"
    activation_dir.mkdir()
    activation = activation_dir / "activation.json"
    raw = _canonical(manifest) + b"\n"
    activation.write_bytes(raw)
    activation_sha256 = hashlib.sha256(raw).hexdigest()
    top_complete = {
        "activation_sha256": activation_sha256,
        "schema": "wam_production_runtime_activation_completion_v1",
        "status": "COMPLETE",
    }
    (activation_dir / "COMPLETE").write_bytes(_canonical(top_complete) + b"\n")
    return {
        "activation": activation,
        "activation_sha256": activation_sha256,
        "capture_mode": capture_mode,
        "completion_paths": {
            "camera_latency": latency_complete,
            "depth_noise": depth_complete,
            "handeye": handeye_complete,
            "occlusion_identity": identity_complete,
        },
    }


def _load(fixture: dict):
    return load_production_runtime_assembly(
        fixture["activation"],
        expected_activation_sha256=fixture["activation_sha256"],
        point_names=POINT_NAMES,
        expected_camera_id="CL8326300A0",
        expected_capture_mode=fixture["capture_mode"],
    )


def test_loads_complete_left_runtime_into_real_validated_components(
    tmp_path: Path,
) -> None:
    fixture = _activation_fixture(tmp_path)

    runtime = _load(fixture)

    assert isinstance(runtime.robot_capture_coordinator, ProductionRobotCaptureCoordinator)
    assert isinstance(runtime.robot_mask_provider, ProductionRobotMaskProvider)
    assert isinstance(
        runtime.back_mask_provider, IndependentPosteriorBackMaskProvider
    )
    assert isinstance(runtime.patient_pose_provider, MetricPatientPoseProvider)
    assert isinstance(runtime.occlusion_runtime, OcclusionRuntime)
    identity = runtime.occlusion_runtime.activated_identity
    handeye = runtime.robot_mask_provider.handeye
    assert runtime.handeye is handeye
    assert identity.handeye_capability_schema == handeye.capability_schema
    assert identity.handeye_calibration_tier == handeye.calibration_tier
    assert identity.handeye_policy_sha256 == handeye.policy_sha256
    contract = runtime.robot_capture_coordinator.contract
    assert (contract.side, contract.leader_can_interface, contract.follower_can_interface) == (
        "left", "can1", "can3",
    )
    assert contract.expected_arm_rx_motor_ids == tuple(range(17, 24))
    assert contract.expected_gripper_rx_motor_id == 24
    assert runtime.occlusion_runtime.point_names == POINT_NAMES
    assert runtime.activation_sha256 == fixture["activation_sha256"]
    assert runtime.robot_capture_coordinator.receiver_health().status == "STOPPED"
    assert callable(runtime.prepare_robot_receiver)
    assert callable(runtime.attach_and_start_robot_receiver)


def test_legacy_identity_v1_derives_policy_only_for_trusted_legacy_handeye(
    tmp_path: Path,
) -> None:
    runtime = _load(_activation_fixture(tmp_path / "runtime"))
    identity = runtime.occlusion_runtime.activated_identity
    legacy_handeye = runtime.robot_mask_provider.handeye
    omitted = {
        "handeye_capability_schema",
        "handeye_calibration_tier",
        "handeye_policy_sha256",
    }
    payload = {
        "schema": "wam_activated_occlusion_identity_v1",
        **{
            field.name: getattr(identity, field.name)
            for field in fields(ActivatedOcclusionIdentity)
            if field.name not in omitted
        },
    }

    replayed = _build_identity(payload, handeye=legacy_handeye)

    assert replayed.handeye_capability_schema == legacy_handeye.capability_schema
    assert replayed.handeye_calibration_tier == legacy_handeye.calibration_tier
    assert replayed.handeye_policy_sha256 == legacy_handeye.policy_sha256


def test_provisional_handeye_requires_policy_bound_identity_v2(
    tmp_path: Path,
) -> None:
    runtime = _load(_activation_fixture(tmp_path / "runtime"))
    identity = runtime.occlusion_runtime.activated_identity
    omitted = {
        "handeye_capability_schema",
        "handeye_calibration_tier",
        "handeye_policy_sha256",
    }
    legacy_payload = {
        "schema": "wam_activated_occlusion_identity_v1",
        **{
            field.name: getattr(identity, field.name)
            for field in fields(ActivatedOcclusionIdentity)
            if field.name not in omitted
        },
    }
    entry, release = _write_provisional_handeye_release(tmp_path / "provisional")
    try:
        payload, artifact_sha256 = _load_artifact("handeye", entry)
        provisional = _build_handeye(
            payload, artifact_sha256=artifact_sha256,
        )

        with pytest.raises(
            ProductionRuntimeActivationError,
            match="policy-bound occlusion identity v2",
        ):
            _build_identity(legacy_payload, handeye=provisional)
    finally:
        _make_provisional_release_writable(release)


@pytest.mark.parametrize(
    "artifact_name",
    ("handeye", "depth_noise", "camera_latency", "occlusion_identity"),
)
def test_missing_physical_complete_marker_blocks_before_runtime_is_returned(
    tmp_path: Path, artifact_name: str,
) -> None:
    fixture = _activation_fixture(tmp_path)
    fixture["completion_paths"][artifact_name].unlink()

    with pytest.raises(
        ProductionRuntimeActivationError,
        match=rf"{artifact_name}.*COMPLETE",
    ):
        _load(fixture)


def test_manifest_and_identity_bytes_are_independently_hash_pinned(
    tmp_path: Path,
) -> None:
    fixture = _activation_fixture(tmp_path)
    manifest = json.loads(fixture["activation"].read_text())
    identity = Path(manifest["artifacts"]["occlusion_identity"]["path"])
    identity.write_bytes(identity.read_bytes() + b"\n")

    with pytest.raises(
        ProductionRuntimeActivationError,
        match=r"occlusion_identity.*SHA-256",
    ):
        _load(fixture)


def test_staging_marker_blocks_even_with_a_matching_external_pin(
    tmp_path: Path,
) -> None:
    fixture = _activation_fixture(tmp_path)
    (fixture["activation"].parent / "STAGING").write_text("incomplete")

    with pytest.raises(ProductionRuntimeActivationError, match="STAGING"):
        _load(fixture)


def test_top_complete_marker_is_read_once_through_the_stable_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _activation_fixture(tmp_path)
    original = Path.read_bytes

    def reject_unstable_complete_read(path: Path) -> bytes:
        if path == fixture["activation"].parent / "COMPLETE":
            raise AssertionError("top COMPLETE must not use Path.read_bytes")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unstable_complete_read)

    runtime = _load(fixture)

    assert runtime.activation_sha256 == fixture["activation_sha256"]


def test_loads_issuer_v2_as_provisional_robot_mask_capability(tmp_path: Path) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        payload, artifact_sha256 = _load_artifact("handeye", entry)
        handeye = _build_handeye(payload, artifact_sha256=artifact_sha256)

        assert isinstance(handeye, ProvisionalRobotMaskHandEye)
        assert handeye.lifecycle_status == "COMPLETE"
        assert handeye.calibration_status == "USER_ACCEPTED_BELOW_GATE"
        assert handeye.handeye_sha256 == artifact_sha256
        assert handeye.handeye_release_id == release.name
        assert handeye.autonomous_targeting_authorized is False
        assert handeye.workspace_1sigma_max == pytest.approx(0.00605)
    finally:
        _make_provisional_release_writable(release)


def test_public_loader_verifies_and_loads_completed_provisional_handeye(
    tmp_path: Path,
) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        handeye = load_completed_handeye_release(
            release,
            expected_artifact_sha256=entry["sha256"],
            expected_complete_sha256=entry["complete_sha256"],
            expected_camera_id="CL8326300A0",
        )

        assert isinstance(handeye, ProvisionalRobotMaskHandEye)
        assert handeye.handeye_release_id == release.name
        assert handeye.handeye_sha256 == entry["sha256"]
        assert handeye.autonomous_targeting_authorized is False
    finally:
        _make_provisional_release_writable(release)


def test_public_handeye_loader_rejects_wrong_camera_and_marker_pin(
    tmp_path: Path,
) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        with pytest.raises(ProductionRuntimeActivationError, match="camera"):
            load_completed_handeye_release(
                release,
                expected_artifact_sha256=entry["sha256"],
                expected_complete_sha256=entry["complete_sha256"],
                expected_camera_id="WRONG-CAMERA",
            )
        with pytest.raises(ProductionRuntimeActivationError, match="COMPLETE.*SHA-256"):
            load_completed_handeye_release(
                release,
                expected_artifact_sha256=entry["sha256"],
                expected_complete_sha256="0" * 64,
                expected_camera_id="CL8326300A0",
            )
    finally:
        _make_provisional_release_writable(release)


def test_v2_cannot_be_rewrapped_with_generic_complete_marker(tmp_path: Path) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        complete = Path(entry["complete_path"])
        complete.chmod(0o644)
        marker = {
            "artifact_sha256": entry["sha256"],
            "schema": "wam_production_runtime_artifact_completion_v1",
            "status": "COMPLETE",
        }
        raw = _canonical(marker) + b"\n"
        complete.write_bytes(raw)
        complete.chmod(0o444)
        entry["complete_sha256"] = hashlib.sha256(raw).hexdigest()

        with pytest.raises(
            ProductionRuntimeActivationError, match="provisional|handeye.*COMPLETE",
        ):
            _load_artifact("handeye", entry)
    finally:
        _make_provisional_release_writable(release)


def test_v2_requires_release_complete_and_read_only_modes(tmp_path: Path) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        release.chmod(0o755)
        (release / "COMPLETE").unlink()
        release.chmod(0o555)

        with pytest.raises(ProductionRuntimeActivationError, match="release COMPLETE"):
            _load_artifact("handeye", entry)
    finally:
        _make_provisional_release_writable(release)

    entry, release = _write_provisional_handeye_release(tmp_path / "second")
    try:
        release.chmod(0o755)
        with pytest.raises(ProductionRuntimeActivationError, match="read-only|mode"):
            _load_artifact("handeye", entry)
    finally:
        _make_provisional_release_writable(release)


def test_three_mm_runtime_bound_cannot_mislabel_provisional_five_mm_handeye(
    tmp_path: Path,
) -> None:
    entry, release = _write_provisional_handeye_release(tmp_path)
    try:
        payload, artifact_sha256 = _load_artifact("handeye", entry)
        handeye = _build_handeye(payload, artifact_sha256=artifact_sha256)

        with pytest.raises(
            ProductionRuntimeActivationError, match="hand-eye bound.*6.5 mm",
        ):
            _validate_handeye_runtime_bound(handeye, OcclusionRuntimeConfig())

        _validate_handeye_runtime_bound(
            handeye,
            OcclusionRuntimeConfig(
                handeye_bound_m=0.0065,
                timing_sweep_bound_m=0.0015,
            ),
        )
    finally:
        _make_provisional_release_writable(release)


def test_activation_canonical_bytes_agree_across_deploy_and_runtime() -> None:
    """Cross-layer regression guard (reviewer P0): a single activation.json /
    COMPLETE file must satisfy BOTH verifiers at once.

    The deploy preflight (collection_service_unit._stable_canonical_object) accepts
    exactly ``artifact_contract.canonical_json_bytes(value)`` (canonical + one
    trailing newline). The runtime loader accepts exactly
    ``_canonical_json_bytes(value) + b"\n"``. Before the fix these disagreed on the
    newline, so no bundle could pass both. This proves the byte forms are now
    identical for representative activation/COMPLETE payloads.
    """
    import sys as _sys

    _sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")
    )
    from artifact_contract import canonical_json_bytes  # noqa: E402
    from ausc.perception.production_runtime_activation import (  # noqa: E402
        _canonical_json_bytes,
    )

    samples = [
        {
            "schema": "wam_production_runtime_activation_v1",
            "status": "COMPLETE",
            "side": "left",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "artifacts": {"handeye": {"path": "a", "sha256": "0" * 64}},
        },
        {
            "activation_sha256": "0" * 64,
            "schema": "wam_production_runtime_activation_completion_v1",
            "status": "COMPLETE",
        },
        {
            "artifact_sha256": "f" * 64,
            "schema": "wam_production_runtime_artifact_completion_v1",
            "status": "COMPLETE",
        },
    ]
    for value in samples:
        deploy_bytes = canonical_json_bytes(value)
        runtime_bytes = _canonical_json_bytes(value) + b"\n"
        assert deploy_bytes == runtime_bytes, value
