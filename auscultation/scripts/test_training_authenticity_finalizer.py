from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest


import training_authenticity_finalizer as finalizer
from ausc.audio.audio_stream import AudioSignalAuthenticityAccumulator
from ausc.perception.robot.robot_mask_provider import (
    ActivatedHandEye,
    ProvisionalRobotMaskHandEye,
)
from ausc.perception.robot.robot_state_stream import RobotStateContract
from ausc.perception.tracking.detector import LUNG_ORDER


SHA_A = "a" * 64
SHA_B = "b" * 64


def _legacy_handeye() -> ActivatedHandEye:
    return ActivatedHandEye.create(
        camera_id="CL8326300A0",
        side="left",
        transform_camera_from_follower_base=np.eye(4, dtype=np.float64),
        covariance_se3=np.eye(6, dtype=np.float64) * 1.0e-6,
        status="COMPLETE",
    )


def _provisional_handeye_payload() -> dict[str, object]:
    follower_base_from_camera = np.eye(4, dtype=np.float64)
    follower_base_from_camera[:3, 3] = [0.30, -0.20, 0.70]
    camera_from_follower_base = np.linalg.inv(follower_base_from_camera)
    covariance = np.diag(
        (1.0e-6, 1.1e-6, 1.2e-6, 4.0e-6, 5.0e-6, 6.0e-6)
    )
    return {
        "acceptance_note": "William accepted approximately 5 mm for bootstrap use.",
        "authorized_scopes": [
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ],
        "autonomous_targeting_authorized": False,
        "bootstrap_reproducibility": {
            "missing": [
                "bootstrap_seed",
                "bootstrap_sample_count",
                "resample_policy",
            ],
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


def _provisional_handeye() -> tuple[ProvisionalRobotMaskHandEye, bytes]:
    payload = _provisional_handeye_payload()
    raw = finalizer.canonical_json_bytes(payload)
    capability = ProvisionalRobotMaskHandEye.from_activation_v2(
        payload,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        publication_status="COMPLETE",
    )
    return capability, raw


def _manifest(*, contract: str = "wam_promotion_v2") -> dict[str, object]:
    return {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": contract,
        "session_id": "session-test-001",
        "status": "sensors_finalized",
        "training_eligible": False,
        "rehearsal_overrides": [],
    }


def _identity() -> dict[str, str]:
    return {
        "code_release_id": f"code-release-{SHA_A[:20]}",
        "code_manifest_sha256": SHA_A,
        "environment_id": f"linbox-env-{SHA_B[:20]}",
        "environment_lock_sha256": SHA_B,
        "module_sha256": finalizer.module_sha256(),
    }


def _robot_contract() -> RobotStateContract:
    return RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=tuple(
            f"openarm_left_joint{index}" for index in range(1, 8)
        ),
        joint_position_limits_rad=tuple((-4.0, 4.0) for _ in range(7)),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256="1" * 64,
        joint_mapper_sha256="2" * 64,
        camera_profile_sha256="3" * 64,
        timing_artifact_sha256="4" * 64,
        pair_contract_sha256="5" * 64,
        controller_binary_sha256="6" * 64,
        controller_source_manifest_sha256="7" * 64,
        follower_config_sha256="8" * 64,
        openarm_sdk_sha256="9" * 64,
        max_abs_joint_velocity_rad_s=8.0,
        gripper_motor_position_limits_rad=(-0.2, 0.2),
        buffer_duration_ns=2_000_000_000,
        max_packets=256,
        max_gap_ns=20_000_000,
        max_staleness_ns=50_000_000,
        max_exposure_interval_ns=40_000_000,
    )


def _activated_contracts(
    handeye: ActivatedHandEye | ProvisionalRobotMaskHandEye,
    *,
    artifact_sha256: str | None = None,
) -> finalizer.ActivatedCaptureContracts:
    source_sha256 = artifact_sha256 or handeye.handeye_sha256
    return finalizer.ActivatedCaptureContracts(
        robot_contract=_robot_contract(),
        depth_scale_mm=1.0,
        activation_path="/immutable/hardware_activation/activation.json",
        activation_sha256="d" * 64,
        camera_id="CL8326300A0",
        point_names=tuple(LUNG_ORDER),
        handeye=finalizer._binding_from_runtime_handeye(
            handeye,
            artifact_sha256=source_sha256,
        ),
    )


def _write_tracker_handeye_rows(
    root: Path,
    handeye: finalizer.ActivatedHandEyeBinding,
    *,
    mutations: dict[int, dict[str, object]] | None = None,
) -> None:
    (root / "rgb_times.csv").write_text(
        "frame_index,host_arrival_t_s,device_t_ms,time_source\n"
        "10,1.0,1000.0,device\n"
        "11,1.1,1100.0,device\n",
        encoding="utf-8",
    )
    rows = []
    for frame_index in (10, 11):
        robot_mask = handeye.frame_evidence_policy()
        robot_mask.update({
            "frame_index": frame_index,
            "handeye_release_id": handeye.release_id,
            "handeye_sha256": handeye.capability_sha256,
            "mask_sha256": f"{frame_index:064x}",
        })
        if mutations and frame_index in mutations:
            robot_mask.update(mutations[frame_index])
        canonical_robot_mask = {
            key: (
                {"float64_hex": value.hex()}
                if type(value) is float else value
            )
            for key, value in robot_mask.items()
        }
        canonical_metadata = {
            "camera_exposure": {"frame_index": frame_index},
            "camera_frame": {"index": frame_index},
            "robot_mask": canonical_robot_mask,
            "schema": "wam_frame_evidence_bundle_v3",
        }
        canonical_raw = json.dumps(
            canonical_metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        canonical_sha256 = hashlib.sha256(canonical_raw).hexdigest()
        evidence_id = "frame_evidence-" + canonical_sha256
        rows.append({
            "frame_evidence": {
                "bundle_id": evidence_id,
                "canonical_json": canonical_raw.decode("ascii"),
                "canonical_sha256": canonical_sha256,
                "schema": "wam_frame_evidence_ledger_record_v1",
            },
            "frame_evidence_bundle_id": evidence_id,
            "frame_index": frame_index,
            "robot_mask": robot_mask,
        })
    (root / "tracker.jsonl").write_bytes(
        b"".join(finalizer.canonical_json_bytes(row) for row in rows)
    )


def _manifest_with_runtime_activation(root: Path) -> tuple[dict[str, object], Path]:
    activation = root / "hardware_activation" / "activation.json"
    activation.parent.mkdir()
    activation.write_bytes(finalizer.canonical_json_bytes({
        "artifacts": {
            "handeye": {
                "complete_path": "/immutable/handeye.COMPLETE",
                "complete_sha256": "c" * 64,
                "path": "/immutable/handeye.json",
                "sha256": "e" * 64,
            },
        },
    }))
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
    manifest = _manifest()
    manifest.update({
        "hardware_activation": {
            "path": "hardware_activation/activation.json",
            "sha256": hashlib.sha256(activation.read_bytes()).hexdigest(),
        },
        "camera": {
            "camera_id": "CL8326300A0",
            "timing": {"capture_mode": capture_mode},
        },
        "point_names": list(LUNG_ORDER),
    })
    return manifest, activation


def _final_metadata_paths(session: Path) -> list[Path]:
    return [
        session / "manifest.json",
        session / "FINALIZATION.json",
        session / "TRAINING_AUTHENTICITY.json",
        session / "authenticity",
    ]


def test_old_promotion_v1_is_not_ready_and_writes_no_pass_metadata(tmp_path: Path) -> None:
    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(contract="wam_promotion_v1"),
        finalizer_identity=_identity(),
    )

    assert report["status"] == "NOT_READY"
    assert report["training_eligible"] is False
    assert "unsupported_promotion_contract" in report["blockers"]
    with pytest.raises(finalizer.TrainingFinalizationNotReady):
        finalizer.finalize_training_session(
            tmp_path,
            manifest_template=_manifest(contract="wam_promotion_v1"),
            finalizer_identity=_identity(),
        )
    assert not any(path.exists() for path in _final_metadata_paths(tmp_path))


def test_current_capture_depth_timing_layout_is_an_explicit_blocker(tmp_path: Path) -> None:
    (tmp_path / "depth_times.csv").write_text("frame_index,host_t,path\n", encoding="utf-8")

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(),
        finalizer_identity=_identity(),
    )

    assert "depth_timing_path_contract_mismatch" in report["blockers"]
    assert "required_raw_artifacts_missing" in report["blockers"]


def test_lossless_geometry_and_patient_depth_evidence_are_required(tmp_path: Path) -> None:
    for relative in finalizer.REQUIRED_BASE_RAW_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder\n")
    (tmp_path / "depth" / "depth_000000.npy").write_bytes(b"placeholder\n")

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(),
        finalizer_identity=_identity(),
    )

    blockers = set(report["blockers"])
    assert "geometry_lossless_model_inputs_missing" in blockers
    assert "geometry_patient_frame_evidence_missing" in blockers
    assert "geometry_depth_patch_evidence_missing" in blockers
    assert "quality_lossless_model_inputs_missing" in blockers


def test_finalizer_identity_must_content_address_release_and_own_module(tmp_path: Path) -> None:
    identity = _identity()
    identity["code_release_id"] = "code-release-00000000000000000000"
    identity["module_sha256"] = "0" * 64

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(),
        finalizer_identity=identity,
    )

    assert "finalizer_identity_invalid" in report["blockers"]
    assert report["checks"]["finalizer_identity"]["status"] == "FAIL"


def test_audio_component_recomputes_wav_authenticity_and_rejects_self_report(
    tmp_path: Path,
) -> None:
    sample_rate = 16_000
    samples = ((np.arange(sample_rate, dtype=np.int32) % 31) - 15).astype("<i2")
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    analyzer.update(samples)
    receipt = analyzer.finish()
    metadata = {
        "schema_version": "wam_audio_stream_capture_v1",
        "status": "sensors_finalized",
        "training_eligible": False,
        "sample_rate": sample_rate,
        "input_sample_rate": 48_000,
        "partial_sample_rate": 48_000,
        "partial_encoding": "signed_pcm_s16le_mono",
        "t0_monotonic": 10.0,
        "audio_t0_monotonic": 10.0,
        "audio_t_end_monotonic": 11.0,
        "partial_n_samples": 48_000,
        "n_samples": sample_rate,
        "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        "signal_authenticity": receipt,
    }
    (tmp_path / "audio_capture.json").write_bytes(
        finalizer.canonical_json_bytes(metadata)
    )
    (tmp_path / "audio_input_times.jsonl").write_bytes(
        finalizer.canonical_json_bytes({
            "input_sample_offset": 0,
            "input_sample_count": 48_000,
            "t_start": 10.0,
            "t_end": 11.0,
            "source_block_start_t": 10.0,
            "source_block_end_t": 11.0,
        })
    )
    (tmp_path / "audio_times.jsonl").write_bytes(
        finalizer.canonical_json_bytes({
            "sample_offset": 0,
            "sample_count": sample_rate // 2,
            "t_start": 10.0,
            "t_end": 10.5,
            "source_input_sample_count": 24_000,
            "input_sample_rate": 48_000,
            "sample_rate": sample_rate,
        })
        + finalizer.canonical_json_bytes({
            "sample_offset": sample_rate // 2,
            "sample_count": sample_rate // 2,
            "t_start": 10.5,
            "t_end": 11.0,
            "source_input_sample_count": 24_000,
            "input_sample_rate": 48_000,
            "sample_rate": sample_rate,
        })
    )

    evidence = finalizer.verify_audio_component(tmp_path)
    assert evidence["signal_authenticity"] == receipt
    assert evidence["n_samples"] == sample_rate
    assert evidence["input_timeline_rows"] == 1
    assert set(evidence["timeline_sha256"]) == {
        "audio_input_times.jsonl", "audio_times.jsonl",
    }

    metadata["signal_authenticity"] = dict(receipt)
    metadata["signal_authenticity"]["evidence"] = dict(receipt["evidence"])
    metadata["signal_authenticity"]["evidence"]["rms_pcm"] = 123.0
    (tmp_path / "audio_capture.json").write_bytes(
        finalizer.canonical_json_bytes(metadata)
    )
    with pytest.raises(finalizer.TrainingFinalizationError, match="audio authenticity"):
        finalizer.verify_audio_component(tmp_path)


def test_audio_component_rejects_missing_or_discontinuous_timelines(tmp_path: Path) -> None:
    sample_rate = 16_000
    samples = ((np.arange(sample_rate, dtype=np.int32) % 29) - 14).astype("<i2")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    analyzer.update(samples)
    metadata = {
        "schema_version": "wam_audio_stream_capture_v1",
        "status": "sensors_finalized",
        "training_eligible": False,
        "sample_rate": sample_rate,
        "input_sample_rate": 48_000,
        "partial_sample_rate": 48_000,
        "partial_encoding": "signed_pcm_s16le_mono",
        "t0_monotonic": 1.0,
        "audio_t0_monotonic": 1.0,
        "audio_t_end_monotonic": 2.0,
        "partial_n_samples": 48_000,
        "n_samples": sample_rate,
        "sha256": hashlib.sha256((tmp_path / "audio.wav").read_bytes()).hexdigest(),
        "signal_authenticity": analyzer.finish(),
    }
    (tmp_path / "audio_capture.json").write_bytes(finalizer.canonical_json_bytes(metadata))

    with pytest.raises(finalizer.TrainingFinalizationError, match="audio input timeline"):
        finalizer.verify_audio_component(tmp_path)

    (tmp_path / "audio_input_times.jsonl").write_bytes(
        finalizer.canonical_json_bytes({
            "input_sample_offset": 1,
            "input_sample_count": 48_000,
            "t_start": 1.0,
            "t_end": 2.0,
            "source_block_start_t": 1.0,
            "source_block_end_t": 2.0,
        })
    )
    (tmp_path / "audio_times.jsonl").write_bytes(b"{}\n")
    with pytest.raises(finalizer.TrainingFinalizationError, match="audio input timeline"):
        finalizer.verify_audio_component(tmp_path)


def test_readiness_requires_physical_calibration_and_activation_artifacts(
    tmp_path: Path,
) -> None:
    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(),
        finalizer_identity=_identity(),
    )

    blockers = set(report["blockers"])
    assert "physical_audio_force_sync_artifact_missing" in blockers
    assert "camera_led_latency_artifact_missing" in blockers
    assert "handeye_robot_base_artifact_missing" in blockers
    assert "hardware_activation_artifact_missing" in blockers
    assert "model_activation_artifacts_missing" in blockers
    assert report["checks"]["production_artifact_bindings"]["status"] == "FAIL"


def test_provisional_handeye_cannot_pass_without_trusted_runtime_binding(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": "wam_openarm_handeye_activation_v2",
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "provisional": True,
        "quality_gate_passed": False,
        "failed_quality_gate_m": 0.003,
        "authorized_scopes": [
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ],
        "autonomous_targeting_authorized": False,
        "source_frame": "orbbec_color_optical_frame",
        "target_frame": "openarm_left_link0",
        "covariance_scope": "random_only",
        "covariance_se3": {
            "bound_transform": "transform_follower_base_from_camera",
        },
        "workspace_1sigma_m": {"median": 0.00496, "max": 0.00605},
        "systematic_bias_status": "NOT_CHARACTERIZED",
        "systematic_bias_bound_m": None,
        "tip_offset_frame": "openarm_left_hand_tcp",
    }
    calibration = tmp_path / "calibration" / "handeye.json"
    calibration.parent.mkdir()
    calibration.write_bytes(finalizer.canonical_json_bytes(payload))
    manifest = _manifest()
    manifest["handeye"] = {
        "calibration_path": "calibration/handeye.json",
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "capability_sha256": "c" * 64,
        "policy_sha256": "d" * 64,
        "release_id": "handeye_activation-" + "c" * 20,
    }

    blockers, checks = finalizer._production_artifact_binding_checks(
        tmp_path, manifest,
    )

    assert "handeye_robot_base_artifact_missing" in blockers
    assert checks["handeye"]["status"] == "FAIL"
    assert "trusted hardware activation" in checks["handeye"]["reason"]

    payload["autonomous_targeting_authorized"] = True
    calibration.write_bytes(finalizer.canonical_json_bytes(payload))
    manifest["handeye"]["calibration_sha256"] = hashlib.sha256(
        calibration.read_bytes()
    ).hexdigest()
    blockers, _checks = finalizer._production_artifact_binding_checks(
        tmp_path, manifest,
    )
    assert "handeye_robot_base_artifact_missing" in blockers


def test_provisional_handeye_passes_only_when_exact_runtime_identity_policy_and_frames_bind(
    tmp_path: Path,
) -> None:
    capability, raw = _provisional_handeye()
    calibration = tmp_path / "calibration" / "handeye.json"
    calibration.parent.mkdir()
    calibration.write_bytes(raw)
    activated = _activated_contracts(capability)
    manifest = _manifest()
    manifest["handeye"] = {
        "calibration_path": "calibration/handeye.json",
        "calibration_sha256": hashlib.sha256(raw).hexdigest(),
        "capability_sha256": capability.handeye_sha256,
        "policy_sha256": capability.policy_sha256,
        "release_id": capability.handeye_release_id,
    }
    _write_tracker_handeye_rows(tmp_path, activated.handeye)

    blockers, checks = finalizer._production_artifact_binding_checks(
        tmp_path,
        manifest,
        activated_contracts=activated,
    )

    assert "handeye_robot_base_artifact_missing" not in blockers
    assert "handeye_frame_evidence_invalid" not in blockers
    assert checks["handeye"]["status"] == "PASS_PROVISIONAL"
    assert checks["handeye"]["release_id"] == capability.handeye_release_id
    assert checks["handeye"]["artifact_sha256"] == capability.handeye_sha256
    assert checks["handeye"]["policy_sha256"] == capability.policy_sha256
    assert checks["handeye_frame_evidence"] == {
        "status": "PASS",
        "frame_count": 2,
        "first_frame_index": 10,
        "last_frame_index": 11,
        "release_id": capability.handeye_release_id,
        "capability_sha256": capability.handeye_sha256,
        "policy_sha256": capability.policy_sha256,
    }


def test_legacy_runtime_cannot_authorize_an_unrelated_provisional_artifact(
    tmp_path: Path,
) -> None:
    provisional, raw = _provisional_handeye()
    calibration = tmp_path / "calibration" / "handeye.json"
    calibration.parent.mkdir()
    calibration.write_bytes(raw)
    legacy = _legacy_handeye()
    activated = _activated_contracts(legacy, artifact_sha256="e" * 64)
    manifest = _manifest()
    manifest["handeye"] = {
        "calibration_path": "calibration/handeye.json",
        "calibration_sha256": hashlib.sha256(raw).hexdigest(),
        "capability_sha256": provisional.handeye_sha256,
        "policy_sha256": provisional.policy_sha256,
        "release_id": provisional.handeye_release_id,
    }

    blockers, checks = finalizer._production_artifact_binding_checks(
        tmp_path,
        manifest,
        activated_contracts=activated,
    )

    assert "handeye_robot_base_artifact_missing" in blockers
    assert checks["handeye"]["status"] == "FAIL"
    assert "trusted hardware activation" in checks["handeye"]["reason"]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("handeye_release_id", "handeye_activation-" + "0" * 20),
        ("handeye_sha256", "0" * 64),
        ("handeye_capability_schema", "wam_openarm_handeye_v1"),
        ("handeye_calibration_tier", "QUALITY_GATE_COMPLETE"),
        ("handeye_policy_sha256", "0" * 64),
        ("handeye_calibration_status", "COMPLETE"),
        ("handeye_provisional", False),
        ("handeye_quality_gate_passed", True),
        ("handeye_workspace_1sigma_max_m", 0.003),
        ("handeye_covariance_bound_transform", "wrong_transform"),
        ("handeye_covariance_scope", "legacy_production"),
        ("handeye_systematic_bias_status", "CHARACTERIZED"),
        ("handeye_systematic_bias_bound_m", 0.001),
        ("handeye_autonomous_targeting_authorized", True),
        ("handeye_provisional_policy_allowance_m", 0.0),
    ],
)
def test_raw_tracker_revalidates_every_provisional_handeye_policy_field(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    capability, _raw = _provisional_handeye()
    activated = _activated_contracts(capability)
    _write_tracker_handeye_rows(
        tmp_path,
        activated.handeye,
        mutations={11: {field: bad_value}},
    )

    with pytest.raises(
        finalizer.TrainingFinalizationError,
        match="tracker hand-eye evidence",
    ):
        finalizer.verify_recorded_handeye_frame_evidence(
            tmp_path,
            activated.handeye,
        )


def test_raw_tracker_requires_replayable_frame_ledger_for_every_rgb_frame(
    tmp_path: Path,
) -> None:
    capability, _raw = _provisional_handeye()
    activated = _activated_contracts(capability)
    _write_tracker_handeye_rows(tmp_path, activated.handeye)
    rows = finalizer._read_jsonl(
        tmp_path / "tracker.jsonl", label="test tracker"
    )
    rows[1].pop("frame_evidence")
    (tmp_path / "tracker.jsonl").write_bytes(
        b"".join(finalizer.canonical_json_bytes(row) for row in rows)
    )

    with pytest.raises(
        finalizer.TrainingFinalizationError,
        match="tracker hand-eye evidence",
    ):
        finalizer.verify_recorded_handeye_frame_evidence(
            tmp_path,
            activated.handeye,
        )


def test_raw_tracker_derived_handeye_metadata_must_match_replayed_ledger(
    tmp_path: Path,
) -> None:
    capability, _raw = _provisional_handeye()
    activated = _activated_contracts(capability)
    _write_tracker_handeye_rows(tmp_path, activated.handeye)
    rows = finalizer._read_jsonl(
        tmp_path / "tracker.jsonl", label="test tracker"
    )
    rows[1]["robot_mask"]["handeye_policy_sha256"] = "0" * 64
    (tmp_path / "tracker.jsonl").write_bytes(
        b"".join(finalizer.canonical_json_bytes(row) for row in rows)
    )

    with pytest.raises(
        finalizer.TrainingFinalizationError,
        match="derived robot-mask metadata",
    ):
        finalizer.verify_recorded_handeye_frame_evidence(
            tmp_path,
            activated.handeye,
        )


def test_raw_tracker_ledger_digest_must_match_exact_canonical_bytes(
    tmp_path: Path,
) -> None:
    capability, _raw = _provisional_handeye()
    activated = _activated_contracts(capability)
    _write_tracker_handeye_rows(tmp_path, activated.handeye)
    rows = finalizer._read_jsonl(
        tmp_path / "tracker.jsonl", label="test tracker"
    )
    canonical = json.loads(rows[1]["frame_evidence"]["canonical_json"])
    canonical["tampered"] = True
    rows[1]["frame_evidence"]["canonical_json"] = json.dumps(
        canonical, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    )
    (tmp_path / "tracker.jsonl").write_bytes(
        b"".join(finalizer.canonical_json_bytes(row) for row in rows)
    )

    with pytest.raises(
        finalizer.TrainingFinalizationError,
        match="ledger hash/schema",
    ):
        finalizer.verify_recorded_handeye_frame_evidence(
            tmp_path,
            activated.handeye,
        )


def test_tracker_replay_streams_without_whole_file_jsonl_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, _raw = _provisional_handeye()
    activated = _activated_contracts(capability)
    _write_tracker_handeye_rows(tmp_path, activated.handeye)

    def reject_whole_file_loader(*_args, **_kwargs):
        raise AssertionError("tracker replay must not load a bounded whole file")

    monkeypatch.setattr(finalizer, "_read_jsonl", reject_whole_file_loader)

    assert finalizer.verify_recorded_handeye_frame_evidence(
        tmp_path,
        activated.handeye,
    )["frame_count"] == 2


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"value":1}', "invalid JSONL framing"),
        (b'{"value":1}\r\n', "invalid JSONL framing"),
        (b'{"value":1,"value":2}\n', "duplicate JSON field"),
        (b'{"value": 1}\n', "not canonical JSON"),
    ],
)
def test_streaming_jsonl_rejects_noncanonical_or_unsafe_framing(
    tmp_path: Path,
    raw: bytes,
    message: str,
) -> None:
    tracker = tmp_path / "tracker.jsonl"
    tracker.write_bytes(raw)

    with pytest.raises(finalizer.TrainingFinalizationError, match=message):
        list(finalizer._iter_stable_jsonl(tracker, label="test tracker"))


def test_streaming_jsonl_bounds_each_row_and_detects_mid_replay_change(
    tmp_path: Path,
) -> None:
    tracker = tmp_path / "tracker.jsonl"
    first = finalizer.canonical_json_bytes({"frame": 1})
    second = finalizer.canonical_json_bytes({"frame": 2})
    tracker.write_bytes(first + second)

    with pytest.raises(finalizer.TrainingFinalizationError, match="size bound"):
        list(
            finalizer._iter_stable_jsonl(
                tracker,
                label="test tracker",
                maximum_line_bytes=len(first) - 1,
            )
        )

    rows = finalizer._iter_stable_jsonl(tracker, label="test tracker")
    assert next(rows) == {"frame": 1}
    tracker.write_bytes(first + second + finalizer.canonical_json_bytes({"frame": 3}))
    with pytest.raises(finalizer.TrainingFinalizationError, match="changed while reading"):
        list(rows)


def test_runtime_activation_contract_loader_uses_exact_manifest_identity_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, activation = _manifest_with_runtime_activation(tmp_path)
    contract = _robot_contract()
    observed: dict[str, object] = {}

    class FakeAssembly:
        robot_capture_coordinator = SimpleNamespace(contract=contract)
        robot_mask_provider = SimpleNamespace(handeye=_legacy_handeye())
        patient_pose_provider = SimpleNamespace(
            intrinsics=SimpleNamespace(depth_scale=1.0),
        )
        camera_exposure_contract = SimpleNamespace(camera_id="CL8326300A0")
        occlusion_runtime = SimpleNamespace(point_names=tuple(LUNG_ORDER))

        def close(self) -> None:
            observed["closed"] = True

    def fake_load(path, **kwargs):  # noqa: ANN001, ANN202
        observed["path"] = path
        observed["kwargs"] = kwargs
        return FakeAssembly()

    monkeypatch.setattr(
        "ausc.perception.production_runtime_activation."
        "load_production_runtime_assembly",
        fake_load,
    )

    result = finalizer.load_activated_capture_contracts(tmp_path, manifest)

    assert result.robot_contract is contract
    assert result.depth_scale_mm == 1.0
    assert result.camera_id == "CL8326300A0"
    assert result.point_names == tuple(LUNG_ORDER)
    assert observed["path"] == activation.resolve()
    assert observed["kwargs"] == {
        "expected_activation_sha256": manifest["hardware_activation"]["sha256"],
        "point_names": tuple(LUNG_ORDER),
        "expected_camera_id": "CL8326300A0",
        "expected_capture_mode": manifest["camera"]["timing"]["capture_mode"],
    }
    assert observed["closed"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.pop("point_names"), "point_names"),
        (
            lambda manifest: manifest.__setitem__(
                "point_names", list(reversed(LUNG_ORDER))
            ),
            "canonical",
        ),
        (lambda manifest: manifest.pop("camera"), "camera identity"),
        (
            lambda manifest: manifest.__setitem__(
                "hardware_activation",
                {**manifest["hardware_activation"], "release_id": "untrusted"},
            ),
            "exact path/SHA",
        ),
    ],
)
def test_runtime_activation_contract_loader_fails_closed_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    manifest, _ = _manifest_with_runtime_activation(tmp_path)
    mutate(manifest)
    called = False

    def fake_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid manifest identity reached runtime loader")

    monkeypatch.setattr(
        "ausc.perception.production_runtime_activation."
        "load_production_runtime_assembly",
        fake_load,
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match=message):
        finalizer.load_activated_capture_contracts(tmp_path, manifest)
    assert called is False


def test_runtime_activation_contract_loader_closes_before_rejecting_bad_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _manifest_with_runtime_activation(tmp_path)
    closed = False

    class FakeAssembly:
        robot_capture_coordinator = SimpleNamespace(contract="self-reported")
        patient_pose_provider = SimpleNamespace(
            intrinsics=SimpleNamespace(depth_scale=1.0),
        )
        camera_exposure_contract = SimpleNamespace(camera_id="CL8326300A0")
        occlusion_runtime = SimpleNamespace(point_names=tuple(LUNG_ORDER))

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        "ausc.perception.production_runtime_activation."
        "load_production_runtime_assembly",
        lambda *_args, **_kwargs: FakeAssembly(),
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match="RobotStateContract"):
        finalizer.load_activated_capture_contracts(tmp_path, manifest)
    assert closed is True


def test_readiness_reports_missing_exact_runtime_identity_without_fallback(
    tmp_path: Path,
) -> None:
    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=_manifest(),
        finalizer_identity=_identity(),
    )

    assert "production_runtime_activation_revalidation_failed" in report["blockers"]
    check = report["checks"]["production_runtime_activation"]
    assert check["status"] == "FAIL"
    assert check["expected_point_names"] == list(LUNG_ORDER)
    assert check["observed_point_names"] is None


def test_readiness_passes_only_activated_depth_scale_to_depth_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, activation = _manifest_with_runtime_activation(tmp_path)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "depth_times.csv").write_text("placeholder\n")
    (depth_dir / "depth_000000.npy").write_bytes(b"placeholder\n")
    activated = finalizer.ActivatedCaptureContracts(
        robot_contract=_robot_contract(),
        depth_scale_mm=0.5,
        activation_path=str(activation.resolve()),
        activation_sha256=manifest["hardware_activation"]["sha256"],
        camera_id="CL8326300A0",
        point_names=tuple(LUNG_ORDER),
        handeye=finalizer._binding_from_runtime_handeye(
            _legacy_handeye(), artifact_sha256="e" * 64,
        ),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        finalizer,
        "load_activated_capture_contracts",
        lambda *_args, **_kwargs: activated,
    )

    def fake_depth(_session, *, expected_depth_scale_mm):  # noqa: ANN001, ANN202
        observed["depth_scale"] = expected_depth_scale_mm
        return {"status": "PASS", "depth_scale_mm": expected_depth_scale_mm}

    monkeypatch.setattr(
        finalizer, "verify_depth_session_authenticity", fake_depth
    )

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=manifest,
        finalizer_identity=_identity(),
    )

    assert observed["depth_scale"] == 0.5
    assert report["checks"]["production_runtime_activation"]["status"] == "PASS"
    assert report["checks"]["component_revalidation"]["depth_session"] == {
        "status": "PASS",
        "depth_scale_mm": 0.5,
    }


def test_quality_replay_does_not_claim_production_authorization_without_model_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "packet_count": 3,
        "sha256": SHA_A,
        "last_packet_chain_sha256": SHA_B,
        "model_input_inventory": {"count": 3},
        "force_input_identity": {"sha256": "c" * 64},
        "chain_verification": {"training_signal_eligible": True},
        "quality_authenticity": {
            "model": {
                "release_id": "quality-model-" + "d" * 20,
                "model_sha256": "d" * 64,
            },
            "policy_sha256": "e" * 64,
        },
    }
    monkeypatch.setattr(
        "ausc.realtime_q.verify_quality_packet_log", lambda _session: receipt,
    )
    monkeypatch.setattr(
        "ausc.realtime_q.read_quality_packet_log", lambda _path: [{"sequence": 1}],
    )
    monkeypatch.setattr(
        finalizer,
        "verify_quality_force_rows",
        lambda _session, _packets: {"status": "PASS"},
    )

    evidence = finalizer.verify_quality_packet_component(tmp_path)

    assert evidence["status"] == "PASS"
    assert evidence["production_authorized"] is False
    assert evidence["blocker"] == (
        "activated quality model replay against lossless audio windows is required"
    )


def test_existing_capture_manifest_is_verified_not_treated_as_final_output(
    tmp_path: Path,
) -> None:
    source = _manifest()
    raw = json.dumps(source, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (tmp_path / "manifest.json").write_bytes(raw)

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=source,
        finalizer_identity=_identity(),
    )

    assert "finalization_metadata_already_present" not in report["blockers"]
    assert report["checks"]["source_manifest"]["status"] == "PASS"
    assert report["checks"]["source_manifest"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "source_manifest_controlled_replacement_not_activated" in report["blockers"]

    changed = dict(source)
    changed["session_id"] = "different"
    mismatch = finalizer.inspect_training_finalization_readiness(
        tmp_path,
        manifest_template=changed,
        finalizer_identity=_identity(),
    )
    assert "source_manifest_mismatch" in mismatch["blockers"]


def test_depth_authenticity_rejects_near_empty_frame(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    frame = np.zeros((10, 10), dtype=np.uint16)
    frame[5, 5] = 700
    np.save(depth_dir / "depth_000000.npy", frame, allow_pickle=False)
    (depth_dir / "depth_times.csv").write_text(
        "frame_index,host_arrival_t_s,device_t_ms,depth_scale,path\n"
        "0,1.000000000,1000.0,1.0,depth/depth_000000.npy\n",
        encoding="utf-8",
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match="near-empty"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_rejects_long_bit_identical_run(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    frame = np.full((10, 10), 700, dtype=np.uint16)
    lines = [
        "frame_index,host_arrival_t_s,device_t_ms,depth_scale,path"
    ]
    for index in range(60):
        np.save(depth_dir / f"depth_{index:06d}.npy", frame, allow_pickle=False)
        lines.append(
            f"{index},{1.0 + index * 0.04:.9f},"
            f"{1000.0 + index * 40.0:.3f},1.0,"
            f"depth/depth_{index:06d}.npy"
        )
    (depth_dir / "depth_times.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(finalizer.TrainingFinalizationError, match="bit-identical"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_allows_static_scene_with_sensor_noise(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    lines = [
        "frame_index,host_arrival_t_s,device_t_ms,depth_scale,path"
    ]
    for index in range(60):
        frame = np.full((10, 10), 700, dtype=np.uint16)
        frame[index % 10, (index * 3) % 10] += 1
        np.save(depth_dir / f"depth_{index:06d}.npy", frame, allow_pickle=False)
        lines.append(
            f"{index},{1.0 + index * 0.04:.9f},"
            f"{1000.0 + index * 40.0:.3f},1.0,"
            f"depth/depth_{index:06d}.npy"
        )
    (depth_dir / "depth_times.csv").write_text("\n".join(lines) + "\n")

    evidence = finalizer.verify_depth_session_authenticity(
        tmp_path, expected_depth_scale_mm=1.0,
    )
    assert evidence["status"] == "PASS"
    assert evidence["frame_count"] == 60
    assert evidence["minimum_valid_fraction"] == 1.0


def _write_valid_production_depth_session(
    root: Path,
    *,
    frame_count: int = 60,
    device_time=lambda index: 1000.0 + index * 40.0,
    depth_scale=lambda _index: 1.0,
    shape: tuple[int, int] = (20, 20),
    heartbeat_only: bool = False,
) -> None:
    depth_dir = root / "depth"
    depth_dir.mkdir()
    lines = [
        "frame_index,host_arrival_t_s,device_t_ms,depth_scale,path"
    ]
    for index in range(frame_count):
        frame = np.full(shape, 700, dtype=np.uint16)
        if heartbeat_only:
            frame[0, 0] += index * 2
        else:
            frame[index % shape[0], (index * 3) % shape[1]] += 1
        np.save(depth_dir / f"depth_{index:06d}.npy", frame, allow_pickle=False)
        lines.append(
            f"{index},{1.0 + index * 0.04:.9f},"
            f"{device_time(index)},{depth_scale(index)},"
            f"depth/depth_{index:06d}.npy"
        )
    (depth_dir / "depth_times.csv").write_text("\n".join(lines) + "\n")


def test_depth_authenticity_rejects_frozen_device_clock(tmp_path: Path) -> None:
    _write_valid_production_depth_session(
        tmp_path, device_time=lambda _index: 1000.0,
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match="device clock"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_rejects_scale_drift_or_activation_mismatch(
    tmp_path: Path,
) -> None:
    _write_valid_production_depth_session(
        tmp_path, depth_scale=lambda index: 1.0 if index < 30 else 0.5,
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match="depth scale"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_rejects_too_short_session(tmp_path: Path) -> None:
    _write_valid_production_depth_session(tmp_path, frame_count=29)

    with pytest.raises(finalizer.TrainingFinalizationError, match="minimum"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_rejects_single_pixel_heartbeat_bypass(
    tmp_path: Path,
) -> None:
    _write_valid_production_depth_session(
        tmp_path,
        shape=(200, 200),
        heartbeat_only=True,
    )

    with pytest.raises(finalizer.TrainingFinalizationError, match="near-zero-change"):
        finalizer.verify_depth_session_authenticity(
            tmp_path, expected_depth_scale_mm=1.0,
        )


def test_depth_authenticity_requires_activated_depth_scale(tmp_path: Path) -> None:
    _write_valid_production_depth_session(tmp_path)

    with pytest.raises(finalizer.TrainingFinalizationError, match="activated"):
        finalizer.verify_depth_session_authenticity(tmp_path)


def test_quality_force_rows_must_exist_in_revalidated_force_csv(tmp_path: Path) -> None:
    (tmp_path / "force.csv").write_text(
        "host_t_s,raw,hall_v,displacement_mm,force_n,status,device_t_ms\n"
        "1,100,0.10000000000000001,0.20000000000000001,0.3,OK,10\n"
        "2,101,0.11,0.20999999999999999,0.31,OK,20\n",
        encoding="utf-8",
    )
    packets = [{
        "force_sample": {
            "host_t_s": 2.0,
            "raw": 101,
            "hall_v": 0.11,
            "displacement_mm": 0.21,
            "force_n": 0.31,
            "status": "OK",
            "device_t_ms": 20.0,
        },
    }]

    evidence = finalizer.verify_quality_force_rows(tmp_path, packets)
    assert evidence["matched_packet_count"] == 1

    packets[0]["force_sample"]["force_n"] = 0.32
    with pytest.raises(finalizer.TrainingFinalizationError, match="force.csv"):
        finalizer.verify_quality_force_rows(tmp_path, packets)


def test_offline_robot_replay_uses_activated_contract_not_summary_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "schema": "wam_openarm_exact_control_capture_summary_v4",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": [f"openarm_left_joint{index}" for index in range(1, 8)],
        "joint_position_limits_rad": [[-4.0, 4.0] for _ in range(7)],
        "urdf_sha256": "1" * 64,
        "joint_mapper_sha256": "2" * 64,
        "camera_profile_sha256": "3" * 64,
        "timing_artifact_sha256": "4" * 64,
        "pair_contract_sha256": "5" * 64,
        "controller_binary_sha256": "6" * 64,
        "controller_source_manifest_sha256": "7" * 64,
        "follower_config_sha256": "8" * 64,
        "openarm_sdk_sha256": "9" * 64,
        "max_abs_joint_velocity_rad_s": 8.0,
        "gripper_motor_position_limits_rad": [-0.2, 0.2],
        "buffer_duration_ns": 2_000_000_000,
        "max_packets": 256,
        "max_gap_ns": 20_000_000,
        "max_staleness_ns": 50_000_000,
        "max_exposure_interval_ns": 40_000_000,
        "training_eligible": True,
        "frame_count": 30,
        "control_tick_count": 30,
        "feedback_clean": True,
        "trajectory_nondegenerate": True,
        "command_response_consistent": True,
    }
    (tmp_path / "robot_capture_summary.json").write_bytes(
        finalizer.canonical_json_bytes(summary)
    )
    manifest = _manifest()
    manifest["robot"] = summary
    observed: dict[str, object] = {}

    def fake_replay(_session, contract):  # noqa: ANN001, ANN202
        observed["contract"] = contract
        return dict(summary)

    monkeypatch.setattr(
        "ausc.perception.robot.robot_session_capture."
        "validate_robot_capture_artifacts",
        fake_replay,
    )
    contract = _robot_contract()
    evidence = finalizer._verify_robot_component_against_contract(
        tmp_path, manifest, contract
    )
    assert observed["contract"] is contract
    assert evidence["activated_policy"]["max_staleness_ns"] == 50_000_000

    forged = dict(summary)
    forged["max_staleness_ns"] = 500_000_000
    (tmp_path / "robot_capture_summary.json").write_bytes(
        finalizer.canonical_json_bytes(forged)
    )
    manifest["robot"] = forged
    with pytest.raises(finalizer.TrainingFinalizationError, match="activation"):
        finalizer._verify_robot_component_against_contract(
            tmp_path, manifest, contract
        )


def test_canonical_exclusive_write_never_overwrites_existing_receipt(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    finalizer.write_exclusive_canonical_json(destination, {"status": "PASS"})
    first = destination.read_bytes()

    with pytest.raises(finalizer.TrainingFinalizationError, match="already exists"):
        finalizer.write_exclusive_canonical_json(destination, {"status": "FAIL"})

    assert destination.read_bytes() == first
    assert json.loads(first) == {"status": "PASS"}


# --- Finalizer hidden gates: production writer + quality authorization (#3) ---

def _quality_branch_files(session: Path) -> None:
    """Create the minimal artifacts that reach the quality-verifier branch."""

    inputs = session / "quality_model_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "window_000000.npy").write_bytes(b"\x00\x01")
    (session / "force.csv").write_text("host_t,newtons\n0.0,1.0\n", encoding="utf-8")
    (session / "quality_inference_authenticity.json").write_text(
        "{}", encoding="utf-8"
    )
    (session / "quality_inference_packets.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )


def test_quality_component_not_production_authorized_is_a_blocker(
    tmp_path: Path, monkeypatch
) -> None:
    _quality_branch_files(tmp_path)
    monkeypatch.setattr(
        finalizer,
        "verify_quality_packet_component",
        lambda _session: {
            "status": "PASS",
            "production_authorized": False,
            "blocker": "activated quality model replay is required",
        },
    )

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path, manifest_template=_manifest(), finalizer_identity=_identity(),
    )

    assert "quality_component_not_production_authorized" in report["blockers"]
    check = report["checks"]["lossless_model_evidence"]["quality_exact_input_verifier"]
    assert check["status"] == "FAIL"
    assert check["reason"] == "activated quality model replay is required"


def test_quality_component_production_authorized_passes_the_readiness_check(
    tmp_path: Path, monkeypatch
) -> None:
    _quality_branch_files(tmp_path)
    monkeypatch.setattr(
        finalizer,
        "verify_quality_packet_component",
        lambda _session: {"status": "PASS", "production_authorized": True},
    )

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path, manifest_template=_manifest(), finalizer_identity=_identity(),
    )

    assert (
        "quality_component_not_production_authorized" not in report["blockers"]
    )
    check = report["checks"]["lossless_model_evidence"]["quality_exact_input_verifier"]
    assert check["status"] == "PASS"


def test_diagnostic_capture_kind_can_never_be_promoted(tmp_path: Path) -> None:
    report = finalizer.inspect_training_finalization_readiness(
        tmp_path, manifest_template=_manifest(), finalizer_identity=_identity(),
    )

    assert "capture_kind_not_production" in report["blockers"]
    assert report["checks"]["production_capture_kind"]["status"] == "FAIL"


def test_production_capture_kind_satisfies_the_immutable_writer_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["capture_kind"] = "production_teleop_v4"

    report = finalizer.inspect_training_finalization_readiness(
        tmp_path, manifest_template=manifest, finalizer_identity=_identity(),
    )

    assert "capture_kind_not_production" not in report["blockers"]
    check = report["checks"]["production_capture_kind"]
    assert check["status"] == "PASS"
    assert check["capture_kind"] == "production_teleop_v4"


def test_tracker_jsonl_is_a_geometry_file():
    assert "tracker.jsonl" in finalizer._GEOMETRY_FILES
