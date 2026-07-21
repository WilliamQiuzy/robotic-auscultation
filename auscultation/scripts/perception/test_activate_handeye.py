from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import activate_handeye as issuer  # noqa: E402


RUN_ID = "handeye_left_20260718T195334Z_70cf253b"
CAMERA_ID = "CL8326300A0"
RESULT_NAME = "cam2base_USER_ACCEPTED_5mm.json"
SOURCE_KEYS = {
    "T_base_to_cam",
    "T_cam_to_base",
    "arm",
    "code_release",
    "convention",
    "covariance_se3_T_cam_to_base",
    "inputs",
    "note",
    "quality",
    "run_id",
    "schema",
    "status",
    "tip_offset_tcp_m",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _valid_run(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    runs = tmp_path / "runs"
    run = runs / RUN_ID
    captures = run / "captures"
    captures.mkdir(parents=True)

    samples = []
    for sample_id in range(1, 24):
        prefix = f"{sample_id:04d}"
        rgb_relative = f"captures/{prefix}_rgb.png"
        depth_relative = f"captures/{prefix}_depth.npy"
        rgb_body = b"rgb-capture-" + str(sample_id).encode("ascii")
        depth_body = b"depth-capture-" + str(sample_id).encode("ascii")
        (run / rgb_relative).write_bytes(rgb_body)
        (run / depth_relative).write_bytes(depth_body)
        fit = sample_id <= 20
        position = ((sample_id - 1) // 2 + 1) if fit else sample_id - 20
        orientation = ("A" if sample_id % 2 else "B") if fit else None
        camera_sequence = sample_id
        tcp = np.eye(4)
        if fit and orientation == "B":
            angle = math.radians(45.0)
            tcp[:3, :3] = np.array(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        samples.append(
            {
                "T_tcp_base": tcp.tolist(),
                "camera_age_s": 0.01,
                "camera_device_t_ms": 1000.0 + camera_sequence,
                "camera_frame_index": camera_sequence,
                "camera_host_monotonic_s": 500.0 + camera_sequence / 30.0,
                "camera_time_source": "device",
                "depth_m": 0.7,
                "depth_patch_span_mm": 1.0,
                "depth_patch_valid_count": 25,
                "depth_patch_window_size": 5,
                "depth_path": depth_relative,
                "depth_raw": 700.0,
                "depth_scale_mm": 1.0,
                "depth_sha256": _sha(depth_body),
                "orientation": orientation,
                "phase": "fit" if fit else "heldout",
                "pixel": [640, 360],
                "position_index": position,
                "q7_rad": [0.01 * sample_id] * 7,
                "rgb_path": rgb_relative,
                "rgb_sha256": _sha(rgb_body),
                "sample_id": sample_id,
                "tip_cam_m": [0.1, 0.2, 0.7],
            }
        )
    samples_body = b"".join(_canonical(sample) for sample in samples)
    (run / "samples.jsonl").write_bytes(samples_body)
    (run / "joint_stream_snapshot.csv").write_bytes(
        b"0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.0\n"
    )

    training_errors = [8.0 + index / 10.0 for index in range(20)]
    training_rms = math.sqrt(
        sum(value * value for value in training_errors) / len(training_errors)
    )
    quality = {
        "fit_count": 20,
        "gate_thresholds_mm": {
            "each_point": 5.0,
            "heldout": 5.0,
            "training_rms": 3.0,
        },
        "heldout_count": 3,
        "heldout_errors_mm": [6.1, 5.9, 6.2],
        "result_precision_mm_1sigma_max": 5.0,
        "result_precision_mm_1sigma_median": 4.2,
        "solver_condition": 1234.0,
        "solver_rank": 9,
        "training_errors_mm": training_errors,
        "training_max_mm": max(training_errors),
        "training_rms_mm": training_rms,
    }
    failure_reasons = [
        "training_rms_above_3mm",
        "training_max_error_above_5mm",
        "heldout_error_above_5mm",
    ]
    provenance = {
        "arm": "left",
        "camera": {
            "device_identity": CAMERA_ID,
            "requested_serial": CAMERA_ID,
        },
        "can_interface": "can3",
        "capture_contract": {
            "fit_3d_rms_max_mm": 3.0,
            "fit_count": 20,
            "fit_each_point_max_mm": 5.0,
            "heldout_count": 3,
            "heldout_each_point_max_mm": 5.0,
            "maximum_pair_contact_delta_mm": 5.0,
            "minimum_pair_rotation_deg": 30.0,
            "pair_count": 10,
            "rigid_nonhuman_fixture_required": True,
            "two_wrist_orientations_per_position": True,
        },
        "mode": "tiptouch",
        "production_activation": False,
        "run_id": RUN_ID,
        "sample_count": 23,
        "software_provenance": {
            "code_release": {
                "release_id": "handeye_calibration_code-" + "a" * 20,
            },
        },
    }
    failed = {
        "failure_reasons": failure_reasons,
        "provenance": provenance,
        "schema": "wam_handeye_failure_v1",
        "status": "REJECTED",
    }
    _write_json(run / "FAILED.json", failed)

    T_cam_to_base = np.eye(4)
    T_cam_to_base[:3, 3] = [0.30, -0.20, 0.70]
    T_base_to_cam = np.linalg.inv(T_cam_to_base)
    covariance = np.diag([1e-6, 1e-6, 1e-6, 5.5e-6, 5.5e-6, 5.5e-6])
    skew = np.array(
        [[0.0, -0.7, 0.2], [0.7, 0.0, -0.1], [-0.2, 0.1, 0.0]]
    )
    jacobian = np.hstack((-skew, np.eye(3)))
    propagated_mm = float(
        np.sqrt(np.trace(jacobian @ covariance @ jacobian.T)) * 1000.0
    )
    quality["result_precision_mm_1sigma_median"] = propagated_mm
    quality["result_precision_mm_1sigma_max"] = propagated_mm
    source = {
        "T_base_to_cam": T_base_to_cam.tolist(),
        "T_cam_to_base": T_cam_to_base.tolist(),
        "arm": "left",
        "code_release": "handeye_calibration_code-" + "a" * 20,
        "convention": (
            "p_base = T_cam_to_base @ p_cam ; a point measured in the "
            "COLOR-camera optical frame (metric xyz, meters) maps into the ARM "
            "BASE (openarm_left_link0) frame. tip_offset_tcp_m is the solved "
            "tool point in the openarm_left_hand_tcp frame."
        ),
        "covariance_se3_T_cam_to_base": {
            "definition": (
                "empirical bootstrap cov of delta=[Rodrigues(R_ref^T R_boot), "
                "t_boot - t_ref]; random precision only, excludes systematic bias"
            ),
            "matrix": covariance.tolist(),
            "order": ["rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m"],
        },
        "inputs": {
            "captures_dir": str(captures),
            "full_provenance": str(run / "FAILED.json"),
            "joint_stream_snapshot": str(run / "joint_stream_snapshot.csv"),
            "samples_jsonl": str(run / "samples.jsonl"),
        },
        "note": "William explicitly accepted approximately 5 mm for bootstrap use.",
        "quality": quality,
        "run_id": RUN_ID,
        "schema": "wam_handeye_user_accepted_v1",
        "status": "USER_ACCEPTED_BELOW_GATE",
        "tip_offset_tcp_m": [0.002, 0.028, 0.180],
    }
    assert set(source) == SOURCE_KEYS
    _write_json(run / RESULT_NAME, source)
    return runs, run, source


def _dry_run(
    runs: Path, activation_root: Path,
) -> dict[str, object]:
    return issuer.activate_handeye_run(
        RUN_ID,
        runs_root=runs,
        activation_root=activation_root,
        execute=False,
        test_only_allow_nonproduction_roots=True,
    )


def _execute(
    runs: Path, activation_root: Path, dry_run: dict[str, object], **overrides,
) -> dict[str, object]:
    arguments = {
        "confirm_run_id": RUN_ID,
        "confirm_source_sha256": dry_run["semantic_source_hashes"]["closure"],
        "test_only_allow_nonproduction_roots": True,
    }
    arguments.update(overrides)
    return issuer.activate_handeye_run(
        RUN_ID,
        runs_root=runs,
        activation_root=activation_root,
        execute=True,
        **arguments,
    )


def _rewrite_samples(run: Path, mutate) -> None:
    samples = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
    mutate(samples)
    (run / "samples.jsonl").write_bytes(
        b"".join(_canonical(sample) for sample in samples)
    )


def test_dry_run_builds_provisional_payload_without_writing(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"

    report = _dry_run(runs, activation_root)

    assert report["status"] == "DRY_RUN"
    assert not activation_root.exists()
    assert report["release_id"] == "handeye_activation-" + report["artifact_sha256"][:20]
    artifact = report["artifact"]
    assert _sha(_canonical(artifact)) == report["artifact_sha256"]
    assert artifact["schema"] == "wam_openarm_handeye_activation_v2"
    assert artifact["camera_id"] == CAMERA_ID
    assert artifact["side"] == "left"
    assert artifact["source_frame"] == "orbbec_color_optical_frame"
    assert artifact["target_frame"] == "openarm_left_link0"
    assert artifact["calibration_status"] == "USER_ACCEPTED_BELOW_GATE"
    assert artifact["provisional"] is True
    assert artifact["quality_gate_passed"] is False
    assert artifact["failed_quality_gate_m"] == 0.003
    assert artifact["authorized_scopes"] == [
        "bootstrap_teleop_logging",
        "conservative_robot_mask",
    ]
    assert artifact["autonomous_targeting_authorized"] is False
    assert artifact["systematic_bias_status"] == "NOT_CHARACTERIZED"
    assert artifact["systematic_bias_bound_m"] is None
    assert artifact["bootstrap_reproducibility"]["status"] == "INCOMPLETE"
    assert artifact["acceptance_note"] == _source["note"]
    assert artifact["source_evidence"]["sample_count"] == 23
    assert len(artifact["source_evidence"]["captures"]) == 46


def test_owner_signoff_metadata_does_not_expand_activation_scope(
    tmp_path: Path,
) -> None:
    runs, run, source = _valid_run(tmp_path)
    source["owner_authorization"] = {
        "authorized_by": "william (project owner)",
        "autonomous_targeting_authorized_by_owner": True,
        "conditions": (
            "human-supervised, bounded step, foam/dummy patient first, "
            "right arm (can0/can2) de-energized"
        ),
        "date_utc": "2026-07-18",
        "explicit_signoff": (
            "William explicitly signed off in conversation on 2026-07-18."
        ),
        "note": (
            "Owner override of the issuer default; supervised bring-up does "
            "not require autonomous authorization."
        ),
    }
    _write_json(run / RESULT_NAME, source)

    report = _dry_run(runs, tmp_path / "activations")

    assert report["artifact"]["autonomous_targeting_authorized"] is False
    assert report["artifact"]["authorized_scopes"] == [
        "bootstrap_teleop_logging",
        "conservative_robot_mask",
    ]


def test_equal_camera_evidence_for_a_pair_remains_live_compatible(tmp_path: Path) -> None:
    runs, run, _source = _valid_run(tmp_path)

    def reuse_pair_evidence(samples) -> None:
        samples[1]["camera_frame_index"] = samples[0]["camera_frame_index"]
        samples[1]["camera_host_monotonic_s"] = samples[0]["camera_host_monotonic_s"]

    _rewrite_samples(run, reuse_pair_evidence)

    report = _dry_run(runs, tmp_path / "activations")

    assert report["status"] == "DRY_RUN"


def test_pair_requires_minimum_wrist_rotation(tmp_path: Path) -> None:
    runs, run, _source = _valid_run(tmp_path)

    def remove_rotation(samples) -> None:
        samples[1]["T_tcp_base"] = samples[0]["T_tcp_base"]

    _rewrite_samples(run, remove_rotation)

    with pytest.raises(issuer.HandEyeActivationError, match="pair rotation"):
        _dry_run(runs, tmp_path / "activations")


def test_pair_requires_same_contact_within_five_mm(tmp_path: Path) -> None:
    runs, run, _source = _valid_run(tmp_path)

    def move_contact(samples) -> None:
        samples[1]["tip_cam_m"][0] += 0.006

    _rewrite_samples(run, move_contact)

    with pytest.raises(issuer.HandEyeActivationError, match="pair contact"):
        _dry_run(runs, tmp_path / "activations")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_pair_rotation_deg", 0.0),
        ("maximum_pair_contact_delta_mm", 10.0),
        ("rigid_nonhuman_fixture_required", False),
        ("two_wrist_orientations_per_position", False),
    ],
)
def test_capture_contract_geometry_cannot_be_relaxed(
    tmp_path: Path, field: str, value: object,
) -> None:
    runs, run, _source = _valid_run(tmp_path)
    failed = json.loads((run / "FAILED.json").read_text())
    failed["provenance"]["capture_contract"][field] = value
    _write_json(run / "FAILED.json", failed)

    with pytest.raises(issuer.HandEyeActivationError, match="capture contract"):
        _dry_run(runs, tmp_path / "activations")


def test_successful_publish_is_canonical_complete_and_read_only(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)

    report = _execute(runs, activation_root, dry)

    release = Path(report["release_path"])
    artifact_raw = (release / "handeye.json").read_bytes()
    artifact = json.loads(artifact_raw)
    assert artifact_raw == _canonical(artifact)
    assert _sha(artifact_raw) == report["artifact_sha256"]
    artifact_complete_raw = (release / "handeye.COMPLETE").read_bytes()
    artifact_complete = json.loads(artifact_complete_raw)
    assert artifact_complete_raw == _canonical(artifact_complete)
    assert artifact_complete == {
        "artifact_sha256": report["artifact_sha256"],
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "provisional": True,
        "schema": "wam_openarm_handeye_artifact_completion_v2",
        "status": "COMPLETE",
    }
    release_complete = json.loads((release / "COMPLETE").read_bytes())
    assert release_complete["schema"] == "wam_openarm_handeye_release_completion_v2"
    assert release_complete["status"] == "COMPLETE"
    assert release_complete["release_id"] == report["release_id"]
    assert set(release_complete["inventory"]) == {"handeye.COMPLETE", "handeye.json"}
    assert stat.S_IMODE(release.stat().st_mode) == 0o555
    for path in release.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert list((activation_root / ".staging").iterdir()) == []


def test_publish_renames_private_staging_before_final_read_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    original_rename = issuer.os.rename
    observed_modes: list[int] = []

    def require_owner_writable_staging(source: str, destination: str) -> None:
        observed_modes.append(stat.S_IMODE(Path(source).stat().st_mode))
        assert observed_modes[-1] & stat.S_IWUSR
        original_rename(source, destination)

    monkeypatch.setattr(issuer.os, "rename", require_owner_writable_staging)

    report = _execute(runs, activation_root, dry)

    assert observed_modes == [0o700]
    assert stat.S_IMODE(Path(report["release_path"]).stat().st_mode) == 0o555


@pytest.mark.parametrize("confirmation", [None, "wrong-run"])
def test_execute_rejects_wrong_run_confirmation(
    tmp_path: Path, confirmation: str | None,
) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)

    with pytest.raises(issuer.HandEyeActivationError, match="confirm-run-id"):
        _execute(
            runs,
            activation_root,
            dry,
            confirm_run_id=confirmation,
        )

    assert not activation_root.exists()


def test_execute_rejects_wrong_source_confirmation(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)

    with pytest.raises(issuer.HandEyeActivationError, match="source SHA-256"):
        _execute(
            runs,
            activation_root,
            dry,
            confirm_source_sha256="0" * 64,
        )

    assert not activation_root.exists()


def test_confirmation_binds_joint_and_full_source_closure(tmp_path: Path) -> None:
    runs, run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    joint = run / "joint_stream_snapshot.csv"
    joint.write_bytes(joint.read_bytes() + b"1.0,0,0,0,0,0,0,0,0\n")

    with pytest.raises(issuer.HandEyeActivationError, match="source SHA-256"):
        _execute(runs, activation_root, dry)

    assert not activation_root.exists()


def test_library_rejects_nonproduction_roots_without_explicit_test_flag(
    tmp_path: Path,
) -> None:
    runs, _run, _source = _valid_run(tmp_path)

    with pytest.raises(issuer.HandEyeActivationError, match="production roots"):
        issuer.activate_handeye_run(
            RUN_ID,
            runs_root=runs,
            activation_root=tmp_path / "activations",
            execute=False,
        )


@pytest.mark.parametrize("target", [RESULT_NAME, "captures/0001_rgb.png"])
def test_source_or_capture_tamper_is_rejected(tmp_path: Path, target: str) -> None:
    runs, run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    path = run / target
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(issuer.HandEyeActivationError, match="SHA-256|JSON"):
        _execute(runs, activation_root, dry)

    assert not activation_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source["covariance_se3_T_cam_to_base"].update(
                {"order": ["tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad"]}
            ),
            "covariance order",
        ),
        (
            lambda source: source["covariance_se3_T_cam_to_base"].update(
                {"matrix": np.diag([1, 1, 1, 1, 1, -1]).tolist()}
            ),
            "positive semidefinite",
        ),
        (
            lambda source: source["T_base_to_cam"][0].__setitem__(3, 1.0),
            "inverse",
        ),
    ],
)
def test_covariance_and_transform_semantics_fail_closed(
    tmp_path: Path, mutation, message: str,
) -> None:
    runs, run, source = _valid_run(tmp_path)
    mutation(source)
    _write_json(run / RESULT_NAME, source)

    with pytest.raises(issuer.HandEyeActivationError, match=message):
        _dry_run(runs, tmp_path / "activations")


def test_covariance_is_preserved_and_explicitly_bound(tmp_path: Path) -> None:
    runs, _run, source = _valid_run(tmp_path)

    artifact = _dry_run(runs, tmp_path / "activations")["artifact"]

    covariance = artifact["covariance_se3"]
    assert covariance["matrix"] == source["covariance_se3_T_cam_to_base"]["matrix"]
    assert covariance["order"] == [
        "rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m",
    ]
    assert covariance["definition"] == source["covariance_se3_T_cam_to_base"]["definition"]
    assert covariance["bound_transform"] == "transform_follower_base_from_camera"
    assert artifact["covariance_scope"] == "random_only"
    assert artifact["transform_follower_base_from_camera"] == source["T_cam_to_base"]
    assert artifact["transform_camera_from_follower_base"] == source["T_base_to_cam"]
    expected = round(
        float(source["quality"]["result_precision_mm_1sigma_median"]) / 1000.0,
        12,
    )
    assert artifact["workspace_1sigma_m"] == {"max": expected, "median": expected}


def test_identical_release_is_verified_and_reused(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    first = _execute(runs, activation_root, dry)

    second = _execute(runs, activation_root, dry)

    assert first["release_path"] == second["release_path"]
    assert first["reused"] is False
    assert second["reused"] is True


def test_writable_existing_release_is_not_reused(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    first = _execute(runs, activation_root, dry)
    release = Path(first["release_path"])
    release.chmod(0o755)
    (release / "handeye.json").chmod(0o644)

    with pytest.raises(issuer.HandEyeActivationError, match="immutable|mode|tamper"):
        _execute(runs, activation_root, dry)


def test_ten_mm_claim_is_not_issued_as_five_mm_tier(tmp_path: Path) -> None:
    runs, run, source = _valid_run(tmp_path)
    source["quality"]["result_precision_mm_1sigma_median"] = 5.0
    source["quality"]["result_precision_mm_1sigma_max"] = 10.0
    _write_json(run / RESULT_NAME, source)

    with pytest.raises(issuer.HandEyeActivationError, match="5mm|precision"):
        _dry_run(runs, tmp_path / "activations")


def test_existing_release_collision_or_tamper_fails_closed(tmp_path: Path) -> None:
    runs, _run, _source = _valid_run(tmp_path)
    activation_root = tmp_path / "activations"
    dry = _dry_run(runs, activation_root)
    collision = activation_root / "releases" / dry["release_id"]
    collision.mkdir(parents=True)
    (collision / "handeye.json").write_bytes(b"not-the-artifact")

    with pytest.raises(issuer.HandEyeActivationError, match="collision|tamper"):
        _execute(runs, activation_root, dry)


@pytest.mark.parametrize("extra", ["UNTRACKED.bin", "captures/9999_rgb.png"])
def test_exact_run_and_capture_inventory_rejects_extra_files(
    tmp_path: Path, extra: str,
) -> None:
    runs, run, _source = _valid_run(tmp_path)
    target = run / extra
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"untracked")

    with pytest.raises(issuer.HandEyeActivationError, match="inventory"):
        _dry_run(runs, tmp_path / "activations")


def test_final_closure_revalidation_catches_cross_file_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, run, _source = _valid_run(tmp_path)
    original = issuer._verify_samples

    def mutate_after_first_capture_pass(*args, **kwargs):
        result = original(*args, **kwargs)
        capture = run / "captures/0001_rgb.png"
        capture.write_bytes(capture.read_bytes() + b"changed-after-first-pass")
        return result

    monkeypatch.setattr(issuer, "_verify_samples", mutate_after_first_capture_pass)

    with pytest.raises(issuer.HandEyeActivationError, match="closure|changed|SHA-256"):
        _dry_run(runs, tmp_path / "activations")


def test_zero_covariance_cannot_claim_five_mm_precision(tmp_path: Path) -> None:
    runs, run, source = _valid_run(tmp_path)
    source["covariance_se3_T_cam_to_base"]["matrix"] = np.zeros((6, 6)).tolist()
    _write_json(run / RESULT_NAME, source)

    with pytest.raises(issuer.HandEyeActivationError, match="covariance|precision"):
        _dry_run(runs, tmp_path / "activations")


def test_code_release_must_match_failed_provenance(tmp_path: Path) -> None:
    runs, run, source = _valid_run(tmp_path)
    source["code_release"] = "handeye_calibration_code-" + "b" * 20
    _write_json(run / RESULT_NAME, source)

    with pytest.raises(issuer.HandEyeActivationError, match="code release"):
        _dry_run(runs, tmp_path / "activations")


def test_capture_symlink_is_rejected(tmp_path: Path) -> None:
    runs, run, _source = _valid_run(tmp_path)
    capture = run / "captures/0001_rgb.png"
    body = capture.read_bytes()
    capture.unlink()
    outside = tmp_path / "outside.png"
    outside.write_bytes(body)
    capture.symlink_to(outside)

    with pytest.raises(issuer.HandEyeActivationError, match="symlink"):
        _dry_run(runs, tmp_path / "activations")
