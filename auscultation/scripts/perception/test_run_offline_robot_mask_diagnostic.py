"""Contract tests for the non-release official robot-mask diagnostic."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.perception import run_offline_robot_mask_diagnostic as diagnostic


def _point(xyz=(0.1, 0.2, 1.0), *, measured=True, position_std=0.003):
    return diagnostic.TrackerTargetSample(
        measured=measured,
        xyz_camera_m=np.asarray(xyz, dtype=np.float64),
        position_std_m=position_std,
    )


def _evidence_inputs(rows):
    raw_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    for index, row in enumerate(rows):
        row.setdefault("gt_x_camera_m", 0.0)
        row.setdefault("gt_y_camera_m", 0.0)
        row.setdefault("gt_z_camera_m", 1.0)
        for initialized_key, measured_key, status_key, prefix in (
            (
                "baseline_initialized",
                "baseline_measured",
                "baseline_status",
                "baseline",
            ),
            (
                "candidate_initialized",
                "candidate_raw_measured",
                "candidate_raw_status",
                "candidate_raw",
            ),
        ):
            status = str(row.get(status_key, "INIT"))
            row.setdefault(measured_key, status == "TRACKING")
            valid = bool(row.get(initialized_key)) and status in {
                "TRACKING",
                "COASTING",
            }
            error = row.get(f"{prefix}_error_mm")
            if valid and error is not None:
                row.setdefault(
                    f"{prefix}_x_camera_m",
                    float(row["gt_x_camera_m"]) + float(error) / 1000.0,
                )
                row.setdefault(f"{prefix}_y_camera_m", row["gt_y_camera_m"])
                row.setdefault(f"{prefix}_z_camera_m", row["gt_z_camera_m"])
        display_status = str(row.get("candidate_status", "LOST"))
        display_error = row.get("candidate_error_mm")
        if (
            display_status in {"MEASURED", "PATIENT_RELATIVE_COAST"}
            and display_error is not None
        ):
            row.setdefault(
                "candidate_x_camera_m",
                float(row["gt_x_camera_m"]) + float(display_error) / 1000.0,
            )
            row.setdefault("candidate_y_camera_m", row["gt_y_camera_m"])
            row.setdefault("candidate_z_camera_m", row["gt_z_camera_m"])
        row.setdefault("rendered_input_sha256", f"{index + 1:064x}")
        row["frame_evidence_sha256"] = diagnostic.semantic_frame_sha256(row)
        raw_digest.update(bytes.fromhex(str(row["rendered_input_sha256"])))
        semantic_digest.update(bytes.fromhex(str(row["frame_evidence_sha256"])))
    return {
        "schema": diagnostic.INPUT_SCHEMA,
        "raw_rendered_input_stream_sha256": raw_digest.hexdigest(),
        "rendered_evidence_stream_sha256": semantic_digest.hexdigest(),
    }


def test_static_contract_is_explicitly_nonproduction_and_noncompletion() -> None:
    contract = diagnostic.static_contract()

    assert contract["schema"] == "wam_offline_robot_mask_diagnostic_v3"
    assert contract["oracle_2d"] is True
    assert contract["real_v9_executed"] is False
    assert contract["nominal_root_not_commissioned"] is True
    assert contract["production_activation"] is False
    assert contract["completion"] is False
    assert contract["target_name"] == "R1"
    assert contract["wrist_index"] == 0
    assert contract["camera_pose_name"] == "primary"
    assert contract["banner"] == "ORACLE 2D / NOT V9 / NOMINAL ROOT / NOT ISAAC"
    assert contract["conversion_release_id"] == "official_meshes-b5e826af2539bceafea1"
    assert contract["robot_mask_source"] == "exact_simulator_instance_segmentation"
    assert contract["robot_mask_provider_commissioned"] is False
    assert contract["mask_false_negative_sweep_executed"] is False
    assert contract["stable_wrong_surface_defense_validated"] is False
    assert contract["production_robot_mask_call_path_executed"] is False
    assert contract["simulator"] == "MuJoCo_kinematic_official_mesh_not_Isaac"
    assert contract["isaac_equivalence"] is False
    assert contract["isaac_surface_equivalence"] is False
    assert contract["human_pose"] == "default_equals_rest"
    assert contract["human_rest_pose"] is True
    assert contract["method_banner"] == "SIM PATIENT TF / METHOD UPPER BOUND ONLY"
    assert contract["human_banner"] == "OFFICIAL NVIDIA HUMAN / DEFAULT REST POSE"
    assert contract["outcome_banner"] == (
        "NEGATIVE: RESET / NO RECOVERY / NOT RECOMMENDED"
    )
    assert contract["display_policy"] == (
        "mask_tracker_plus_simulator_gt_patient_relative_bounded_coast"
    )
    assert contract["outcome_class"] == "NEGATIVE_RESET_NO_RECOVERY"
    assert contract["recommendation_supported"] is False


def test_patient_transform_is_declared_as_simulator_only_upper_bound() -> None:
    contract = diagnostic.static_contract()

    assert contract["patient_transform_source"] == "simulator_ground_truth"
    assert contract["patient_transform_method_sensitivity_upper_bound_only"] is True
    assert contract["patient_transform_available_in_production"] is False
    assert contract["simulator_patient_transform_used_by_bounded_wrapper"] is True
    assert contract["anchor_three_d_gt_in_trackers"] is False
    assert contract["production_patient_pose_requirement"] == (
        "tracker_or_external_estimator_plus_latency_and_hand_eye_validation"
    )
    assert diagnostic.METHOD_BANNER == "SIM PATIENT TF / METHOD UPPER BOUND ONLY"


def test_tool_version_parser_binds_first_nonempty_version_line() -> None:
    raw = "\nffmpeg version 8.0.1 Copyright\nconfiguration: --locked\n"

    assert diagnostic.parse_tool_version(raw) == "ffmpeg version 8.0.1 Copyright"
    with pytest.raises(diagnostic.DiagnosticError, match="version"):
        diagnostic.parse_tool_version("\n\n")


def test_input_manifest_binds_every_execution_critical_source_file() -> None:
    assert tuple(diagnostic.BOUND_SOURCE_FILES) == (
        "diagnostic_runner",
        "ausc_init",
        "tracking_init",
        "canonical",
        "detector",
        "ekf",
        "experimental_visibility_geometry",
        "flow",
        "geometry",
        "intrinsics",
        "model_tracker",
        "official_asset_contract",
        "official_openarm_experiment",
        "offline_mesh_simulation",
        "offline_official_mesh_assets",
        "offline_official_mesh_experiment",
        "rgbd_odometry",
        "runtime",
        "tracker",
    )
    for path in diagnostic.BOUND_SOURCE_FILES.values():
        assert path.is_file()
        assert path.is_relative_to(diagnostic.WORKSPACE_ROOT)


def test_bound_source_verifier_rejects_any_post_run_source_drift() -> None:
    manifest = diagnostic._bound_source_manifest()
    payload = {
        "schema": diagnostic.INPUT_SCHEMA,
        "bound_source_files": manifest,
        "diagnostic_script_sha256": manifest["diagnostic_runner"]["sha256"],
    }
    diagnostic.validate_bound_source_inputs(payload)

    first = next(iter(payload["bound_source_files"]))
    payload["bound_source_files"][first]["sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError, match="source binding"):
        diagnostic.validate_bound_source_inputs(payload)


def test_bound_source_verifier_rejects_forged_persisted_runner_sha() -> None:
    manifest = diagnostic._bound_source_manifest()
    payload = {
        "schema": diagnostic.INPUT_SCHEMA,
        "bound_source_files": manifest,
        "diagnostic_script_sha256": "0" * 64,
    }

    with pytest.raises(diagnostic.DiagnosticError, match="diagnostic runner"):
        diagnostic.validate_bound_source_inputs(payload)

    payload["diagnostic_script_sha256"] = manifest["diagnostic_runner"]["sha256"]
    diagnostic.validate_bound_source_inputs(payload)


def test_input_schema_is_required_by_both_persisted_verifiers() -> None:
    row = {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "phase": "approach",
        "baseline_initialized": True,
        "candidate_initialized": True,
        "baseline_status": "TRACKING",
        "candidate_raw_status": "TRACKING",
        "baseline_measured": True,
        "candidate_raw_measured": True,
        "candidate_status": "MEASURED",
        "target_robot_mask_5x5": False,
        "baseline_error_mm": 2.0,
        "candidate_raw_error_mm": 3.0,
        "candidate_error_mm": 3.0,
    }
    metrics = diagnostic.base_metrics()
    metrics.update(diagnostic.derive_row_metrics([row]))
    inputs = _evidence_inputs([row])
    inputs["schema"] = diagnostic.INPUT_SCHEMA
    csv_raw = diagnostic.per_frame_csv_bytes([row])
    diagnostic.verify_metrics_against_csv(metrics, csv_raw, inputs)

    manifest = diagnostic._bound_source_manifest()
    source_inputs = {
        "schema": diagnostic.INPUT_SCHEMA,
        "bound_source_files": manifest,
        "diagnostic_script_sha256": manifest["diagnostic_runner"]["sha256"],
    }
    diagnostic.validate_bound_source_inputs(source_inputs)

    for invalid in (None, "wam_offline_robot_mask_diagnostic_inputs_v2", 3):
        forged_evidence = dict(inputs)
        forged_sources = dict(source_inputs)
        if invalid is None:
            forged_evidence.pop("schema")
            forged_sources.pop("schema")
        else:
            forged_evidence["schema"] = invalid
            forged_sources["schema"] = invalid
        with pytest.raises(diagnostic.DiagnosticError, match="schema"):
            diagnostic.verify_metrics_against_csv(metrics, csv_raw, forged_evidence)
        with pytest.raises(diagnostic.DiagnosticError, match="schema"):
            diagnostic.validate_bound_source_inputs(forged_sources)


def test_timeline_is_12_init_then_official_20hz_with_ten_frame_contact_hold() -> None:
    trajectory_q = np.arange(121 * 7, dtype=np.float64).reshape(121, 7)
    timeline = diagnostic.build_timeline(trajectory_q)

    assert len(timeline) == 12 + 121 + 10
    np.testing.assert_array_equal(
        np.asarray([row.timestamp_s for row in timeline]),
        np.arange(len(timeline), dtype=np.float64) * 0.05,
    )
    assert [row.phase for row in timeline[:12]] == ["init_home"] * 12
    assert [row.phase for row in timeline[12:73]] == ["approach"] * 61
    assert [row.phase for row in timeline[73:83]] == ["contact_hold"] * 10
    assert [row.phase for row in timeline[83:]] == ["depart"] * 60
    for row in timeline[:12]:
        np.testing.assert_array_equal(row.left_joint_positions_rad, diagnostic.HOME_JOINT_POSITIONS_RAD)
    for row in timeline[73:83]:
        np.testing.assert_array_equal(row.left_joint_positions_rad, trajectory_q[60])
    np.testing.assert_array_equal(timeline[72].left_joint_positions_rad, trajectory_q[60])
    np.testing.assert_array_equal(timeline[83].left_joint_positions_rad, trajectory_q[61])


def test_oracle_2d_noise_is_deterministic_and_does_not_mutate_projection() -> None:
    projected = np.arange(20, dtype=np.float64).reshape(10, 2) + 100.0
    original = projected.copy()

    first = diagnostic.noisy_oracle_pixels(projected, frame_index=37)
    second = diagnostic.noisy_oracle_pixels(projected, frame_index=37)
    other = diagnostic.noisy_oracle_pixels(projected, frame_index=38)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    np.testing.assert_array_equal(projected, original)
    assert np.max(np.abs(first - projected)) <= 2.5 * diagnostic.ORACLE_NOISE_STD_PX


def test_patient_relative_coast_ignores_raw_tracker_coast_and_fails_on_age_bound() -> None:
    coast = diagnostic.PatientRelativeBoundedCoast(max_age_s=0.40, max_bound_m=0.010)
    world_from_patient = np.eye(4)
    world_from_camera = np.eye(4)

    valid = coast.step(
        timestamp_s=0.0,
        sample=_point((0.1, 0.2, 1.0), measured=True),
        world_from_patient=world_from_patient,
        world_from_cv_camera=world_from_camera,
    )
    assert valid.status == "MEASURED"
    assert valid.valid is True

    moved_patient = np.eye(4)
    moved_patient[0, 3] = 0.004
    bounded = coast.step(
        timestamp_s=0.20,
        sample=_point((9.0, 9.0, 9.0), measured=False, position_std=0.009),
        world_from_patient=moved_patient,
        world_from_cv_camera=world_from_camera,
    )
    assert bounded.status == "PATIENT_RELATIVE_COAST"
    assert bounded.valid is True
    np.testing.assert_allclose(bounded.xyz_camera_m, [0.104, 0.2, 1.0])
    assert not np.allclose(bounded.xyz_camera_m, [9.0, 9.0, 9.0])

    lost = coast.step(
        timestamp_s=0.45,
        sample=_point((8.0, 8.0, 8.0), measured=False),
        world_from_patient=moved_patient,
        world_from_cv_camera=world_from_camera,
    )
    assert lost.status == "LOST"
    assert lost.valid is False
    assert lost.xyz_camera_m is None
    assert lost.reason == "age_bound_exceeded"


def test_patient_relative_coast_fails_on_uncertainty_bound_before_age() -> None:
    coast = diagnostic.PatientRelativeBoundedCoast(
        max_age_s=0.40,
        max_bound_m=0.010,
        drift_bound_m_s=0.030,
    )
    identity = np.eye(4)
    coast.step(
        timestamp_s=0.0,
        sample=_point(measured=True, position_std=0.006),
        world_from_patient=identity,
        world_from_cv_camera=identity,
    )
    output = coast.step(
        timestamp_s=0.20,
        sample=_point(measured=False),
        world_from_patient=identity,
        world_from_cv_camera=identity,
    )

    assert output.status == "LOST"
    assert output.reason == "uncertainty_bound_exceeded"
    assert output.bound_m == pytest.approx(0.012)


def test_csv_serialization_is_byte_deterministic_and_uses_fixed_field_order() -> None:
    rows = [
        {
            "frame_index": 1,
            "timestamp_s": 0.05,
            "phase": "approach",
            "candidate_status": "MEASURED",
        },
        {
            "frame_index": 2,
            "timestamp_s": 0.10,
            "phase": "approach",
            "candidate_status": "PATIENT_RELATIVE_COAST",
        },
    ]

    first = diagnostic.per_frame_csv_bytes(rows)
    second = diagnostic.per_frame_csv_bytes(list(reversed(list(reversed(rows)))))

    assert first == second
    parsed = list(csv.DictReader(io.StringIO(first.decode("utf-8"))))
    assert list(parsed[0]) == list(diagnostic.PER_FRAME_FIELDS)
    assert parsed[1]["candidate_status"] == "PATIENT_RELATIVE_COAST"


def test_fail_closed_output_retains_only_staging_warning(tmp_path: Path) -> None:
    destination = tmp_path / "diagnostic"
    destination.mkdir()
    (destination / "STAGING_WARNING").write_text("not complete\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in diagnostic.REQUIRED_ARTIFACTS:
        (scratch / name).write_bytes(b"placeholder")

    with pytest.raises(diagnostic.DiagnosticError, match="meaningful"):
        diagnostic.publish_nonrelease_artifacts(
            scratch=scratch,
            destination=destination,
            meaningful=False,
        )

    assert [path.name for path in destination.iterdir()] == ["STAGING_WARNING"]


def test_nonrelease_artifact_publish_has_hashes_but_no_completion_marker(tmp_path: Path) -> None:
    destination = tmp_path / "diagnostic"
    destination.mkdir()
    (destination / "STAGING_WARNING").write_text("not complete\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in diagnostic.REQUIRED_ARTIFACTS:
        (scratch / name).write_bytes(("artifact:" + name).encode("utf-8"))

    diagnostic.publish_nonrelease_artifacts(
        scratch=scratch,
        destination=destination,
        meaningful=True,
    )

    names = {path.name for path in destination.iterdir()}
    assert names == set(diagnostic.REQUIRED_ARTIFACTS) | {"STAGING_WARNING", "SHA256SUMS"}
    assert not any("COMPLETE" in name or name == "current" for name in names)
    sums = (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in sums] == sorted(diagnostic.REQUIRED_ARTIFACTS)


def test_metrics_contract_rejects_any_success_or_production_claim() -> None:
    metrics = diagnostic.base_metrics()
    diagnostic.validate_metrics_contract(metrics)

    for key in ("completion", "production_activation", "real_v9_executed"):
        forged = json.loads(json.dumps(metrics))
        forged[key] = True
        with pytest.raises(diagnostic.DiagnosticError, match=key):
            diagnostic.validate_metrics_contract(forged)


@pytest.mark.parametrize(
    ("key", "forged_value"),
    (
        ("simulator", "Isaac"),
        ("isaac_equivalence", True),
        ("isaac_surface_equivalence", True),
        ("human_pose", "animated"),
        ("human_rest_pose", False),
        ("robot_mask_provider_commissioned", True),
        ("mask_false_negative_sweep_executed", True),
        ("stable_wrong_surface_defense_validated", True),
        ("production_robot_mask_call_path_executed", True),
        ("patient_transform_source", "production_tracker"),
        ("patient_transform_method_sensitivity_upper_bound_only", False),
        ("patient_transform_available_in_production", True),
        ("simulator_patient_transform_used_by_bounded_wrapper", False),
        ("recommendation_supported", True),
        ("outcome_class", "POSITIVE"),
    ),
)
def test_metrics_contract_rejects_forged_scope_and_safety_fields(
    key: str, forged_value: object
) -> None:
    metrics = diagnostic.base_metrics()
    metrics[key] = forged_value

    with pytest.raises(diagnostic.DiagnosticError, match=key):
        diagnostic.validate_metrics_contract(metrics)


def test_lifecycle_metrics_ignore_preinit_lost_and_report_postinit_reset() -> None:
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "candidate_initialized": False,
            "candidate_raw_measured": False,
            "candidate_status": "LOST",
        },
        {
            "frame_index": 1,
            "timestamp_s": 0.10,
            "candidate_initialized": True,
            "candidate_raw_measured": True,
            "candidate_status": "MEASURED",
        },
        {
            "frame_index": 2,
            "timestamp_s": 0.15,
            "candidate_initialized": True,
            "candidate_raw_measured": False,
            "candidate_status": "PATIENT_RELATIVE_COAST",
        },
        {
            "frame_index": 3,
            "timestamp_s": 0.55,
            "candidate_initialized": True,
            "candidate_raw_measured": False,
            "candidate_status": "LOST",
        },
        {
            "frame_index": 4,
            "timestamp_s": 0.60,
            "candidate_initialized": False,
            "candidate_raw_measured": False,
            "candidate_status": "LOST",
        },
    ]

    lifecycle = diagnostic.candidate_lifecycle_metrics(rows)

    assert lifecycle["candidate_ever_initialized"] is True
    assert lifecycle["candidate_first_initialized_frame_index"] == 1
    assert lifecycle["candidate_final_initialized"] is False
    assert lifecycle["candidate_tracker_reset_frame_index"] == 4
    assert lifecycle["candidate_tracker_reinitialized_frame_index"] is None
    assert lifecycle["candidate_first_lost_age_from_last_measurement_s"] == pytest.approx(0.45)
    assert lifecycle["candidate_display_lost_after_first_initialization_frame_count"] == 2
    assert lifecycle["candidate_not_measured_after_first_initialization_frame_count"] == 3


def test_row_metrics_use_after_both_initialized_denominator_and_partition_statuses() -> None:
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "baseline_initialized": False,
            "candidate_initialized": False,
            "candidate_raw_measured": False,
            "baseline_status": "INIT",
            "candidate_raw_status": "INIT",
            "candidate_status": "LOST",
            "baseline_error_mm": None,
            "candidate_raw_error_mm": None,
            "candidate_error_mm": None,
        },
        {
            "frame_index": 1,
            "timestamp_s": 0.10,
            "baseline_initialized": True,
            "candidate_initialized": True,
            "candidate_raw_measured": True,
            "baseline_status": "TRACKING",
            "candidate_raw_status": "TRACKING",
            "candidate_status": "MEASURED",
            "baseline_error_mm": 4.0,
            "candidate_raw_error_mm": 4.0,
            "candidate_error_mm": 4.0,
        },
        {
            "frame_index": 2,
            "timestamp_s": 0.15,
            "baseline_initialized": True,
            "candidate_initialized": True,
            "candidate_raw_measured": False,
            "baseline_status": "TRACKING",
            "candidate_raw_status": "COASTING",
            "candidate_status": "PATIENT_RELATIVE_COAST",
            "baseline_error_mm": 11.0,
            "candidate_raw_error_mm": 7.0,
            "candidate_error_mm": 6.0,
        },
        {
            "frame_index": 3,
            "timestamp_s": 0.55,
            "baseline_initialized": True,
            "candidate_initialized": True,
            "candidate_raw_measured": False,
            "baseline_status": "COASTING",
            "candidate_raw_status": "LOST",
            "candidate_status": "LOST",
            "baseline_error_mm": 3.0,
            "candidate_raw_error_mm": 8.0,
            "candidate_error_mm": None,
        },
        {
            "frame_index": 4,
            "timestamp_s": 0.60,
            "baseline_initialized": True,
            "candidate_initialized": False,
            "candidate_raw_measured": False,
            "baseline_status": "TRACKING",
            "candidate_raw_status": "INIT",
            "candidate_status": "LOST",
            "baseline_error_mm": 2.0,
            "candidate_raw_error_mm": None,
            "candidate_error_mm": None,
        },
    ]
    for row in rows:
        row["target_robot_mask_5x5"] = False

    metrics = diagnostic.derive_row_metrics(rows)

    assert metrics["result_window_first_frame_index"] == 1
    assert metrics["result_window_frame_count"] == 4
    assert metrics["preinitialization_display_lost_frame_count"] == 1
    assert metrics["raw_tracker_result_window_frame_count"] == 4
    assert metrics["raw_mask_unmeasured_frame_count"] == 3
    assert metrics["raw_mask_unmeasured_while_tracker_initialized_frame_count"] == 2
    assert metrics["raw_no_mask_valid_frame_count"] == 4
    assert metrics["raw_mask_valid_frame_count"] == 2
    assert metrics["raw_no_mask_valid_coverage"] == pytest.approx(1.0)
    assert metrics["raw_mask_valid_coverage"] == pytest.approx(0.5)
    assert metrics["raw_common_valid_frame_count"] == 2
    assert metrics["raw_common_valid_coverage"] == pytest.approx(0.5)
    assert metrics["raw_common_no_mask_error_median_mm"] == pytest.approx(7.5)
    assert metrics["raw_common_mask_error_median_mm"] == pytest.approx(5.5)
    assert metrics["display_measured_frame_count"] == 1
    assert metrics["display_bounded_coast_frame_count"] == 1
    assert metrics["display_lost_frame_count"] == 2
    assert metrics["display_valid_coverage"] == pytest.approx(0.5)
    assert metrics["raw_no_mask_false_valid_gt_5mm_rate"] == pytest.approx(0.25)
    assert metrics["raw_no_mask_false_valid_gt_10mm_rate"] == pytest.approx(0.25)
    assert metrics["raw_mask_false_valid_gt_5mm_rate"] == pytest.approx(0.5)
    assert metrics["raw_mask_false_valid_gt_10mm_rate"] == pytest.approx(0.0)
    assert metrics["display_false_valid_gt_5mm_rate"] == pytest.approx(0.5)
    assert metrics["display_false_valid_gt_10mm_rate"] == pytest.approx(0.0)
    assert metrics["candidate_first_lost_age_from_last_measurement_s"] == pytest.approx(0.45)
    assert (
        metrics["display_measured_frame_count"]
        + metrics["display_bounded_coast_frame_count"]
        + metrics["display_lost_frame_count"]
        == metrics["result_window_frame_count"]
    )


def test_negative_outcome_gate_requires_overlap_reset_and_no_recovery() -> None:
    result = {
        "baseline_initialized": True,
        "candidate_initialized": True,
        "target_mask_raw_dropout_overlap_frame_count": 4,
        "target_mask_raw_dropout_overlap_while_initialized_frame_count": 2,
        "display_bounded_coast_frame_count": 3,
        "display_lost_frame_count": 7,
        "candidate_tracker_reset_frame_index": 10,
        "candidate_tracker_reinitialized_frame_index": None,
        "candidate_final_initialized": False,
    }

    assert diagnostic.negative_reset_no_recovery_is_meaningful(result) is True
    for key, invalid in (
        ("target_mask_raw_dropout_overlap_frame_count", 0),
        ("target_mask_raw_dropout_overlap_frame_count", None),
        ("target_mask_raw_dropout_overlap_frame_count", "1"),
        ("target_mask_raw_dropout_overlap_while_initialized_frame_count", 0),
        ("display_bounded_coast_frame_count", 0),
        ("display_lost_frame_count", 0),
        ("candidate_tracker_reset_frame_index", None),
        ("candidate_tracker_reset_frame_index", True),
        ("candidate_tracker_reinitialized_frame_index", 20),
        ("candidate_final_initialized", True),
    ):
        forged = dict(result)
        forged[key] = invalid
        assert diagnostic.negative_reset_no_recovery_is_meaningful(forged) is False


def test_metrics_contract_accepts_only_observed_negative_outcome() -> None:
    metrics = diagnostic.base_metrics()
    metrics.update(
        {
            "baseline_initialized": True,
            "candidate_initialized": True,
            "target_mask_raw_dropout_overlap_frame_count": 4,
            "target_mask_raw_dropout_overlap_while_initialized_frame_count": 2,
            "display_bounded_coast_frame_count": 3,
            "display_lost_frame_count": 7,
            "candidate_tracker_reset_frame_index": 10,
            "candidate_tracker_reinitialized_frame_index": None,
            "candidate_final_initialized": False,
            "meaningful_diagnostic": True,
            "negative_reset_no_recovery_observed": True,
        }
    )
    diagnostic.validate_metrics_contract(metrics)

    metrics["candidate_tracker_reinitialized_frame_index"] = 20
    with pytest.raises(diagnostic.DiagnosticError, match="negative reset/no-recovery"):
        diagnostic.validate_metrics_contract(metrics)


def test_every_overlay_frame_carries_rest_pose_and_negative_outcome_banners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered_text: list[str] = []

    def capture_text(frame, text, *args, **kwargs):
        rendered_text.append(text)
        return frame

    monkeypatch.setattr(diagnostic.cv2, "putText", capture_text)
    intrinsics = diagnostic.Intrinsics(
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=180.0,
        width=640,
        height=360,
        depth_scale=1000.0,
    )
    diagnostic._overlay_frame(
        rgb=np.zeros((360, 640, 3), dtype=np.uint8),
        robot_mask=np.zeros((360, 640), dtype=bool),
        oracle_pixels=np.full((10, 2), np.nan),
        truth_pixel=np.full(2, np.nan),
        baseline_xyz=None,
        candidate=diagnostic.BoundedTargetOutput(
            status="LOST",
            valid=False,
            xyz_camera_m=None,
            reason="test",
            age_s=None,
            bound_m=None,
        ),
        intrinsics=intrinsics,
        phase="approach",
        timestamp_s=1.0,
    )

    assert diagnostic.BANNER in rendered_text
    assert diagnostic.METHOD_BANNER in rendered_text
    assert diagnostic.HUMAN_BANNER in rendered_text
    assert diagnostic.OUTCOME_BANNER in rendered_text


def test_readme_declares_negative_result_and_conditional_noncomparable_errors() -> None:
    metrics = {
        "nominal_home_exact_mask_blocked_points_first_frame": ["R4", "R5"],
        "nominal_home_candidate_depth_association_point_count_first_frame": 8,
        "candidate_first_lost_age_from_last_measurement_s": 0.45,
        "baseline_initialized": True,
        "candidate_initialized": True,
        "candidate_final_initialized": False,
        "candidate_tracker_reset_frame_index": 67,
        "candidate_tracker_reinitialized_frame_index": None,
        "target_robot_mask_frame_count": 92,
        "result_window_frame_count": 132,
        "raw_mask_unmeasured_frame_count": 112,
        "raw_mask_unmeasured_while_tracker_initialized_frame_count": 36,
        "target_mask_raw_dropout_overlap_frame_count": 92,
        "target_mask_raw_dropout_overlap_while_initialized_frame_count": 36,
        "display_bounded_coast_frame_count": 12,
        "display_lost_frame_count": 100,
        "preinitialization_display_lost_frame_count": 11,
        "raw_no_mask_valid_coverage": 1.0,
        "raw_mask_valid_coverage": 0.333333,
        "raw_common_valid_coverage": 0.333333,
        "display_valid_coverage": 0.242424,
        "raw_no_mask_error_p95_mm": 10.03,
        "raw_mask_error_p95_mm": 6.0,
        "raw_common_no_mask_error_p95_mm": 8.0,
        "raw_common_mask_error_p95_mm": 6.0,
        "display_error_p95_mm": 5.05,
        "raw_no_mask_false_valid_gt_5mm_rate": 0.47,
        "raw_no_mask_false_valid_gt_10mm_rate": 0.053,
        "raw_mask_false_valid_gt_5mm_rate": 0.20,
        "raw_mask_false_valid_gt_10mm_rate": 0.01,
        "display_false_valid_gt_5mm_rate": 0.094,
        "display_false_valid_gt_10mm_rate": 0.0,
        "outcome_class": "NEGATIVE_RESET_NO_RECOVERY",
        "recommendation_supported": False,
    }

    readme = diagnostic._readme(metrics)

    assert "NEGATIVE_RESET_NO_RECOVERY" in readme
    assert "recommendation_supported=false" in readme
    assert "default_equals_rest" in readme
    assert "common-valid" in readme
    assert "conditional" in readme
    assert "must not be interpreted as an improvement" in readme


def test_publish_verifier_reparses_csv_and_rejects_metric_drift() -> None:
    rows = []
    for index, status in enumerate(("MEASURED", "PATIENT_RELATIVE_COAST", "LOST")):
        rows.append(
            {
                "frame_index": index,
                "timestamp_s": index * 0.05,
                "phase": "approach",
                "baseline_initialized": True,
                "candidate_initialized": True,
                "baseline_status": "TRACKING",
                "candidate_raw_status": (
                    "TRACKING"
                    if status == "MEASURED"
                    else "COASTING"
                    if status == "PATIENT_RELATIVE_COAST"
                    else "LOST"
                ),
                "candidate_raw_measured": status == "MEASURED",
                "candidate_status": status,
                "target_robot_mask_5x5": False,
                "baseline_error_mm": 2.0,
                "candidate_raw_error_mm": (
                    None if status == "LOST" else 3.0
                ),
                "candidate_error_mm": None if status == "LOST" else 3.0,
                "frame_evidence_sha256": f"{index + 1:064x}",
            }
        )
    inputs = _evidence_inputs(rows)
    csv_bytes = diagnostic.per_frame_csv_bytes(rows)
    metrics = diagnostic.base_metrics()
    metrics.update(diagnostic.derive_row_metrics(rows))
    metrics["frame_count"] = len(rows)

    diagnostic.verify_metrics_against_csv(metrics, csv_bytes, inputs)
    metrics["raw_mask_unmeasured_frame_count"] = 99
    with pytest.raises(diagnostic.DiagnosticError, match="raw_mask_unmeasured_frame_count"):
        diagnostic.verify_metrics_against_csv(metrics, csv_bytes, inputs)


def test_publish_verifier_rejects_frame_digest_or_stream_digest_drift() -> None:
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "phase": "approach",
            "baseline_initialized": True,
            "candidate_initialized": True,
            "baseline_status": "TRACKING",
            "candidate_raw_status": "TRACKING",
            "baseline_measured": True,
            "candidate_raw_measured": True,
            "candidate_status": "MEASURED",
            "target_robot_mask_5x5": False,
            "baseline_error_mm": 2.0,
            "candidate_raw_error_mm": 3.0,
            "candidate_error_mm": 3.0,
            "frame_evidence_sha256": "1" * 64,
        }
    ]
    metrics = diagnostic.base_metrics()
    metrics.update(diagnostic.derive_row_metrics(rows))
    inputs = _evidence_inputs(rows)
    csv_bytes = diagnostic.per_frame_csv_bytes(rows)
    diagnostic.verify_metrics_against_csv(metrics, csv_bytes, inputs)

    forged_inputs = dict(inputs)
    forged_inputs["rendered_evidence_stream_sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError, match="evidence stream"):
        diagnostic.verify_metrics_against_csv(metrics, csv_bytes, forged_inputs)

    forged_raw_inputs = dict(inputs)
    forged_raw_inputs["raw_rendered_input_stream_sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError, match="raw rendered-input stream"):
        diagnostic.verify_metrics_against_csv(metrics, csv_bytes, forged_raw_inputs)

    forged_rows = [dict(rows[0], frame_evidence_sha256="2" * 64)]
    with pytest.raises(diagnostic.DiagnosticError, match="evidence"):
        diagnostic.verify_metrics_against_csv(
            metrics,
            diagnostic.per_frame_csv_bytes(forged_rows),
            inputs,
        )


def test_publish_verifier_rejects_unbound_semantic_row_drift() -> None:
    row = {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "phase": "approach",
        "trajectory_index": 0,
        "oracle_u_px": 123.0,
        "oracle_v_px": 87.0,
        "baseline_initialized": True,
        "candidate_initialized": True,
        "baseline_status": "TRACKING",
        "candidate_raw_status": "TRACKING",
        "baseline_measured": True,
        "candidate_raw_measured": True,
        "candidate_status": "MEASURED",
        "target_robot_mask_5x5": False,
        "baseline_error_mm": 2.0,
        "candidate_raw_error_mm": 3.0,
        "candidate_error_mm": 3.0,
        "frame_evidence_sha256": "1" * 64,
    }
    metrics = diagnostic.base_metrics()
    metrics.update(diagnostic.derive_row_metrics([row]))
    inputs = _evidence_inputs([row])
    diagnostic.verify_metrics_against_csv(
        metrics, diagnostic.per_frame_csv_bytes([row]), inputs
    )

    forged_row = dict(row, oracle_u_px=999.0)
    with pytest.raises(diagnostic.DiagnosticError, match="semantics"):
        diagnostic.verify_metrics_against_csv(
            metrics,
            diagnostic.per_frame_csv_bytes([forged_row]),
            inputs,
        )


def test_raw_point_extraction_never_scores_init_or_lost_coordinates() -> None:
    finite = np.asarray([0.1, 0.2, 1.0], dtype=np.float64)

    for status in ("INIT", "LOST"):
        point = SimpleNamespace(status=status, xyz_camera=finite)
        assert diagnostic._xyz_or_none(point, initialized=True) is None

    for status in ("TRACKING", "COASTING"):
        point = SimpleNamespace(status=status, xyz_camera=finite)
        np.testing.assert_array_equal(
            diagnostic._xyz_or_none(point, initialized=True), finite
        )


def test_publish_verifier_rejects_finite_scored_coordinates_for_lost_point() -> None:
    row = {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "phase": "approach",
        "baseline_initialized": True,
        "candidate_initialized": True,
        "baseline_measured": True,
        "candidate_raw_measured": False,
        "baseline_status": "TRACKING",
        "candidate_raw_status": "LOST",
        "candidate_status": "LOST",
        "target_robot_mask_5x5": True,
        "gt_x_camera_m": 0.1,
        "gt_y_camera_m": 0.2,
        "gt_z_camera_m": 1.0,
        "baseline_x_camera_m": 0.1,
        "baseline_y_camera_m": 0.2,
        "baseline_z_camera_m": 1.0,
        "baseline_error_mm": 0.0,
        "candidate_raw_x_camera_m": 0.1,
        "candidate_raw_y_camera_m": 0.2,
        "candidate_raw_z_camera_m": 1.0,
        "candidate_raw_error_mm": 0.0,
    }
    metrics = diagnostic.base_metrics()
    metrics.update(diagnostic.derive_row_metrics([row]))
    inputs = _evidence_inputs([row])

    with pytest.raises(diagnostic.DiagnosticError, match="LOST.*empty"):
        diagnostic.verify_metrics_against_csv(
            metrics,
            diagnostic.per_frame_csv_bytes([row]),
            inputs,
        )


def test_ffmpeg_command_locks_h264_yuv420p_640x360_and_single_thread(tmp_path: Path) -> None:
    command = diagnostic.ffmpeg_command(tmp_path / "out.mp4")

    assert command[0] == "/opt/homebrew/bin/ffmpeg"
    assert "libx264" in command
    assert "yuv420p" in command
    assert "640x360" in command
    assert command[command.index("-r") + 1] == "20"
    assert command[command.index("-threads") + 1] == "1"
