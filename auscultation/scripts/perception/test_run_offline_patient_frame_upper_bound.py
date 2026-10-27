"""Contracts for the non-release patient-frame freeze upper-bound study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ausc.perception.tracking.offline_mesh_simulation import camera_pose_transform
from scripts.perception import run_offline_patient_frame_upper_bound as study


def _result_row(
    frame_index: int,
    *,
    camera_error_mm: float,
    patient_error_mm: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_s": frame_index * 0.05,
        "phase": "contact_hold",
        "trajectory_index": 60,
        "patient_translation_x_m": 0.001 * frame_index,
        "reference_ready": True,
        "target_robot_mask_5x5": True,
        "camera_lock_valid": True,
        "patient_lock_valid": True,
        "camera_lock_error_mm": camera_error_mm,
        "patient_lock_error_mm": patient_error_mm,
        "rendered_input_sha256": f"{frame_index + 101:064x}",
    }
    row["frame_evidence_sha256"] = study.semantic_frame_sha256(row)
    return row


def _evidence_stream(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(str(row["frame_evidence_sha256"])))
    return digest.hexdigest()


def _raw_rendered_input_stream(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(str(row["rendered_input_sha256"])))
    return digest.hexdigest()


def _full_experiment_rows() -> list[dict[str, object]]:
    timeline = study.build_timeline(np.zeros((121, 7), dtype=np.float64))
    world_from_camera = camera_pose_transform(study.CAMERA_POSE_NAME)
    rows: list[dict[str, object]] = []
    for frame in timeline:
        ready = frame.frame_index >= 11
        offset = frame.patient_translation_x_m
        camera_delta = world_from_camera[:3, :3].T @ np.asarray(
            [offset, 0.0, 0.0]
        )
        truth = np.asarray([0.1, 0.2, 1.0]) + camera_delta
        camera = np.asarray([0.1, 0.2, 1.0]) if ready else None
        patient = (
            np.asarray([0.1, 0.2, 1.0]) + camera_delta if ready else None
        )
        candidate = (
            np.asarray([0.1 + 0.001 * ((frame.frame_index % 3) - 1), 0.2, 1.0])
            if frame.phase == "confirm_home"
            else None
        )
        row: dict[str, object] = {
            "frame_index": frame.frame_index,
            "timestamp_s": frame.timestamp_s,
            "phase": frame.phase,
            "trajectory_index": frame.trajectory_index,
            "patient_speed_m_s": frame.patient_speed_m_s,
            "patient_translation_x_m": offset,
            "robot_pixel_count": 10,
            "target_robot_mask_5x5": False,
            "reference_candidate_valid": candidate is not None,
            "reference_candidate_x_camera_m": (
                None if candidate is None else candidate[0]
            ),
            "reference_candidate_y_camera_m": (
                None if candidate is None else candidate[1]
            ),
            "reference_candidate_z_camera_m": (
                None if candidate is None else candidate[2]
            ),
            "reference_ready": ready,
            "reference_confirmed_this_frame": frame.frame_index == 11,
            "oracle_u_px": 100.0,
            "oracle_v_px": 100.0,
            "gt_x_camera_m": truth[0],
            "gt_y_camera_m": truth[1],
            "gt_z_camera_m": truth[2],
            "camera_lock_valid": ready,
            "camera_lock_x_camera_m": None if camera is None else camera[0],
            "camera_lock_y_camera_m": None if camera is None else camera[1],
            "camera_lock_z_camera_m": None if camera is None else camera[2],
            "camera_lock_error_mm": (
                None if camera is None else np.linalg.norm(camera - truth) * 1000.0
            ),
            "patient_lock_valid": ready,
            "patient_lock_x_camera_m": None if patient is None else patient[0],
            "patient_lock_y_camera_m": None if patient is None else patient[1],
            "patient_lock_z_camera_m": None if patient is None else patient[2],
            "patient_lock_error_mm": (
                None if patient is None else np.linalg.norm(patient - truth) * 1000.0
            ),
            "rendered_input_sha256": f"{frame.frame_index + 1001:064x}",
        }
        row["frame_evidence_sha256"] = study.semantic_frame_sha256(row)
        rows.append(row)
    return rows


def test_contract_is_permanently_oracle_nonrelease_upper_bound() -> None:
    contract = study.static_contract()

    assert contract["schema"] == "wam_patient_frame_freeze_upper_bound_v1"
    assert contract["oracle_2d"] is True
    assert contract["real_v9_executed"] is False
    assert contract["production_activation"] is False
    assert contract["completion"] is False
    assert contract["simulator"] == "MuJoCo_kinematic_official_mesh_not_Isaac"
    assert contract["isaac_equivalence"] is False
    assert contract["official_openarm"] is True
    assert contract["official_nvidia_human_pose"] == "default_equals_rest"
    assert contract["patient_transform_source"] == "simulator_ground_truth"
    assert contract["patient_transform_available_in_production"] is False
    assert contract["reference_consensus_max_deviation_m"] == pytest.approx(0.015)
    assert contract["comparison"] == (
        "camera_frame_3d_freeze_vs_ideal_patient_frame_canonical_freeze"
    )
    assert contract["recommendation_supported"] is False
    assert study.BANNER == "ORACLE2D / NOT V9 / NOMINAL ROOT / NOT ISAAC"
    assert study.HUMAN_BANNER == "OFFICIAL NVIDIA REST"
    assert study.METHOD_BANNER == (
        "IDEAL SIM PATIENT TF / UPPER BOUND / NOT AVAILABLE PRODUCTION"
    )


def test_timeline_is_exactly_143_frames_and_motion_stays_within_bound() -> None:
    trajectory = np.arange(121 * 7, dtype=np.float64).reshape(121, 7)
    timeline = study.build_timeline(trajectory)

    assert len(timeline) == 143
    assert [row.phase for row in timeline[:12]] == ["confirm_home"] * 12
    assert [row.phase for row in timeline[12:73]] == ["approach"] * 61
    assert [row.phase for row in timeline[73:83]] == ["contact_hold"] * 10
    assert [row.phase for row in timeline[83:]] == ["depart"] * 60
    np.testing.assert_allclose(
        [row.timestamp_s for row in timeline], np.arange(143) * 0.05
    )
    assert max(row.patient_speed_m_s for row in timeline) == pytest.approx(0.012)
    assert max(row.patient_speed_m_s for row in timeline) <= 0.013
    assert timeline[82].patient_translation_x_m == pytest.approx(0.006)
    assert timeline[-1].patient_translation_x_m == pytest.approx(0.006)


def test_trusted_depth_lift_uses_rgbd_masks_and_never_anchor_truth() -> None:
    intrinsics = study.Intrinsics(
        fx=100.0,
        fy=100.0,
        cx=4.0,
        cy=4.0,
        width=9,
        height=9,
        depth_scale=1000.0,
    )
    depth_m = np.ones((9, 9), dtype=np.float64)
    human_mask = np.ones((9, 9), dtype=bool)
    robot_mask = np.zeros((9, 9), dtype=bool)

    lifted = study.trusted_depth_lift(
        pixel=np.asarray([4.0, 4.0]),
        depth_m=depth_m,
        robot_mask=robot_mask,
        human_mask=human_mask,
        intrinsics=intrinsics,
    )
    np.testing.assert_allclose(lifted, [0.0, 0.0, 1.0])

    robot_mask[2, 2] = True
    assert (
        study.trusted_depth_lift(
            pixel=np.asarray([4.0, 4.0]),
            depth_m=depth_m,
            robot_mask=robot_mask,
            human_mask=human_mask,
            intrinsics=intrinsics,
        )
        is None
    )

    robot_mask[:] = False
    human_mask[:] = False
    human_mask[4, 4] = True
    assert (
        study.trusted_depth_lift(
            pixel=np.asarray([4.0, 4.0]),
            depth_m=depth_m,
            robot_mask=robot_mask,
            human_mask=human_mask,
            intrinsics=intrinsics,
        )
        is None
    )


def test_freeze_comparator_uses_one_confirmation_then_separates_frames() -> None:
    comparator = study.FreezeComparator(required_confirmation_frames=3)
    identity = np.eye(4)
    samples = (
        np.asarray([0.10, 0.20, 1.00]),
        np.asarray([0.11, 0.20, 1.00]),
        np.asarray([0.09, 0.20, 1.00]),
    )

    assert comparator.observe(samples[0], identity, identity) is False
    assert comparator.observe(samples[1], identity, identity) is False
    assert comparator.observe(samples[2], identity, identity) is True
    assert comparator.reference_frame_index == 2

    moved_patient = np.eye(4)
    moved_patient[0, 3] = 0.006
    output = comparator.predict(moved_patient, identity)
    np.testing.assert_allclose(output.camera_lock_xyz_camera_m, [0.10, 0.20, 1.00])
    np.testing.assert_allclose(output.patient_lock_xyz_camera_m, [0.106, 0.20, 1.00])

    # A later sample cannot silently replace the fixed comparison reference.
    with pytest.raises(study.UpperBoundError, match="already frozen"):
        comparator.observe(np.asarray([9.0, 9.0, 9.0]), identity, identity)


def test_confirmation_accepts_declared_15mm_bound_but_rejects_larger_spread() -> None:
    identity = np.eye(4)
    accepted = study.FreezeComparator(required_confirmation_frames=3)
    assert accepted.observe(np.asarray([0.0, 0.0, 1.0]), identity, identity) is False
    assert accepted.observe(np.asarray([0.0128, 0.0, 1.0]), identity, identity) is False
    assert accepted.observe(np.asarray([0.0, 0.0, 1.0]), identity, identity) is True
    assert accepted.consensus_max_deviation_m == pytest.approx(0.0128)

    rejected = study.FreezeComparator(required_confirmation_frames=3)
    assert rejected.observe(np.asarray([0.0, 0.0, 1.0]), identity, identity) is False
    assert rejected.observe(np.asarray([0.0151, 0.0, 1.0]), identity, identity) is False
    with pytest.raises(study.UpperBoundError, match="spatially stable"):
        rejected.observe(np.asarray([0.0, 0.0, 1.0]), identity, identity)


def test_both_methods_are_derived_from_one_identical_physical_reference() -> None:
    comparator = study.FreezeComparator(required_confirmation_frames=3)
    patient = np.eye(4)
    camera = np.eye(4)
    angle = np.deg2rad(45.0)
    camera[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    samples = (
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.010, 0.0, 1.0]),
        np.asarray([0.0, 0.010, 1.0]),
    )
    for sample in samples:
        comparator.observe(sample, patient, camera)

    output = comparator.predict(patient, camera)
    np.testing.assert_allclose(
        output.camera_lock_xyz_camera_m,
        output.patient_lock_xyz_camera_m,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_metric_denominator_and_validity_policy_are_identical() -> None:
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "phase": "confirm_home",
            "trajectory_index": None,
            "patient_translation_x_m": 0.0,
            "reference_ready": False,
            "target_robot_mask_5x5": False,
            "camera_lock_valid": False,
            "patient_lock_valid": False,
            "camera_lock_error_mm": None,
            "patient_lock_error_mm": None,
            "frame_evidence_sha256": "0" * 64,
        },
        _result_row(1, camera_error_mm=2.0, patient_error_mm=2.0),
        _result_row(2, camera_error_mm=6.0, patient_error_mm=3.0),
        _result_row(3, camera_error_mm=12.0, patient_error_mm=4.0),
    ]

    metrics = study.derive_row_metrics(rows)

    assert metrics["result_window_first_frame_index"] == 1
    assert metrics["result_window_frame_count"] == 3
    assert metrics["common_denominator"] == "all_frames_at_or_after_reference_confirmation"
    assert metrics["camera_lock_valid_frame_count"] == 3
    assert metrics["patient_lock_valid_frame_count"] == 3
    assert metrics["camera_lock_coverage"] == pytest.approx(1.0)
    assert metrics["patient_lock_coverage"] == pytest.approx(1.0)
    assert metrics["camera_lock_error_p50_mm"] == pytest.approx(6.0)
    assert metrics["camera_lock_error_p95_mm"] == pytest.approx(11.4)
    assert metrics["patient_lock_error_p95_mm"] == pytest.approx(3.9)
    assert metrics["camera_lock_false_valid_gt_5mm_count"] == 2
    assert metrics["camera_lock_false_valid_gt_10mm_count"] == 1
    assert metrics["patient_lock_false_valid_gt_5mm_count"] == 0
    assert metrics["patient_lock_false_valid_gt_10mm_count"] == 0


def test_motion_and_pose_delay_sensitivity_are_small_declared_sweeps() -> None:
    reference_patient = np.asarray([0.1, 0.2, 1.0])
    reference_camera = reference_patient.copy()
    identity = np.eye(4)

    motion = study.motion_sensitivity(
        reference_patient_xyz_m=reference_patient,
        reference_camera_xyz_m=reference_camera,
        world_from_cv_camera=identity,
    )
    assert [row["patient_speed_m_s"] for row in motion] == [0.0, 0.006, 0.012, 0.013]
    assert all(row["patient_speed_m_s"] <= 0.013 for row in motion)
    assert motion[-1]["final_camera_lock_error_mm"] == pytest.approx(6.5)
    assert motion[-1]["final_ideal_patient_lock_error_mm"] == pytest.approx(0.0)

    delay = study.pose_delay_sensitivity(
        reference_patient_xyz_m=reference_patient,
        world_from_cv_camera=identity,
    )
    assert [row["delay_s"] for row in delay] == [0.0, 0.05, 0.1, 0.2]
    assert delay[0]["max_error_mm"] == pytest.approx(0.0)
    assert delay[-1]["max_error_mm"] == pytest.approx(2.4)


def test_csv_reverse_verifier_rejects_metric_and_evidence_drift() -> None:
    rows = [
        _result_row(0, camera_error_mm=2.0, patient_error_mm=2.0),
        _result_row(1, camera_error_mm=6.0, patient_error_mm=3.0),
    ]
    csv_raw = study.per_frame_csv_bytes(rows)
    parsed = list(csv.DictReader(io.StringIO(csv_raw.decode("utf-8"))))
    assert list(parsed[0]) == list(study.PER_FRAME_FIELDS)
    metrics = study.base_metrics()
    metrics.update(study.derive_row_metrics(rows))
    inputs = {
        "raw_rendered_input_stream_sha256": _raw_rendered_input_stream(rows),
        "rendered_evidence_stream_sha256": _evidence_stream(rows),
    }
    study.verify_metrics_against_csv(metrics, csv_raw, inputs)

    forged = json.loads(json.dumps(metrics))
    forged["camera_lock_error_p95_mm"] = 999.0
    with pytest.raises(study.UpperBoundError, match="camera_lock_error_p95_mm"):
        study.verify_metrics_against_csv(forged, csv_raw, inputs)

    inputs["rendered_evidence_stream_sha256"] = "f" * 64
    with pytest.raises(study.UpperBoundError, match="evidence stream"):
        study.verify_metrics_against_csv(metrics, csv_raw, inputs)


def test_frame_evidence_binds_every_persisted_row_semantic() -> None:
    row = _result_row(0, camera_error_mm=2.0, patient_error_mm=2.0)
    row.update(
        {
            "rendered_input_sha256": "a" * 64,
            "gt_x_camera_m": 0.1,
            "camera_lock_x_camera_m": 0.1,
            "patient_lock_x_camera_m": 0.1,
        }
    )
    row["frame_evidence_sha256"] = study.semantic_frame_sha256(row)
    rows = [row]
    csv_raw = study.per_frame_csv_bytes(rows)
    metrics = study.base_metrics()
    metrics.update(study.derive_row_metrics(rows))
    inputs = {
        "raw_rendered_input_stream_sha256": hashlib.sha256(
            bytes.fromhex(str(row["rendered_input_sha256"]))
        ).hexdigest(),
        "rendered_evidence_stream_sha256": _evidence_stream(rows),
    }
    study.verify_metrics_against_csv(metrics, csv_raw, inputs)

    for key, forged_value in (
        ("phase", "forged_phase"),
        ("gt_x_camera_m", "-999.000000000"),
        ("camera_lock_x_camera_m", "999.000000000"),
    ):
        parsed = list(csv.DictReader(io.StringIO(csv_raw.decode("utf-8"))))
        parsed[0][key] = forged_value
        forged_csv = study.per_frame_csv_bytes(parsed)
        with pytest.raises(study.UpperBoundError, match="frame evidence"):
            study.verify_metrics_against_csv(metrics, forged_csv, inputs)


def test_full_experiment_verifier_enforces_timeline_and_geometry_invariants() -> None:
    rows = _full_experiment_rows()
    assert float(rows[82]["gt_x_camera_m"]) < float(rows[11]["gt_x_camera_m"])
    study.validate_full_experiment_rows(rows)

    attacks = (
        (82, "phase", "forged_phase", "timeline"),
        (20, "timestamp_s", 99.0, "timeline"),
        (10, "reference_ready", True, "reference"),
        (12, "reference_confirmed_this_frame", True, "reference"),
        (82, "camera_lock_error_mm", 999.0, "error"),
        (82, "patient_lock_x_camera_m", None, "validity"),
    )
    for index, key, value, match in attacks:
        forged = [dict(row) for row in rows]
        forged[index][key] = value
        forged[index]["frame_evidence_sha256"] = study.semantic_frame_sha256(
            forged[index]
        )
        with pytest.raises(study.UpperBoundError, match=match):
            study.validate_full_experiment_rows(forged)

    # Even a self-consistent forged estimate cannot turn the camera freeze into motion.
    forged = [dict(row) for row in rows]
    forged[82]["camera_lock_x_camera_m"] = 0.101
    truth = np.asarray(
        [
            forged[82]["gt_x_camera_m"],
            forged[82]["gt_y_camera_m"],
            forged[82]["gt_z_camera_m"],
        ],
        dtype=np.float64,
    )
    estimate = np.asarray(
        [
            forged[82]["camera_lock_x_camera_m"],
            forged[82]["camera_lock_y_camera_m"],
            forged[82]["camera_lock_z_camera_m"],
        ],
        dtype=np.float64,
    )
    forged[82]["camera_lock_error_mm"] = np.linalg.norm(estimate - truth) * 1000.0
    forged[82]["frame_evidence_sha256"] = study.semantic_frame_sha256(forged[82])
    with pytest.raises(study.UpperBoundError, match="camera freeze"):
        study.validate_full_experiment_rows(forged)

    # A false validity flag cannot hide a complete published xyz.
    forged = [dict(row) for row in rows]
    forged[0]["camera_lock_x_camera_m"] = 0.1
    forged[0]["camera_lock_y_camera_m"] = 0.2
    forged[0]["camera_lock_z_camera_m"] = 1.0
    forged[0]["frame_evidence_sha256"] = study.semantic_frame_sha256(forged[0])
    with pytest.raises(study.UpperBoundError, match="validity"):
        study.validate_full_experiment_rows(forged)


def test_current_input_schema_forces_full_semantic_csv_verification() -> None:
    rows = [_result_row(0, camera_error_mm=2.0, patient_error_mm=2.0)]
    metrics = study.base_metrics()
    metrics.update(study.derive_row_metrics(rows))
    inputs = {
        "schema": study.INPUT_SCHEMA,
        "raw_rendered_input_stream_sha256": _raw_rendered_input_stream(rows),
        "rendered_evidence_stream_sha256": _evidence_stream(rows),
    }

    with pytest.raises(study.UpperBoundError, match="143"):
        study.verify_metrics_against_csv(
            metrics, study.per_frame_csv_bytes(rows), inputs
        )


def test_full_verifier_recomputes_nonrow_metrics_and_reference_consensus() -> None:
    rows = _full_experiment_rows()
    metrics = study.base_metrics()
    metrics.update(study.derive_row_metrics(rows))
    metrics.update(study.derive_nonrow_metrics(rows))
    assert metrics["reference_consensus_max_deviation_mm"] == pytest.approx(1.0)
    assert metrics["reference_confirmation_frame_index"] == 11
    assert metrics["report_frame_index"] == 82
    assert metrics["video_contract"]["nb_read_frames"] == "143"
    inputs = {
        "schema": study.INPUT_SCHEMA,
        "raw_rendered_input_stream_sha256": _raw_rendered_input_stream(rows),
        "rendered_evidence_stream_sha256": _evidence_stream(rows),
    }
    csv_raw = study.per_frame_csv_bytes(rows)
    study.verify_metrics_against_csv(metrics, csv_raw, inputs)

    attacks = {
        "frame_rate_hz": 99,
        "reference_confirmation_frame_index": 99,
        "reference_consensus_max_deviation_mm": 99.0,
        "report_frame_index": 0,
        "official_trajectory_frame_count": 1,
    }
    for key, value in attacks.items():
        forged = json.loads(json.dumps(metrics))
        forged[key] = value
        with pytest.raises(study.UpperBoundError, match=key):
            study.verify_metrics_against_csv(forged, csv_raw, inputs)

    for key in (
        "motion_sensitivity",
        "pose_delay_sensitivity",
        "pose_translation_noise_sensitivity",
    ):
        forged = json.loads(json.dumps(metrics))
        first_numeric = next(
            field
            for field, value in forged[key][0].items()
            if isinstance(value, (int, float))
        )
        forged[key][0][first_numeric] = 999.0
        with pytest.raises(study.UpperBoundError, match=key):
            study.verify_metrics_against_csv(forged, csv_raw, inputs)

    forged = json.loads(json.dumps(metrics))
    forged["video_contract"]["width"] = 1
    with pytest.raises(study.UpperBoundError, match="video_contract"):
        study.verify_metrics_against_csv(forged, csv_raw, inputs)


def test_source_binding_is_exact_complete_local_import_closure() -> None:
    assert tuple(study.BOUND_SOURCE_FILES) == (
        "upper_bound_runner",
        "ausc_init",
        "tracking_init",
        "canonical",
        "detector",
        "ekf",
        "experimental_visibility_geometry",
        "flow",
        "geometry",
        "geometry_channel_packet",
        "intrinsics",
        "model_tracker",
        "official_asset_contract",
        "official_openarm_experiment",
        "offline_mesh_simulation",
        "offline_official_mesh_assets",
        "offline_official_mesh_experiment",
        "production_geometry_gate",
        "rgbd_odometry",
        "runtime",
        "tracker",
    )
    manifest = study.bound_source_manifest()
    assert len(manifest) == 21
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
from pathlib import Path
from scripts.perception import run_offline_patient_frame_upper_bound
root = Path('auscultation').resolve()
loaded = []
for module in sys.modules.values():
    source = getattr(module, '__file__', None)
    if source is None:
        continue
    path = Path(source).resolve()
    if path.suffix != '.py':
        continue
    try:
        loaded.append(str(path.relative_to(root)))
    except ValueError:
        pass
print(json.dumps(sorted(set(loaded))))
""",
        ],
        cwd=study.WORKSPACE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_paths = set(json.loads(probe.stdout))
    bound_paths = {
        str(path.resolve().relative_to(study.AUSCULTATION_ROOT.resolve()))
        for path in study.BOUND_SOURCE_FILES.values()
    }
    assert loaded_paths == bound_paths
    camera_transform_sha = study.camera_pose_transform_sha256()
    study.validate_bound_source_inputs(
        {
            "schema": study.INPUT_SCHEMA,
            "camera_pose_transform_sha256": camera_transform_sha,
            "bound_source_files": manifest,
            "upper_bound_runner_sha256": manifest["upper_bound_runner"]["sha256"],
        }
    )

    for dependency in tuple(study.BOUND_SOURCE_FILES):
        forged = json.loads(json.dumps(manifest))
        forged[dependency]["sha256"] = "0" * 64
        with pytest.raises(study.UpperBoundError, match="source binding"):
            study.validate_bound_source_inputs(
                {
                    "schema": study.INPUT_SCHEMA,
                    "camera_pose_transform_sha256": camera_transform_sha,
                    "bound_source_files": forged,
                    "upper_bound_runner_sha256": manifest["upper_bound_runner"][
                        "sha256"
                    ],
                }
            )

    for dependency in tuple(study.BOUND_SOURCE_FILES):
        forged = json.loads(json.dumps(manifest))
        forged.pop(dependency)
        with pytest.raises(study.UpperBoundError, match="source binding"):
            study.validate_bound_source_inputs(
                {
                    "schema": study.INPUT_SCHEMA,
                    "camera_pose_transform_sha256": camera_transform_sha,
                    "bound_source_files": forged,
                    "upper_bound_runner_sha256": manifest["upper_bound_runner"][
                        "sha256"
                    ],
                }
            )

    for dependency in tuple(study.BOUND_SOURCE_FILES):
        forged = json.loads(json.dumps(manifest))
        forged[dependency]["path"] = "forged/path.py"
        with pytest.raises(study.UpperBoundError, match="source binding"):
            study.validate_bound_source_inputs(
                {
                    "schema": study.INPUT_SCHEMA,
                    "camera_pose_transform_sha256": camera_transform_sha,
                    "bound_source_files": forged,
                    "upper_bound_runner_sha256": manifest["upper_bound_runner"][
                        "sha256"
                    ],
                }
            )

    forged = json.loads(json.dumps(manifest))
    forged["extra"] = dict(next(iter(forged.values())))
    with pytest.raises(study.UpperBoundError, match="source binding"):
        study.validate_bound_source_inputs(
            {
                "schema": study.INPUT_SCHEMA,
                "camera_pose_transform_sha256": camera_transform_sha,
                "bound_source_files": forged,
                "upper_bound_runner_sha256": manifest["upper_bound_runner"]["sha256"],
            }
        )

    with pytest.raises(study.UpperBoundError, match="input schema"):
        study.validate_bound_source_inputs(
            {
                "bound_source_files": manifest,
                "upper_bound_runner_sha256": manifest["upper_bound_runner"]["sha256"],
            }
        )

    with pytest.raises(study.UpperBoundError, match="camera transform"):
        study.validate_bound_source_inputs(
            {
                "schema": study.INPUT_SCHEMA,
                "camera_pose_transform_sha256": "0" * 64,
                "bound_source_files": manifest,
                "upper_bound_runner_sha256": manifest["upper_bound_runner"]["sha256"],
            }
        )


def test_publish_keeps_warning_and_never_writes_release_state(tmp_path: Path) -> None:
    destination = tmp_path / "staging"
    destination.mkdir()
    (destination / "STAGING_WARNING").write_text("not complete\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in study.REQUIRED_ARTIFACTS:
        (scratch / name).write_bytes(("artifact:" + name).encode())

    study.publish_staging_artifacts(scratch=scratch, destination=destination)

    names = {path.name for path in destination.iterdir()}
    assert names == set(study.REQUIRED_ARTIFACTS) | {"STAGING_WARNING", "SHA256SUMS"}
    assert not any(name in {"COMPLETE", "COMPLETED", "COMPLETION.json", "current"} for name in names)


def test_overlay_carries_all_three_permanent_banners(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered_text: list[str] = []

    def capture_text(frame, value, *args, **kwargs):
        rendered_text.append(str(value))
        return frame

    monkeypatch.setattr(study.cv2, "putText", capture_text)
    study.overlay_frame(
        rgb=np.zeros((360, 640, 3), dtype=np.uint8),
        robot_mask=np.zeros((360, 640), dtype=bool),
        oracle_pixels=np.full((10, 2), np.nan),
        truth_pixel=np.full(2, np.nan),
        camera_lock_xyz=None,
        patient_lock_xyz=None,
        truth_xyz=np.asarray([0.0, 0.0, 1.0]),
        intrinsics=study.Intrinsics(100, 100, 320, 180, 640, 360, 1000),
        phase="confirm_home",
        timestamp_s=0.0,
    )

    assert study.BANNER in rendered_text
    assert study.HUMAN_BANNER in rendered_text
    assert study.METHOD_BANNER in rendered_text


def test_ffmpeg_contract_is_h264_yuv420p_640x360_20hz(tmp_path: Path) -> None:
    command = study.ffmpeg_command(tmp_path / "video.mp4")

    assert command[0] == "/opt/homebrew/bin/ffmpeg"
    assert command[command.index("-s") + 1] == "640x360"
    assert command[command.index("-r") + 1] == "20"
    assert "libx264" in command
    assert "yuv420p" in command
    assert command[command.index("-threads") + 1] == "1"
