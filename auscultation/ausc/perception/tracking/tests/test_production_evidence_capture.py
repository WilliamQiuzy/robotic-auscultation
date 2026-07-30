from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.tracking.back_mask_packet import BackMaskPacket
from ausc.perception.tracking.detector import DetectorResult, LUNG_ORDER
from ausc.perception.tracking.geometry import CropTransform, Intrinsics
from ausc.perception.tracking.model_tracker import ModelBasedTracker
from ausc.perception.tracking.patient_frame_evidence import PatientFrameEvidence
from ausc.perception.tracking.production_evidence_capture import (
    ProductionEvidenceCaptureError,
    ProductionGeometryEvidenceCapture,
)
from ausc.perception.tracking.production_geometry_gate import (
    PRODUCTION_DECISION_SURFACE,
    ProductionGeometryProfile,
    production_geometry_profile_semantic_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
EPISODE = "episode-" + "1" * 32
POINT_NAMES = tuple(LUNG_ORDER)

_ROBOT_TEST_PATH = (
    Path(__file__).parents[2]
    / "robot"
    / "tests"
    / "test_frame_evidence_bundle.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_wam_evidence_capture_robot_helpers", _ROBOT_TEST_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_ROBOT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ROBOT)


def _profile() -> ProductionGeometryProfile:
    values = {
        "camera_profile_sha256": SHA_A,
        "camera_projection_sha256": SHA_B,
        "depth_noise_model_sha256": SHA_C,
        "independent_model_sha256": SHA_D,
        "lateral_noise_std_m": 1.0e-3,
        "axial_noise_std_m": 2.0e-3,
        "robust_delta_m": 3.0e-3,
        "max_scale_deviation_relative": 0.02,
        "depth_sample_window_px": 2,
        "min_depth_m": 0.05,
        "max_depth_m": 3.0,
        "min_depth_patch_valid_pixels": 9,
        "min_depth_patch_valid_fraction": 0.60,
        "max_depth_patch_mad_m": 0.004,
        "max_depth_patch_p90_p10_m": 0.020,
        "min_back_mask_support_fraction": 0.60,
        "max_independent_model_innovation_m": 0.010,
        "max_independent_model_innovation_sigma": 3.0,
        "decision_surface": PRODUCTION_DECISION_SURFACE,
    }
    values["semantic_sha256"] = production_geometry_profile_semantic_sha256(
        **values
    )
    return ProductionGeometryProfile(**values)


def _intrinsics() -> Intrinsics:
    return Intrinsics(
        fx=100.0,
        fy=100.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
        depth_scale=1.0,
    )


def _patient_evidence(
    *, frame_index: int = 12, camera_host_monotonic_ns: int = 1_050_000_000,
) -> PatientFrameEvidence:
    host_s = camera_host_monotonic_ns / 1_000_000_000
    device_t_ms = 400.0 + (
        camera_host_monotonic_ns - 1_050_000_000
    ) / 1_000_000
    exposure = _ROBOT._exposure(
        frame_index=frame_index,
        camera_host_t_s=host_s,
        camera_host_monotonic_ns=camera_host_monotonic_ns,
        camera_device_t_ms=device_t_ms,
        exposure_lower_host_t_s=(camera_host_monotonic_ns - 40_000_000)
        / 1_000_000_000,
        exposure_nominal_host_t_s=(camera_host_monotonic_ns - 35_000_000)
        / 1_000_000_000,
        exposure_upper_host_t_s=(camera_host_monotonic_ns - 30_000_000)
        / 1_000_000_000,
        exposure_lower_monotonic_ns=camera_host_monotonic_ns - 40_000_000,
        exposure_nominal_monotonic_ns=camera_host_monotonic_ns - 35_000_000,
        exposure_upper_monotonic_ns=camera_host_monotonic_ns - 30_000_000,
    )
    trajectory = _ROBOT._trajectory(
        times=tuple(
            camera_host_monotonic_ns - 41_000_000 + offset * 1_000_000
            for offset in range(13)
        ),
        lower_ns=camera_host_monotonic_ns - 40_000_000,
        nominal_ns=camera_host_monotonic_ns - 35_000_000,
        upper_ns=camera_host_monotonic_ns - 30_000_000,
    )
    yy, xx = np.indices((64, 64), dtype=np.uint16)
    rgb = np.stack(
        [
            (xx % 251).astype(np.uint8),
            (yy % 251).astype(np.uint8),
            ((xx + yy) % 251).astype(np.uint8),
        ],
        axis=2,
    )
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    camera = _ROBOT._camera_frame(
        index=frame_index,
        t=host_s,
        device_t_ms=device_t_ms,
        rgb=rgb,
        depth=depth,
    )
    robot_packet = _ROBOT._mask_packet(
        exposure=exposure,
        trajectory=trajectory,
        mask=np.zeros((64, 64), dtype=np.bool_),
    )
    bundle = _ROBOT._bundle(
        camera_frame=camera,
        exposure=exposure,
        trajectory=trajectory,
        robot_mask_packet=robot_packet,
    )
    back = BackMaskPacket.build(
        camera_frame=bundle.camera_frame,
        mask=np.ones((64, 64), dtype=np.bool_),
        back_mask_release_id="back_mask-" + "9" * 20,
        back_mask_model_sha256="8" * 64,
        operator_gate_decision_id="operator_back_gate-" + "7" * 64,
    )
    return PatientFrameEvidence.build(
        frame_bundle=bundle,
        back_mask_packet=back,
    )


class _NoBackDetector:
    def __init__(self) -> None:
        self.seen_rgb = None
        self.seen_depth = None

    def detect(self, rgb, depth, intr):  # noqa: ANN001
        del intr
        self.seen_rgb = rgb
        self.seen_depth = depth
        return DetectorResult(
            xy_det=np.zeros((len(POINT_NAMES), 2), dtype=np.float64),
            crop=CropTransform(0, 0, 64, 64),
            confidence=1.0,
            back_present=False,
        )


def _capture(tmp_path: Path) -> ProductionGeometryEvidenceCapture:
    return ProductionGeometryEvidenceCapture(
        tmp_path,
        intrinsics=_intrinsics(),
        geometry_profile=_profile(),
        point_names=POINT_NAMES,
    )


def test_model_consumes_exact_lossless_bytes_reloaded_from_content_addressed_file(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    patient = _patient_evidence()
    model_input = capture.stage_model_input(
        patient, episode_id=EPISODE, sequence=1,
    )
    detector = _NoBackDetector()
    tracker = ModelBasedTracker(detector, _intrinsics())

    reloaded = model_input.reload_for_inference()
    tracker.process_persisted_model_input(model_input)

    assert model_input.rgb_relative_path == (
        f"geometry/model_inputs/{patient.frame_bundle.camera_frame.rgb_sha256}.npy"
    )
    assert model_input.depth_relative_path == "depth/depth_000012.npy"
    assert np.array_equal(detector.seen_rgb, patient.rgb)
    assert np.array_equal(detector.seen_depth, patient.aligned_depth)
    assert detector.seen_rgb.flags.writeable is False
    assert detector.seen_depth.flags.writeable is False
    assert reloaded.back_mask is not None
    assert reloaded.robot_mask is not None
    assert reloaded.back_mask.flags.writeable is False
    assert reloaded.robot_mask.flags.writeable is False


def test_model_refuses_tampered_content_addressed_input_before_inference(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    patient = _patient_evidence()
    model_input = capture.stage_model_input(
        patient, episode_id=EPISODE, sequence=1,
    )
    rgb_path = tmp_path / model_input.rgb_relative_path
    forged = np.zeros_like(patient.rgb)
    with rgb_path.open("wb") as fp:
        np.save(fp, forged, allow_pickle=False)

    detector = _NoBackDetector()
    with pytest.raises(ProductionEvidenceCaptureError, match="RGB"):
        ModelBasedTracker(detector, _intrinsics()).process_persisted_model_input(
            model_input
        )
    assert detector.seen_rgb is None
    assert detector.seen_depth is None


def test_existing_exact_depth_is_reused_but_mismatch_and_symlink_are_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "depth").mkdir()
    exact = _patient_evidence().aligned_depth
    with (tmp_path / "depth/depth_000012.npy").open("xb") as fp:
        np.save(fp, exact, allow_pickle=False)
    capture = _capture(tmp_path)
    capture.stage_model_input(
        _patient_evidence(), episode_id=EPISODE, sequence=1,
    )

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    (mismatch_root / "depth").mkdir()
    with (mismatch_root / "depth/depth_000012.npy").open("xb") as fp:
        np.save(fp, np.full((64, 64), 900, dtype=np.uint16), allow_pickle=False)
    with pytest.raises(ProductionEvidenceCaptureError, match="depth"):
        _capture(mismatch_root).stage_model_input(
            _patient_evidence(), episode_id=EPISODE, sequence=1,
        )

    symlink_root = tmp_path / "symlink-depth"
    symlink_root.mkdir()
    (symlink_root / "depth").mkdir()
    target = symlink_root / "actual.npy"
    with target.open("xb") as fp:
        np.save(fp, exact, allow_pickle=False)
    (symlink_root / "depth/depth_000012.npy").symlink_to(target)
    with pytest.raises(ProductionEvidenceCaptureError, match="symlink"):
        _capture(symlink_root).stage_model_input(
            _patient_evidence(), episode_id=EPISODE, sequence=1,
        )


# --- Geometry per-frame ledger (Step 2a) -------------------------------------

from ausc.perception.tracking.production_evidence_capture import (  # noqa: E402
    GEOMETRY_MODEL_INPUT_LEDGER_RELATIVE,
    PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA,
    PersistedGeometryModelInput,
    ProductionEvidenceCaptureError,
    append_geometry_model_input_ledger,
    read_geometry_model_input_ledger,
)


def _ledger_record(session_dir, *, sequence, frame_index, episode_id="episode-xyz"):
    rgb_sha = f"{sequence:064x}"
    return PersistedGeometryModelInput(
        schema=PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA,
        session_dir=Path(session_dir).absolute(),
        episode_id=episode_id,
        sequence=sequence,
        patient_frame_evidence_id="patient_frame_evidence-" + "a" * 20,
        frame_index=frame_index,
        camera_t=1.0 + sequence,
        depth_scale=0.001,
        rgb_relative_path=f"geometry/model_inputs/{rgb_sha}.npy",
        rgb_sha256=rgb_sha,
        rgb_file_sha256=f"{(sequence + 2) * 3:064x}",
        rgb_dtype=np.dtype(np.uint8).str,
        rgb_shape=(4, 4, 3),
        depth_relative_path=f"depth/depth_{frame_index:06d}.npy",
        depth_sha256=f"{(sequence + 1) * 7:064x}",
        depth_file_sha256=f"{(sequence + 3) * 5:064x}",
        depth_dtype=np.dtype(np.uint16).str,
        depth_shape=(4, 4),
        back_mask_relative_path=None,
        back_mask_sha256=None,
        back_mask_file_sha256=None,
        robot_mask_relative_path=None,
        robot_mask_sha256=None,
        robot_mask_file_sha256=None,
    )


def _ledger_path(session_dir):
    return Path(session_dir) / GEOMETRY_MODEL_INPUT_LEDGER_RELATIVE


def test_ledger_round_trips_in_strict_episode_order(tmp_path):
    (tmp_path / "geometry").mkdir()
    records = [
        _ledger_record(tmp_path, sequence=i, frame_index=i - 1) for i in (1, 2, 3)
    ]
    for record in records:
        append_geometry_model_input_ledger(tmp_path, record)

    read_back = read_geometry_model_input_ledger(tmp_path)

    assert [r.to_manifest() for r in read_back] == [r.to_manifest() for r in records]
    # to_manifest carries no absolute session_dir.
    assert "session_dir" not in records[0].to_manifest()


def test_ledger_rejects_a_sequence_gap(tmp_path):
    (tmp_path / "geometry").mkdir()
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=0))
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=3, frame_index=2))
    with pytest.raises(ProductionEvidenceCaptureError, match="contiguous"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_a_duplicate_sequence(tmp_path):
    (tmp_path / "geometry").mkdir()
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=0))
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=1))
    with pytest.raises(ProductionEvidenceCaptureError, match="contiguous"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_reordered_records(tmp_path):
    (tmp_path / "geometry").mkdir()
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=2, frame_index=1))
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=0))
    with pytest.raises(ProductionEvidenceCaptureError, match="contiguous"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_more_than_one_episode(tmp_path):
    (tmp_path / "geometry").mkdir()
    append_geometry_model_input_ledger(
        tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=0, episode_id="episode-a"))
    append_geometry_model_input_ledger(
        tmp_path, _ledger_record(tmp_path, sequence=2, frame_index=1, episode_id="episode-b"))
    with pytest.raises(ProductionEvidenceCaptureError, match="one episode"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_non_increasing_frame_index(tmp_path):
    (tmp_path / "geometry").mkdir()
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=1, frame_index=5))
    append_geometry_model_input_ledger(tmp_path, _ledger_record(tmp_path, sequence=2, frame_index=5))
    with pytest.raises(ProductionEvidenceCaptureError, match="strictly increasing"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_a_non_canonical_line(tmp_path):
    (tmp_path / "geometry").mkdir()
    record = _ledger_record(tmp_path, sequence=1, frame_index=0)
    # Same content on one line, but with non-canonical whitespace (default
    # json.dumps inserts ", " / ": " separators).
    import json as _json
    _ledger_path(tmp_path).write_bytes(
        (_json.dumps(record.to_manifest()) + "\n").encode("utf-8"))
    with pytest.raises(ProductionEvidenceCaptureError, match="canonical"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_rejects_a_tampered_field_set(tmp_path):
    (tmp_path / "geometry").mkdir()
    import json as _json
    manifest = _ledger_record(tmp_path, sequence=1, frame_index=0).to_manifest()
    manifest.pop("depth_sha256")  # drop a required identity
    _ledger_path(tmp_path).write_bytes(
        (_json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    with pytest.raises(ProductionEvidenceCaptureError, match="field set is not exact"):
        read_geometry_model_input_ledger(tmp_path)


def test_ledger_read_fails_closed_when_missing(tmp_path):
    with pytest.raises(ProductionEvidenceCaptureError, match="missing"):
        read_geometry_model_input_ledger(tmp_path)
