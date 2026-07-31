from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.tracking.back_mask_packet import BackMaskPacket
from ausc.perception.tracking.detector import DetectorResult, LUNG_ORDER
from ausc.perception.tracking.geometry import CropTransform, Intrinsics
from ausc.perception.tracking.model_tracker import ModelBasedTracker
from ausc.perception.tracking.patient_frame_evidence import PatientFrameEvidence
from ausc.perception.tracking.geometry_channel_packet import (
    verify_geometry_packet_log_files,
)
from ausc.perception.tracking.production_evidence_capture import (
    ProductionEvidenceCaptureError,
    ProductionGeometryEvidenceCapture,
    read_geometry_model_input_ledger,
)
from ausc.perception.tracking.production_geometry_recorder import (
    GeometryChannelIdentity,
    ProductionGeometryRecorderError,
    ProductionGeometrySessionRecorder,
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


# --- ProductionGeometrySessionRecorder (write-side composition) ---------------

_PERCEPTION_CHANNEL = GeometryChannelIdentity(
    producer_id="v9-landmark-worker",
    release_id="v9-landmark-worker-" + "e" * 20,
    model_sha256="e" * 64,
)
_INDEPENDENT_CHANNEL = GeometryChannelIdentity(
    producer_id="metric-patient-model",
    release_id="metric-patient-model-" + "d" * 20,
    model_sha256="d" * 64,
)


def _points(base: float) -> np.ndarray:
    n = len(POINT_NAMES)
    return np.array(
        [[base + i, base + i + 0.1, 0.5 + i * 0.01] for i in range(n)],
        dtype=np.float64,
    )


def _covariances() -> np.ndarray:
    n = len(POINT_NAMES)
    return np.stack([np.eye(3, dtype=np.float64) * 1.0e-4 for _ in range(n)])


def _recorder(tmp_path: Path) -> ProductionGeometrySessionRecorder:
    return ProductionGeometrySessionRecorder(
        tmp_path,
        intrinsics=_intrinsics(),
        geometry_profile=_profile(),
        point_names=POINT_NAMES,
        perception_channel=_PERCEPTION_CHANNEL,
        independent_channel=_INDEPENDENT_CHANNEL,
    )


def _record_frames(recorder: ProductionGeometrySessionRecorder, count: int) -> None:
    for i in range(count):
        evidence = _patient_evidence(
            frame_index=10 + i,
            camera_host_monotonic_ns=1_050_000_000 + i * 1_000_000,
        )
        recorder.record_frame(
            evidence,
            episode_id=EPISODE,
            sequence=i + 1,
            perception_points_m=_points(0.10 * i),
            perception_covariances_m2=_covariances(),
            independent_points_m=_points(0.10 * i + 0.001),
            independent_covariances_m2=_covariances(),
        )


def test_recorder_writes_a_replayable_geometry_session(tmp_path):
    recorder = _recorder(tmp_path)
    _record_frames(recorder, 3)
    assert recorder.frame_count == 3
    receipt = recorder.finalize()

    # The packet log verifies and the ledger reads back in strict episode order.
    entries, verified_receipt = verify_geometry_packet_log_files(tmp_path)
    assert len(entries) == 3
    assert verified_receipt == receipt
    assert receipt.perception_release_id == _PERCEPTION_CHANNEL.release_id
    assert receipt.perception_model_sha256 == _PERCEPTION_CHANNEL.model_sha256
    assert receipt.independent_release_id == _INDEPENDENT_CHANNEL.release_id

    ledger = read_geometry_model_input_ledger(tmp_path)
    assert len(ledger) == 3
    assert [record.sequence for record in ledger] == [1, 2, 3]
    assert all(record.episode_id == EPISODE for record in ledger)

    # Both required artifacts + the Step 2a ledger are on disk.
    assert (tmp_path / "geometry_frame_packets.jsonl").is_file()
    assert (tmp_path / "geometry_frame_authenticity.json").is_file()
    assert (tmp_path / "geometry" / "model_input_ledger.jsonl").is_file()


def test_recorder_ledger_and_packet_counts_stay_aligned(tmp_path):
    recorder = _recorder(tmp_path)
    _record_frames(recorder, 4)
    recorder.finalize()
    entries, _ = verify_geometry_packet_log_files(tmp_path)
    ledger = read_geometry_model_input_ledger(tmp_path)
    assert len(entries) == len(ledger) == 4


def test_recorder_rejects_noncontiguous_sequence(tmp_path):
    recorder = _recorder(tmp_path)
    evidence = _patient_evidence(frame_index=10)
    with pytest.raises(ProductionGeometryRecorderError, match="contiguous"):
        recorder.record_frame(
            evidence, episode_id=EPISODE, sequence=2,  # must start at 1
            perception_points_m=_points(0.0), perception_covariances_m2=_covariances(),
            independent_points_m=_points(0.001), independent_covariances_m2=_covariances(),
        )


def test_recorder_rejects_episode_change_mid_session(tmp_path):
    recorder = _recorder(tmp_path)
    _record_frames(recorder, 1)
    evidence = _patient_evidence(frame_index=11, camera_host_monotonic_ns=1_051_000_000)
    with pytest.raises(ProductionGeometryRecorderError, match="episode id changed"):
        recorder.record_frame(
            evidence, episode_id="episode-" + "2" * 32, sequence=2,
            perception_points_m=_points(0.1), perception_covariances_m2=_covariances(),
            independent_points_m=_points(0.101), independent_covariances_m2=_covariances(),
        )


def test_recorder_finalize_requires_at_least_two_frames(tmp_path):
    recorder = _recorder(tmp_path)
    _record_frames(recorder, 1)
    with pytest.raises(ProductionGeometryRecorderError, match="at least two frames"):
        recorder.finalize()


def test_recorder_refuses_frames_after_finalize(tmp_path):
    recorder = _recorder(tmp_path)
    _record_frames(recorder, 2)
    recorder.finalize()
    evidence = _patient_evidence(frame_index=99, camera_host_monotonic_ns=1_060_000_000)
    with pytest.raises(ProductionGeometryRecorderError, match="already finalized"):
        recorder.record_frame(
            evidence, episode_id=EPISODE, sequence=3,
            perception_points_m=_points(0.3), perception_covariances_m2=_covariances(),
            independent_points_m=_points(0.301), independent_covariances_m2=_covariances(),
        )


def test_finalizer_reverifies_geometry_model_inputs_against_tamper(tmp_path):
    # A recorded session's RGB/depth model inputs must not be modifiable after the
    # fact without detection: the finalizer's re-verification re-hashes each
    # ledger-referenced npy and fails closed on any tamper.
    import importlib

    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    finalizer = importlib.import_module("training_authenticity_finalizer")

    recorder = _recorder(tmp_path)
    _record_frames(recorder, 3)
    recorder.finalize()

    # Clean session re-verifies.
    assert finalizer._reverify_geometry_model_inputs(tmp_path) == 3

    # Tamper one persisted RGB model input; re-verification must now fail closed.
    npys = sorted((tmp_path / "geometry" / "model_inputs").glob("*.npy"))
    assert npys
    npys[0].write_bytes(npys[0].read_bytes() + b"\x00")
    with pytest.raises(Exception):
        finalizer._reverify_geometry_model_inputs(tmp_path)


def test_recorded_session_is_replayable_by_the_runner(tmp_path):
    # Closes the loop: what the write-side records is exactly what the Step 3c
    # replay runner reconstructs (ledger inputs matched to the recorded packets).
    from ausc.perception.replay_runner import reconstruct_geometry_items

    recorder = _recorder(tmp_path)
    _record_frames(recorder, 3)
    recorder.finalize()

    items, identity = reconstruct_geometry_items(tmp_path)
    assert len(items) == 3
    assert identity == {
        "release_id": _PERCEPTION_CHANNEL.release_id,
        "model_sha256": _PERCEPTION_CHANNEL.model_sha256,
    }
    assert set(items[0].recorded_outputs) == {"points_m", "covariances_m2"}
    # The verified RGB/depth model inputs reload from the persisted content-addressed npys.
    assert items[0].model_input.rgb.shape[2] == 3
    assert items[0].model_input.depth.ndim == 2


def test_recorder_accepts_the_real_patient_pose_independent_identity(tmp_path):
    # The real activated independent model id is "patient_pose-<sha20>" (with an
    # underscore); it must validate as a geometry channel release id.
    sha = "b" * 64
    recorder = ProductionGeometrySessionRecorder(
        tmp_path, intrinsics=_intrinsics(), geometry_profile=_profile(),
        point_names=POINT_NAMES,
        perception_channel=_PERCEPTION_CHANNEL,
        independent_channel=GeometryChannelIdentity(
            producer_id="metric-patient-model",
            release_id="patient_pose-" + sha[:20],
            model_sha256=sha,
        ),
    )
    recorder.record_frame(
        _patient_evidence(frame_index=10),
        episode_id=EPISODE, sequence=1,
        perception_points_m=_points(0.0), perception_covariances_m2=_covariances(),
        independent_points_m=_points(0.001), independent_covariances_m2=_covariances(),
    )
    assert recorder.frame_count == 1


def test_recorder_requires_independent_channels(tmp_path):
    same = GeometryChannelIdentity(
        producer_id="v9-landmark-worker",
        release_id="v9-landmark-worker-" + "e" * 20,
        model_sha256="e" * 64,
    )
    with pytest.raises(ProductionGeometryRecorderError, match="independent"):
        ProductionGeometrySessionRecorder(
            tmp_path, intrinsics=_intrinsics(), geometry_profile=_profile(),
            point_names=POINT_NAMES,
            perception_channel=same, independent_channel=same,
        )


# --- ProductionEvidenceSink (geometry + quality composition) ------------------

_FORCE_FEEDBACK = Path(__file__).resolve().parents[4] / "mechanical_sensor" / "force_feedback"
sys.path.insert(0, str(_FORCE_FEEDBACK))
import force_capture  # noqa: E402
import force_sensor  # noqa: E402

from ausc.perception.production_evidence_sink import ProductionEvidenceSink  # noqa: E402
from ausc.realtime_q import verify_quality_packet_log  # noqa: E402

_Q_MODEL_SHA = "a" * 64
_Q_MODEL_RELEASE = "clinical-q-capture-test-" + _Q_MODEL_SHA[:20]
_Q_CAL_SHA = "c" * 64
_Q_POLICY = {
    "schema_version": "wam_quality_policy_v1",
    "policy_id": "clinical-q-capture-test-v1",
    "force_band_n": [3.0, 5.0],
    "window_s": 0.25,
    "ema": 0.5,
    "max_audio_age_s": 0.2,
    "max_inference_age_s": 0.5,
    "max_force_age_s": 0.2,
    "min_packets_for_receipt": 2,
    "output_contract": {
        "model_role": "clinical_usability_q",
        "output_semantics": "class_probability",
        "positive_class": "usable",
        "calibration": {
            "method": "embedded_in_model",
            "release_id": "clinical-q-calibration-test-" + _Q_CAL_SHA[:20],
            "sha256": _Q_CAL_SHA,
        },
        "transform_to_p_usable": "identity",
        "output_range": [0.0, 1.0],
    },
}


class _PoseStub:
    def __init__(self):
        self.rotation_camera_from_patient = np.eye(3, dtype=np.float64)
        self.translation_camera_from_patient_m = np.zeros(3, dtype=np.float64)
        self.covariance_se3 = np.diag(
            [3e-4, 3e-4, 3e-4, 1e-4, 1e-4, 1e-4]
        ).astype(np.float64)

    def camera_from_patient(self, point):
        return self.rotation_camera_from_patient @ np.asarray(
            point, dtype=np.float64
        ) + self.translation_camera_from_patient_m


def _write_quality_force_csv(tmp_path: Path) -> None:
    recorder = force_capture.ForceCaptureRecorder(
        tmp_path, t0=7.2,
        calibration_identity={"sha256": "e" * 64, "calibrated": True,
                              "source_name": "hall-calibration.json"},
        device_identity={"port": "/dev/serial/by-id/usb-RP2040-quality-test",
                         "baud": 115200, "usb_vid": "2e8a", "usb_pid": "0005",
                         "serial_number": "quality-test"},
        max_device_gap_ms=400.0, max_host_gap_ms=400.0,
    )
    recorder.start()
    for index, inference_t in enumerate((7.25, 7.55, 7.85)):
        recorder.submit(force_sensor.ForceSample(
            t=inference_t - 0.05, raw=31_000 + index, hall_v=1.56 + index * 0.001,
            displacement_mm=1.20 + index * 0.01, force_n=4.0, status="OK",
            device_t_ms=100.0 + index * 300.0,
        ))
    recorder.finish(t_end=7.8)


def _q_window(sequence, value, inference_t):
    samples = np.linspace(value, value + 0.01, 4_000, dtype=np.float32)
    window_index = sequence - 30
    return SimpleNamespace(
        samples=samples, sr=16_000, t_start=inference_t - 0.25, t_end=inference_t,
        status="OK",
        raw_source={
            "source_channel": 1,
            "source_sample_start": window_index * 12_000,
            "source_sample_end": (window_index + 1) * 12_000,
            "source_sample_rate": 48_000,
            "resampler_identity": {"algorithm": "polyphase_fir_v1",
                                   "input_sample_rate": 48_000,
                                   "output_sample_rate": 16_000},
        },
    )


def _q_force(sequence, inference_t):
    return SimpleNamespace(
        t=inference_t - 0.05, raw=31_000 + (sequence - 30),
        hall_v=1.56 + (sequence - 30) * 0.001,
        displacement_mm=1.20 + (sequence - 30) * 0.01, force_n=4.0, status="OK",
        device_t_ms=100.0 + (sequence - 30) * 300.0,
    )


def _sink(tmp_path):
    return ProductionEvidenceSink(
        tmp_path,
        intrinsics=_intrinsics(),
        geometry_profile=_profile(),
        point_names=POINT_NAMES,
        perception_channel=_PERCEPTION_CHANNEL,
        independent_channel=_INDEPENDENT_CHANNEL,
        quality_scorer=lambda samples, _sr: float(np.mean(samples)),
        quality_model_release_id=_Q_MODEL_RELEASE,
        quality_model_sha256=_Q_MODEL_SHA,
        quality_policy=_Q_POLICY,
    )


def test_sink_records_both_channels_into_one_promotable_session(tmp_path):
    sink = _sink(tmp_path)
    n = len(POINT_NAMES)
    pose = _PoseStub()
    template = _points(0.05)

    # Geometry: 3 frames, sink owns the sequence + applies both adapters.
    for i in range(3):
        evidence = _patient_evidence(
            frame_index=10 + i,
            camera_host_monotonic_ns=1_050_000_000 + i * 1_000_000,
        )
        sink.record_geometry_frame(
            evidence,
            episode_id=EPISODE,
            perception_points_xyz_camera_m=_points(0.10 * i),
            perception_position_std_m=np.full(n, 0.01),
            patient_pose_packet=pose,
            template_points_patient_m=template,
        )
    # Quality: 3 windows.
    for sequence, value, inference_t in ((30, 0.1, 7.25), (31, 0.2, 7.55), (32, 0.3, 7.85)):
        sink.record_quality_window(
            audio_window=_q_window(sequence, value, inference_t),
            force_sample=_q_force(sequence, inference_t),
            inference_sequence=sequence,
            inference_monotonic_s=inference_t,
            observed_monotonic_s=inference_t + 0.01,
        )
    _write_quality_force_csv(tmp_path)

    assert sink.geometry_frame_count == 3
    assert sink.quality_window_count == 3
    geo_receipt = sink.finalize_geometry()
    sink.finalize_quality(now_monotonic_s=7.9)

    # Both channels land in one session and both verify + replay.
    from ausc.perception.replay_runner import (
        reconstruct_geometry_items,
        reconstruct_quality_items,
    )

    geo_entries, verified = verify_geometry_packet_log_files(tmp_path)
    assert len(geo_entries) == 3 and verified == geo_receipt
    assert verify_quality_packet_log(tmp_path)["packet_count"] == 3

    geo_items, geo_identity = reconstruct_geometry_items(tmp_path)
    q_items, q_identity = reconstruct_quality_items(tmp_path)
    assert len(geo_items) == 3 and len(q_items) == 3
    assert geo_identity["release_id"] == _PERCEPTION_CHANNEL.release_id
    assert q_identity["release_id"] == _Q_MODEL_RELEASE
