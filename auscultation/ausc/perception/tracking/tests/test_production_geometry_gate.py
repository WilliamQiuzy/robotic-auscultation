from __future__ import annotations

from dataclasses import replace
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.tracking import geometry_channel_packet as geometry_packet_module
from ausc.perception.tracking import production_geometry_gate as geometry_gate_module
from ausc.perception.tracking.back_mask_packet import BackMaskPacket
from ausc.perception.tracking.patient_frame_evidence import PatientFrameEvidence
from ausc.perception.tracking.production_geometry_gate import (
    PRODUCTION_DECISION_SURFACE,
    ProductionGeometryGate,
    ProductionGeometryProfile,
    production_geometry_profile_semantic_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
POINT_NAMES = ("R1", "L1", "R2", "L2")
EPISODE_A = "episode-" + "1" * 32
EPISODE_B = "episode-" + "2" * 32

_ROBOT_TEST_PATH = (
    Path(__file__).parents[2]
    / "robot"
    / "tests"
    / "test_frame_evidence_bundle.py"
)
_ROBOT_SPEC = importlib.util.spec_from_file_location(
    "_wam_geometry_frame_bundle_helpers", _ROBOT_TEST_PATH
)
assert _ROBOT_SPEC is not None and _ROBOT_SPEC.loader is not None
_ROBOT_HELPERS = importlib.util.module_from_spec(_ROBOT_SPEC)
_ROBOT_SPEC.loader.exec_module(_ROBOT_HELPERS)


def _profile(**changes) -> ProductionGeometryProfile:
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
    values.update(changes)
    values.setdefault(
        "semantic_sha256",
        production_geometry_profile_semantic_sha256(**values),
    )
    return ProductionGeometryProfile(**values)


def _broad_four() -> np.ndarray:
    return np.array(
        [
            [-0.12, -0.15, 1.00],
            [0.12, -0.15, 1.00],
            [-0.12, 0.15, 1.00],
            [0.12, 0.15, 1.00],
        ],
        dtype=np.float64,
    )


def _covariances(count: int, std_m: float = 5.0e-4) -> np.ndarray:
    return np.repeat(
        (np.eye(3, dtype=np.float64) * std_m**2)[None, :, :],
        count,
        axis=0,
    )


def _bound_evidence(
    gate: ProductionGeometryGate,
    reference: np.ndarray,
    observed: np.ndarray,
    targets: np.ndarray,
    *,
    independent_model_points_m: np.ndarray | None = None,
    uncertainty_std_m: float = 5.0e-4,
    aligned_depth_frame: np.ndarray | None = None,
    aligned_back_mask: np.ndarray | None = None,
    aligned_robot_mask: np.ndarray | None = None,
    sample_pixels_uv: np.ndarray | None = None,
):
    height = width = 64
    depth = (
        np.full((height, width), 1000, dtype=np.uint16)
        if aligned_depth_frame is None
        else np.asarray(aligned_depth_frame, dtype=np.uint16)
    )
    camera = np.zeros((height, width, 3), dtype=np.uint8)
    pixels = (
        np.array(
            [[12.0, 12.0], [51.0, 12.0], [12.0, 51.0], [51.0, 51.0]]
            + [
                [32.0, float(8 + index * 8)]
                for index in range(max(0, len(reference) - 4))
            ],
            dtype=np.float64,
        )[: len(reference)]
        if sample_pixels_uv is None
        else np.asarray(sample_pixels_uv, dtype=np.float64)
    )
    back = (
        np.ones(depth.shape, dtype=np.bool_)
        if aligned_back_mask is None
        else aligned_back_mask
    )
    robot = (
        np.zeros(depth.shape, dtype=np.bool_)
        if aligned_robot_mask is None
        else aligned_robot_mask
    )
    model = (
        observed
        if independent_model_points_m is None
        else independent_model_points_m
    )
    return gate.bind_evidence(
        frame_id="frame-000042",
        camera_frame=camera,
        aligned_depth_frame=depth,
        depth_scale_mm_per_unit=1.0,
        sample_pixels_uv=pixels,
        aligned_back_mask=back,
        aligned_robot_mask=robot,
        reference_points_m=reference,
        observed_points_m=observed,
        target_reference_points_m=targets,
        mask_confidence=np.full(len(reference), 0.95),
        independent_model_points_m=model,
        reference_covariances_m2=_covariances(
            len(reference), uncertainty_std_m,
        ),
        target_covariances_m2=_covariances(len(targets), uncertainty_std_m),
        independent_model_covariances_m2=_covariances(
            len(reference), uncertainty_std_m,
        ),
    )


def _evaluate(
    gate: ProductionGeometryGate,
    reference: np.ndarray,
    observed: np.ndarray,
    targets: np.ndarray,
    **evidence_changes,
):
    evidence = _bound_evidence(
        gate,
        reference,
        observed,
        targets,
        **evidence_changes,
    )
    return gate.evaluate_diagnostic(
        reference_points_m=reference,
        observed_points_m=observed,
        target_reference_points_m=targets,
        evidence=evidence,
    )


def _rehash_evidence(evidence, **changes):
    values = {
        name: changes.get(name, getattr(evidence, name))
        for name in evidence.__dataclass_fields__
        if name != "evidence_sha256"
    }
    digest = geometry_gate_module._canonical_sha256(
        geometry_gate_module._evidence_payload(values)
    )
    return replace(evidence, **changes, evidence_sha256=digest)


def _rehash_production_evidence(evidence, **changes):
    values = {
        name: changes.get(name, getattr(evidence, name))
        for name in evidence.__dataclass_fields__
        if name != "evidence_sha256"
    }
    provisional = object.__new__(type(evidence))
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    digest = geometry_gate_module._canonical_sha256(
        geometry_gate_module._production_evidence_payload(provisional)
    )
    return replace(evidence, **changes, evidence_sha256=digest)


def _patient_frame_evidence(
    *,
    frame_index: int = 12,
    camera_host_monotonic_ns: int = 1_050_000_000,
) -> PatientFrameEvidence:
    host_s = camera_host_monotonic_ns / 1_000_000_000
    device_t_ms = 400.0 + (camera_host_monotonic_ns - 1_050_000_000) / 1_000_000
    exposure = _ROBOT_HELPERS._exposure(
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
    trajectory = _ROBOT_HELPERS._trajectory(
        times=tuple(
            camera_host_monotonic_ns - 41_000_000 + offset * 1_000_000
            for offset in range(13)
        ),
        lower_ns=camera_host_monotonic_ns - 40_000_000,
        nominal_ns=camera_host_monotonic_ns - 35_000_000,
        upper_ns=camera_host_monotonic_ns - 30_000_000,
    )
    camera = _ROBOT_HELPERS._camera_frame(
        index=frame_index,
        t=host_s,
        device_t_ms=device_t_ms,
        rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        depth=np.full((64, 64), 1000, dtype=np.uint16),
    )
    robot_packet = _ROBOT_HELPERS._mask_packet(
        exposure=exposure,
        trajectory=trajectory,
        mask=np.zeros((64, 64), dtype=np.bool_),
    )
    frame_bundle = _ROBOT_HELPERS._bundle(
        camera_frame=camera,
        exposure=exposure,
        trajectory=trajectory,
        robot_mask_packet=robot_packet,
    )
    back_packet = BackMaskPacket.build(
        camera_frame=frame_bundle.camera_frame,
        mask=np.ones((64, 64), dtype=np.bool_),
        back_mask_release_id="back_mask-" + "9" * 20,
        back_mask_model_sha256="8" * 64,
        operator_gate_decision_id="operator_back_gate-" + "7" * 64,
    )
    return PatientFrameEvidence.build(
        frame_bundle=frame_bundle,
        back_mask_packet=back_packet,
    )


def _geometry_packet(
    *,
    channel: str,
    producer_id: str,
    model_sha256: str,
    patient_frame_evidence: PatientFrameEvidence,
    episode_id: str,
    sequence: int,
    points_m: np.ndarray,
    point_names: tuple[str, ...] = POINT_NAMES,
):
    return geometry_gate_module.GeometryChannelPacket.build(
        channel=channel,
        producer_id=producer_id,
        release_id=f"{channel.replace('_', '-')}-{model_sha256[:20]}",
        model_sha256=model_sha256,
        patient_frame_evidence=patient_frame_evidence,
        episode_id=episode_id,
        sequence=sequence,
        point_names=point_names,
        points_m=points_m,
        covariances_m2=_covariances(len(points_m)),
    )


def _bind_production(
    gate: ProductionGeometryGate,
    *,
    patient_frame_evidence: PatientFrameEvidence,
    episode_id: str,
    sequence: int,
    reference: np.ndarray,
    targets: np.ndarray,
    perception_packet=None,
    independent_packet=None,
):
    perception_packet = perception_packet or _geometry_packet(
        channel="perception_observation",
        producer_id="v9-landmark-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=patient_frame_evidence,
        episode_id=episode_id,
        sequence=sequence,
        points_m=reference,
    )
    independent_packet = independent_packet or _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_D,
        patient_frame_evidence=patient_frame_evidence,
        episode_id=episode_id,
        sequence=sequence,
        points_m=reference,
    )
    evidence = gate.bind_production_evidence(
        patient_frame_evidence=patient_frame_evidence,
        episode_id=episode_id,
        sequence=sequence,
        point_names=POINT_NAMES,
        perception_observation_packet=perception_packet,
        independent_model_packet=independent_packet,
        sample_pixels_uv=np.array(
            [[12.0, 12.0], [51.0, 12.0], [12.0, 51.0], [51.0, 51.0]],
            dtype=np.float64,
        ),
        reference_points_m=reference,
        target_reference_points_m=targets,
        mask_confidence=np.full(len(reference), 0.95),
        reference_covariances_m2=_covariances(len(reference)),
        target_covariances_m2=_covariances(len(targets)),
    )
    return perception_packet, independent_packet, evidence


def _geometry_packet_log(*, count: int = 3, episode_id: str = EPISODE_A):
    points = _broad_four()
    entries = []
    packets = []
    for offset in range(count):
        patient_evidence = _patient_frame_evidence(
            frame_index=12 + offset,
            camera_host_monotonic_ns=1_050_000_000 + offset * 10_000_000,
        )
        perception = _geometry_packet(
            channel="perception_observation",
            producer_id="v9-landmark-worker",
            model_sha256=SHA_E,
            patient_frame_evidence=patient_evidence,
            episode_id=episode_id,
            sequence=10 + offset,
            points_m=points,
        )
        independent = _geometry_packet(
            channel="independent_patient_model",
            producer_id="metric-patient-model",
            model_sha256=SHA_D,
            patient_frame_evidence=patient_evidence,
            episode_id=episode_id,
            sequence=10 + offset,
            points_m=points,
        )
        entry = geometry_packet_module.GeometryFramePacketLogEntry.build(
            perception_observation_packet=perception,
            independent_model_packet=independent,
            previous_entry=entries[-1] if entries else None,
        )
        entries.append(entry)
        packets.append((perception, independent))
    return tuple(entries), tuple(packets)


def test_four_well_distributed_correspondences_can_pass_without_a_six_point_rule() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    result = _evaluate(gate, points, points, points)

    assert result.accepted is True
    assert result.mathematical_support_count == 4
    assert result.effective_support < 6.0
    assert result.max_target_error_p95_m <= 0.010
    assert result.camera_profile_sha256 == SHA_A
    assert result.decision == "diagnostic_geometry_surface_pass"


def test_six_nearly_collinear_points_fail_despite_higher_count() -> None:
    points = np.array(
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
    target = np.array([[0.0, 0.14, 1.0]])

    gate = ProductionGeometryGate(_profile())
    result = _evaluate(gate, points, points, target)

    assert result.accepted is False
    assert result.mathematical_support_count == 6
    assert "geometry_surface_rejected" in result.failure_reasons


def test_three_points_fail_the_mathematical_floor_before_quality_authorization() -> None:
    points = _broad_four()[:3]

    gate = ProductionGeometryGate(_profile())
    result = _evaluate(gate, points, points, points)

    assert result.accepted is False
    assert result.failure_reasons == ("insufficient_mathematical_support",)


def test_scale_change_is_rejected_even_when_shape_and_count_are_good() -> None:
    reference = _broad_four()
    observed = reference.copy()
    centre = observed.mean(axis=0)
    observed = centre + 1.03 * (observed - centre)

    gate = ProductionGeometryGate(_profile())
    result = _evaluate(
        gate,
        reference,
        observed,
        reference,
        independent_model_points_m=observed,
    )

    assert result.accepted is False
    assert result.scale_audit == pytest.approx(1.03)
    assert "scale_audit_failed" in result.failure_reasons


@pytest.mark.parametrize(
    "profile",
    (
        lambda: _profile(
            decision_surface=replace(
                PRODUCTION_DECISION_SURFACE,
                max_target_error_p95_m=0.0101,
            )
        ),
        lambda: _profile(max_scale_deviation_relative=0.0201),
        lambda: _profile(
            depth_sample_window_px=0,
            min_depth_patch_valid_pixels=1,
        ),
        lambda: _profile(min_depth_patch_valid_pixels=8),
        lambda: _profile(camera_profile_sha256=""),
    ),
)
def test_profile_cannot_loosen_or_omit_activated_safety_identity(profile) -> None:
    with pytest.raises(ValueError):
        profile()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lateral_noise_std_m", 9.0e-4),
        ("axial_noise_std_m", 1.9e-3),
        ("robust_delta_m", 2.9e-3),
        ("max_scale_deviation_relative", 0.019),
        ("depth_sample_window_px", 1),
        ("min_depth_m", 0.051),
        ("max_depth_m", 2.9),
        ("min_depth_patch_valid_pixels", 10),
        ("min_depth_patch_valid_fraction", 0.61),
        ("max_depth_patch_mad_m", 3.9e-3),
        ("max_depth_patch_p90_p10_m", 0.019),
        ("min_back_mask_support_fraction", 0.61),
        ("max_independent_model_innovation_m", 0.009),
        ("max_independent_model_innovation_sigma", 2.9),
    ),
)
def test_semantic_hash_rejects_tampered_numeric_profile_values(
    field: str,
    value: float,
) -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="semantic"):
        replace(profile, **{field: value})


@pytest.mark.parametrize(
    "field",
    tuple(vars(PRODUCTION_DECISION_SURFACE)),
)
def test_semantic_hash_binds_every_numeric_decision_surface_value(field: str) -> None:
    profile = _profile()
    surface = replace(
        profile.decision_surface,
        **{field: getattr(profile.decision_surface, field) * 0.99},
    )
    with pytest.raises(ValueError, match="semantic"):
        replace(profile, decision_surface=surface)


def test_evaluate_requires_bound_same_frame_depth_and_mask_evidence() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())

    missing = gate.evaluate_diagnostic(
        reference_points_m=points,
        observed_points_m=points,
        target_reference_points_m=points,
        evidence=None,
    )
    evidence = _bound_evidence(gate, points, points, points)
    altered_observation = points.copy()
    altered_observation[0, 0] += 1.0e-4
    mismatch = gate.evaluate_diagnostic(
        reference_points_m=points,
        observed_points_m=altered_observation,
        target_reference_points_m=points,
        evidence=evidence,
    )

    assert missing.accepted is False
    assert missing.failure_reasons == ("missing_bound_geometry_evidence",)
    assert mismatch.accepted is False
    assert mismatch.failure_reasons == ("geometry_evidence_identity_mismatch",)


def test_raw_bound_evidence_cannot_authorize_production_geometry() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    evidence = _bound_evidence(gate, points, points, points)

    result = gate.evaluate(
        reference_points_m=points,
        observed_points_m=points,
        target_reference_points_m=points,
        evidence=evidence,
    )

    assert result.accepted is False
    assert result.failure_reasons == (
        "production_bound_geometry_evidence_required",
    )


def test_typed_production_bind_authorizes_independent_same_frame_packet() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    perception_packet, independent_packet, evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    assert evidence.frame_bundle_id == patient_evidence.frame_bundle.bundle_id
    assert evidence.patient_frame_evidence_id == patient_evidence.evidence_id
    assert evidence.episode_id == EPISODE_A
    assert evidence.frame_index == patient_evidence.index
    assert evidence.camera_host_monotonic_ns == (
        patient_evidence.frame_bundle.camera_frame.camera_host_monotonic_ns
    )
    assert evidence.sequence == 1
    assert evidence.point_names == POINT_NAMES
    assert evidence.perception_inference_id == perception_packet.inference_id
    assert (
        evidence.independent_model_inference_id
        == independent_packet.inference_id
    )
    assert independent_packet.release_id.endswith(SHA_D[:20])
    assert independent_packet.model_sha256 == SHA_D
    assert independent_packet.input_rgb_sha256 == (
        patient_evidence.frame_bundle.camera_frame.rgb_sha256
    )
    assert independent_packet.input_depth_sha256 == (
        patient_evidence.frame_bundle.camera_frame.depth_sha256
    )
    assert independent_packet.point_names == POINT_NAMES
    assert independent_packet.covariances_m2.shape == (len(POINT_NAMES), 3, 3)
    assert independent_packet.points_m.flags.writeable is False
    assert independent_packet.covariances_m2.flags.writeable is False

    result = gate.evaluate(
        reference_points_m=points,
        observed_points_m=perception_packet.points_m,
        target_reference_points_m=points,
        evidence=evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
    )

    assert result.accepted is True
    assert result.decision == "accepted_by_production_geometry_surface"


def test_geometry_session_receipt_verifies_every_packet_pair_in_full_chain() -> None:
    entries, packets = _geometry_packet_log()
    receipt = geometry_packet_module.GeometrySessionReceipt.build(entries)

    verified = geometry_packet_module.verify_geometry_packet_log(
        entries,
        receipt,
    )

    assert verified == receipt
    assert receipt.packet_count == 3
    assert receipt.first_sequence == 10
    assert receipt.last_sequence == 12
    assert receipt.episode_id == EPISODE_A
    assert receipt.perception_producer_id == "v9-landmark-worker"
    assert receipt.perception_model_sha256 == SHA_E
    assert receipt.perception_release_id == packets[-1][0].release_id
    assert receipt.independent_producer_id == "metric-patient-model"
    assert receipt.independent_model_sha256 == SHA_D
    assert receipt.independent_release_id == packets[-1][1].release_id
    assert receipt.last_entry_chain_sha256 == entries[-1].entry_chain_sha256
    assert len(receipt.packet_log_sha256) == 64
    assert receipt.production_authorized is False
    assert receipt.authorization_blockers == (
        "geometry packet receipt alone is not finalization-bound for production",
    )
    with pytest.raises(ValueError, match="cannot authorize production"):
        replace(receipt, production_authorized=True)


def test_geometry_packet_log_files_round_trip_full_numeric_evidence(
    tmp_path: Path,
) -> None:
    entries, packets = _geometry_packet_log()

    written = geometry_packet_module.write_geometry_packet_log_files(
        tmp_path, entries
    )
    loaded_entries, loaded_receipt = (
        geometry_packet_module.verify_geometry_packet_log_files(tmp_path)
    )

    assert written == loaded_receipt
    assert len(loaded_entries) == len(entries)
    np.testing.assert_array_equal(
        loaded_entries[1].perception_observation_packet.points_m,
        packets[1][0].points_m,
    )
    np.testing.assert_array_equal(
        loaded_entries[1].independent_model_packet.covariances_m2,
        packets[1][1].covariances_m2,
    )
    raw_entry = json.loads(
        (tmp_path / "geometry_frame_packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1]
    )
    assert "points_m" in raw_entry["perception_observation_packet"]
    assert "covariances_m2" in raw_entry["independent_model_packet"]


def test_geometry_packet_log_file_rejects_numeric_tamper_even_if_hash_claims_stay(
    tmp_path: Path,
) -> None:
    entries, _ = _geometry_packet_log()
    geometry_packet_module.write_geometry_packet_log_files(tmp_path, entries)
    path = tmp_path / "geometry_frame_packets.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["independent_model_packet"]["points_m"][0][0] += 0.05
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="points SHA-256|changed|receipt",
    ):
        geometry_packet_module.verify_geometry_packet_log_files(tmp_path)


def test_geometry_session_receipt_rejects_log_with_missing_middle_packet() -> None:
    entries, _ = _geometry_packet_log()
    receipt = geometry_packet_module.GeometrySessionReceipt.build(entries)

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="continuous|count|chain",
    ):
        geometry_packet_module.verify_geometry_packet_log(
            (entries[0], entries[2]),
            receipt,
        )


def test_geometry_session_receipt_cannot_be_issued_from_only_last_packet() -> None:
    entries, _ = _geometry_packet_log(count=1)

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="at least 2",
    ):
        geometry_packet_module.GeometrySessionReceipt.build(entries)


def test_geometry_full_chain_binds_changed_middle_packet_content() -> None:
    entries, packets = _geometry_packet_log()
    receipt = geometry_packet_module.GeometrySessionReceipt.build(entries)
    middle_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    changed_independent = _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_D,
        patient_frame_evidence=middle_frame,
        episode_id=EPISODE_A,
        sequence=11,
        points_m=_broad_four() + np.array([0.001, 0.0, 0.0]),
    )
    changed_middle = geometry_packet_module.GeometryFramePacketLogEntry.build(
        perception_observation_packet=packets[1][0],
        independent_model_packet=changed_independent,
        previous_entry=entries[0],
    )
    changed_last = geometry_packet_module.GeometryFramePacketLogEntry.build(
        perception_observation_packet=packets[2][0],
        independent_model_packet=packets[2][1],
        previous_entry=changed_middle,
    )

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="does not match receipt",
    ):
        geometry_packet_module.verify_geometry_packet_log(
            (entries[0], changed_middle, changed_last),
            receipt,
        )


@pytest.mark.parametrize(
    ("producer_id", "model_sha256"),
    [
        ("alternate-patient-model", SHA_D),
        ("metric-patient-model", SHA_C),
    ],
)
def test_geometry_full_chain_rejects_mid_session_source_identity_drift(
    producer_id,
    model_sha256,
) -> None:
    entries, packets = _geometry_packet_log()
    middle_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    changed_independent = _geometry_packet(
        channel="independent_patient_model",
        producer_id=producer_id,
        model_sha256=model_sha256,
        patient_frame_evidence=middle_frame,
        episode_id=EPISODE_A,
        sequence=11,
        points_m=_broad_four(),
    )
    changed_middle = geometry_packet_module.GeometryFramePacketLogEntry.build(
        perception_observation_packet=packets[1][0],
        independent_model_packet=changed_independent,
        previous_entry=entries[0],
    )

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="session identity changed",
    ):
        geometry_packet_module.GeometrySessionReceipt.build(
            (entries[0], changed_middle)
        )


def test_geometry_full_chain_rejects_cross_episode_packet_reuse() -> None:
    first_entries, _ = _geometry_packet_log(count=1, episode_id=EPISODE_A)
    _, second_packets = _geometry_packet_log(count=2, episode_id=EPISODE_B)
    cross_episode = geometry_packet_module.GeometryFramePacketLogEntry.build(
        perception_observation_packet=second_packets[1][0],
        independent_model_packet=second_packets[1][1],
        previous_entry=first_entries[0],
    )

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="session identity changed",
    ):
        geometry_packet_module.GeometrySessionReceipt.build(
            (first_entries[0], cross_episode)
        )


def test_geometry_packet_log_pair_rejects_cross_frame_rgbd_binding() -> None:
    _, packets = _geometry_packet_log(count=2)

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="one frame",
    ):
        geometry_packet_module.GeometryFramePacketLogEntry.build(
            perception_observation_packet=packets[0][0],
            independent_model_packet=packets[1][1],
        )


def test_geometry_packet_log_rejects_replayed_frame_with_new_sequence() -> None:
    entries, packets = _geometry_packet_log(count=2)
    first_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    replay_perception = _geometry_packet(
        channel="perception_observation",
        producer_id="v9-landmark-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=11,
        points_m=_broad_four(),
    )
    replay_independent = _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_D,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=11,
        points_m=_broad_four(),
    )
    replay_entry = geometry_packet_module.GeometryFramePacketLogEntry.build(
        perception_observation_packet=replay_perception,
        independent_model_packet=replay_independent,
        previous_entry=entries[0],
    )

    with pytest.raises(
        geometry_packet_module.GeometryPacketLogError,
        match="frame time did not advance|replays frame evidence",
    ):
        geometry_packet_module.GeometrySessionReceipt.build(
            (entries[0], replay_entry)
        )


@pytest.mark.parametrize("caller_sequence", [1.0, True])
def test_production_evaluate_rejects_non_integer_caller_sequence(
    caller_sequence,
) -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    perception_packet, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    result = gate.evaluate(
        reference_points_m=points,
        observed_points_m=perception_packet.points_m,
        target_reference_points_m=points,
        evidence=evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=caller_sequence,
    )

    assert result.accepted is False
    assert result.failure_reasons == ("production_geometry_context_invalid",)


def test_production_bind_rejects_same_geometry_producer_circular_self_proof() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    perception_packet = _geometry_packet(
        channel="perception_observation",
        producer_id="shared-geometry-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )
    independent_packet = _geometry_packet(
        channel="independent_patient_model",
        producer_id="shared-geometry-worker",
        model_sha256=SHA_D,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )

    with pytest.raises(ValueError, match="not independent"):
        _bind_production(
            gate,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            reference=points,
            targets=points,
            perception_packet=perception_packet,
            independent_packet=independent_packet,
        )


def test_production_bind_rejects_same_model_identity_circular_self_proof() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile(independent_model_sha256=SHA_E))
    perception_packet = _geometry_packet(
        channel="perception_observation",
        producer_id="landmark-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )
    independent_packet = _geometry_packet(
        channel="independent_patient_model",
        producer_id="patient-model-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )

    with pytest.raises(ValueError, match="not independent"):
        _bind_production(
            gate,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            reference=points,
            targets=points,
            perception_packet=perception_packet,
            independent_packet=independent_packet,
        )


def test_geometry_packet_rejects_release_not_bound_to_model_sha() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()

    with pytest.raises(ValueError, match="release identity"):
        geometry_gate_module.GeometryChannelPacket.build(
            channel="independent_patient_model",
            producer_id="metric-patient-model",
            release_id=f"independent-patient-model-{SHA_E[:20]}",
            model_sha256=SHA_D,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            point_names=POINT_NAMES,
            points_m=points,
            covariances_m2=_covariances(len(points)),
        )


def test_geometry_packet_rejects_zero_covariance_claim() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()

    with pytest.raises(ValueError, match="positive semidefinite and nonzero"):
        geometry_gate_module.GeometryChannelPacket.build(
            channel="independent_patient_model",
            producer_id="metric-patient-model",
            release_id=f"independent-patient-model-{SHA_D[:20]}",
            model_sha256=SHA_D,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            point_names=POINT_NAMES,
            points_m=points,
            covariances_m2=np.zeros((len(points), 3, 3), dtype=np.float64),
        )


def test_production_bind_rejects_independent_model_sha_outside_profile() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    independent_packet = _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_C,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )

    with pytest.raises(ValueError, match="does not match profile"):
        _bind_production(
            gate,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            reference=points,
            targets=points,
            independent_packet=independent_packet,
        )


def test_production_bind_rejects_independent_point_order_drift() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    independent_packet = _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_D,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points[::-1],
        point_names=tuple(reversed(POINT_NAMES)),
    )

    with pytest.raises(ValueError, match="point names/order"):
        _bind_production(
            gate,
            patient_frame_evidence=patient_evidence,
            episode_id=EPISODE_A,
            sequence=1,
            reference=points,
            targets=points,
            independent_packet=independent_packet,
        )


def test_production_bind_rejects_packet_from_different_patient_frame() -> None:
    points = _broad_four()
    source_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    target_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    independent_packet = _geometry_packet(
        channel="independent_patient_model",
        producer_id="metric-patient-model",
        model_sha256=SHA_D,
        patient_frame_evidence=source_frame,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=points,
    )

    with pytest.raises(ValueError, match="not bound to this frame"):
        _bind_production(
            gate,
            patient_frame_evidence=target_frame,
            episode_id=EPISODE_A,
            sequence=1,
            reference=points,
            targets=points,
            independent_packet=independent_packet,
        )


def test_production_gate_rejects_exact_geometry_evidence_replay() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    perception_packet, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )
    arguments = {
        "reference_points_m": points,
        "observed_points_m": perception_packet.points_m,
        "target_reference_points_m": points,
        "evidence": evidence,
        "patient_frame_evidence": patient_evidence,
        "episode_id": EPISODE_A,
        "sequence": 1,
    }

    first = gate.evaluate(**arguments)
    replay = gate.evaluate(**arguments)

    assert first.accepted is True
    assert replay.accepted is False
    assert replay.failure_reasons == ("production_geometry_evidence_replayed",)


def test_production_gate_rejects_non_monotonic_sequence_in_same_episode() -> None:
    points = _broad_four()
    first_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    next_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    first_packet, _, first_evidence = _bind_production(
        gate,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=2,
        reference=points,
        targets=points,
    )
    next_packet, _, next_evidence = _bind_production(
        gate,
        patient_frame_evidence=next_frame,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    first = gate.evaluate(
        reference_points_m=points,
        observed_points_m=first_packet.points_m,
        target_reference_points_m=points,
        evidence=first_evidence,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=2,
    )
    rollback = gate.evaluate(
        reference_points_m=points,
        observed_points_m=next_packet.points_m,
        target_reference_points_m=points,
        evidence=next_evidence,
        patient_frame_evidence=next_frame,
        episode_id=EPISODE_A,
        sequence=1,
    )

    assert first.accepted is True
    assert rollback.accepted is False
    assert rollback.failure_reasons == (
        "production_geometry_sequence_not_monotonic",
    )


def test_production_gate_rejects_reused_frame_with_newer_sequence() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    first_packet, _, first_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )
    reused_packet, _, reused_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=2,
        reference=points,
        targets=points,
    )

    first = gate.evaluate(
        reference_points_m=points,
        observed_points_m=first_packet.points_m,
        target_reference_points_m=points,
        evidence=first_evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
    )
    reused = gate.evaluate(
        reference_points_m=points,
        observed_points_m=reused_packet.points_m,
        target_reference_points_m=points,
        evidence=reused_evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=2,
    )

    assert first.accepted is True
    assert reused.accepted is False
    assert reused.failure_reasons == (
        "production_geometry_frame_not_monotonic",
    )


def test_production_gate_accepts_strictly_monotonic_next_frame() -> None:
    points = _broad_four()
    first_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    next_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    first_packet, _, first_evidence = _bind_production(
        gate,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )
    next_packet, _, next_evidence = _bind_production(
        gate,
        patient_frame_evidence=next_frame,
        episode_id=EPISODE_A,
        sequence=2,
        reference=points,
        targets=points,
    )

    first = gate.evaluate(
        reference_points_m=points,
        observed_points_m=first_packet.points_m,
        target_reference_points_m=points,
        evidence=first_evidence,
        patient_frame_evidence=first_frame,
        episode_id=EPISODE_A,
        sequence=1,
    )
    next_result = gate.evaluate(
        reference_points_m=points,
        observed_points_m=next_packet.points_m,
        target_reference_points_m=points,
        evidence=next_evidence,
        patient_frame_evidence=next_frame,
        episode_id=EPISODE_A,
        sequence=2,
    )

    assert first.accepted is True
    assert next_result.accepted is True


def test_production_evidence_cannot_be_evaluated_against_another_live_frame() -> None:
    points = _broad_four()
    source_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    target_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    packet, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=source_frame,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    result = gate.evaluate(
        reference_points_m=points,
        observed_points_m=packet.points_m,
        target_reference_points_m=points,
        evidence=evidence,
        patient_frame_evidence=target_frame,
        episode_id=EPISODE_A,
        sequence=1,
    )

    assert result.accepted is False
    assert result.failure_reasons == ("production_geometry_context_mismatch",)


def test_rejected_geometry_does_not_advance_accepted_episode_watermark() -> None:
    points = _broad_four()
    valid_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    untrusted_future_frame = _patient_frame_evidence(
        frame_index=99,
        camera_host_monotonic_ns=2_000_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    displaced = points + np.array([0.05, 0.0, 0.0], dtype=np.float64)
    untrusted_packet = _geometry_packet(
        channel="perception_observation",
        producer_id="v9-landmark-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=untrusted_future_frame,
        episode_id=EPISODE_A,
        sequence=99,
        points_m=displaced,
    )
    untrusted_packet, _, untrusted_evidence = _bind_production(
        gate,
        patient_frame_evidence=untrusted_future_frame,
        episode_id=EPISODE_A,
        sequence=99,
        reference=points,
        targets=points,
        perception_packet=untrusted_packet,
    )
    valid_packet, _, valid_evidence = _bind_production(
        gate,
        patient_frame_evidence=valid_frame,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    rejected = gate.evaluate(
        reference_points_m=points,
        observed_points_m=untrusted_packet.points_m,
        target_reference_points_m=points,
        evidence=untrusted_evidence,
        patient_frame_evidence=untrusted_future_frame,
        episode_id=EPISODE_A,
        sequence=99,
    )
    valid = gate.evaluate(
        reference_points_m=points,
        observed_points_m=valid_packet.points_m,
        target_reference_points_m=points,
        evidence=valid_evidence,
        patient_frame_evidence=valid_frame,
        episode_id=EPISODE_A,
        sequence=1,
    )

    assert rejected.accepted is False
    assert valid.accepted is True


def test_rejected_evidence_replay_fails_but_new_inference_can_retry_frame() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    displaced = points + np.array([0.05, 0.0, 0.0], dtype=np.float64)
    rejected_packet = _geometry_packet(
        channel="perception_observation",
        producer_id="v9-landmark-worker",
        model_sha256=SHA_E,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        points_m=displaced,
    )
    rejected_packet, _, rejected_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
        perception_packet=rejected_packet,
    )
    retry_packet, _, retry_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=2,
        reference=points,
        targets=points,
    )
    rejected_arguments = {
        "reference_points_m": points,
        "observed_points_m": rejected_packet.points_m,
        "target_reference_points_m": points,
        "evidence": rejected_evidence,
        "patient_frame_evidence": patient_evidence,
        "episode_id": EPISODE_A,
        "sequence": 1,
    }

    rejected = gate.evaluate(**rejected_arguments)
    replay = gate.evaluate(**rejected_arguments)
    retry = gate.evaluate(
        reference_points_m=points,
        observed_points_m=retry_packet.points_m,
        target_reference_points_m=points,
        evidence=retry_evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=2,
    )

    assert rejected.accepted is False
    assert replay.accepted is False
    assert replay.failure_reasons == ("production_geometry_evidence_replayed",)
    assert retry.accepted is True


def test_rehashed_envelope_cannot_retarget_packets_to_another_frame() -> None:
    points = _broad_four()
    source_frame = _patient_frame_evidence(
        frame_index=12,
        camera_host_monotonic_ns=1_050_000_000,
    )
    target_frame = _patient_frame_evidence(
        frame_index=13,
        camera_host_monotonic_ns=1_060_000_000,
    )
    gate = ProductionGeometryGate(_profile())
    _, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=source_frame,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    with pytest.raises(ValueError, match="packet context"):
        _rehash_production_evidence(
            evidence,
            frame_bundle_id=target_frame.frame_bundle.bundle_id,
            patient_frame_evidence_id=target_frame.evidence_id,
            frame_index=target_frame.index,
            camera_host_monotonic_ns=(
                target_frame.frame_bundle.camera_frame.camera_host_monotonic_ns
            ),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_index": 12.0},
        {"camera_host_monotonic_ns": 1_050_000_000.0},
        {"sequence": True},
    ],
)
def test_rehashed_envelope_rejects_numeric_metadata_type_confusion(changes) -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    _, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )

    with pytest.raises(ValueError, match="metadata is invalid"):
        _rehash_production_evidence(evidence, **changes)


def test_rehashed_raw_observation_cannot_diverge_from_perception_packet() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    _, _, evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )
    displaced = points + np.array([0.001, 0.0, 0.0], dtype=np.float64)
    tampered_raw = _rehash_evidence(
        evidence.raw_geometry_evidence,
        observed_points_sha256=geometry_gate_module._array_sha256(
            displaced,
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError, match="packet payload"):
        _rehash_production_evidence(
            evidence,
            raw_geometry_evidence=tampered_raw,
        )


def test_same_patient_frame_cannot_be_rebound_into_another_episode() -> None:
    points = _broad_four()
    patient_evidence = _patient_frame_evidence()
    gate = ProductionGeometryGate(_profile())
    first_packet, _, first_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
        reference=points,
        targets=points,
    )
    reused_packet, _, reused_evidence = _bind_production(
        gate,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_B,
        sequence=1,
        reference=points,
        targets=points,
    )

    first = gate.evaluate(
        reference_points_m=points,
        observed_points_m=first_packet.points_m,
        target_reference_points_m=points,
        evidence=first_evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_A,
        sequence=1,
    )
    reused = gate.evaluate(
        reference_points_m=points,
        observed_points_m=reused_packet.points_m,
        target_reference_points_m=points,
        evidence=reused_evidence,
        patient_frame_evidence=patient_evidence,
        episode_id=EPISODE_B,
        sequence=1,
    )

    assert first.accepted is True
    assert reused.accepted is False
    assert reused.failure_reasons == (
        "production_geometry_frame_reused_across_episodes",
    )


def test_bound_evidence_is_immutable_and_rejects_content_tampering() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    evidence = _bound_evidence(gate, points, points, points)
    tampered_depth = list(evidence.sampled_depth_m)
    tampered_depth[0] += 1.0e-3

    with pytest.raises(ValueError, match="semantic"):
        replace(evidence, sampled_depth_m=tuple(tampered_depth))
    with pytest.raises(ValueError, match="immutable"):
        replace(evidence, sampled_depth_m=list(evidence.sampled_depth_m))


def test_frame_border_cannot_supply_a_clipped_depth_patch() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    evidence = _bound_evidence(
        gate,
        points,
        points,
        points,
        sample_pixels_uv=np.array(
            [[0.0, 0.0], [63.0, 0.0], [0.0, 63.0], [63.0, 63.0]],
            dtype=np.float64,
        ),
    )

    assert evidence.depth_patch_pixel_count == (9, 9, 9, 9)
    result = gate.evaluate_diagnostic(
        reference_points_m=points,
        observed_points_m=points,
        target_reference_points_m=points,
        evidence=evidence,
    )

    assert result.accepted is False
    assert result.failure_reasons == ("depth_patch_evidence_inconsistent",)


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        (
            "wrong configured area",
            {
                "depth_patch_pixel_count": (36, 36, 36, 36),
                "depth_patch_valid_fraction": (25.0 / 36.0,) * 4,
            },
        ),
        (
            "fraction disagrees with counts",
            {"depth_patch_valid_fraction": (0.90,) * 4},
        ),
        (
            "fractional valid pixel count",
            {
                "depth_patch_valid_pixel_count": (24.5,) * 4,
                "depth_patch_valid_fraction": (0.98,) * 4,
            },
        ),
        (
            "negative depth MAD",
            {"depth_patch_mad_m": (-1.0e-3,) * 4},
        ),
        (
            "negative robust depth span",
            {"depth_patch_p90_p10_m": (-1.0e-3,) * 4},
        ),
    ),
)
def test_evaluate_rejects_rehashed_cross_inconsistent_depth_patch_evidence(
    case: str,
    changes: dict,
) -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    evidence = _bound_evidence(gate, points, points, points)
    tampered = _rehash_evidence(evidence, **changes)

    result = gate.evaluate_diagnostic(
        reference_points_m=points,
        observed_points_m=points,
        target_reference_points_m=points,
        evidence=tampered,
    )

    assert result.accepted is False, case
    assert result.failure_reasons == ("depth_patch_evidence_inconsistent",)


@pytest.mark.parametrize("sampled_depth_m", (0.05, 3.0))
def test_evaluate_rejects_rehashed_depth_on_profile_boundary(
    sampled_depth_m: float,
) -> None:
    reference = _broad_four()
    observed = reference.copy()
    observed[:, 2] = sampled_depth_m
    gate = ProductionGeometryGate(_profile())
    evidence = _bound_evidence(
        gate,
        reference,
        observed,
        reference,
        independent_model_points_m=observed,
    )
    tampered = _rehash_evidence(
        evidence,
        sampled_depth_m=(sampled_depth_m,) * len(reference),
    )

    result = gate.evaluate_diagnostic(
        reference_points_m=reference,
        observed_points_m=observed,
        target_reference_points_m=reference,
        evidence=tampered,
    )

    assert result.accepted is False
    assert result.failure_reasons == ("depth_patch_evidence_inconsistent",)


def test_robot_intersection_and_missing_back_support_are_not_fit_support() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    robot = np.zeros((64, 64), dtype=np.bool_)
    robot[10:15, 10:15] = True
    back = np.ones((64, 64), dtype=np.bool_)
    back[49:54, 49:54] = False

    result = _evaluate(
        gate,
        points,
        points,
        points,
        aligned_back_mask=back,
        aligned_robot_mask=robot,
    )

    assert result.accepted is False
    assert result.mathematical_support_count == 2
    assert result.failure_reasons == ("insufficient_mathematical_support",)


def test_depth_is_derived_from_bound_frame_not_supplied_as_observed_z() -> None:
    reference = _broad_four()
    observed = reference.copy()
    observed[:, 2] += 0.020
    gate = ProductionGeometryGate(_profile())

    result = _evaluate(
        gate,
        reference,
        observed,
        reference,
        independent_model_points_m=observed,
    )

    assert result.accepted is False
    assert "geometry_surface_rejected" in result.failure_reasons
    assert any(
        "depth_observation_mismatch" in target.failure_reasons
        for target in result.target_results
    )


def test_depth_patch_with_one_valid_pixel_is_not_geometry_support() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    depth = np.zeros((64, 64), dtype=np.uint16)
    for u, v in ((12, 12), (51, 12), (12, 51), (51, 51)):
        depth[v, u] = 1000

    result = _evaluate(
        gate,
        points,
        points,
        points,
        aligned_depth_frame=depth,
    )

    assert result.accepted is False
    assert result.mathematical_support_count == 0
    assert result.failure_reasons == ("insufficient_mathematical_support",)


def test_depth_patch_crossing_a_large_depth_edge_is_not_geometry_support() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    for u, v in ((12, 12), (51, 12), (12, 51), (51, 51)):
        depth[v - 2 : v + 3, u : u + 3] = 1500

    result = _evaluate(
        gate,
        points,
        points,
        points,
        aligned_depth_frame=depth,
    )

    assert result.accepted is False
    assert result.mathematical_support_count == 0
    assert result.failure_reasons == ("insufficient_mathematical_support",)


def test_depth_patch_robust_statistics_tolerate_one_isolated_outlier() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    for u, v in ((12, 12), (51, 12), (12, 51), (51, 51)):
        depth[v - 2, u - 2] = 2000

    result = _evaluate(
        gate,
        points,
        points,
        points,
        aligned_depth_frame=depth,
    )

    assert result.accepted is True
    assert result.mathematical_support_count == 4


def test_reference_target_and_model_uncertainty_contribute_to_target_p95() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())

    precise = _evaluate(gate, points, points, points, uncertainty_std_m=5.0e-4)
    uncertain = _evaluate(gate, points, points, points, uncertainty_std_m=4.0e-3)

    assert precise.accepted is True
    assert precise.max_target_error_p95_m < 0.010
    assert uncertain.accepted is False
    assert uncertain.max_target_error_p95_m > 0.010
    assert "target_uncertainty_above_surface" in uncertain.failure_reasons


def test_uncertainty_evidence_is_mandatory() -> None:
    points = _broad_four()
    gate = ProductionGeometryGate(_profile())
    with pytest.raises(ValueError, match="target_covariances_m2"):
        gate.bind_evidence(
            frame_id="frame-1",
            camera_frame=np.zeros((64, 64, 3), dtype=np.uint8),
            aligned_depth_frame=np.full((64, 64), 1000, dtype=np.uint16),
            depth_scale_mm_per_unit=1.0,
            sample_pixels_uv=np.array(
                [[12, 12], [51, 12], [12, 51], [51, 51]], dtype=np.float64,
            ),
            aligned_back_mask=np.ones((64, 64), dtype=np.bool_),
            aligned_robot_mask=np.zeros((64, 64), dtype=np.bool_),
            reference_points_m=points,
            observed_points_m=points,
            target_reference_points_m=points,
            mask_confidence=np.ones(4),
            independent_model_points_m=points,
            reference_covariances_m2=_covariances(4),
            target_covariances_m2=None,
            independent_model_covariances_m2=_covariances(4),
        )


def test_stable_coherently_wrong_constellation_is_rejected_by_independent_model() -> None:
    reference = _broad_four()
    coherently_wrong = reference + np.array([0.020, 0.0, 0.0])
    gate = ProductionGeometryGate(_profile())

    result = _evaluate(
        gate,
        reference,
        coherently_wrong,
        reference,
        independent_model_points_m=reference,
    )

    assert result.accepted is False
    assert result.max_independent_model_innovation_m == pytest.approx(0.020)
    assert "independent_model_innovation_above_surface" in result.failure_reasons


def test_nonfinite_or_depth_inconsistent_input_fails_closed() -> None:
    points = _broad_four()
    bad = points.copy()
    bad[0, 0] = np.nan
    gate = ProductionGeometryGate(_profile())

    nonfinite = _evaluate(
        gate,
        points,
        bad,
        points,
        independent_model_points_m=bad,
    )
    mismatch_points = points.copy()
    mismatch_points[:, 2] += 0.020
    mismatch = _evaluate(
        gate,
        points,
        mismatch_points,
        points,
        independent_model_points_m=mismatch_points,
    )

    assert nonfinite.accepted is False
    assert mismatch.accepted is False
    assert "geometry_surface_rejected" in mismatch.failure_reasons
