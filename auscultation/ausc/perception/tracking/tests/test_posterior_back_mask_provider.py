from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ausc.perception.camera_stream import CameraFrame
from ausc.perception.robot.frame_evidence_bundle import ImmutableCameraFrame
from ausc.perception.tracking.posterior_back_mask_provider import (
    IndependentPosteriorBackMaskProvider,
    OperatorConfirmedBackGate,
    PosteriorBackMaskConfig,
    PosteriorBackMaskProviderError,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _camera_frame(
    *,
    index: int = 41,
    t: float = 2.0,
    rgb_value: int = 7,
    depth: np.ndarray | None = None,
) -> ImmutableCameraFrame:
    height, width = (20, 30)
    if depth is None:
        depth = np.full((height, width), 2500, dtype=np.uint16)
        depth[4:18, 6:24] = 1000
    raw = CameraFrame(
        frame=np.full((height, width, 3), rgb_value, dtype=np.uint8),
        t=t,
        index=index,
        status="OK",
        depth=depth,
        depth_scale=1.0,
        age_s=0.01,
        device_t_ms=1000.0 + index,
        time_source="orbbec_device_clock",
    )
    return ImmutableCameraFrame.from_camera_frame(
        raw,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        alignment_sha256=SHA_B,
    )


def _gate(
    frame: ImmutableCameraFrame,
    *,
    posterior_present: bool = True,
    operator_confirmed: bool = True,
    person_box_xyxy: tuple[int, int, int, int] | None = (6, 4, 24, 18),
) -> OperatorConfirmedBackGate:
    return OperatorConfirmedBackGate.for_frame(
        camera_frame=frame,
        posterior_present=posterior_present,
        operator_confirmed=operator_confirmed,
        person_box_xyxy=person_box_xyxy,
    )


def _config(**changes: object) -> PosteriorBackMaskConfig:
    values: dict[str, object] = {
        "minimum_depth_m": 0.3,
        "maximum_depth_m": 3.0,
        "depth_band_m": 0.08,
        "mad_scale": 3.0,
        "maximum_adaptive_depth_band_m": 0.15,
        "central_seed_fraction": 0.4,
        "minimum_valid_depth_pixels": 12,
        "minimum_valid_depth_fraction": 0.2,
        "minimum_component_pixels": 16,
        "morphology_radius_px": 0,
        "maximum_frame_pixels": 1024,
        "maximum_box_pixels": 512,
        "maximum_connected_components": 32,
    }
    values.update(changes)
    return PosteriorBackMaskConfig.create(**values)


def _provider(**changes: object) -> IndependentPosteriorBackMaskProvider:
    return IndependentPosteriorBackMaskProvider(config=_config(**changes))


def test_provider_binds_operator_gate_to_the_exact_immutable_frame() -> None:
    frame = _camera_frame()
    gate = _gate(frame)

    packet = _provider().build(camera_frame=frame, gate=gate)

    assert packet.status == "OK"
    assert packet.frame_index == frame.index == gate.frame_index
    assert packet.camera_host_monotonic_ns == gate.camera_host_monotonic_ns
    assert packet.rgb_sha256 == gate.rgb_sha256
    assert packet.depth_sha256 == frame.depth_sha256
    assert packet.operator_gate_decision_id == gate.decision_id
    assert packet.back_mask_release_id == _config().release_id
    assert packet.back_mask_model_sha256 == _config().semantic_sha256
    assert packet.masked_pixel_count == 14 * 18
    assert np.all(packet.mask[4:18, 6:24])


@pytest.mark.parametrize(
    ("posterior_present", "operator_confirmed"),
    ((False, False), (True, False), (False, True)),
)
def test_unconfirmed_or_absent_posterior_is_explicit_no_back(
    posterior_present: bool, operator_confirmed: bool
) -> None:
    frame = _camera_frame()
    gate = _gate(
        frame,
        posterior_present=posterior_present,
        operator_confirmed=operator_confirmed,
        person_box_xyxy=None,
    )

    packet = _provider().build(camera_frame=frame, gate=gate)

    assert packet.status == "NO_BACK"
    assert packet.depth_sha256 == frame.depth_sha256
    assert packet.operator_gate_decision_id == gate.decision_id
    assert packet.masked_pixel_count == 0
    assert not np.any(packet.mask)


def test_insufficient_metric_depth_is_explicit_no_back() -> None:
    depth = np.zeros((20, 30), dtype=np.uint16)
    depth[8:10, 12:14] = 1000
    frame = _camera_frame(depth=depth)

    packet = _provider().build(camera_frame=frame, gate=_gate(frame))

    assert packet.status == "NO_BACK"
    assert not np.any(packet.mask)


def test_robot_like_foreground_and_every_output_pixel_stay_inside_person_box() -> None:
    depth = np.full((20, 30), 2500, dtype=np.uint16)
    depth[4:18, 6:24] = 1000
    depth[0:20, 0:4] = 450
    depth[2:19, 26:30] = 450
    depth[7:14, 7:9] = 450
    frame = _camera_frame(depth=depth)
    box = (6, 4, 24, 18)

    packet = _provider(morphology_radius_px=2).build(
        camera_frame=frame, gate=_gate(frame, person_box_xyxy=box),
    )

    outside = packet.mask.copy()
    outside[box[1] : box[3], box[0] : box[2]] = False
    assert not np.any(outside)
    assert not np.any(packet.mask[:, : box[0]])
    assert not np.any(packet.mask[:, box[2] :])
    assert not np.any(packet.mask[7:14, 7:9])


@pytest.mark.parametrize(
    "other_frame", (_camera_frame(index=42, t=2.1), _camera_frame(rgb_value=8),),
)
def test_gate_from_a_different_frame_fails_closed(
    other_frame: ImmutableCameraFrame,
) -> None:
    source_frame = _camera_frame()
    gate = _gate(source_frame)

    with pytest.raises(PosteriorBackMaskProviderError, match="same.*frame"):
        _provider().build(camera_frame=other_frame, gate=gate)


@pytest.mark.parametrize(
    ("box", "message"),
    (
        ((6, 4, 31, 18), "bounds"),
        ((6, 4, 6, 18), "positive area"),
        ((6.0, 4, 24, 18), "integer"),
    ),
)
def test_malformed_person_box_fails_closed(
    box: tuple[object, object, object, object], message: str
) -> None:
    frame = _camera_frame()
    with pytest.raises(PosteriorBackMaskProviderError, match=message):
        OperatorConfirmedBackGate.for_frame(
            camera_frame=frame,
            posterior_present=True,
            operator_confirmed=True,
            person_box_xyxy=box,
        )


def test_provider_never_mutates_inputs_and_packet_mask_is_hard_readonly() -> None:
    frame = _camera_frame()
    gate = _gate(frame)
    rgb_before = frame.frame.copy()
    depth_before = frame.depth.copy()
    gate_before = gate

    packet = _provider().build(camera_frame=frame, gate=gate)

    np.testing.assert_array_equal(frame.frame, rgb_before)
    np.testing.assert_array_equal(frame.depth, depth_before)
    assert gate == gate_before
    assert packet.mask.flags.writeable is False
    with pytest.raises(ValueError):
        packet.mask.setflags(write=True)


def test_algorithm_configuration_has_stable_content_addressed_identity() -> None:
    first = _config()
    second = _config()

    assert first.semantic_sha256 == second.semantic_sha256
    assert first.release_id == second.release_id
    assert first.release_id == f"back_mask-{first.semantic_sha256[:20]}"
    with pytest.raises(PosteriorBackMaskProviderError, match="semantic SHA-256"):
        replace(first, semantic_sha256="0" * 64)


def test_runtime_memory_and_component_work_are_bounded_fail_closed() -> None:
    frame = _camera_frame()

    with pytest.raises(PosteriorBackMaskProviderError, match="frame pixel"):
        _provider(maximum_frame_pixels=599).build(camera_frame=frame, gate=_gate(frame))
    with pytest.raises(PosteriorBackMaskProviderError, match="box pixel"):
        _provider(maximum_box_pixels=251).build(camera_frame=frame, gate=_gate(frame))
    with pytest.raises(PosteriorBackMaskProviderError, match="morphology"):
        _config(morphology_radius_px=9)
