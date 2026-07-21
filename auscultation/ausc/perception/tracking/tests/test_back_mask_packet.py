from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ausc.perception.camera_stream import CameraFrame
from ausc.perception.robot.frame_evidence_bundle import ImmutableCameraFrame
from ausc.perception.tracking.back_mask_packet import (
    BACK_MASK_PACKET_SCHEMA,
    BackMaskPacket,
    BackMaskPacketError,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GATE_ID = "operator_back_gate-" + "d" * 64


def _camera_frame() -> ImmutableCameraFrame:
    raw = CameraFrame(
        frame=np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3),
        t=1.25,
        index=17,
        status="OK",
        depth=np.full((8, 12), 800, dtype=np.uint16),
        depth_scale=0.001,
        age_s=0.01,
        device_t_ms=1234.5,
        time_source="orbbec_device_clock",
    )
    return ImmutableCameraFrame.from_camera_frame(
        raw,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        alignment_sha256=SHA_B,
    )


def _mask() -> np.ndarray:
    result = np.zeros((8, 12), dtype=np.bool_)
    result[2:7, 3:10] = True
    return result


def _packet(mask: np.ndarray | None = None) -> BackMaskPacket:
    return BackMaskPacket.build(
        camera_frame=_camera_frame(),
        mask=_mask() if mask is None else mask,
        back_mask_release_id="back_mask-" + "1" * 20,
        back_mask_model_sha256=SHA_C,
        operator_gate_decision_id=GATE_ID,
    )


def test_back_mask_packet_binds_exact_rgb_frame_and_immutable_mask() -> None:
    source = _mask()
    packet = _packet(source)

    assert packet.schema == BACK_MASK_PACKET_SCHEMA
    assert packet.frame_index == 17
    assert packet.camera_host_monotonic_ns == 1_250_000_000
    assert packet.camera_id == "CL8326300A0"
    assert packet.camera_profile_sha256 == SHA_A
    assert packet.alignment_sha256 == SHA_B
    assert packet.rgb_sha256 == _camera_frame().rgb_sha256
    assert packet.depth_sha256 == _camera_frame().depth_sha256
    assert packet.operator_gate_decision_id == GATE_ID
    assert packet.status == "OK"
    assert packet.masked_pixel_count == int(np.count_nonzero(source))
    assert packet.mask.flags.writeable is False
    assert packet.packet_id.startswith("back_mask_packet-")

    source[:] = False
    assert packet.masked_pixel_count > 0
    assert np.any(packet.mask)
    with pytest.raises(ValueError):
        packet.mask.setflags(write=True)


def test_empty_back_mask_is_an_explicit_no_back_packet() -> None:
    packet = _packet(np.zeros((8, 12), dtype=np.bool_))

    assert packet.status == "NO_BACK"
    assert packet.masked_pixel_count == 0
    assert not np.any(packet.mask)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"frame_index": 18}, "frame index"),
        ({"camera_host_monotonic_ns": 1_250_000_001}, "host timestamp"),
        ({"rgb_sha256": SHA_C}, "RGB"),
        ({"depth_sha256": SHA_C}, "depth SHA"),
        (
            {"operator_gate_decision_id": "operator_back_gate-" + "e" * 64},
            "packet ID",
        ),
        ({"mask_sha256": SHA_A}, "mask SHA"),
        ({"masked_pixel_count": 1}, "pixel count"),
        ({"status": "OK"}, "status"),
        ({"packet_id": "back_mask_packet-" + "0" * 64}, "packet ID"),
    ),
)
def test_back_mask_packet_rejects_metadata_or_byte_binding_tamper(
    change: dict[str, object], message: str
) -> None:
    packet = _packet(np.zeros((8, 12), dtype=np.bool_))
    with pytest.raises(BackMaskPacketError, match=message):
        replace(packet, **change)


@pytest.mark.parametrize(
    ("mask", "message"),
    (
        (np.zeros((8, 12), dtype=np.uint8), "bool"),
        (np.zeros((7, 12), dtype=np.bool_), "shape"),
        (np.zeros((8, 12, 1), dtype=np.bool_), "shape"),
    ),
)
def test_back_mask_packet_rejects_wrong_mask_contract(
    mask: np.ndarray, message: str
) -> None:
    with pytest.raises(BackMaskPacketError, match=message):
        _packet(mask)


@pytest.mark.parametrize(
    ("release_id", "digest", "message"),
    (
        ("working-tree", SHA_C, "release"),
        ("back_mask-" + "1" * 20, "not-a-sha", "model SHA"),
    ),
)
def test_back_mask_packet_requires_immutable_model_identity(
    release_id: str, digest: str, message: str
) -> None:
    with pytest.raises(BackMaskPacketError, match=message):
        BackMaskPacket.build(
            camera_frame=_camera_frame(),
            mask=_mask(),
            back_mask_release_id=release_id,
            back_mask_model_sha256=digest,
            operator_gate_decision_id=GATE_ID,
        )


@pytest.mark.parametrize(
    "decision_id",
    ("", "operator_back_gate-not-a-sha", "other_gate-" + "d" * 64),
)
def test_back_mask_packet_requires_canonical_operator_gate_identity(
    decision_id: str,
) -> None:
    with pytest.raises(BackMaskPacketError, match="operator gate decision"):
        BackMaskPacket.build(
            camera_frame=_camera_frame(),
            mask=_mask(),
            back_mask_release_id="back_mask-" + "1" * 20,
            back_mask_model_sha256=SHA_C,
            operator_gate_decision_id=decision_id,
        )


def test_back_mask_packet_rejects_a_raw_mutable_camera_frame() -> None:
    raw = CameraFrame(
        frame=np.zeros((8, 12, 3), dtype=np.uint8),
        t=1.25,
        index=17,
        status="OK",
        depth=np.ones((8, 12), dtype=np.uint16),
    )
    with pytest.raises(BackMaskPacketError, match="immutable camera"):
        BackMaskPacket.build(
            camera_frame=raw,
            mask=_mask(),
            back_mask_release_id="back_mask-" + "1" * 20,
            back_mask_model_sha256=SHA_C,
            operator_gate_decision_id=GATE_ID,
        )


def test_back_mask_packet_revalidates_embedded_depth_bytes_on_replay() -> None:
    packet = _packet()
    original = packet.camera_frame
    forged = object.__new__(ImmutableCameraFrame)
    for name in original.__dataclass_fields__:
        object.__setattr__(
            forged,
            name,
            np.zeros_like(original.depth)
            if name == "depth"
            else getattr(original, name),
        )

    with pytest.raises(BackMaskPacketError, match="depth SHA"):
        replace(packet, camera_frame=forged)
