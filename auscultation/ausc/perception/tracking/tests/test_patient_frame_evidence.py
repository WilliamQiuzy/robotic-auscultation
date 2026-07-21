from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.tracking.back_mask_packet import BackMaskPacket
from ausc.perception.tracking.patient_frame_evidence import (
    PATIENT_FRAME_EVIDENCE_SCHEMA,
    PatientFrameEvidence,
    PatientFrameEvidenceError,
)


_ROBOT_TEST_PATH = (
    Path(__file__).parents[2]
    / "robot"
    / "tests"
    / "test_frame_evidence_bundle.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_wam_frame_bundle_test_helpers", _ROBOT_TEST_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

GATE_ID = "operator_back_gate-" + "7" * 64


def _evidence() -> PatientFrameEvidence:
    frame_bundle = _HELPERS._bundle()
    mask = np.ones(
        frame_bundle.camera_frame.frame.shape[:2], dtype=np.bool_
    )
    back_mask = BackMaskPacket.build(
        camera_frame=frame_bundle.camera_frame,
        mask=mask,
        back_mask_release_id="back_mask-" + "9" * 20,
        back_mask_model_sha256="8" * 64,
        operator_gate_decision_id=GATE_ID,
    )
    return PatientFrameEvidence.build(
        frame_bundle=frame_bundle,
        back_mask_packet=back_mask,
    )


def test_patient_frame_evidence_binds_bundle_and_independent_back_mask() -> None:
    evidence = _evidence()

    assert evidence.schema == PATIENT_FRAME_EVIDENCE_SCHEMA
    assert evidence.index == evidence.frame_bundle.index
    assert evidence.t == evidence.frame_bundle.t
    assert evidence.rgb is evidence.frame_bundle.camera_frame.frame
    assert evidence.aligned_depth is evidence.frame_bundle.camera_frame.depth
    assert evidence.back_mask is evidence.back_mask_packet.mask
    assert evidence.robot_mask is evidence.frame_bundle.robot_mask_packet.mask
    assert evidence.back_mask_packet.depth_sha256 == (
        evidence.frame_bundle.camera_frame.depth_sha256
    )
    assert evidence.back_mask_packet.operator_gate_decision_id == GATE_ID
    assert evidence.canonical_metadata()["operator_gate_decision_id"] == GATE_ID
    assert evidence.evidence_id.startswith("patient_frame_evidence-")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("frame_index", "frame index"),
        ("camera_host_monotonic_ns", "host timestamp"),
        ("camera_profile_sha256", "camera profile"),
        ("alignment_sha256", "alignment"),
        ("rgb_sha256", "RGB"),
        ("depth_sha256", "depth"),
    ),
)
def test_patient_frame_evidence_rejects_cross_frame_back_mask(
    field: str, message: str
) -> None:
    evidence = _evidence()
    value = getattr(evidence.back_mask_packet, field)
    changed = value + 1 if isinstance(value, int) else "0" * 64
    forged = object.__new__(BackMaskPacket)
    for name in evidence.back_mask_packet.__dataclass_fields__:
        object.__setattr__(
            forged,
            name,
            changed
            if name == field
            else getattr(evidence.back_mask_packet, name),
        )

    with pytest.raises(PatientFrameEvidenceError, match=message):
        PatientFrameEvidence.build(
            frame_bundle=evidence.frame_bundle,
            back_mask_packet=forged,
        )


def test_patient_frame_evidence_rejects_identity_tamper() -> None:
    evidence = _evidence()
    with pytest.raises(PatientFrameEvidenceError, match="evidence ID"):
        replace(
            evidence,
            evidence_id="patient_frame_evidence-" + "0" * 64,
        )


def test_patient_frame_evidence_accepts_explicit_no_back_without_fallback() -> None:
    evidence = _evidence()
    frame = evidence.frame_bundle.camera_frame
    empty = BackMaskPacket.build(
        camera_frame=frame,
        mask=np.zeros(frame.frame.shape[:2], dtype=np.bool_),
        back_mask_release_id="back_mask-" + "9" * 20,
        back_mask_model_sha256="8" * 64,
        operator_gate_decision_id=GATE_ID,
    )

    result = PatientFrameEvidence.build(
        frame_bundle=evidence.frame_bundle,
        back_mask_packet=empty,
    )

    assert result.back_mask_packet.status == "NO_BACK"
    assert not np.any(result.back_mask)
