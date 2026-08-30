import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.robot import session_bundle_replay as replay
from ausc.perception.robot.session_bundle_replay import (
    ExactRecordWindow,
    SessionBundleReplayError,
    iter_jsonl,
    load_recorded_ledger_hashes,
    reconstruct_session_bundles,
    session_artifact_path,
)


# ---------------------------------------------------------------------------
# artifact path + jsonl
# ---------------------------------------------------------------------------


def test_session_artifact_path_prefers_finalized_over_partial(tmp_path: Path):
    (tmp_path / "x.jsonl.partial").write_text("{}\n")
    assert session_artifact_path(tmp_path, "x.jsonl").name == "x.jsonl.partial"
    (tmp_path / "x.jsonl").write_text("{}\n")
    assert session_artifact_path(tmp_path, "x.jsonl").name == "x.jsonl"


def test_session_artifact_path_missing_fails_closed(tmp_path: Path):
    with pytest.raises(SessionBundleReplayError):
        session_artifact_path(tmp_path, "absent.jsonl")


def test_iter_jsonl_rejects_bad_json_and_non_objects(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot-json\n')
    with pytest.raises(SessionBundleReplayError):
        list(iter_jsonl(bad))
    arrays = tmp_path / "arrays.jsonl"
    arrays.write_text("[1, 2]\n")
    with pytest.raises(SessionBundleReplayError):
        list(iter_jsonl(arrays))


# ---------------------------------------------------------------------------
# ExactRecordWindow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRecord:
    control_tick_seq: int
    payload: str = "x"


def _envelope(*ticks: int, payload: str = "x") -> SimpleNamespace:
    return SimpleNamespace(
        records=tuple(_FakeRecord(tick, payload) for tick in ticks)
    )


def test_window_brackets_across_envelopes():
    window = ExactRecordWindow(
        [_envelope(1, 2, 3), _envelope(4, 5), _envelope(6, 7, 8)]
    )
    bracket = window.bracket(2, 6)
    assert [record.control_tick_seq for record in bracket] == [2, 3, 4, 5, 6]


def test_window_tolerates_identical_resend_but_rejects_regression():
    window = ExactRecordWindow([_envelope(1, 2), _envelope(2, 3)])
    assert [r.control_tick_seq for r in window.bracket(1, 3)] == [1, 2, 3]
    conflicting = ExactRecordWindow(
        [
            _envelope(1, 2),
            SimpleNamespace(
                records=(_FakeRecord(2, "DIFFERENT"), _FakeRecord(3))
            ),
        ]
    )
    with pytest.raises(SessionBundleReplayError):
        conflicting.bracket(1, 3)


def test_window_missing_tick_fails_closed():
    window = ExactRecordWindow([_envelope(1, 2), _envelope(4, 5)])
    with pytest.raises(SessionBundleReplayError):
        window.bracket(1, 5)


def test_window_eviction_forbids_rewinding():
    window = ExactRecordWindow([_envelope(1, 2, 3, 4, 5, 6)])
    window.bracket(4, 6)
    with pytest.raises(SessionBundleReplayError):
        window.bracket(1, 2)


def test_window_inverted_range_fails_closed():
    window = ExactRecordWindow([_envelope(1, 2)])
    with pytest.raises(SessionBundleReplayError):
        window.bracket(2, 1)


# ---------------------------------------------------------------------------
# reconstruct_session_bundles orchestration (production seams stubbed)
# ---------------------------------------------------------------------------

_LO, _NOM, _HI = 1_000_000_000, 1_005_000_000, 1_010_000_000


def _frame(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.full((2, 2), 700, dtype=np.uint16),
        depth_scale=1.0,
        t=1.015,
        device_t_ms=42.0,
    )


def _row(frame_index: int, *, lo: int = _LO, nom: int = _NOM, hi: int = _HI):
    return {
        "frame_index": frame_index,
        "exposure_lower_ns": lo,
        "exposure_nominal_ns": nom,
        "exposure_upper_ns": hi,
        "exact_action_tick_first": 10,
        "exact_action_tick_last": 12,
        "camera_id": "orbbec-main",
        "camera_profile_sha256": "b" * 64,
        "timing_artifact_sha256": "c" * 64,
    }


class _ExposureContractStub:
    def __init__(self, lo: int = _LO, nom: int = _NOM, hi: int = _HI):
        self.calls = []
        self._triplet = (lo, nom, hi)

    def for_frame(self, *, frame_index, camera_host_t_s, camera_device_t_ms):
        self.calls.append((frame_index, camera_host_t_s, camera_device_t_ms))
        lo, nom, hi = self._triplet
        return SimpleNamespace(
            exposure_lower_monotonic_ns=lo,
            exposure_nominal_monotonic_ns=nom,
            exposure_upper_monotonic_ns=hi,
            alignment_sha256="d" * 64,
        )


class _MaskProviderStub:
    def __init__(self):
        self.calls = []

    def build(self, *, exposure, trajectory, aligned_depth, depth_scale):
        self.calls.append(
            {"aligned_depth": aligned_depth, "depth_scale": depth_scale}
        )
        return SimpleNamespace(kind="mask-packet")


@pytest.fixture()
def _stubbed_seams(monkeypatch):
    """Replace the production seams so orchestration is testable anywhere."""
    bundle_calls = []

    class _BundleStub:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def canonical_metadata_bytes(self):
            return b"canonical:" + str(
                self.kwargs["exposure"].exposure_nominal_monotonic_ns
            ).encode("ascii")

    def fake_build(**kwargs):
        bundle_calls.append(kwargs)
        return _BundleStub(kwargs)

    monkeypatch.setattr(
        replay, "FrameEvidenceBundle", SimpleNamespace(build=fake_build),
    )
    monkeypatch.setattr(
        replay,
        "BracketedExactControlTrajectory",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    validator_calls = []
    monkeypatch.setattr(
        replay,
        "_validate_streamed_frame_bracket",
        lambda row, bracket, contract: validator_calls.append(
            (row["frame_index"], tuple(r.control_tick_seq for r in bracket))
        ),
    )
    return bundle_calls, validator_calls


def _run(
    rows,
    *,
    frames=None,
    depth_veto=True,
    exposure_contract=None,
    mask_provider=None,
    recorded_ledger_sha256=None,
    require_ledger_match=False,
):
    frames = frames or {row["frame_index"]: _frame(row["frame_index"]) for row in rows}
    return list(
        reconstruct_session_bundles(
            frame_index_rows=rows,
            envelopes=[_envelope(*range(9, 14))],
            contract=SimpleNamespace(),
            exposure_contract=exposure_contract or _ExposureContractStub(),
            activated_identity=SimpleNamespace(),
            mask_provider=mask_provider or _MaskProviderStub(),
            frame_loader=lambda row: frames[row["frame_index"]],
            depth_veto=depth_veto,
            recorded_ledger_sha256=recorded_ledger_sha256,
            require_ledger_match=require_ledger_match,
        )
    )


def test_reconstruct_orders_production_calls_and_seals(_stubbed_seams):
    bundle_calls, validator_calls = _stubbed_seams
    mask_provider = _MaskProviderStub()
    results = _run([_row(5), _row(8)], mask_provider=mask_provider)
    assert [r.frame_index for r in results] == [5, 8]
    assert validator_calls == [(5, (10, 11, 12)), (8, (10, 11, 12))]
    assert len(bundle_calls) == 2
    sealed = bundle_calls[0]
    assert sealed["camera_id"] == "orbbec-main"
    assert sealed["camera_profile_sha256"] == "b" * 64
    assert sealed["timing_artifact_sha256"] == "c" * 64
    assert sealed["alignment_sha256"] == "d" * 64
    assert sealed["robot_mask_packet"].kind == "mask-packet"
    # Depth veto ON: the mask consumed the frame's own aligned depth.
    assert all(
        call["aligned_depth"] is not None and call["depth_scale"] == 1.0
        for call in mask_provider.calls
    )


def test_reconstruct_depth_veto_off_reproduces_geometric_mask(_stubbed_seams):
    mask_provider = _MaskProviderStub()
    _run([_row(5)], depth_veto=False, mask_provider=mask_provider)
    assert mask_provider.calls == [{"aligned_depth": None, "depth_scale": 1.0}]


def test_reconstruct_rejects_exposure_interval_drift(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        _run(
            [_row(5)],
            exposure_contract=_ExposureContractStub(nom=_NOM + 1),
        )


def test_reconstruct_rejects_frame_index_mismatch(_stubbed_seams):
    frames = {5: _frame(6)}
    with pytest.raises(SessionBundleReplayError):
        _run([_row(5)], frames=frames)


def test_reconstruct_rejects_missing_depth(_stubbed_seams):
    frame = _frame(5)
    frame.depth = None
    with pytest.raises(SessionBundleReplayError):
        _run([_row(5)], frames={5: frame})


def test_reconstruct_rejects_non_increasing_rows(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        _run([_row(8), _row(5)])


def test_reconstruct_wraps_production_validator_rejection(
    _stubbed_seams, monkeypatch,
):
    def raising(row, bracket, contract):
        raise ValueError("recorded trajectory hash mismatch")

    monkeypatch.setattr(replay, "_validate_streamed_frame_bracket", raising)
    with pytest.raises(SessionBundleReplayError):
        _run([_row(5)])


def test_reconstruct_ledger_match_flags_and_enforces(_stubbed_seams):
    matching = hashlib.sha256(b"canonical:" + str(_NOM).encode()).hexdigest()
    results = _run(
        [_row(5)], recorded_ledger_sha256={5: matching},
    )
    assert results[0].ledger_match is True
    assert results[0].ledger_canonical_sha256 == matching

    flagged = _run(
        [_row(5)], recorded_ledger_sha256={5: "0" * 64},
    )
    assert flagged[0].ledger_match is False

    with pytest.raises(SessionBundleReplayError):
        _run(
            [_row(5)],
            recorded_ledger_sha256={5: "0" * 64},
            require_ledger_match=True,
        )


def test_reconstruct_require_match_without_hashes_fails_closed(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        _run([_row(5)], require_ledger_match=True)


def test_reconstruct_frames_without_ledger_are_unflagged(_stubbed_seams):
    results = _run([_row(5)], recorded_ledger_sha256={})
    assert results[0].ledger_match is None
    assert results[0].ledger_canonical_sha256 is None


# ---------------------------------------------------------------------------
# recorded ledger hashes
# ---------------------------------------------------------------------------


def test_load_recorded_ledger_hashes_extracts_model_frames(tmp_path: Path):
    rows = [
        {"frame_index": 7, "tracking": {"frame_evidence": {"canonical_sha256": "a" * 64}}},
        {"frame_index": 9, "tracking": {"no_ledger": True}},
        {"tracking": {"frame_index": 11, "frame_evidence": {"canonical_sha256": "b" * 64}}},
        {"not_tracking": 1},
    ]
    path = tmp_path / "inference.jsonl.inprogress"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    hashes = load_recorded_ledger_hashes(tmp_path)
    assert hashes == {7: "a" * 64, 11: "b" * 64}


def test_load_recorded_ledger_hashes_accepts_top_level_shape(tmp_path: Path):
    rows = [
        {"frame_index": 3, "frame_evidence": {"canonical_sha256": "c" * 64}},
        {"frame_index": 4, "frame_evidence": "not-a-dict"},
    ]
    path = tmp_path / "inference.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_recorded_ledger_hashes(tmp_path) == {3: "c" * 64}


def test_iter_jsonl_rejects_truncated_final_line(tmp_path: Path):
    path = tmp_path / "truncated.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}')  # no trailing newline
    with pytest.raises(SessionBundleReplayError):
        list(iter_jsonl(path))


def test_reconstruct_zero_frames_fails_closed(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        _run([])


def test_reconstruct_minimum_frames_enforced(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        list(
            reconstruct_session_bundles(
                frame_index_rows=[_row(5)],
                envelopes=[_envelope(*range(9, 14))],
                contract=SimpleNamespace(),
                exposure_contract=_ExposureContractStub(),
                activated_identity=SimpleNamespace(),
                mask_provider=_MaskProviderStub(),
                frame_loader=lambda row: _frame(row["frame_index"]),
                minimum_frames=2,
            )
        )


# ---------------------------------------------------------------------------
# REAL-MACHINERY reconstruction (2026-08-30): real ExactControlRecords, real
# BracketedExactControlTrajectory, real production bracket validator — only
# the mask provider and bundle seal stay stubbed (covered elsewhere + linbox).
# ---------------------------------------------------------------------------

from ausc.perception.robot.robot_session_capture import (
    _exact_action_sha256,
    _exact_trajectory_sha256,
    _observation_payload,
)
from ausc.perception.robot.robot_state_stream import (
    EXACT_CONTROL_RECORD_SCHEMA,
    VOID_SEND_STATUS_REASON,
    BracketedExactControlTrajectory,
    ExactControlRecord,
    ExactFollowerAction,
    RobotStateContract,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_JOINTS = tuple("openarm_left_joint%d" % i for i in range(1, 8))


def _real_record(seq, t_ns, q0=0.0):
    q = (q0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qd = (0.0,) * 7
    return ExactControlRecord(
        schema=EXACT_CONTROL_RECORD_SCHEMA,
        control_tick_seq=seq,
        observation_monotonic_ns=t_ns,
        reference_generation=seq,
        reference_source_monotonic_ns=t_ns - 100_000,
        reference_snapshot_valid=True,
        control_call_begin_monotonic_ns=t_ns + 50_000,
        arm_call_return_monotonic_ns=t_ns + 100_000,
        gripper_call_begin_monotonic_ns=t_ns + 110_000,
        gripper_call_return_monotonic_ns=t_ns + 160_000,
        control_call_end_monotonic_ns=t_ns + 170_000,
        follower_q_rad=q,
        follower_qd_rad_s=qd,
        follower_gripper_q_rad=0.0,
        follower_gripper_qd_rad_s=0.0,
        follower_gripper_motor_q_rad=0.0,
        follower_gripper_motor_qd_rad_s=0.0,
        action=ExactFollowerAction(
            arm_kp=(240.0,) * 4 + (40.0, 48.0, 40.0),
            arm_kd=(3.0,) * 4 + (0.3, 0.3, 0.3),
            arm_q=q,
            arm_dq=qd,
            arm_tau=(0.0,) * 7,
            gripper_kp=25.0,
            gripper_kd=0.3,
            gripper_q=0.0,
            gripper_dq=0.0,
            gripper_tau=0.0,
        ),
        envelope_seq=10,
        publisher_monotonic_ns=t_ns + 500_000,
        controller_pid=456,
        controller_start_token="openarm-left-commissioning",
        side="left",
        controller_mode="unilateral",
        leader_can_interface="can1",
        follower_can_interface="can3",
        joint_names=_JOINTS,
        urdf_sha256=_SHA_A,
        joint_mapper_sha256=_SHA_C,
        pair_contract_sha256=_SHA_D,
        controller_binary_sha256=_SHA_E,
        controller_source_manifest_sha256=_SHA_A,
        follower_config_sha256=_SHA_B,
        openarm_sdk_sha256=_SHA_C,
        send_status_observable=False,
        send_status_reason=VOID_SEND_STATUS_REASON,
    )


def _real_contract():
    return RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=_JOINTS,
        joint_position_limits_rad=tuple((-3.2, 3.2) for _ in range(7)),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256=_SHA_A,
        joint_mapper_sha256=_SHA_C,
        camera_profile_sha256=_SHA_B,
        timing_artifact_sha256=_SHA_C,
        pair_contract_sha256=_SHA_D,
        controller_binary_sha256=_SHA_E,
        max_abs_joint_velocity_rad_s=6.0,
        gripper_motor_position_limits_rad=(-2.0, 2.0),
        controller_source_manifest_sha256=_SHA_A,
        follower_config_sha256=_SHA_B,
        openarm_sdk_sha256=_SHA_C,
    )


def _real_session_fixture():
    records = tuple(
        _real_record(100 + i, 1_009_000_000 + i * 1_000_000) for i in range(13)
    )
    lo, nom, hi = 1_010_500_000, 1_015_000_000, 1_019_500_000
    first_tick, last_tick = 101, 111
    bracket = records[1:12]
    trajectory = BracketedExactControlTrajectory(
        exposure_lower_ns=lo, exposure_nominal_ns=nom, exposure_upper_ns=hi,
        records=bracket,
    )
    row = {
        "schema": "wam_openarm_exact_control_frame_evidence_v2",
        "frame_index": 5,
        "camera_id": "orbbec-main",
        "camera_profile_sha256": _SHA_B,
        "timing_artifact_sha256": _SHA_C,
        "exposure_lower_ns": lo,
        "exposure_nominal_ns": nom,
        "exposure_upper_ns": hi,
        "exact_action_tick_first": first_tick,
        "exact_action_tick_last": last_tick,
        "observation_tick_first": first_tick,
        "observation_tick_last": last_tick,
        "observation_timestamp_first_ns": bracket[0].observation_monotonic_ns,
        "observation_timestamp_last_ns": bracket[-1].observation_monotonic_ns,
        "observation_max_gap_ns": trajectory.max_gap_ns,
        "exact_action_sha256": _exact_action_sha256(bracket),
        "exact_control_trajectory_sha256": _exact_trajectory_sha256(bracket),
        "observation_at_nominal": _observation_payload(
            trajectory.sample_observation_at(nom)
        ),
        "nominal_alignment_rule": (
            "interpolate_observation_only_at_exposure_nominal_ns"
        ),
        "action_interpolation": "forbidden",
        "controller_pid": 456,
        "controller_start_token": "openarm-left-commissioning",
        "publisher_identity": bracket[0].publisher_identity,
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "urdf_sha256": _SHA_A,
        "joint_mapper_sha256": _SHA_C,
        "pair_contract_sha256": _SHA_D,
        "controller_binary_sha256": _SHA_E,
        "controller_source_manifest_sha256": _SHA_A,
        "follower_config_sha256": _SHA_B,
        "openarm_sdk_sha256": _SHA_C,
        "send_status_observable": False,
        "send_status_reason": VOID_SEND_STATUS_REASON,
    }
    return records, row, (lo, nom, hi)


@pytest.fixture()
def _real_machinery(monkeypatch):
    """Stub ONLY the mask provider seam and the bundle seal."""
    sealed = []

    class _BundleStub:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def canonical_metadata_bytes(self):
            return b"real-machinery"

    monkeypatch.setattr(
        replay,
        "FrameEvidenceBundle",
        SimpleNamespace(build=lambda **kw: (sealed.append(kw), _BundleStub(kw))[1]),
    )
    return sealed


def _run_real(records, row, triplet, **overrides):
    lo, nom, hi = triplet
    return list(
        reconstruct_session_bundles(
            frame_index_rows=[row],
            envelopes=[SimpleNamespace(records=records)],
            contract=_real_contract(),
            exposure_contract=_ExposureContractStub(lo=lo, nom=nom, hi=hi),
            activated_identity=SimpleNamespace(),
            mask_provider=_MaskProviderStub(),
            frame_loader=lambda r: _frame(int(r["frame_index"])),
            **overrides,
        )
    )


def test_real_validator_accepts_faithful_reconstruction(_real_machinery):
    records, row, triplet = _real_session_fixture()
    results = _run_real(records, row, triplet)
    assert len(results) == 1
    sealed = _real_machinery[0]
    trajectory = sealed["robot_trajectory"]
    assert isinstance(trajectory, BracketedExactControlTrajectory)
    assert trajectory.records[0].control_tick_seq == 101
    assert trajectory.records[-1].control_tick_seq == 111
    assert _exact_trajectory_sha256(trajectory.records) == (
        row["exact_control_trajectory_sha256"]
    )


def test_real_validator_rejects_tampered_trajectory_hash(_real_machinery):
    records, row, triplet = _real_session_fixture()
    row = dict(row)
    row["exact_control_trajectory_sha256"] = "0" * 64
    with pytest.raises(SessionBundleReplayError):
        _run_real(records, row, triplet)


def test_real_validator_rejects_mutated_record_bytes(_real_machinery):
    records, row, triplet = _real_session_fixture()
    from dataclasses import replace as dc_replace

    poisoned = list(records)
    victim = poisoned[5]
    poisoned[5] = dc_replace(
        victim,
        follower_q_rad=(0.001,) + tuple(victim.follower_q_rad[1:]),
    )
    with pytest.raises(SessionBundleReplayError):
        _run_real(tuple(poisoned), row, triplet)


def test_real_validator_rejects_wrong_nominal_interpolation(_real_machinery):
    records, row, triplet = _real_session_fixture()
    row = dict(row)
    nominal = dict(row["observation_at_nominal"])
    nominal["follower_q_rad"] = list(nominal["follower_q_rad"])
    nominal["follower_q_rad"][0] += 0.01
    row["observation_at_nominal"] = nominal
    with pytest.raises(SessionBundleReplayError):
        _run_real(records, row, triplet)


def test_real_validator_rejects_provenance_field_drift(_real_machinery):
    records, row, triplet = _real_session_fixture()
    row = dict(row)
    row["controller_pid"] = 999
    with pytest.raises(SessionBundleReplayError):
        _run_real(records, row, triplet)


# ---------------------------------------------------------------------------
# 2026-08-30 workflow-discovered gaps
# ---------------------------------------------------------------------------

from itertools import islice


@pytest.mark.parametrize(
    ("lo_off", "nom_off", "hi_off"),
    ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
)
def test_reconstruct_rejects_any_exposure_bound_drift(
    _stubbed_seams, lo_off, nom_off, hi_off,
):
    with pytest.raises(SessionBundleReplayError):
        _run(
            [_row(5)],
            exposure_contract=_ExposureContractStub(
                lo=_LO + lo_off, nom=_NOM + nom_off, hi=_HI + hi_off,
            ),
        )


def test_reconstruct_rejects_crossed_exposure_interval(_stubbed_seams):
    with pytest.raises(SessionBundleReplayError):
        _run(
            [_row(5)],
            exposure_contract=_ExposureContractStub(lo=_HI, nom=_NOM, hi=_LO),
        )


def test_reconstruct_rejects_nonpositive_minimum_frames(_stubbed_seams):
    for bad in (0, -1, True):
        with pytest.raises(SessionBundleReplayError):
            list(
                reconstruct_session_bundles(
                    frame_index_rows=[_row(5)],
                    envelopes=[_envelope(*range(9, 14))],
                    contract=SimpleNamespace(),
                    exposure_contract=_ExposureContractStub(),
                    activated_identity=SimpleNamespace(),
                    mask_provider=_MaskProviderStub(),
                    frame_loader=lambda row: _frame(row["frame_index"]),
                    minimum_frames=bad,
                )
            )


def test_reconstruct_minimum_frames_only_fires_on_full_drain(_stubbed_seams):
    generator = reconstruct_session_bundles(
        frame_index_rows=[_row(5)],
        envelopes=[_envelope(*range(9, 14))],
        contract=SimpleNamespace(),
        exposure_contract=_ExposureContractStub(),
        activated_identity=SimpleNamespace(),
        mask_provider=_MaskProviderStub(),
        frame_loader=lambda row: _frame(row["frame_index"]),
        minimum_frames=2,
    )
    first = list(islice(generator, 1))
    assert len(first) == 1                 # partial consumption: no verdict yet
    with pytest.raises(SessionBundleReplayError):
        list(generator)                    # full drain enforces the floor


def test_window_identical_resend_after_eviction_fails_closed():
    window = ExactRecordWindow(
        [_envelope(1, 2, 3, 4, 5, 6), _envelope(2, 7)]
    )
    window.bracket(4, 6)                   # evicts ticks 1-3
    with pytest.raises(SessionBundleReplayError):
        window.bracket(5, 7)               # pull hits resent (evicted) tick 2


def test_ledger_hashes_conflicting_digests_fail_closed(tmp_path: Path):
    rows = [
        {"frame_index": 3, "frame_evidence": {"canonical_sha256": "a" * 64}},
        {"frame_index": 3, "frame_evidence": {"canonical_sha256": "b" * 64}},
    ]
    path = tmp_path / "inference.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(SessionBundleReplayError):
        load_recorded_ledger_hashes(tmp_path)


def test_ledger_hashes_identical_duplicates_and_container_precedence(
    tmp_path: Path,
):
    rows = [
        {"frame_index": 3, "frame_evidence": {"canonical_sha256": "a" * 64}},
        {"frame_index": 3, "frame_evidence": {"canonical_sha256": "a" * 64}},
        {
            "frame_index": 4,
            "frame_evidence": {"canonical_sha256": "c" * 64},
            "tracking": {"frame_evidence": {"canonical_sha256": "d" * 64}},
        },
    ]
    path = tmp_path / "inference.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    hashes = load_recorded_ledger_hashes(tmp_path)
    assert hashes == {3: "a" * 64, 4: "c" * 64}   # top-level wins


def test_require_ledger_match_is_vacuous_without_coverage(_stubbed_seams):
    """Pinned semantics: 'require' binds only frames the ledger covers; a
    mapping covering nothing yields unflagged frames (the standing gate must
    therefore ALSO check ledger coverage when it wants byte-proof)."""
    results = _run(
        [_row(5)], recorded_ledger_sha256={999: "0" * 64},
        require_ledger_match=True,
    )
    assert results[0].ledger_match is None


def test_artifact_path_partial_beats_inprogress_and_skips_directories(
    tmp_path: Path,
):
    (tmp_path / "x.jsonl.partial").write_text("{}\n")
    (tmp_path / "x.jsonl.inprogress").write_text("{}\n")
    assert session_artifact_path(tmp_path, "x.jsonl").name == "x.jsonl.partial"
    (tmp_path / "y.jsonl").mkdir()          # directory under the final name
    (tmp_path / "y.jsonl.inprogress").write_text("{}\n")
    assert session_artifact_path(tmp_path, "y.jsonl").name == (
        "y.jsonl.inprogress"
    )
