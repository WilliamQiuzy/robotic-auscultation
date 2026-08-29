# Session-gate audit and loosening (2026-08-29)

Authority: operator directive 2026-08-24/29 — "过严的门禁会否掉医生确认过的录制,
浪费医生时间;P0-P3 全部执行". Scope: every gate that can reject, invalidate, or
block a recorded auscultation session on its way to training data.

## 1. Terminal-by-construction blockers (NOT bugs — go-live train locks)

No session can reach training data today regardless of quality. These five points
are deliberate activation locks; do not debug "rejections" that are actually these:

| # | Where | Lock |
|---|-------|------|
| 1 | `training_authenticity_finalizer.finalize_training_session` | hardened status forced to `NOT_READY` with `production_publish_not_activated` even on a READY report |
| 2 | `training_authenticity_finalizer` source-manifest check | `source_manifest_controlled_replacement_not_activated` appended unconditionally in the PASS branch |
| 3 | `session_to_lerobot_v2._build_target_split_segments` | called with `quality_fn=None`, which is forbidden ("activated quality model required") — converter always dies |
| 4 | `training_authenticity_finalizer._unconfigured_replay_trust_root` | returns `False` for every replay attestation; geometry/quality can never self-authorize |
| 5 | `collection_supervisor.finalize_session` | `_require_observable_robot_feedback_liveness()` unconditionally raises (legacy path is retired) |

When the activation packages land, unlocking is a deliberate release-train step.

## 2. Zero-tolerance counters — what stays hard and why

These reject on a single count because they protect the **lossless capture /
evidence chain** (a gap here is unfixable after the fact):

- camera delivery FIFO overflow, camera index gaps (lossless RGB-D contract)
- arm/gripper TX write failures, socketcan enqueue failures (action fidelity)
- capture boundary exact-index closure proofs (Start/End evidence)
- robot v4 receipt: 20 ms feedback freshness/gap over 8 motors, `robot_frames ==
  rgbd_frames` (feedback liveness contract; loosening requires a schema bump and
  a release-train pass — documented follow-up, NOT changed here)

Softened in this change set (see §4-§6): model-bound inference drops, audio
degraded-real classes, Start-path person friction.

## 3. Known inconsistency (left in place, path is terminal anyway)

`collection_supervisor` rejects force rows with `SCREENED_LIMIT`/`HARD_LIMIT`
status while the finalizer and converter accept them per P0-4 ("an
over-threshold reading is a VALID physician-reference measurement"). The legacy
supervisor path is terminal (§1.5), so this inconsistency has no live effect.

## 4. P0 — audio authenticity: fabrication vs degraded-real (IMPLEMENTED)

`ausc/audio/audio_stream.py`, policy `audio_signal_degraded_real_v3`:

- **Hard fail (fabrication-class, unchanged):** no samples, all-zero PCM,
  constant PCM.
- **Degraded-real (new — recorded, never voids the session):** frozen dropout
  blocks (with a per-region ledger: start_s/duration_s, cap 64), low AC energy,
  severe clipping, severe DC. Receipt `status` becomes `PASS_DEGRADED` with a
  `degradations` list; `evidence.frozen_regions[]` lets the reward/Q layer mask
  exact windows downstream.
- Rationale: the 2026-08-11 End policy already made Scarlett dropouts soft
  (`audio_end_unverified`), but the finalizer's authenticity replay still killed
  the whole session for a >=2.0 s digital-silence dropout (verified by edge
  test). That contradiction is now resolved at the shared accumulator, so the
  producer receipt and every verifier replay (finalizer, converter,
  collect_server diagnostic seal) stay byte-identical by construction.
- Consumers needed only one change: `collect_server` diagnostic seal pinned the
  old policy literal and `status == "PASS"`; now pins v3 and accepts
  `PASS_DEGRADED`.

**Follow-up (recommended, not yet implemented):** force authenticity
(`mechanical_sensor/force_feedback/force_capture.py`) has the same
degraded-real classes (5 s freeze, near-constant, near-zero) hard-killing
sessions via `verify_force_component`, which contradicts P0-4 ("force is
advisory, never gates"). It should receive the identical fabrication-vs-degraded
split once test execution is available.

## 5. P1 — model-bound frame drops: bounded tolerance (IMPLEMENTED, GATED OFF)

`collect_server.py --model-bound-drop-tolerance N` (default **0** = historical
zero-tolerance, byte-identical behavior):

- Mid-recording and End-drain: a drop delta `<= N` records a
  `model_bound_frames_dropped` session-quality flag instead of aborting; `> N`
  still aborts with the tolerance in the message.
- **DIAGNOSTIC-ONLY above 0 for now:** the trusted finalizer requires exactly
  one `tracker.jsonl` evidence row per recorded RGB frame
  (`handeye_frame_evidence_invalid`), so a production session recorded with
  tolerated drops would pass End and then die post-hoc — the exact
  doctor-time-waste this audit is eliminating. Enabling tolerance for
  production requires teaching the finalizer to accept a documented drop
  ledger; that is a coupled change on the finalizer evidence contract.

## 6. P2 — Start-path person friction (IMPLEMENTED)

`scripts/posterior_gate.py`:

1. **Distant-bystander exemption:** `select_single_person` gains
   `ambiguity_min_area_fraction=0.05`. Exactly one dominant person (>=5 % of
   frame) plus smaller confident detections (e.g. a colleague passing 4-5 m
   behind) stays `ONE_PERSON`; the count is recorded as `bystander_people`.
   Two dominant people are still `AMBIGUOUS_PEOPLE`.
2. **Malformed-detection downgrade:** a glitched detection row no longer voids
   a clean dominant decision (`invalid_detections` stays on the record);
   fail-closed is preserved when no clean dominant person exists.
3. **Flicker tolerance:** `max_missed_frames` default 5 → 10 (~1.4-2 s at the
   deployed model rate).
4. **Lean tolerance:** `min_area_ratio` default 0.35 → 0.25 (a confirmed
   patient leaning forward shrinks the detector box; edge tests put the old
   cliff right in natural-motion range).

Unchanged on purpose: observation staleness 4.0 s, IoU 0.20, center-shift 0.20,
preview HMAC binding, all confirmation-revocation semantics.

## 7. Verification status (2026-08-29)

Edge/boundary regression tests live in
`scripts/tests/test_gate_edge_boundaries.py` (ported from the 2026-08-24
21-case audit harness, updated to the new policies).

Executed on this change set:

- `test_audio_stream.py` + `test_posterior_gate.py` +
  `test_gate_edge_boundaries.py`: **46/46 green**
- `test_training_authenticity_finalizer.py` + `test_perception_capture.py`:
  **201/201 green** (degraded receipts flow through the byte-equality replays
  with zero finalizer changes, as designed)
- `test_collect_server.py`: 185 green / 24 red **with or without this change
  set** (baseline stash-run fails the same 24 plus the audio-seal pin test that
  this change set fixes; net effect of this audit: +1 fixed, 0 broken). The 24
  pre-existing failures all stub the older `_runtime_health_snapshot` Start
  probe and predate the in-flight recording-perf commits (`fed5dd3`..`a7a48cb`)
  that extended the Start path; they belong to that workstream.
