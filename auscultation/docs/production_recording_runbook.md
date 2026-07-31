# Production Recording Runbook (`production_teleop_v4`)

Turnkey guide for the on-box session: build the production write path, record a
real teleop session, finalize, and (only after the trust root is wired) promote.

Status as of this writing: **every framework-free block is built, tested, and
committed; promotion is HARD-BLOCKED by construction** (the replay-attestation
trust root is unconfigured → refuse, and the finalizer returns NOT_READY). Nothing
can be mis-promoted while the live path comes together.

---

## 0. What is already done (Mac, tested)

- **Replay gate** (framework-free): `ausc/perception/replay_attestation.py`
  (contract + verifier, now with a **mandatory trust root** + session-id binding),
  `ausc/perception/replay_runner.py` (compare + `run_replay` + geometry/quality
  reconstruction), `ausc/perception/trusted_model_loader.py`.
- **Write-side blocks**: `production_geometry_recorder.py`,
  `geometry_channel_adapters.py` (perception isotropic + independent SE(3)
  covariance), `production_quality_recorder.py`, `production_evidence_sink.py`
  (the single object the writer drives).
- **Contract**: `PRODUCTION_TELEOP_CONTRACT` in `collection_platform.py`.
- **Finalizer wiring**: `verify_geometry_packet_component` /
  `verify_quality_packet_component` gate on a valid attestation + re-verify
  geometry model-input npys; both default to the **unconfigured trust root**.
- **Hardening from review**: unforgeable attestation, `patient_pose-<sha>` id
  accepted, negative-std / bit-exact-dtype rejected, protocol-level mandatory
  `request_cancel`, and the Start-arm health window closed up to ingress open.

---

## 1. The one hard architectural fact

`LatestFrameWorker.latest()` is **JSON-only** (perception_worker.py:147-151). The
`ProductionEvidenceSink` geometry path needs live objects
(`PatientFrameEvidence`, `PatientPosePacket`, np arrays) that exist **only inside
the worker thread** in `_process_production_evidence` (perception_worker.py:1119-1303).
So the geometry sink **must be driven inside the worker**; `collect_server` cannot
reconstruct those objects from the JSON result. The quality sink (audio+force) is
separable and can be driven from `collect_server`.

---

## 2. On-box build steps (5g/5h)

### 2a. `ProductionTeleopCapture` writer (a SEPARATE writer, not a subclass)
`DiagnosticPerceptionCapture` is deliberately un-promotable (hardcoded
`training_eligible=False` at perception_capture.py:3687-3688, 1061-1062; only ever
emits `DIAGNOSTIC_FINALIZATION.json`; refuses a production `FINALIZATION.json` at
2680-2681). A subclass would inherit those guards. Build a distinct writer that:
- reuses the RGB-D/depth/robot/audio write machinery (factor the sensor writers
  out of `DiagnosticPerceptionCapture` into a shared mixin, OR compose),
- stamps `capture_kind="production_teleop_v4"`, `training_eligible=True`,
- emits a production `FINALIZATION.json` seal,
- implements the full `_Capture` protocol INCLUDING `request_cancel` + a
  terminal-CAS `abort` (now mandatory — the platform refuses a writer without
  `request_cancel`),
- holds a `ProductionEvidenceSink` and finalizes it inside `finalize()`.
Register it: `writer_types={"ProductionTeleopCapture": ProductionTeleopCapture}`.

### 2b. Drive the geometry sink inside the worker
Inject the sink (or just the geometry recorder + `point_names`) into
`PerceptionProcessor`; at the end of `_process_production_evidence`, extract the
perception landmark points + scalar std from `landmark_frame.points`, get the
canonical patient-frame template + the two `GeometryChannelIdentity` values, and
call `sink.record_geometry_frame(...)`. The worker owns the monotonic sequence.
`collect_server` sets the sink on the processor at Start (when `session_dir` is
known) and clears it at End; it finalizes geometry at End.

### 2c. Drive the quality sink from `collect_server`
Add a per-window audio tap during recording (near the audio ownership,
collect_server.py ~2560-2600) + the live force sample, and call
`sink.record_quality_window(...)`. Needs a **production** `quality_scorer` +
`quality_policy` + model release/sha (see 2e).

### 2d. `--production` mode + preflight
Add a `--production` flag; in `main`, when set (and not fake/preview), select
`PRODUCTION_TELEOP_CONTRACT`, register the production writer, point
`capture_factory` at a production factory, and compute real `production_blockers`.
`start_authorization` already enforces: training-eligible contract, non-diagnostic
writer, a registered production writer class, and `not production_blockers`
(collection_platform.py:792-814); `_start` binds by **exact class** for production.

### 2e. Activation artifacts (the pieces that only exist on the box)
Extend `ProductionRuntimeAssembly` (production_runtime_activation.py:136) to supply,
from the activated runtime:
- `geometry_profile` + the two `GeometryChannelIdentity` (perception =
  `v9-landmark-worker` + the detector's `TrustedModelActivation` id; independent =
  `metric-patient-model` + `patient_pose-<sha>`),
- a **production** `quality_scorer` (the activated Q model), `quality_policy`, and
  its `model_release_id`/`model_sha256` (today only the *diagnostic* proxy exists
  at realtime_q.py:2283),
- plumb camera intrinsics onto `self` at camera open so the factory can build the
  sink.

Validate 2a–2d on Mac with stub artifacts (same injected-seam pattern as the
recorders), then plug the real 2e artifacts in on the box.

---

## 3. Record → finalize → (later) promote

1. **Sync code**: Mac and Linbox must match (`deploy_remote.py --production ...`,
   see `docs/production_activation_deploy.md`); H200 synced before any use.
2. **Preflight** clear (robot/force/perception activation), `--production` mode.
3. **Record** one teleop session via `stethoscope/scripts/teleop.sh` (left arm
   only — see the OpenArm launch memory) → a `production_teleop_v4` session under
   `/home/robot/wam_data/sessions/`.
4. **Inspect readiness**: run `inspect_training_finalization_readiness` on it; the
   blockers ARE the remaining gap list. Expect it to be READY on the raw
   artifacts + geometry/Q evidence, then blocked on the replay attestations.
5. **Step 3d — replay + calibration** (on box, real models): run the box-side
   runner (`run_replay` with the real Torch `model_run` via `TrustedModelLoader`)
   to produce the attestations; measure bit-exact vs epsilon and freeze
   `FROZEN_EPSILON_CEILINGS`.
6. **Wire the trust root** (see §4) — until then, `production_authorized` stays
   refused BY CONSTRUCTION.
7. **Promote**: build the finalizer publish path, `finalize_training_session` →
   `TRAINING_AUTHENTICITY.json` + `FINALIZATION.json`; then
   `session_to_lerobot_v2`.

---

## 4. MUST-fix before enabling promotion (open review items)

Promotion stays blocked until these are closed AND a trust root is wired:

- **Trust root** (P0): implement the real `trust_verifier` — the attestation must
  be signed by, or registered under, the activated production runtime (a
  signature over its canonical bytes verified against a runtime-held key, or a
  trusted-registry membership check like `trusted_finalization_registry`). Inject
  it into `verify_geometry/quality_packet_component`. Until then the default
  refuses everything (correct).
- **Q window ↔ audio.wav** (P0): re-derive the window from `audio.wav[range]` via
  the exact resampler (`polyphase_fir_v1`) and compare (bit-exact-then-frozen-
  epsilon). Today a zero `audio.wav` + unrelated window passes.
- **Independent geometry channel** (P0): replay the independent `patient_pose`
  model too (a second `model_run` seam) — today it is caller-self-reported.
- **Publish integrity** (P1): add the new production-chain files to the hardened
  SHA inventory; make the geometry packet log durable per-frame (today it is
  written only at finalize; a crash loses all model outputs); give geometry+Q a
  unified session owner + atomic finalize transaction.

The geometry model-input npy re-verification and the protocol-level terminal CAS
and Start-arm window are already closed.
