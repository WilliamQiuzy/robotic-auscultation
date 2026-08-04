# Closed-Loop P0 Fix Plan — safe, recoverable, trainable teleop capture

**Status:** proposal for review. No code changed yet. All work stays in the branch /
`~/wam_canary`; the live `wam-collector` (56f7299 preview) and `stethoscope/bin`
are never touched; no real robot motion happens without the lab operator at the arm.

**Why this exists.** The 8801 stack today is an honest *commissioning bench*: camera
preview (~30 fps) + Scarlett audio + Hall force are live and the software gate
framework is strong, but it is **not** a closed loop that can safely record a real
robot, recover from faults, or persist a trainable session. A read-only audit
surfaced four safety P0s plus systemic gaps; every one was reproduced against the
code (line refs below marked **[verified]** were read directly; **[audit]** are cited
from the audit and will be confirmed as the first step of that item's implementation).

The ordering principle: **the arm's safety semantics must be real before any real-arm
session** — those fixes are pure software + unit tests and need no hardware. Only then
do we connect the robot link, then build the production writer, then validate on the
real arm with the operator.

---

## Ground truth verified during the audit

| Claim | Evidence | Verdict |
|---|---|---|
| Depth codec 12.9× bit-exact, 30 fps, camera+audio live | soak: 9020 frames, 29.87 fps, decompress bit-exact, encode P95 13.7 ms | real progress |
| Controller telemetry bring-up | historical HIL: 3252 envelopes, control tick 0→32516, 7 joints RX rising, TX enqueue no failures | **component-level only** — used a simplified JSON decoder, not the production v4 receiver; **no** contract/attestation/ledger/gap/trajectory/all-zero/variance gate ran, so it does **not** prove follower joints are non-frozen/non-zero |
| Left arm selection | `can1` leader → `can3` follower | correct |
| Software gates exist | writer type binding, single Start authority, full depth decode, non-degenerate trajectory check, geometry/Q replay verifier, fail-closed finalizer | real, but **not exercised** by a real session |

**Bottom line:** "arm recording tested" currently means "the controller can *emit*
telemetry," not "a real session was safely captured and validated." That gap is what
this plan closes.

---

## P0-1 — 8801 robot link cannot connect (socket mismatch + launch race)

**Root cause**
- The activated pair fixes `telemetry_socket = /run/user/1000/wam/openarm-left-telemetry.sock`
  and the receiver *requires* the socket to be a direct child of `/run/user/<uid>/wam`
  — `robot_socket_receiver.py:526` **[verified]**.
- My `~/wam_canary/hil_teleop.sh:6` exports the controller's output as
  `WAM_OPENARM_TELEMETRY_SOCKET=/tmp/wam_hil_telemetry.sock` **[verified]**.
- Receiver waits on `/run/user/1000/wam/...`, controller writes to `/tmp/...` → they
  never rendezvous → model stays `WAITING_ROBOT` after Bring Up. The historical HIL
  decoder only worked because `hil_decode_listen.py` bound `/tmp/...` (matching the
  controller, bypassing the production receiver entirely).
- **Launch-race (worse than a race):** the launcher runs `bash hil_teleop.sh`;
  `bring_up()` attests the PID **immediately** after `Popen`
  (`trusted_controller_companion.py:145→149` **[verified]**), but the script runs
  `clear_faults` and only *then* `exec`s the real binary. At attest time `/proc/pid/exe`
  is **always** still `bash` (the exec hasn't happened) → the exe-sha never matches the
  contract binary → the companion path can **never** bring up through a wrapper. So the
  wrapper is not viable; the binary must be launched directly.

**Fix**
1. Point the controller's telemetry env at the **activated** socket:
   `WAM_OPENARM_TELEMETRY_SOCKET=/run/user/1000/wam/openarm-left-telemetry.sock`
   (derive it from the activation, do not hardcode a second copy). Ensure
   `/run/user/1000/wam` exists (`mkdir -p`, mode 0700) before bind.
2. Remove the shell wrapper from the trust path. Add a `--controller-preflight-cmd`
   collect_server option (peer to `--teleop-launch-cmd`/`--controller-disable-cmd`) that
   the companion runs **synchronously before** launch — it does
   `clear_faults can1 can3` + `mkdir -p /run/user/1000/wam`. Then launch the controller
   **binary directly** (`argv = [unilateral_control, urdf, urdf_follower, left_arm, can1,
   can3]`, env carrying `WAM_OPENARM_*` + the activation socket) so `/proc/pid/exe` is
   the controller from t=0 and the exe-sha attestation
   (`robot_socket_receiver.py:709`) is sound. **Discovery:** without this the companion
   path cannot bring up at all (it always attests `bash`), so P0-1 is a small branch
   addition + canary reconfig, and is verifiable only at a real HIL with the operator.
3. Single source of truth: the launch argv + telemetry socket + binary sha are all
   read from the activation, never independently restated in a script.

**New tests (TDD)**
- `test_launch_socket_matches_activation`: the resolved controller telemetry socket ==
  activated pair `telemetry_socket`.
- `test_attested_executable_is_controller_not_shell`: the attested path resolves to the
  activated controller binary; a shell-wrapper argv is rejected.
- `test_clear_faults_runs_before_launch`: preflight ordering (clear-faults completes,
  then launch).

**Regression risk:** low (launch config + ordering). Must confirm `clear_faults` still
runs (moved from script into preflight) and the direct binary launch keeps the same
process-group signalability (`start_new_session=True`, already in
`controller_launch.py:81` **[verified]**).

---

## P0-2 — E-stop reports "energy-isolated" without any proof

**Root cause**
- `controller_launch.py:125` **[verified]**: `direct_disable` returns
  `DisableResult(outcome="energy_isolated", returncode=code)` whenever the disable
  **command** exits 0. `hil_disable.sh` is a test-grade `SIGINT`/`pkill` — killing the
  process is not a CAN-level motor disable and is not independent energy isolation, yet
  it yields the string `energy_isolated`. **Zero motor readback.**
- `trusted_controller_companion.py:_fault_disable:225-236` **[verified]** sets
  `_disabled=True` *before* running disable; if `_direct_disable` raises, it stores
  `outcome="direct_disable_error: …"` but **still constructs a `DisableOutcome`** — so a
  consumer that treats "an outcome exists" as "isolated" gets a false positive.

**Fix — split the single boolean into an honest state ladder** (exactly as the audit proposed):

| State | Meaning | Set when |
|---|---|---|
| `disable_requested` | we asked to disable | request issued |
| `disable_command_succeeded` | the disable command returned rc 0 | command rc==0 |
| `motor_disable_verified` | all 7 motors read back torque-off / zero-current / disabled | independent STATE readback confirms |
| `energy_isolated` | de-energized | **only** after `motor_disable_verified` (or a hardware energy-isolation proof) |

Changes:
- `make_subprocess_direct_disable` returns `outcome="disable_command_succeeded"` (not
  `energy_isolated`) on rc==0. rc≠0 / timeout / OSError → command_succeeded False.
- `DisableOutcome` carries explicit booleans `requested / command_succeeded /
  motor_verified / energy_isolated` (derived), plus the raw reason string.
- Add a **post-disable motor readback** verifier: after the disable command, the
  receiver reads STATE frames for all 7 activated motor IDs and asserts torque-off /
  near-zero current. `motor_disable_verified` is set only if all 7 confirm within a
  short window. **Absent a real readback path, `energy_isolated` is never claimed** —
  the surface shows `disable_command_succeeded` + `MOTOR-DISABLE UNVERIFIED`.
- `collect_server` power/controller snapshot renders the honest state; the string
  "energy-isolated" is forbidden unless `motor_verified` is true.

**New tests (TDD)**
- disable command raises / rc≠0 → `command_succeeded=False`, `energy_isolated=False`,
  state `faulted`.
- disable rc==0 but no motor readback → `command_succeeded=True`, `motor_verified=False`,
  `energy_isolated=False`; UI shows UNVERIFIED.
- disable rc==0 + all-7 torque-off readback → `energy_isolated=True`.

**Regression risk:** medium — this is the safety status surface. Every consumer of
`disable_outcome` and the UI power state must be updated to the ladder. This is
intentionally conservative: it can only make the surface *more* honest.

---

## P0-3 — controller crash during capture does not stop the arm

**Root cause** `poll()` (`trusted_controller_companion.py:181` **[verified]**) calls
`_assert_controller_alive()` at line 189 **outside** its try/disable block. A controller
exit raises `CompanionError` (`:217` **[verified]**), *not* `CompanionFault`, so it
propagates straight out of `poll()` **without** calling `_fault_disable`. The collector
supervision only catches `CompanionFault`. `bring_up()` is covered (its
`except BaseException` at `:157` **[verified]** funnels into `_fault_disable`), but
`poll()` is not. Fault-injection reproduces: controller exits code 9 → companion stays
`ready`, no disable.

**Fix** In `poll()`, treat a controller exit as a fault on the **same disable-first
path** as a telemetry fault: wrap/catch the aliveness check so a controller exit calls
`_fault_disable(reason)` then raises `CompanionFault`. Belt-and-suspenders: the
collector supervision loop should also catch `CompanionError` and route it to an
external-fault stop, so no companion exception can ever leave the arm live.

**New tests (TDD)**
- inject controller exit (code 9) during `poll()` → assert `_fault_disable` invoked
  (disable-first) and `CompanionFault` raised.
- collector supervision receiving `CompanionError` → arm stop path taken.

**Regression risk:** low — strictly closes a bypass; the healthy path is unchanged.

---

## P0-4 — force is advisory-only: decouple it from arm motion AND data collection

**Owner decision (supersedes the audit framing).** The Hall force reading is a
**passive reference for physicians**, not a software safety interlock. Contact-force
safety is provided **mechanically** by the passive-compliance stethoscope mount, with
human-in-the-loop teleop and the E-stop/disable path (P0-2/P0-3) as the active safety.
Therefore force must **never** stop the arm and **never** abort data collection. It is
recorded as a reference channel when present; nothing depends on it.

> Consequence: P0-2 (verified motor-disable) and P0-3 (controller-crash → disable) are
> now the load-bearing active safety and must be solid. Force carries no safety duty.

**Current wrong behavior (to remove)**
- `_on_force_safety_event:5063` (force over threshold → `UNSAFE`) and
  `_on_force_capture_fault:5070` (Hall serial dropout) both call
  `_abort_recording(reason)` (`collect_server.py` **[verified]**) — i.e. an over-force
  reading or a sensor dropout currently **kills the recording**. That coupling is
  exactly what must go. (The earlier plan revision proposing to *also* stop the arm on
  force is withdrawn.)

**Fix — decouple entirely**
1. Remove the `_abort_recording` call from both `_on_force_safety_event` and
   `_on_force_capture_fault`.
2. Force over-threshold: update the advisory status/display and keep recording the
   value; take **no** action on the arm or the session.
3. Hall serial dropout mid-session: mark the force channel **degraded/absent in the
   recorded metadata** (honest provenance — never claim force data we don't have), keep
   recording every other modality, do not abort, do not touch the arm.
4. No force path ever calls `report_external_fault()` or `_abort_recording()`.
5. Force also currently **blocks Start**: `require_force` (default `True`) →
   `collect_server.py:2298-2301` rejects Start with "Hall-force is not ready," and the
   Start control path re-checks at `:2587-2600` **[verified]**. Blocking Start is
   "affecting data collection," so force must **not** gate Start either. Fix: make force
   advisory (`require_force=False` semantics) — Start never waits on force; the force
   channel is recorded opportunistically when present and honestly marked absent when
   not.
   - **One point to confirm with the owner:** the audio↔force *sync-calibration* timing
     gate and the training pipeline currently treat force as a synchronized channel.
     Making force fully optional means a session may legitimately have **no** force
     channel. Default in this plan: keep recording force when present, never block on
     it; leave the sync-calibration as a separate *data-quality* note (not a
     safety/collection gate). Flagged for a yes/no.

**New tests (TDD)**
- force `UNSAFE` during RECORDING → recording continues, arm untouched, only advisory
  status changes.
- Hall serial dropout during RECORDING → recording continues (other channels), force
  channel marked degraded in metadata, arm untouched.
- assert **no** force path invokes `report_external_fault` or `_abort_recording`.

**Regression risk:** low — a decoupling/removal. Must ensure removing the abort still
leaves the advisory `_force_status`/display updated, and the recorded manifest honestly
marks force presence/degradation.

---

## P0-5 (contract) — RobotStateContract does not pin the activated motor IDs

**Root cause [audit]** the activation fixes RX IDs `0x11–0x17` (+ gripper `0x18`), but
the runtime `RobotStateContract` only requires "7 distinct IDs," not an exact match to
the activated set/mapping. A joint permutation or mis-mapping can pass — the exact class
of silent-garbage bug seen before. (`robot_session_capture.py:1692` **[audit]** already
holds all-zero / bit-identical / variance / commanded-motion / tracking-correlation
checks — they are simply not run by the current HIL/session.)

**Fix** `RobotStateContract` validates decoded STATE IDs **==** the activated pair's
expected RX-ID set *in the expected leader/follower mapping* (0x11..0x17 + gripper
0x18), rejecting permutation/mis-map. Wire the existing all-zero/variance/gap/trajectory
gates into the production receiver path so they actually run per session.

**New tests (TDD)** permuted IDs → contract rejects; exact activated mapping → passes;
all-zero or frozen STATE stream → rejected by the variance/all-zero gate.

**Regression risk:** low–medium — tightens acceptance; must match the real activated
mapping exactly (confirm from the pair contract at implementation).

---

## #6 (large) — production capture writer, End semantics, finalizer activation

**Root cause [audit]**
- The console only builds `DiagnosticPerceptionCapture` (`collect_server.py:1271`).
- There is no `ProductionTeleopCapture`. Even with a production writer injected, `End`
  hardcodes a `diagnostic_complete` requirement and a production result goes to `ERROR`
  (`collection_platform.py:1037`).
- The finalizer correctly fail-closes: even if all upstream checks read READY it
  rewrites to `NOT_READY / production_publish_not_activated` and refuses to publish
  (`training_authenticity_finalizer.py:3046`).

**Design (sequenced AFTER the safety P0s)**
- New `ProductionTeleopCapture` (peer to `DiagnosticPerceptionCapture`): on Start →
  RECORDING; streams synchronized lossless RGB-D + audio + Hall force + robot
  obs/action **from the production v4 receiver with the all-zero/variance/gap/trajectory
  gates actually running**, plus a raw-envelope ledger; on End → finalize with a
  production completion contract (not `diagnostic_complete`).
- `collection_platform` End path branches on capture kind; the production capture
  finalizes via its own completion contract.
- Define what "activates" production publish: an owner-signed production-publish
  authorization **plus** the session passing every gate. This is the real go-live gate;
  keep the finalizer fail-closed until both are present.

**Scope reality (measured):** the `CollectionPlatform` is already
contract-parameterized and production-capable (`PRODUCTION_TELEOP_CONTRACT` with
`capture_kind="production_teleop_v4"`, `writer_identity="ProductionTeleopCapture"`
already exists; training-eligibility is "realized, not declared"). The two concrete
blockers are small: (1) the End path hardcodes `result["status"] ==
"diagnostic_complete"` and `training_eligible is False`
(`collection_platform.py:1049-1058`) — it should be **driven by `self.contract`**;
(2) the finalizer's production-publish activation gate. **But** the writer itself,
`DiagnosticPerceptionCapture` in `perception_capture.py`, is **~1800 lines** of
training-data-integrity machinery (pinned-ffmpeg stitching, segment validation,
durable JSONL, force authenticity, robot-summary gates, cancellation-checkpointed
finalize), with `training_eligible=False` asserted at many points. A correct
`ProductionTeleopCapture` is therefore a **large, high-risk subsystem** — not a
single implementation pass — and it sits on the training-data-authenticity path
where rushing would *weaken* the very fail-closed guarantees this whole plan exists
to protect.

**Recommended approach:** treat #6 as its own scoped effort with its own test suite,
sequenced strictly after the safety work (which is done). A safe first increment is
the contract-driven End path (make finalize validation read `self.contract`'s
expected completion status + `training_intent` rather than hardcoding diagnostic) —
but that is only meaningful once the production completion contract + writer are
designed, so it should land *with* the writer, not before it.

**Regression risk:** high — new capture path + platform End branching + finalizer
activation. Do it last, behind the safety work, with its own test suite.

### #6 implementation status (this branch)

- **Step A — contract-driven End validation (DONE, committed, tested).** `CaptureContract`
  gains a derived `completion_status`; the platform validates the writer's sealed status
  + `training_eligible` against the contract instead of hardcoding diagnostic. Production
  sessions complete to COMPLETE with realized eligibility; diagnostic/preview unchanged.
- **Step B — `ProductionTeleopCapture` writer (DONE, committed, tested).** The ~1800-line
  writer's completion policy is parameterized via hooks that DEFAULT to the diagnostic
  values (the in-use diagnostic writer is byte-identical — 128 diagnostic tests
  unchanged); a thin subclass reuses all machinery and seals `production_teleop_complete`.
  Training eligibility is realized from the v4 OpenArm receiver's certification
  (fail-closed: no certified v4 robot summary → saved but not eligible).
- **Step C — console production wiring (DONE, committed, tested).** A `--production-teleop`
  flag (gated on a real production activation, incompatible with fake/preview) makes the
  console build the production contract + `ProductionTeleopCapture`. Default is
  byte-identical diagnostic behavior.
- **Step D — finalizer production-publish activation (INTENTIONALLY NOT ACTIVATED).**
  `finalize_training_session` (`training_authenticity_finalizer.py:3074-3085`) hardcodes
  `NOT_READY / production_publish_not_activated` even when every readiness gate passes —
  by design. Activating it *mints* a training-eligible session (auto-promotes training
  data); that is the owner's **go-live** decision requiring explicit authorization plus a
  real promotion-publish implementation, not an autonomous code flip. Left fail-closed.

**Net:** the production **recording** path is complete and tested end-to-end at the
code level (record → `ProductionTeleopCapture` → `production_teleop_complete` → platform
COMPLETE + training-eligible). The **promotion** of a recorded session into a training
dataset stays deliberately gated on owner authorization + the real commissioning.

## STOP-SHIP review response (round 2)

A read-only review correctly flagged NO-GO items my first P0-4/#6 passes left half-done.
Fixed + committed + tested this round:

- **Force is advisory end-to-end (P0-4 completed).** Removed every remaining force→
  control-flow coupling: the `/state` polling path no longer `_abort_recording()` on
  over-threshold; a force Start-boundary failure no longer rolls back Start; a non-OK
  force End no longer aborts the session; and `force_capture.py` no longer stops +
  DROPS the over-threshold sample (the physician's datum) — `submit` writes it,
  `end_recording`/`bind_start` no longer fail on it, and `_copy_and_validate_csv`
  preserves it. UI no longer says "Stops if force is unsafe"; `HARD_LIMIT` reads
  "Very high". Old-policy force tests flipped.
- **Robot-fault safety.** A dead collector loop now energy-isolates the arm
  (disable-first), and `disable_controller` reports `ok:false` when the disable command
  itself fails (never a false "controller_disabled").
- **Production Start manifest identity.** `ProductionTeleopCapture` now emits
  `capture_kind=production_teleop_v4` + `promotion_contract_version=wam_promotion_v2` +
  `capture_mode=production_teleop_v4` (was a diagnostic manifest under a production
  seal, which the converter/finalizer would reject).

Still open — all on the **promotion path the finalizer keeps fail-closed by design**,
and genuinely part of the go-live activation, NOT a rushed code flip:

- **Force must not disqualify promotion.** The writer (`perception_capture` require-force
  gates) and converter (`session_to_lerobot_v2` "force raw authenticity is not
  training-signal eligible") still treat force as a hard training gate. Honoring the
  contract requires splitting `require_force`'s two conflated meanings — *record force*
  (advisory) vs *gate promotion on force* (never) — a careful training-authenticity
  semantics change.
- **CLI cannot clear `production_blockers`** (5 static blockers) — but those encode real
  commissioning requirements; clearing them is the commissioning, not a CLI flag.
- **Deployment generator** (`collection_service_unit`) does not emit the production
  teleop/launch/disable/preflight ExecStart; the canary `hil_*.sh` are not in release
  provenance.
- **Production evidence sink** has no runtime caller; **finalizer** stays
  `production_publish_not_activated`; **completion_validator=None** for production means
  the authoritative finalizer is not yet run in the console path.

**Verdict stands: NO-GO for production recording / training promotion** until the
promotion path is activated with owner authorization + real commissioning. 8801 remains
a valid diagnostic-observation bench.

## STOP-SHIP review response (round 3)

A third review found real integration gaps my round-2 pass missed. Fixed + committed +
tested this round (742 passed / 3 skipped across the touched suites):

- **True disable-first on collector crash.** The crash path energy-isolates the arm
  BEFORE data teardown; teardown is guarded so an `_abort_recording()` exception can no
  longer skip the disable (previously `disable_calls=0` on abort failure). A fatal
  collector loop now refuses a fresh Bring-up. The arm UI no longer shows "DISABLED"
  when the disable **command** failed — it uses `disable_command_succeeded`.
- **Production manifest schema.** `schema_version` corrected to `wam_session_v2` (was
  `wam_production_teleop_v1`); the converter's `validate_manifest_for_conversion` and the
  finalizer both require exactly `wam_session_v2`.
- **Writer never fails a session on force.** A missing / degraded / low-authenticity /
  over-threshold force channel now only sets `contact_force_valid=False` (in the manifest
  + finalize result); it no longer ERRORs the session in background finalize or discards
  the video/audio/robot data. Force raw values + status are preserved verbatim.

Still open — coherent packages that must NOT be rushed (this is where round-1/2 broke):

1. **Force → promotion decoupling (converter/finalizer).** `session_to_lerobot_v2`
   (`:2374`, `:2977`) and the finalizer force-row check still reject a non-OK/WARN force
   status. Correct decoupling is not "stop raising" — `contact_force_valid=False` must
   EXCLUDE/mark the force channel in the training data (else unsafe readings would be
   served as valid training signal). That is a training-data-format feature on the
   go-live promotion path.
2. **Trusted-chain seal format.** The writer seals `PRODUCTION_FINALIZATION.json` /
   `production_teleop_complete`, but the trusted chain expects `FINALIZATION.json` +
   `TRAINING_AUTHENTICITY.json` / `complete`; `completion_validator=None`, the evidence
   sink has no runtime caller, and the finalizer stays `production_publish_not_activated`.
   A recovery bug also exists: if the process exits after the writer seal but before the
   platform lifecycle commit, the reconciler can rewrite a production manifest to
   `diagnostic_aborted`, conflicting with the production seal.
3. **Robot-safety concurrency hardening.** E-stop must not block on the Bring-up
   `_controller_lock`; disable must be retryable after a first failure; a crash+preflight
   race must not launch a new controller after disable; no motor-disable readback is
   wired.

These three are the remaining go-live / promotion-path + concurrency work. They need
careful, cross-component-tested passes (like the reviewer's), not more piecemeal edits.

---

## Gate verification matrix

| Gate | Enforced where | Exercised by |
|---|---|---|
| exe-sha attestation == contract binary | `robot_socket_receiver.py:709` | P0-1 direct-launch + unit test |
| telemetry socket == activation | receiver bind + launch env | P0-1 unit test |
| controller-exit → arm disable | companion `poll()` | P0-3 fault-injection test |
| force is advisory-only: never stops arm, never aborts recording | `collect_server` force handlers | P0-4 unit test (asserts no stop/abort) |
| motor-disable verified before "energy_isolated" | companion + receiver readback | P0-2 unit test + real-arm HIL |
| activated RX-ID exact mapping | `RobotStateContract` | P0-5 unit test |
| all-zero / variance / gap / trajectory | production receiver + `robot_session_capture` | #6 + real-arm session |
| production publish activation | `training_authenticity_finalizer` | #6 + owner authorization |

---

## Recommended merge order

1. **Safety semantics** — P0-2, P0-3, P0-4, P0-5 (pure software + unit tests; the hard
   gate for *any* real-arm session; no hardware needed).
2. **Robot link** — P0-1 socket + direct-launch (unblocks 8801 → connected), then a
   real HIL through the **production** receiver so the gates run on the real arm
   (operator present).
3. **Production writer** — #6 capture + End semantics + finalizer activation.
4. **Real-arm validation session** (operator at the arm): prove follower joints are
   non-frozen through the production gates; prove the controller-exit / force-unsafe /
   Hall-dropout stops actually de-energize and are motor-verified.

Each phase lands as small, self-contained commits (RED test → GREEN fix → refactor), no
attribution, canary/branch only.

---

## What stays untouched / guardrails

- Live `wam-collector` (56f7299 preview) and `stethoscope/bin` — never modified.
- Right arm `can0/can2` — never touched; left `can1/can3` only.
- No real robot motion without the lab operator at the arm.
- The old activation `7becd916` is immutable; any re-issue is a new immutable release.
- The finalizer stays fail-closed until #6 + owner authorization are both real.
