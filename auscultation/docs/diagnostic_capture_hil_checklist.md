# DIAGNOSTIC_CAPTURE — HIL bring-up checklist

Milestone: record the first real-machine **diagnostic** session (camera + audio +
force + left-arm state). **Not** production/training data. Everything up to the
HIL boundary is done in software with no real motion; the steps marked
**[OPERATOR]** require William / a lab operator on-site.

## 0. What is already done (committed, tested; live service untouched)

| Piece | Commit | Proof |
|---|---|---|
| Canonical activation re-finalizer (hardened) | `f9e9faf` | 22 unit + full on-box loader integration; new durable release passes `verify-activation` |
| Frame-worker `stop()` failure + no replacement over a live thread | `95cc666` | 32 perception_worker tests |
| Atomic Start arm on the health epoch | `6a90a3b` | unit + integration; 170 collector/worker tests |
| Duration-aware storage gate at Start | `a838211` | unit + integration |
| Trusted controller companion (vcan shadow) | `604854a` | 11 shadow tests |

New canonical activation release (durable, live package untouched):
`/home/robot/wam_data/commissioning/activation_canonical/activation-...`,
activation SHA pinned in provenance; passes the real `verify-activation` gate.

## 1. Build + RENDER the diagnostic unit (no install) — Task 5, pre-HIL

Because `collect_server.py` and `perception_worker.py` changed (both are in the
semantic pin), a **new** code release + linbox runtime must be built from the
clean HEAD containing the commits above, then the unit **rendered** (not
installed). Do NOT activate the current `d7…` runtime — it embeds `b39…`, behind
Mac HEAD.

1. **DONE (Mac)** — pin re-baselined + arm-UI robot-safety files pinned; the
   clean-HEAD code release builds and passes the v4 hardened verify (40 files).
2. **DONE (box, running service UNTOUCHED — still `56f7299`)** — deployed + rebound:
   - **`code_release-b5cecd19e3c2af4dc8e4`** activated at
     `.../.wam-releases/` (`.wam-current`: b39c616 → b5cecd19). Payload
     `b5cecd19e3c2af4dc8e4d5946c08813ee37af8307da431b9a7d77a910fbf001b`.
     Includes the console arm bring-up / E-stop UI.
   - **`perception_runtime-5bd6bcebf804bf974e62`** built (same model
     perc_lmh_v9_occ + gate + dino + env `8ea23197`; new code b5cecd19). NOT
     activated. (If HEAD advances, recompute id via `select_code_release_sources`
     + `build_manifest`, then `write_release` + `deploy_remote --production`.)
   The new re-finalizer, controller companion, controller_launch, their tests, and
   the telemetry-publisher C++ are all in the payload / pinned.
3. **RENDER + hand-edit the DIAGNOSTIC unit — do NOT install here** (install is §3
   with the operator). Render/derive the ExecStart for code b5cecd19 + runtime
   5bd6bceb + the canonical activation
   (`.../commissioning/activation_canonical/activation-...`, sha `71b063f6...`),
   back up `~/.config/systemd/user/wam-collector.service.bak.preview.*`, then:
   - remove `--commissioning-preview-only` + `--commissioning-hall-monitor-only` +
     all `--commissioning-handeye-*`;
   - add `--production-activation <act>/activation.json --production-activation-sha256 71b063f6...`
     + `--sync-calibration ... --sync-calibration-sha256 6b539d55...`;
   - to enable the **console arm bring-up / E-stop button**, add
     `--teleop-launch-cmd "<stethoscope>/scripts/teleop.sh left"` +
     `--controller-disable-cmd "<confirmed left-arm energy-isolation command>"`
     (both required together);
   - activate the runtime: `publish_perception_runtime.py activate` (CAS).
   Save the unit SHA, full argv, the five ExecStartPre verify-\* commands, and the
   rollback recipe (`cp .bak.preview.* back + daemon-reload + restart`).

Deploy identity guards to enforce at every layer (planner / remote validator /
activation / postverify): `artifact_kind == code_release`, `source_role == mac`,
`target_role == linbox` (and `runtime` verifier must check
`TRANSFER_COMPLETE.target_role == linbox`).

## 2. [OPERATOR] Controller bring-up (real motion — William on-site)

Use the trusted companion (`TrustedControllerCompanion`) or its wiring; **only**
launch teleop via `stethoscope/scripts/teleop.sh` **left** (never the shared
Openarm_project). Left arm = `can1` leader → `can3` follower. **Right arm
`can0`/`can2` is DANGEROUS — never touch.**

- [ ] Bring up the left controller; confirm `/proc/<pid>/exe` → `stethoscope/bin`.
- [ ] Verify `can1`/`can3` RX and TX counters both **increasing**.
- [ ] Verify **per-motor** real state for all 7 joints is **not all-zero / not frozen**
      (companion gates on receiver `status == READY` + `preflight_ready`).
- [ ] **Test direct-disable**: trigger a fault and confirm the arm energy-isolates
      (companion calls direct-disable *first*, then stops receiver + terminates controller).
- [ ] Confirm all-ready: audio, Hall force + sync, camera, perception, robot.

## 3. [OPERATOR] Switch to DIAGNOSTIC_CAPTURE (CAS install)

- [ ] CAS-install the rendered unit only after §1 fully verified and rollback staged.
- [ ] `daemon-reload` + restart `wam-collector.service`.
- [ ] Confirm: `capture_mode == DIAGNOSTIC_CAPTURE`; `training_eligible == false`;
      `start_authorized == true`; no `--commissioning-preview-only` in the argv.

## 4. [OPERATOR] First short recording (10–20 s)

- [ ] `Start` → `Confirm`/`Redo` → `End`.
- [ ] Storage gate passes for the planned duration (§ `--planned-session-duration-s`;
      remember the disk is ~61 GB free until the HF cache is cleared — keep it short).
- [ ] Session contains: RGB, native + aligned depth, `audio.wav`, force, robot v4
      ledger, tracker output, hash inventory, and a single terminal state.

## 5. [OPERATOR] Failure-path tests

- [ ] `Abort` mid-session → clean abort, no partial "complete".
- [ ] Telemetry disconnect → companion direct-disables + Start refuses.
- [ ] Service restart mid-session → orphan recovery reconciles, no phantom session.
- [ ] Only then: a target-duration **soak** (keep `training_eligible == false`).

## 6. Guardrails (do not violate)

- No `production-ready` / `training-eligible` / `H200-ready` language until
  ProductionTeleopCapture + trusted replay + finalizer exist (Phase 2, later).
- Never touch the right arm (`can0`/`can2`) or the other project's `lerobot-record`.
- Storage reclaim (HF cache) is William's; the ≤06-24 stale subset needs the
  right-arm owner's OK.
