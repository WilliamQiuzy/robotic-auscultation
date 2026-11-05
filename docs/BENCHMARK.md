# Evaluation and benchmark

This release exposes actual experiments and a protocol for the next comparison.
It does **not** claim a successful learned auscultation policy, completed independent-body benchmark, or clinical validation.

## Released results

| Track | Evidence | Result | Scope |
|---|---|---|---|
| Analytic expert | Eight complete native Isaac episodes, Oct 28 | 8/8 physically qualified | Fixed diagnostic qualification set |
| State-only ACT | Actual selected weights driving Isaac, Oct 29 | 0/2 | One optimizer TRAIN body and one checkpoint-selection VAL body; both stopped before contact |
| ACT without input augmentation | Second completed 5,000-step run | VAL H20 MAE 155.331 mrad | Offline only; not run in Isaac in the released evidence |
| Independent held-out benchmark | Protocol only in this release | Not reported | No success rate inferred from missing receipts |
| Astra + local skills | API request and validation prototype | Not evaluated | No live model or physical execution result |

The expert and ACT rows have **different denominators**. Do not rank their aggregate success rates as a controlled head-to-head comparison. VAL means unseen by optimizer updates; it was used for checkpoint selection and is not an independent test.

Rebuild the table and verify the original media checksums:

```sh
python -m demo.benchmark --verify-assets --output /tmp/ausc-results.json
```

This command checks the retained public summaries and hashes. It neither reruns Isaac nor independently requalifies the original 240 Hz raw physics records.

## A physically successful episode

An episode must approach the requested target, pass the existing parking/handoff check, make allowed contact, hold **physical surface-normal force 3.5–4.5 N continuously for at least 3 s**, unload, withdraw and return home. Non-allowed parts must carry exactly zero contact force. Total contact force stays ≤6 N, Hall-derived axial force stays <5.5 N and spring stroke stays ≤6 mm. Joint, posture, collision, dynamics and emergency gates remain active throughout.

The tool uses the approved protruding **C retainer** geometry (5 mm); **B is recessed by about 1 mm**. C is an allowed contact surface; a nonzero B load is not a success prerequisite. Flatness uses alignment with the local 23 mm disk plane, ≤3°. Ring gaps are diagnostics rather than a rigid-body sealing requirement. This is a mechanical skill test, not an acoustic-quality test.

Parking uses 25 actual TCP poses / 24 intervals: speed ≤100 µm/s, accumulated path ≤20 µm, full SO(3) angular path ≤0.05°, within the original 5 s budget. Solver settings in the expert qualification were position64 / velocity1, armature0.02.

## Next controlled benchmark

[`benchmarks/protocol-v1.json`](../benchmarks/protocol-v1.json) specifies the tracks and measurements.

1. Freeze scene IDs, body identities, target IDs, initial states, seeds and reachable/unreachable classification before launching any method. Split **identities**, not overlapping windows. Preserve the attempted denominator and all failures; report preclassified unreachable scenarios separately.
2. Evaluate the analytic expert, state-only ACT with local contact control, and Astra with the same local skills. Credit expert interventions separately; no expert approach fallback is counted as autonomous ACT success.
3. Use identical geometry, camera, patient/seat distributions and physical gates. Report contact timing, force range, forbidden force, tilt, parking, minimum joint margin, withdrawal/home, wall time and simulated time.
4. Report checkpoint-selection VAL and independently frozen identities as separate strata. The earlier candidate plan has 30 scenarios across 10 VAL identities, including six scenarios on two identities excluded from the declared training/selection pool. No execution receipt from that plan is included here. Its available plans cover **eight** target names, not all ten.
5. For Astra record model ID, prompt, observation representation, API calls, token use/cost, interventions, network/inference delay and whether simulation paused. Do not compare paused-physics task time to a live-clock policy without identifying that difference.
6. RGB masking, depth/geometry perturbation and Hall-voltage perturbation are future input-use ablations. These are meaningful only for policies that actually consume the affected input; recorded sensors do not prove modality use.

The original ACT pilot trained on seven identities (3,908 windows) and selected on one identity (500 windows). It consumed state/target geometry, **not RGB, raw depth or Hall directly**. Acoustic simulation is not used. This small pilot cannot establish generalization to moving patients, clothing or the real robot.

## Evidence and provenance

- [Original expert public summary](data/expert_summary.json), including eight per-case physical measurements.
- [Original ACT public summary](data/act_summary.json), including all full-validation selections, checkpoint digests and actual failed native episodes.
- [Derived public result table](data/results.json).
- [Media/source-summary SHA-256 inventory](data/assets_manifest.json).

The original expert summary's `ACT_formally_trained: false` describes that experiment on Oct 28. The ACT summary records later actual training on Oct 29. These snapshots are retained without rewriting their historical fields.

All public clips are simulation recordings, not real patients. Playback follows simulated time and does not show the full wall-clock cost. Raw arrays, model weights and licensed body assets are not distributed with this release.
