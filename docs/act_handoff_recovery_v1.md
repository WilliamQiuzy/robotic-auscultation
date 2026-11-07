# ACT handoff and recovery experiment

This version keeps prescribed 3D goals, state17/H2, H20/E10, official ACT and
the original analytic contact controller. It changes the supervision and actual
recovery coverage. It is a state-only approach policy followed by analytic L3,
not a learned RGB-D/Hall VLA. Experimental results and simulator recordings
belong to the [experiment notebook](drafts/act-recovery-20261106.html).

## What changed

The former full-H20 export removed real query states near the L2 boundary.
The opt-in `m2_given_contact_masked_dataset` export permits a nonempty contiguous
prefix of 1–20 complete, actually executed teacher cells. Its seventh array,
`action_is_pad`, masks the unavailable suffix in official ACT loss. Action
normalization and validation metrics use real cells only. Two actual causal
observations remain mandatory; STOP, alignment, perturbation and partial cells
never become future action labels. Legacy full-H20 data remain immutable.

Late DART changes actual joint target commands in Isaac and asks the expert to
replan from measured state. Numerical alternative waypoints still receive the
original full collision, posture and derivative preview. A corrected setter
ledger accounts for actual constant STOP-tail samples without relabeling them
as L2. Only complete episodes passing original physical/source/reset checks
enter the new export; every failed attempt remains in its outcome ledger.

The [historical source19 root selection](drafts/act-student-recovery-source-selection-20261106.json)
binds the distributed production/test bytes to the experiment source. Private
per-run orchestration, credentials, licensed bodies, raw arrays and checkpoints
are not distributed. Historical source8 and source17 inventories remain intact;
the source17 public snapshot is commit `64c8c764fc6d27dad32e12b3b1700ed0bd6544f6`.

The later source26 code is isolated in a [partial CPU capsule](SOURCE26_REPRODUCTION.md),
with its own exact-file inventory and an original-owner reproduction guide. It
does not relabel the root snapshot or earlier results as source26.

The first repaired ACT completed20,000 steps but passed0/4 actual development
cycles. The next recovery mode authenticates raw target prefixes from closed
failed TRAIN student rollouts, executes them from a fresh physical home start,
and queries the expert from the resulting measured state. It is replay for data
collection, not a new student evaluation. Internal history waits retain their
true role and held target. Every replay cell is a perturbation and supplies no
teacher future. Only subsequent expert actions from an episode passing the
original full qualifier become labels. VALIDATION and TEST inputs are rejected
before native payloads open. The actual failed source episode stays negative.

The optional bounded `--cpu-workers 3` export uses the same full physical and
setter-fidelity reader as serial export and preserves declared row ordering.
It changes decoding throughput, not admission. Before the second training run,
all seven arrays of previously admitted TRAIN and selection VAL origins must
remain bit identical as an ordered prefix of the expanded export.

## Export and train on an independently qualified Linux installation

Obtain the required licensed assets and use the existing
[reconstruction/capture workflow](m2_rebuild_capture_dataset_20261105.md).
Create an explicit hash-bound request with schema
`m2_given_contact_masked_export_request_v1`, a qualified rebuilt nominal
`source_dataset` reference, and a finite list of TRAIN-only `recovery_manifests`.
References must describe actual local bytes; copying identifiers cannot recreate
the assets or qualify a new installation.

```sh
CUDA_VISIBLE_DEVICES='' "$AUSC_CPU_PYTHON" -B -m sim.ausc_policy.m2_given_contact_masked_dataset \
  --request "$AUSC_MASKED_REQUEST" --sha256 "$AUSC_REQUEST_SHA256" \
  --output "$AUSC_NEW_EXPORT"
```

Read the export's `outcomes.json`, `COVERAGE.json` and `COMPLETE.json`. Require
actual qualified recovery episodes at both requested sites before this pilot.
The data schema is `m2_given_contact_masked_preparation_v1`; legacy data are
accepted only through their separate declared schema. Freeze new eligible
identity metadata using `m2_act_recovery_scope.validate` **before training**.
The earlier evaluated identities are no longer untouched evaluation data.

Create a fresh expanded-training owner config from
`m2_given_contact_expanded_pilot_train.SCHEDULE`, binding the actual export's
full reference, base directory, a freshly issued replacement GPU binding and
the pretraining `evaluation_cohort` reference. `validate_config` checks this
contract. Keep the pinned official ACT installation and source bindings.
Run that config through the original owner, never by manufacturing its marker:

```sh
"$AUSC_ACT_PYTHON" -u -B -m sim.ausc_policy.run \
  --config "$AUSC_TRAIN_CONFIG" --run-id "$AUSC_NEW_TRAIN_RUN"
```

The experiment schedule is 20,000 steps, batch64, full selection-VAL evaluation
every500 steps, earliest minimum, and actual selected CPU/CUDA equivalence.
Half each minibatch samples real origins within60 mm and half samples uniformly.
TRAIN supplies all normalization. No held-out identities or fake state noise
enter optimization or checkpoint selection.

## Actual closed-loop evaluation

First test the selected ACT on explicit TRAIN/selection scenes. A software
authentication or offline fit is not a successful contact cycle. In this round,
four actual development cycles must pass before the new identities are exposed.

Use the [paired evaluation helper](m2_rebuild_paired_eval_v1.md) with explicit
`identity_cohort` in the case ledger and the same reference bound in the training
schedule. Its v2 protocol supports an expert/ACT-only comparison:

```sh
"$AUSC_CPU_PYTHON" -B -m sim.ausc_policy.m2_rebuild_paired_eval --prepare \
  --complete "$AUSC_TRAIN_COMPLETE" --cases "$AUSC_FROZEN_CASES" \
  --expert-act-only --parallelism 3 \
  --output "$AUSC_NEW_COMPARISON" --run-id "$AUSC_NEW_EVAL_RUN"
```

This prepares configs; it runs no simulator or model. Execute each prepared
batch once through `sim.ausc_policy.run`, then use the existing `--qualify` and
`--summarize` modes against actual outputs. Keep every planned identity × pose ×
target, CPU preparation failure, native failure and unlaunched case. Report
planned and CPU-prepared denominators separately. Do not call planner timeouts
proved geometric unreachability. Do not retry a negative case in place or tune
against this cohort and still call it untouched.

Every positive still requires original full240 Hz qualification: normal contact
3.5–4.5 N continuously for≥3s, total≤6N, axial Hall<5.5N, compression≤6mm,
forbidden forces exactly0, local23mm-disk alignment≤3°, actual parking,
joint/posture/collision/derivative checks, unloading and home. Geometry and the
approved camera remain fixed. The added native measurement summary is
descriptive; it cannot override any qualifier. Its20 Hz plot bins retain
native extrema and keep physical contact and axial Hall separate.

The predeclared pilot target is at least24 CPU-prepared cases, every prepared
case actually run by both systems, ACT physical success≥95%, and successful
coverage of at least6 identities, both known poses and both targets. All32
planned slots also remain reported. This is bounded prescribed-goal simulation
evidence; it does not establish unseen-pose, clothing, dynamic-patient,
continuous-ten-point, sensor-conditioned, acoustic or hardware performance.
