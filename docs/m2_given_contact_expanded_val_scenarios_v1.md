# Thirty paired validation scenarios

`sim.ausc_policy.m2_given_contact_expanded_val_scenarios` evaluates the actual
selected 20,000-step expanded ACT on thirty frozen VALIDATION scenarios. It
uses the original identity split and rejects all optimizer TRAIN identities.
It accepts prescribed targets outside the training dataset's positive
episode list through its own explicit route, rather than the eight-case or
expanded training-case adapter.

The thirty cases retain two distinct evidence groups:

- Six scenarios on `p000486` and `p001139`, predeclared absent from both
  optimizer TRAIN and checkpoint-selection VALIDATION. Any subsequent tuning
  on their outcomes invalidates independence. Evidence is limited to two
  bodies and six scenarios.
- Twenty-four scenarios on the eight checkpoint-selection VALIDATION bodies.
  These remain selection-validation evidence.

This route never issues a formal permit or opens sealed TEST payloads. It
does not claim thirty independent TEST scenarios. Target counts and missing
targets remain explicit; the saved candidate cohort lacks L1 and R5.

The cohort helper produces `m2_given_val_scenario_expert_configs_v1` with
exactly thirty ordered, unique identity/target cases, full expert config
references, seed0, and the predeclared strata. All expert configurations must
retain current 5 mm geometry, Native60, physics64/1/0.02, the original
handoff, approved camera, and authenticated patient/plan/reference sources.
Before the first native run, prepare both arms using the actual model:

```sh
python -B -m sim.ausc_policy.m2_given_contact_expanded_val_scenarios \
  --prepare \
  --complete /workspace/wam/runs/ausc_policy/ACT_RUN/output/COMPLETE.json \
  --expert-configs /workspace/wam/configs/m2-given-val-scenarios-20261029-v1/COMPLETE.json \
  --output /workspace/wam/configs/m2-given-val-scenarios-act-20261030-v1 \
  --parallelism 4 \
  --run-id m2-given-val-scenarios-act-20261030-v1
```

`PROTOCOL.json` freezes the selected model, actual CPU-equivalence receipt
through its completed-model binding, numerical source bytes, all thirty
expert configurations, identities, seeds, target coverage and strata.
`learned-batch.json` and `expert-batch.json` share this protocol. The expert
jobs reference the exact unchanged source configurations. Learned configs
remove expert capture and teacher intent, preserve the original physical
configuration, and add the distinct validation deployment route. Every wire
request binds the protocol SHA and scenario ID in addition to the original
checkpoint, source config, target, reference and fixed origin.

Launch each generated batch through the existing experiment owner using a
fresh run ID. Run the arms sequentially under the existing shared GPU lease;
the original guard, session ownership, child deadlines and cleanup remain.
The batch-owned worker declaration authenticates both configurations and
sets Python/NumPy seed0 before native scene reads. It leaves the Native60
child command and physical controller chain intact. ACT uses the original
GivenContactACT hard stops and CPU bridge, with no expert L2 fallback.

`--parallelism` accepts1 through8. Values above4 emit the explicit
`wam_l40s_diagnostic_eight_v1` resource profile. Without that opt-in, the
existing batch runner still rejects values above4. The profile requires
`diagnostic_only=true` and `formal_acceptance=false`; it changes concurrency
only. The server operator must confirm resource availability before an
eight-worker launch. It does not alter physical acceptance or older formal
dataset evidence limits.

Summarize both native outputs after their terminal receipts exist:

```sh
python -B -m sim.ausc_policy.m2_given_contact_expanded_val_scenarios \
  --summarize \
  --prepared /workspace/wam/configs/m2-given-val-scenarios-act-20261030-v1/PREPARED.json \
  --learned-output /workspace/wam/runs/ausc_policy/LEARNED_RUN/output \
  --expert-output /workspace/wam/runs/ausc_policy/EXPERT_RUN/output \
  --output /workspace/wam/diagnostics/val-scenario-outcomes.json
```

The report retains all thirty outcomes, including missing, failed and
unexecuted cases, with no replacements. It reports fixed6/24 strata and
actual episode/physical-success counts for each arm. A learned episode
requires evidence that learned actions actually advanced physics, its frozen
worker config hash, and the exact validation protocol attribution. Merely
preparing configs, starting Isaac or advancing warmup rows does not count.
Physical success still requires the unchanged native result and clean child
exit. Local tests verify contracts and dispatch; they establish no native
execution or physical qualification.
