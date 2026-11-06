# Rebuilt matched-scene evaluation v1

`sim.ausc_policy.m2_rebuild_paired_eval` is an opt-in configuration adapter and
read-only result qualifier/summarizer. It does not collect data, train, start a
service, invoke a model, or own native/GPU execution. No native closed-loop result
is supplied by this change.

The default benchmark contains **only** frozen evaluation identities p001030 and
p000956, each at observed poses `p000474_a001230_f0000` and
`p000476_a001231_f0023`, on an explicit root-selected subset of 2–4 prescribed
targets. Thus its frozen denominator is 8–16 scenarios / 24–48 arm episodes.
Every scenario has expert, ACT, and Astra+GivenContactExpert arms. The same
subject/target at two poses is bound by distinct scenario, source config, source
episode, and patient proposal; it is never looked up by subject/target alone.

The protocol retains all seven identity partitions: optimizer TRAIN4 p001120,
p000758, p001031, p000833; checkpoint-selection VALIDATION1 p000923; frozen
VALIDATION2 p001030, p000956. It rejects a checkpoint whose actual TRAIN/selection
scope differs. Frozen evaluation IDs are never inserted into checkpoint scope.
This is a bounded simulation diagnostic, not sealed TEST or broad generalization.
No tuning or replacement based on frozen outcomes is permitted. A separately
frozen input may explicitly select additional `eval_cohorts` from `train`,
`selection_validation`, `frozen_evaluation`; denominators are reported by cohort.
The default never schedules the training/selection scenes.

## CPU scene preparation

The existing `rebuild_scene_prepare` owner request accepts explicit
`"purpose": "frozen_evaluation"`, which permits only
`"cohorts": ["frozen_evaluation"]` (also the default cohort for that purpose).
Its default purpose remains `"capture"`, restricted to TRAIN and selection
VALIDATION. Frozen scenes remain excluded by the capture/export loader and all
scene manifests retain `training_eligible: false`.

Set request `"parallelism": 3` to prepare/plan at most three scenes concurrently
inside the same original CPU owner. The default is 1; only integer values 1–3
are accepted. Each scene runs one original CPU subprocess at a time, with CUDA
hidden, BLAS threads limited to one, the inherited owner marker/session, unchanged
full collision checks and unchanged physical limits. Manifest scenes, target
slots, failures and attempt records stay in source order despite out-of-order
worker completion. Run through the original owner:

```sh
python3 -m sim.ausc_policy.run --config /workspace/wam/rebuilt-eval/frozen-scene-request.json --run-id rebuilt-frozen-scene-prep-v1
```

The same options are available as direct CPU-driver arguments `--purpose
frozen_evaluation --parallelism 3`; direct invocation still requires the existing
original Linux CPU owner environment and never creates ownership.

## Required inputs

The ACT input must be the **new** actual expanded trainer `COMPLETE.json` from
`m2_given_contact_expanded_pilot_train`. The existing expanded adapter validates
its completed schedule, earliest full-selection-VAL argmin, actual selected
weights, full CPU checkpoint load/inference equivalence receipt, dataset/code
bindings, scope, and model identity. Old or fabricated scope is inadmissible.
Preparation only hashes these artifacts; it does not load or run the model.

The explicit cases file has these top-level fields:

| Field | Required value |
| --- | --- |
| `schema` | `m2_rebuild_paired_eval_cases_v1` |
| `complete` | `true` |
| `model_outcomes_consulted` | `false` |
| `target_ids` | Explicit unique 2–4 IDs from L1–L5 / R1–R5; no default target choice |
| `pose_ids` | Both ordered observed pose IDs above |
| `eval_cohorts` | Optional; defaults to `["frozen_evaluation"]` |
| `fixed_denominator` | All identities in selected cohorts × two poses × selected targets |
| `cases` | Every planned case in fixed order, including unavailable/failed sources |

Each case requires `index` (1-based contiguous integer), unique `scenario_id`,
`subject_id`, `cohort`, original `split` (`train` or `validation`), `pose_id`,
`target_id`, unique original `episode_id`, integer `seed: 0`,
`preparation_status`, `reason`, and `config`. A prepared case has
`preparation_status: "prepared"`, `reason: null`, and full hash-bound
`config: {"path": "/absolute/expert.json", "bytes": 123, "sha256": "..."}`.
A failed/unavailable source has `preparation_status: "failed"` or `"unavailable"`,
`config: null`, and a nonempty `reason`. It stays in every arm denominator.
No native failed episode may be dropped or substituted.

The original expert config must retain Native60, 64 position / 1 velocity solver
iterations, .02 armature, its exact source given reference and strict parking
policy, original geometry epoch, and one exact fresh plan. The bound patient
receipt must have the declared subject/split and
`proposal == subject_id + "_" + pose_id`; full reference sources are independently
rebuilt and compared. Normal scene-store admission and prepared-scene authoring
checks still execute before Kit starts. The adapter admits an integer case wall
budget above 60 and at most `MAX_CASE_SECONDS = 9000` seconds, retaining the
original prepared value unchanged in every arm. This resource bound preserves
exact source authoring; it does not change a physical threshold, controller
watchdog, collision check, or acceptance gate. Expert, ACT, and Astra derive from the
same entire config; only episode metadata, runtime intent, and removal of expert
training capture differ. All arms are excluded from training datasets.

## Prepare and execute

On the final installed Linux source, after native capture, export, training and
CPU-equivalence admission actually finish (paths below are operator-selected):

```sh
python3 -m sim.ausc_policy.m2_rebuild_paired_eval --prepare \
  --complete /workspace/wam/runs/ausc_policy/rebuilt-act/output/COMPLETE.json \
  --cases /workspace/wam/rebuilt-eval/CASES.json \
  --astra-settings /workspace/wam/configs/astra_supervisor_given_contact_v1.merge.json \
  --output /workspace/wam/rebuilt-eval/prepared-v1 --parallelism 1 \
  --run-id rebuilt-frozen-v1
```

The output binds `PROTOCOL.json`, `PREPARED.json`, and three ordinary
`isaac_batch` configurations. `--parallelism` controls expert/ACT (default 1,
maximum 4); `--astra-parallelism` independently defaults to 1 so the serial Astra
service does not queue concurrent inference requests beyond client deadlines.
Per-arm parallelism is frozen in PROTOCOL/PREPARED and checked by each batch. Run the
arms sequentially through the **original owner and lock**, never direct native
worker calls:

```sh
python3 -m sim.ausc_policy.run --config /workspace/wam/rebuilt-eval/prepared-v1/expert-batch.json --run-id rebuilt-frozen-expert-v1
python3 -m sim.ausc_policy.run --config /workspace/wam/rebuilt-eval/prepared-v1/act-batch.json --run-id rebuilt-frozen-act-v1
python3 -m sim.ausc_policy.run --config /workspace/wam/rebuilt-eval/prepared-v1/astra-batch.json --run-id rebuilt-frozen-astra-v1
```

Astra requires the separately authorized Mac service, per-run token and reverse
loopback tunnel from [the original supervisor protocol](astra_native_supervisor_v1.md).
Its top-level `astra_supervisor` config is preserved. Model `gpt-6-astra` must
actually generate each admitted skill through `codex_exec`. GivenContactExpert
provides explicitly labeled analytic local motor/contact assistance. No learned
Astra motor-policy claim is permitted. The adapter never starts that service or
makes a model request during preparation.

## Qualify and summarize

Qualify each actual episode independently, using its exact prepared arm config
and corresponding original owner output directory. Example for scenario
`frozen-p001030-pose1-l1`:

```sh
python3 -m sim.ausc_policy.m2_rebuild_paired_eval --qualify \
  --config /workspace/wam/rebuilt-eval/prepared-v1/frozen-p001030-pose1-l1-astra.json \
  --output /workspace/wam/runs/ausc_policy/rebuilt-frozen-astra-v1/output/frozen-p001030-pose1-l1
```

This writes `MATCHED_NATIVE_QUALIFICATION.json`, including a negative receipt on
failure. Never overwrite terminal evidence or delete a failed qualification.
For all existing actual episodes, an operator may use this CPU-only loop after
the native owners finish:

```python
from pathlib import Path
from sim.ausc_policy import m2_rebuild_paired_eval as paired
from sim.ausc_policy import m2_given_contact_runtime as rt
prepared = rt.read_bound(rt.reference('/workspace/wam/rebuilt-eval/prepared-v1/PREPARED.json'))
for arm in paired.ARMS:
    root = Path('/workspace/wam/runs/ausc_policy/rebuilt-frozen-' + arm + '-v1/output')
    for job in rt.read_bound(prepared['batches'][arm])['jobs']:
        folder = root / job['id']
        if (folder / 'fresh_expert_result.json').is_file():
            ref = paired.qualify(folder, job['config'])
            print(arm, job['id'], rt.read_bound(ref)['passed'])
```

Expert and Astra receive the original full neutral native observer with buffered
240-row writes. It calls each original tick exactly once, retains original
command source and all native fields, and never changes physics. Its binding
explicitly says `model: null` for analytic motors; Astra's supervisor identity is
separate. ACT uses the original CPU child and native observer with its actual
selected checkpoint. Qualification checks the actual trace/index and all 240 Hz
rows, then calls **unchanged** `m2_given_contact_dataset._physical`: prescribed
3D plane, measured parking/full pose, command and observed derivatives,
collision, force, continuous HOLD, release and final measured home all remain
mandatory. ACT also requires actual CPU IPC, executed native L2 ranges, source
identity and no expert fallback. Astra additionally requires actual RGB-D/Hall
artifacts and teardown, source hashes, requests/images/generated proposals,
paused inference state, three executed local skill ranges and honest analytic
motor attribution. A completed owner, skill acknowledgment or software pass is
never physical acceptance.

```sh
python3 -m sim.ausc_policy.m2_rebuild_paired_eval --summarize \
  --prepared /workspace/wam/rebuilt-eval/prepared-v1/PREPARED.json \
  --expert-output /workspace/wam/runs/ausc_policy/rebuilt-frozen-expert-v1/output \
  --act-output /workspace/wam/runs/ausc_policy/rebuilt-frozen-act-v1/output \
  --astra-output /workspace/wam/runs/ausc_policy/rebuilt-frozen-astra-v1/output \
  --output /workspace/wam/rebuilt-eval/FROZEN_OUTCOMES-v1.json
```

Every planned case remains visible, including preparation failures, unlaunched,
missing, infrastructure abort, model abort, physical/execution failure,
unqualified native result and evidence error. Infrastructure, model and physical
outcomes are separate fields. Reported tick counts are separate from intact
neutral native evidence. Only a current independently rechecked successful
qualification contributes to physical successes; success rates use the frozen
planned denominator. No replacements, skipped failed cases, or reuse of these
arms as training examples is allowed.

Remaining proof requires the rebuilt Linux installation, actual native prepared
body scenes and source receipts, a newly trained admitted ACT, normal CPU child
execution, actual Astra-generated dispatch with RGB-D/Hall artifacts, complete
240 Hz traces, and all original physical gates on every planned matched arm.
CPU tests and preparation artifacts establish none of those native outcomes.
