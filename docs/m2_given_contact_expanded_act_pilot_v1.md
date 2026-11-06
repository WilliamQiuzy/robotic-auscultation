# Expanded successful-expert ACT diagnostic

This independent route trains on the original eight cases plus successful,
physically qualified cases from the forty-case expansion. It accepts variable
positive TRAIN/VAL counts and disjoint identities from the frozen identity
split. Excluded or failed attempts remain in the exporter's forty-attempt
outcome denominator. This route does not require or qualify the formal
150-case population and never opens TEST payloads.

The dataset input is the exporter's actual `COMPLETE.json`, schema
`m2_given_diverse_pilot_preparation_v1`. It contains `counts`, `episodes`,
`split_subject_ids`, and full path/bytes/SHA256 references for `arrays.train`,
`arrays.validation`, `normalization`, and `index`. Every episode includes its
source configuration reference, positive `windows`, and a contiguous
`split_sample_range` using an exclusive end. Its index retains the original
`m2_given_contact_causal_dataset_index_v2` schema. The six arrays retain raw
states17, histories2x17, actions20x7, origins7, velocities7, and observation
times. Normalization must exactly equal the fit on the TRAIN arrays.

The official no-VAE, state-only ACT model remains 2,658,823 parameters:
7 state dimensions plus 27 environment dimensions, H20 chunks and E10 native
execution. Model dropout stays 0.1. Q noise and q dropout are both zero.
Defaults are 20,000 AdamW steps, batch64, learning rate and weight decay1e-4,
full TRAIN/VAL evaluation every500 steps, and a finite21,600-second budget.
`steps`, `eval_every`, `batch_size`, `seed`, and `time_limit_seconds` are
configurable finite integers. The final step is evaluated even when it is not
an evaluation-interval multiple. There is no automatic resume.

Create the training config in the frozen server source root using its pinned
official Python environment:

```python
from sim.ausc_policy import m2_given_contact_expanded_pilot_train as train
from sim.ausc_policy import m2_given_contact_runtime as runtime

config = dict(train.SCHEDULE, base='/workspace/wam', dataset=runtime.reference(
    '/workspace/wam/datasets/m2-given-diverse-act-pilot-20261029-v1/COMPLETE.json'))
train.validate_config(config)
runtime.write_new('/workspace/wam/configs/m2-given-diverse-act-20261029-v1.json', config)
```

Launch through the unchanged owner, resource lock and process guard:

```sh
python -B -m sim.ausc_policy.run \
  --config /workspace/wam/configs/m2-given-diverse-act-20261029-v1.json \
  --run-id m2-given-diverse-act-gpu-20261029-v1
```

Every full evaluation reports raw full-chunk and per-horizon MAE for ACT,
HOLD and constant velocity. Selection uses the earliest full-validation ACT
MAE minimum. The selected weights, original config, dataset, numerical source
bytes, official software inventory, protocol and window ordering are bound
in the checkpoint and `SELECTED.json`. A fresh official CPU child loads those
actual selected weights and compares every TRAIN/VAL prediction and metric
against the saved CUDA predictions using the existing runtime tolerances.
`COMPLETE.json` is written only after `CPU_EQUIVALENCE.json` passes. Failed
runs retain their partial checkpoint, completed steps, traceback and failure
receipt.

Prepare actual diagnostic native cases only after that completed training:

```sh
python -B -m sim.ausc_policy.m2_given_contact_expanded_pilot_deployment \
  --prepare \
  --complete /workspace/wam/runs/ausc_policy/m2-given-diverse-act-gpu-20261029-v1/output/COMPLETE.json \
  --manifest /workspace/wam/datasets/m2-given-diverse-act-pilot-20261029-v1/COMPLETE.json \
  --output /workspace/wam/configs/m2-given-diverse-act-native-20261029-v1 \
  --run-id m2-given-diverse-act-native-20261029-v1
```

By default this prepares all successful dataset cases. To select finite cases,
pass `--attempts /absolute/path/attempts.json`, whose content is a JSON list
such as `[["p000449","R3"],["p001005","L3"]]`. The manifest must be exactly
the trained dataset reference; every requested case must occur exactly once
in it with its actual source configuration. Run the generated `batch.json`
through `sim.ausc_policy.run` with a fresh run ID.

The route key is `m2_given_contact_expanded_pilot_deployment`. Its checkpoint,
deployment and observation-pipe schemas are distinct from the original
eight-case pilot. It reuses that route's FramedPipe child ownership/cleanup
and the original GivenContactACT physical hook, force, collision, parking,
joint-limit and handoff logic. The native parent remains Native60; the
official Python3.12 CPU child stays in its original session with CUDA hidden.
No expert L2 fallback or formal permit is produced. TRAIN/VAL diagnostic
closed-loop outcomes remain separate from formal and held-out acceptance.

Local verification covers expanded schemas, split/count/range rejection,
selection and actual weight hash binding, CPU receipt requirements, distinct
protocol dispatch, E10 consumption, shared controller construction, and the
nearest existing pilot/runtime/controller/scene tests. Local checks do not
establish training or Isaac execution; those require their own terminal
receipts.

Diagnostic learned native runs attach the original full neutral NativeAudit
observer with a gzip buffer flushed every240 rows and at close. Full native
rows, measured/commanded poses, force/witness evidence and persistent raw
intent/interpolator readbacks remain. Only an explicitly installed pilot
observer skips duplicate bulk `env.save` and control-row JSON dumps.
The10Hz640x360 video, first-contact/failure evidence and controller diagnostics
remain. Append/close/source failures revoke acceptance and retain a negative
audit receipt. The original formal observer and default finalizer are unchanged.

After the native worker closes, independently recheck its actual evidence:

```sh
python -B -m sim.ausc_policy.m2_given_contact_pilot_native_audit \
  --qualify --config /absolute/native-case.json --output /absolute/native-case-output
```

`qualify(output, config, receipt_path=None)` accepts a config dictionary or
config-file path and returns a full reference to the new receipt. The CLI
writes `PILOT_NATIVE_QUALIFICATION.json` by default and exits0 only when passed;
use `--receipt` for an explicit fresh receipt path. It requires matching closed
native results, intact full audit/source/model hashes, actual normal CPU-child
exit and bound IPC/query/action counts, then calls the unchanged dataset
`_physical(run, events, metadata)` gate. The receipt records observed force
maxima, HOLD, release and home evidence. Failure yields a negative receipt;
it never creates raw labels, a formal permit or a sealed TEST claim.
