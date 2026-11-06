# Fresh diagnostic expert capture and ACT preparation

The destroyed Pod's original eight arrays, expanded arrays and ACT weights are
not inputs. `sim.ausc_policy.m2_rebuild_capture_dataset` builds a new bounded
capture batch and exports fresh positive windows through the original
`m2_given_contact_dataset._read_payload` reader. It does not run a simulator,
launch training, issue a hardware permit, or qualify a formal population.

The new diagnostic cohort is explicit: TRAIN may use a declared nonempty subset
of `p001120,p000758,p001031,p000833`; selection validation is `p000923`.
Each selected subject has both observed poses `p000474_a001230_f0000` and
`p000476_a001231_f0023`. Declared frozen evaluation IDs `p001030,p000956` are
rejected by scope preflight. Header checks may read falsely labeled metadata
before rejecting it; no zero-metadata-access claim is made. Their source
reconstruction metadata may exist in the shared authoring bundle, while their
body/scene/capture payloads are excluded from this route. This is a new
reconstruction epoch and does not recover the lost original depth-plane bytes.

Choose two to four named targets before building, such as `R3,L3`. Every
subject × pose × target is retained. Failed scene preparation, unavailable or
failed planning, configuration failure, unlaunched cases, missing receipts,
physical rejection, fidelity rejection and zero-window episodes remain visible
in the denominator. A smaller set of accepted positives does not redefine the
requested population or the paired comparison scope.

## Required request

The JSON request has exactly these keys:

```json
{
  "schema": "m2_rebuild_expert_capture_request_v1",
  "run_id": "m2-rebuild-expert-20261105-v1",
  "base": "/workspace/wam",
  "source": {
    "code_commit": "ACTUAL_40_CHARACTER_ARCHIVED_CODE_COMMIT",
    "code_commit_file": {"path": "/ARCHIVE/CODE_COMMIT", "bytes": 41, "sha256": "ACTUAL_SHA256"},
    "source_manifest": {"path": "/ARCHIVE/SOURCE_MANIFEST.json", "bytes": 1, "sha256": "ACTUAL_SHA256"}
  },
  "reconstruction_source": {"path": "/REBUILD/RECONSTRUCTION.json", "bytes": 1, "sha256": "ACTUAL_SHA256"},
  "subject_ids": ["p001120", "p000758", "p000923"],
  "targets": ["R3", "L3"],
  "scenes": [],
  "parallelism": 2,
  "time_limit_seconds": 21600,
  "max_frames": 5000,
  "retainer_witness_contract": {},
  "scene_store": {"root": "/workspace/wam/scenes/m2-rebuild-expert-20261105-v1", "batch_id": "m2-rebuild-expert-20261105-v1"}
}
```

The placeholders are deliberately non-runnable. Populate every reference from
the actual bytes using `m2_given_contact_runtime.reference(path)`. Do not invent
source or qualification hashes. `source` must describe the exact imported
collector archive containing `CODE_COMMIT` and `SOURCE_MANIFEST.json`; the helper
authenticates every archived inventory entry. Use the original source-freeze
packaging workflow before capture. Build from that imported archive, not from
an overlay that claims a different source root.

`reconstruction_source` binds `RECONSTRUCTION.json` with schema
`m2_observed_reconstruction_sources_v1`, not the reconstruction's COMPLETE.
Provide the actual qualified native `retainer_witness_contract` accepted by
`retainer_witness.load_contract`, preserving its `{file,sha256}` domain
references. A CPU diagnostic witness cannot qualify exported native episodes.
An optional `scene_store.sdk_mdl_manifest` uses the existing scene-store
contract. No additional capture, collision, force, joint, parking or contact
overrides are accepted by this request.

`scenes` contains exactly one row per declared subject and pose. The new
`rebuild_scene_prepare` driver can supply these rows. A successful scene row is:

```json
{
  "subject_id": "p001120",
  "split": "train",
  "cohort": "train",
  "pose_id": "p000474_a001230_f0000",
  "preparation_config": {"path": "/PREP/config.json", "bytes": 1, "sha256": "ACTUAL_SHA256"},
  "prepared_complete": {"path": "/PREP/output/COMPLETE.json", "bytes": 1, "sha256": "ACTUAL_SHA256"},
  "patient_receipt": {"path": "/PREP/output/patient_receipt.json", "bytes": 1, "sha256": "ACTUAL_SHA256"},
  "plans": {
    "R3": {"plan": {"path": "/PLANS/R3.json", "bytes": 1, "sha256": "ACTUAL_SHA256"}},
    "L3": {"status": "unavailable", "reason": "Original planner not run or no plan receipt"}
  }
}
```

Use split `validation` and cohort `selection_validation` only for `p000923`.
Every target is represented by a bound actual plan or an explicit unavailable
reason; a plan with `success:false` stays a failure. A failed scene-preparation
row may include a nonempty `preparation_error`; its three preparation references
may then be null, and all target entries must be unavailable. Such a scene is
retained without opening missing files. Prepared patient/spec headers must bind
the exact new reconstruction source, role, subject and observed pose.

## Execute in the rebuilt Linux archive

First build the bounded request with actual references. Freeze it before any
ACT/Astra evaluation, then run these commands from the archived code root:

```sh
CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.m2_rebuild_capture_dataset build \
  --request /REQUEST.json --sha256 ACTUAL_REQUEST_SHA256 --output /NEW_BUILD_DIR

/ORIGINAL_OWNER/PYTHON -B -m sim.ausc_policy.run \
  --config /NEW_BUILD_DIR/batch.json --run-id m2-rebuild-expert-20261105-v1

CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.m2_rebuild_capture_dataset export \
  --preparation /NEW_BUILD_DIR/COMPLETE.json --sha256 ACTUAL_BUILD_COMPLETE_SHA256 \
  --owner /workspace/wam/runs/ausc_policy/m2-rebuild-expert-20261105-v1/result.json \
  --owner-sha256 ACTUAL_OWNER_RESULT_SHA256 --output /NEW_EXPORT_DIR \
  --steps 5000 --eval-every 500 --seed 20261105

/OFFICIAL_ACT/PYTHON -B -m sim.ausc_policy.run \
  --config /NEW_EXPORT_DIR/act_config.json --run-id m2-rebuild-act-20261105-v1
```

No runnable jobs means no batch file and no native launch. Export requires the
exact original owner to be terminal, drained and GPU-released. Every actual
positive additionally needs the unchanged reader's normal owner/child process
chain, current geometry, qualified native witness, full contact/force/camera/
joint/parking/home gates, source union, immutable scene storage and complete
capture inventory. The helper does not promote reported status booleans into
physical acceptance. It calls the original payload reader without replacing
or caching any physical, source, fidelity or label validator.

The build's `COMPARISON_SCOPE.json` freezes all requested cases, targets,
source configs and the three arms `expert,act,astra` before model evaluation.
Bind this receipt into the later comparison protocol. Cases without runnable
configs remain declared unavailable. Derive evaluation configs from the bound
expert configs while preserving all scene and physical settings; record actual
per-arm runs and failures. TRAIN/selection comparisons remain diagnostic. A
separate frozen-evaluation protocol may use the excluded IDs after model
selection, without passing them through this capture/export route. The scope
receipt is prelaunch intent, not proof of an executed or successful comparison.

## Receipts and limits

Keep the immutable request; build COMPLETE; config/assembly receipts;
COMPARISON_SCOPE; original owner result, launch and child results; COMPLETE and
partial fresh-expert results; raw compact capture inventories; source-freeze
receipts; and export `outcomes.json`. The outcome ledger preserves every
requested case and binds existing launch/result/partial receipts even when
qualification fails. If no positive TRAIN or no positive selection episode
survives, export writes FAILED and the completed ledger, and emits no ACT config.

Successful export writes `dataset/{train,validation}.npz`, normalization,
causal index, positive source manifest and dataset COMPLETE. Exported episode
rows retain `pose_id`, `observed_pose_id`, `attempt_id`, `source_attempt_id`,
`cohort` and the original bound config. The config's original patient receipt
binds `proposal = subject_id + '_' + pose_id`, and the scene-store binding
authenticates the preparation config/COMPLETE/installed patient sources.
Only the original
six arrays are emitted: states `[N,17]`, histories `[N,2,17]`, raw H20 actions
`[N,20,7]`, origins `[N,7]`, velocities `[N,7]`, observation times `[N]`.
No padding, target relabeling, perturbation/STOP/ALIGN promotion or original
eight-case import occurs. Original TRAIN-only normalization and E10 execution
semantics are unchanged. The official expanded loader rechecks the published
arrays before the helper emits `act_config.json`; q noise/dropout remain zero,
and model/schedule/selection rules stay those of the existing expanded trainer.

All receipts keep `diagnostic_only:true`, `formal_acceptance:false` and
`training_eligible:false`. The last field is the existing formal-population
eligibility flag; the explicitly diagnostic expanded trainer accepts this
contract under its original guarded runtime. CPU tests, a built batch, exported
arrays and a training config are separate from native physical qualification,
trained-policy performance, held-out evidence and real hardware readiness.
