# Proposed source8 public export

**Review snapshot from the original draft.** The follow-up isolated branch now contains a local byte-identical preparation of the explicit allowlist; see the [preparation inventory](source8-preparation.json). It remains unpublished and does not establish native model verification. Inspect the [explicit candidate allowlist](source8-publication-allowlist.json) after the actual source8 freeze, tests and native evidence. The inspected code is the actual source8 freeze `cf551566c0285a086d197553cee3299f4444f464`, including the later per-arm resource option and the semantics note introduced at `4c381a6`. Its reported source archive has 1,597 files and SHA256 `ee2a0b69c322b09ac5f7c89ec0210db491deedf53fb973aedca57646bd1e5c96`. The owner reports 67 critical CPU checks passing; no new physical model outcome is inferred. The public comparison base is `270cbc84dad8be94113ed4dea09b4c49dc21ac57`.

The public repository already has a simulation source tree, but it is missing many current given-contact modules. Exporting only the portable inference bridge does not reproduce the native experiment. A conservative AST import scan from 14 named runtime, reconstruction, rebuild and diagnostic entry modules reaches 178 local Python modules. **71 differ from the public base.** Each is enumerated with its old/new Git blob IDs and an import reason. The list is a review scope, not permission to copy an entire private folder or proof of runtime completeness.

## Core changes that must be represented

| Area | Main files | Reason |
| --- | --- | --- |
| Actual Astra native ownership | `sim/ausc_policy/astra_native.py`, `astra_sensors.py`, `astra_service.py`, `astra_supervisor.py` | Guarded native skill dispatch, actual sensor binding and exact inference implementation |
| Hall lower endpoint | `sim/ausc_policy/hall_adc_boundary.py`, `m2_capture.py` | Shared existing ADC-equivalent conversion with raw state/gates and provenance preserved |
| STOP and jerk repair | `sim/auscultation_pi05_v6/execution.py`, `sim/ausc_policy/m2_given_contact_controller.py`, `m2_given_contact_dataset.py`, `stop_preview_numeric_diagnostic.py` | Typed clone-only rejection, fresh-grid candidates, original shared budget, tighter requested jerk and reader authentication |
| New reconstruction and collection | `observed_reconstruction.py`, `rebuild_scene_prepare.py`, `m2_rebuild_capture_dataset.py` | Explicit new epoch, fixed identities/poses/slots, original physical/phase/source export |
| Replacement resource binding | `rebuild_gpu_binding.py`, `m2_given_contact_runtime.py`, `m2_given_contact_expanded_pilot_train.py` | Actual existing lock/GPU identity without reconstructing the lost weights or bypassing ownership |
| Matched native comparison | `m2_rebuild_paired_eval.py`, expanded/pilot deployment, native audit and evaluation modules | Exact per-scenario source identity, neutral actual physics audit and motor assistance attribution |
| Entry and runtime dependencies | `run.py`, `fresh_expert.py`, native lifecycle, scene store, SDK/MDL and collision/witness dependencies | Preserve the original owner, native source union, geometry, camera and force gates |

The JSON contains all 71 source candidates explicitly, including the necessary modules outside `sim/ausc_policy` and the two imported archival operator helpers. It separately lists **66 focused regression/test-fixture candidates**, **nine documentation candidates**, the generic Astra merge configuration and one CPU SDK bootstrap utility. Test fixture imports are included. These counts describe inspected source files, not new experiment passes.

## Preserve validated provenance constraints

`layered_scene_authoring.py` contains deliberately pinned legacy provenance/admission path constants. The experiment owner confirmed that these non-secret historical references should remain byte-identical. They are not portable runtime configuration and do not make the old assets available on a new machine. Document fresh local qualification and the source/reference requirements; do not silently rewrite validated source or weaken source admission to make an example appear runnable.

The loopback defaults in the Astra merge config and paired helper are generic local endpoints. They contain no token bytes or real SSH/server address. Keep account authentication and per-run capabilities outside Git. Existing portable `demo/astra_service.py` and `demo/astra_supervisor.py` should continue to match the actual frozen protocol/service hashes; do not let a second namespace drift silently.

## Excluded from direct export

The one-off `scripts/rebuild_runtime_20261105.py`, `scripts/rebuild_cpu_chain_20261105.py` and `scripts/rebuild_train_chain_20261105.py` remain private operator scripts. They embed particular run directories, source versions and queue ownership. Provide parameterized commands for the existing public helpers instead of presenting those scripts as general reproduction tools.

Do not export `.env`, token/account-auth files, server addresses, actual private launch configs, source-archive inventories containing absolute private paths, raw human/body assets, raw RGB/depth/physics arrays, checkpoints or untracked private files. The selected small diagnostic extracts and explicitly reviewed recorded media are separate deliverables; they are not a license to distribute their raw inputs.

## Preparation and final export procedure

1. Recompute this comparison against the actual source8 freeze and latest public HEAD. Check new dynamic imports and the declared static camera/calibration dependencies. Bind exact source bytes and document public portability limits.
2. Review each explicit path and its privacy/provenance meaning. Export the approved source/config/tests only; publish edited public documentation as a separately described adaptation when needed. Keep historical numerical evidence and README release claims unchanged until the new actual receipts are reviewed.
3. Run the focused native supervisor, Hall/preload, STOP/jerk, reconstruction/export, GPU binding, paired comparison and shared runtime tests in their required CPU environments. The present draft does not claim those source8 tests ran from the public checkout.
4. Verify actual native source/runtime/data/model identities, owner terminal/drain/release receipts, original independent physical audits and sensor finalization. CPU tests and a source freeze do not qualify native model behavior.
5. Append the new matched cohort/media using the separate result schema and guide. Review rendered page links, source attribution, captions and playback before publication.

The eventual TRAINING note should describe **an environment rebuilt on the replacement A40**, using newly collected data and a newly trained model. It must not describe old weights as migrated or recovered.
