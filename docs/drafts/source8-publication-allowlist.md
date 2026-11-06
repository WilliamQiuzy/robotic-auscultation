# Pinned public export: source8 baseline and CODE10 transport

**Unmerged publication preparation.** The isolated branch contains the explicitly selected frozen baselines, with test/template adaptations individually hash-bound in the [preparation inventory](source8-preparation.json). It does not establish native model verification. The [explicit allowlist](source8-publication-allowlist.json) includes the follow-up fixture dependency audit. The baseline is the actual source8 freeze `cf551566c0285a086d197553cee3299f4444f464`, including the later per-arm resource option and the semantics note introduced at `4c381a6`. Its reported source archive has 1,597 files and SHA256 `ee2a0b69c322b09ac5f7c89ec0210db491deedf53fb973aedca57646bd1e5c96`. The owner reports 67 critical CPU checks passing; no new physical model outcome is inferred. The public comparison base is `270cbc84dad8be94113ed4dea09b4c49dc21ac57`.

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

The JSON records all original **71 source selections**, including the necessary modules outside `sim/ausc_policy` and the two imported archival helpers. Three now have explicit CODE10 transport overrides below; the other 72 of the current 75 production/source selections keep exact source8 bytes. A follow-up audit of all 66 initially selected tests found two additional reusable library dependencies imported inside fixtures: `m2_given_contact_recovery_dataset.py` and `layered_runtime_equivalence_probe.py`. The parameterized scene-store fixture also needs the frozen `render_expert.py` updates in both `auscultation_expert_layout_v1` and `auscultation_training`; the public prior versions lacked the scene-reference replay API. All four source dependencies are explicitly selected with their exact frozen Git blobs and SHA256 values. A 1,224-byte generic SDK path recipe is included as the bootstrap test's static reference, not installed automatically.

The current selection is **75 source files, 68 focused test files, ten documentation files, two generic configurations, one CPU SDK bootstrap utility and one static test reference: 157 files total**. Full cloud CI identified the omitted updated frozen fresh-expert companion tests and the sealed formal-training template; the existing gravity test baseline is also recorded for its narrow stub adaptation. The formal template updates only its two stale handoff labels to the required v3 contract, keeping its numerical schedule and null seals/bindings. Five test-only adaptations, that template adaptation and one public transport-document adaptation are separately hash-bound in the preparation inventory. The production/source files match their individual source8 or CODE10 pins exactly.

The private evaluation-operator-only test is excluded as described below. Every native, physical and reusable-library test remains selected. These are file counts, not new experiment passes; the original archival stationary rejection is retained as a strict negative test, alongside explicit current-v3 teacher/student positives with unchanged velocity bounds.

## Explicit CODE10 transport overrides

The source8 archive pin above describes the baseline only. The current selection overrides exactly three production files with root-reviewed bytes from CODE10 commit `ad8707e65ecdb9f2762fe5c99fde9bdbbfee7d2f`, byte-identical to the corresponding `3d2376b` transport change:

- `sim/ausc_policy/m2_given_contact_pilot_native_audit.py`: buffered XZ construction and attachment, retaining legacy gzip callers.
- `sim/ausc_policy/m2_given_contact_evaluation.py`: strict transport opener in `_read_native_events`; all 109 other neutral functions remain unchanged.
- `sim/ausc_policy/m2_rebuild_paired_eval.py`: matched analytic attachment requests the same declared XZ transport as the learned arms.

The exact CODE10 transport test, paired filename assertion and [transport evidence note](../native_audit_xz_transport_20261106.md) are selected too. The note has a separately recorded public adaptation replacing non-distributed operator-test commands with public library commands. Per-file Git blob/SHA256 pins and superseded source8 identities are explicit in both JSON inventories. Thus **72 production files match source8 and three match CODE10**, with no local production edits. Private-source commit IDs identify the reviewed provenance snapshots; the per-file hashes verify distributed bytes even when those commits are not reachable in the public Git history. Licensed assets and a freshly qualified runtime remain separate requirements. The original model numerical bindings, decoded JSONL fields, physical functions and 100 MB whole-job cap remain unchanged. All 24 frozen method slots remain pending.

## Preserve validated provenance constraints

`layered_scene_authoring.py` contains deliberately pinned legacy provenance/admission path constants. The experiment owner confirmed that these non-secret historical references should remain byte-identical. They are not portable runtime configuration and do not make the old assets available on a new machine. Document fresh local qualification and the source/reference requirements; do not silently rewrite validated source or weaken source admission to make an example appear runnable.

The loopback defaults in the Astra merge config and paired helper are generic local endpoints. They contain no token bytes or real SSH/server address. Keep account authentication and per-run capabilities outside Git. Existing portable `demo/astra_service.py` and `demo/astra_supervisor.py` should continue to match the actual frozen protocol/service hashes; do not let a second namespace drift silently.

## Excluded from direct export

The one-off `scripts/rebuild_runtime_20261105.py`, `scripts/rebuild_cpu_chain_20261105.py`, `scripts/rebuild_train_chain_20261105.py` and `scripts/rebuild_eval_chain_20261105.py` are **private operators, not distributed**. They embed particular run directories, source versions and queue ownership. Their command lines are not public reproduction entry points. Use the parameterized scene-preparation, paired-evaluation and original-owner commands in [the paired evaluation guide](../m2_rebuild_paired_eval_v1.md), with fresh local inputs and qualification.

`tests/test_ausc_rebuild_eval_chain.py` is also excluded: it tests only the non-distributed evaluation operator. Executing its fixture reproduced the missing-operator assertion; collection alone did not detect it. The audit covered AST imports and dynamic module/path strings inside all initially selected test bodies. The remaining script imports resolve to the two distributed archival metadata helpers and the SDK bootstrap utility. Library dependencies were added rather than removing their tests or changing safety assertions.

Do not export `.env`, token/account-auth files, server addresses, actual private launch configs, source-archive inventories containing absolute private paths, raw human/body assets, raw RGB/depth/physics arrays, checkpoints or untracked private files. The selected small diagnostic extracts and explicitly reviewed recorded media are separate deliverables; they are not a license to distribute their raw inputs.

## Preparation and final export procedure

1. Recompute this comparison against the actual source8 freeze and latest public HEAD. Check new dynamic imports and the declared static camera/calibration dependencies. Bind exact source bytes and document public portability limits.
2. Review each explicit path and its privacy/provenance meaning. Export the approved source/config/tests only; publish edited public documentation as a separately described adaptation when needed. Keep historical numerical evidence and README release claims unchanged until the new actual receipts are reviewed.
3. Run the focused native supervisor, Hall/preload, STOP/jerk, reconstruction/export, GPU binding, paired comparison and shared runtime tests in their required CPU environments. [Preparation verification](PUBLICATION_PREPARATION.md) records the checks actually run from this public worktree and the remaining coverage gaps; it does not claim a full policy-CI pass.
4. Verify actual native source/runtime/data/model identities, owner terminal/drain/release receipts, original independent physical audits and sensor finalization. CPU tests and a source freeze do not qualify native model behavior.
5. Append the new matched cohort/media using the separate result schema and guide. Review rendered page links, source attribution, captions and playback before publication.

The eventual TRAINING note should describe **an environment rebuilt on the replacement A40**, using newly collected data and a newly trained model. It must not describe old weights as migrated or recovered.
