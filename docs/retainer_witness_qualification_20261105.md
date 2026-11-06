# Fresh native witness qualification

The old qualified receipt cannot be reused after its raw oracle chain was lost.
The original `contact_semantics_oracle` already writes the required evidence;
no exact reusable independent issuer survived in the repository. The new
`retainer_witness_qualification` module issues a receipt only after reading and
independently checking a newly completed actual native run. It never launches
Kit, physics, inference or a GPU workload, and does not modify the oracle,
owner, witness loader or physical gates.

After the original owner reports exit 0, a drained process group and released
GPU, run:

```sh
CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.retainer_witness_qualification \
  --owner /ACTUAL_BASE/runs/ausc_policy/ACTUAL_NEW_ORACLE_RUN/result.json \
  --sha256 ACTUAL_OWNER_RESULT_SHA256 \
  --output /FRESH_DIAGNOSTIC_OUTPUT
```

Run from the committed issuer code tree with NumPy and SciPy. The issuer may
live in a later code archive than the collector: it authenticates the actual
collector archive using the owner's guard path, `CODE_COMMIT` and complete
`SOURCE_MANIFEST.json`. The native oracle, geometry, contact-recorder and guard
source bytes must match the reviewed issuer tree. No historical source,
artifact, binary hash or success count is copied into the new receipt.

Required original run layout:

- `result.json`: actual original owner, complete/exit 0/drained/released.
- `output/native_worker_exit.json`: clean native worker exit, with its PID in
  the original guard's recorded group and no signals/errors.
- `output/review/COMPLETE.json`: completed original 72-case, 12-tick oracle.
- Review artifacts: `cases.json`, `native_contacts.jsonl`,
  `native_journal.jsonl`, `physics_ticks.json`, `residuals.json`, all bound by
  COMPLETE size/SHA. Every raw callback and its post-step update is retained.
- Actual recorded PhysX `.so`: absolute path, matching SHA, Linux ELF bytes.
  The fixture's convexHull wedge and static triangle plate are established by
  the authenticated unchanged native authoring source, not invented shape labels.

The CPU analyzer reconstructs final contacts exactly from callback/post-step
journal records. All 72 original feature × gap × speed × rotation × creation
controls and all 12 controlled ticks must be present. Native fields and raw
shape ordering are required; only retainer shape 0 is qualified. Initialization
callbacks remain separate and do not acquire invented pre/post samples.

The witness candidate is the unmodified `p + separation * normal`, with raw
shape1-to-shape0 normals. Independent analytic six-plane equations and full
linear coordinate solves recompute every stored alpha hypothesis, the plate
point residuals, and the pre/post feature decision. The existing runtime
classifier is then compared with that independent result. No production
classifier or production plane helper supplies the independent verdict.

Qualification requires the original 0.1 mm tolerance, raw-point and midpoint
negative controls, actual positive front/side/back coverage across rotations
and creation orders, every front positive accepted in both frames, and zero
accepted side/back positives. Face indices remain raw cooked indices, never
CAD labels. Shape 1 and other shape combinations remain unqualified. Counts
are recomputed from the fresh raw evidence and cross-checked against COMPLETE.

On success, the fresh output contains `INDEPENDENT_ANALYSIS.json`,
`QUALIFIED.json` and `CONTRACT.json`. The latter is the exact original
`retainer_front_audit_v3` contract value needed by the capture helper. The
issuer calls the unchanged `retainer_witness.load_contract` before returning.
A rejected run emits no new qualification receipt; preserve its raw run and
the analyzer's stderr. CPU diagnostic fixtures are explicitly rejected by the
issuer and cannot initialize native witness authorization.

This qualifies native contact semantics only. It provides no robot episode,
training population, ACT, held-out evaluation or hardware qualification.
