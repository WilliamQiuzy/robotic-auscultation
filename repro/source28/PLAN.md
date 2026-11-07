# Source28 inspection and CPU capsule implementation plan

**Goal:** Publish a partial, exact-byte source28 inspection capsule with isolated CPU fixtures, without changing the historical root or source26 capsule.

**Architecture:** Select original Git blobs from commit `d932993b13030da7873d04a4a9471df99bb836ec`, verify each against the independently authenticated full source28 manifest, and record byte/SHA-256/Git-blob identities. List two external operator copies separately as inspection artifacts. A new public verifier checks the closed selection and runs only explicit original CPU fixtures in an isolated process.

**Tech stack:** Python 3.11, unittest, pytest, NumPy, SciPy, CPU PyTorch.

## Scope and acceptance

- All additions live in `repro/source28/`, `docs/SOURCE28_REPRODUCTION.md`, or `demo_tests/test_source28_capsule.py`.
- The root source19 and historical source26 contents remain unchanged.
- This is `PARTIAL_INSPECTION_CPU_CAPSULE`, never the full 1,673-file runtime, model, dataset, environment, or physical qualification.
- Original project/test/config/URDF bytes come from committed Git blobs and are members of the full manifest. External operators retain exact bytes but are explicitly outside that manifest.
- No raw data, arrays, checkpoints, meshes, licensed body payloads, credentials, server addresses, private Mac paths, symlinks or caches are selected.
- Original source26 bridging requires the complete 1,660-file original archive; the public source26 selection cannot satisfy it.
- Fixture import paths remain inside source28. Skip reasons remain visible. Native/official owner execution is never attempted here.

## Execution

- [x] Write public adapter tests for byte/Git-blob tampering, safe paths, closed inventory, provenance, external-operator separation and import boundaries; observe failure because the source28 verifier is absent.
- [x] Copy the minimal original selection and explicit V4 numerical modules/tests/helpers from Git; authenticate the archive/manifest and all chosen bytes before copying.
- [x] Add the minimal verifier; run adapter tests and the selected original CPU fixtures with `-I -B`, preserving official-environment skips.
- [x] Document source bindings, exact external Linux operator placement requirements, isolated commands and the original-owner prerequisites/gates.
- [x] Record the bounded CPU result, inspect privacy and additive diff, rerun verification, and commit only this capsule change.
- [ ] Root reviews SPEC then assigns distinct QUALITY review before integration/publication.

This plan follows the already authorized single-implementer workflow. Root manages reviews, publication and merge.
