# Source45 partial inspection and synthetic CPU capsule

Goal: add an independent exact-byte selection from Source45 commit
`29024f00a8656865fe79242605580278561f463e`, using the public Source28 capsule
conventions. Preserve the released root, historical capsules, blog and benchmarks.

- [x] Read current capsule patterns and original test contracts.
- [x] Write adapter tests and observe failure because the Source45 verifier is absent.
- [x] Authenticate the complete local archive/manifest and every selected Git blob;
  copy 304 original text files without rewriting their bytes.
- [x] Add a bounded test-node profile and isolated import/inventory verifier.
- [x] Run bounded actual synthetic CPU checks and record results, versions and skips.
- [x] Document Source44/45 deltas and the missing original/native prerequisites.
- [x] Review the additive diff, verify, explicitly commit this capsule, and stop.
- [ ] Root obtains fresh SPEC then distinct QUALITY review before integration/publication.

The complete original context is 1,724 files. This 304-file selection is neither
that complete runtime nor a replacement for original Source40/41 archive owners.
No model training, simulator owner, dataset collection or public publication is
part of this implementation task. Root owns those separately.
