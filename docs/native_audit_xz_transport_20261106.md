# Native-audit transport correction

The source9 comparison stopped because full neutral audit files exceeded the
unchanged 100,000,000-byte whole-job budget. The two closed R3 jobs occupied
109,589,381 and 115,910,667 bytes; their audit gzip files alone occupied
106,629,873 and 112,952,352 bytes. The closed L3 job had only 166,702 bytes of
headroom. All other files were about 2.96 MB per job. These recording failures
remain source9 evidence; no existing artifact, status or qualification is changed.

On the largest closed recording, CPU-only counting tests found:

| Transport | Trace bytes | Projected whole job | JSON preservation |
| --- | ---: | ---: | --- |
| Existing gzip6 | 112,952,352 | 115,910,667 | Original |
| Gzip9, exact original JSON | 109,353,317 | 112,311,632 | Byte identical |
| Compact JSON + gzip9 | 88,810,201 | 91,768,644 | All fields/types/float.hex values equal; different JSON bytes |
| XZ3 / CRC64 | 52,735,204 | 55,693,519 | Byte identical |

The compact-gzip projection includes a conservative 128-byte header allowance.
It misses the requested 85 MB comfort target. Original row length averaged
43,909 bytes, with median 31,024, p95 171,544 and maximum 173,898; 9,623 rows
exceeded gzip's 32 KB history window. Compaction still left 2,695 rows above
that window. XZ3 retains the original strings and provides about 44.3 MB of
whole-job headroom for this observed case, without moving evidence outside the
budget or discarding fields.

Only the rebuilt matched adapter requests `given_control_native.jsonl.xz`, for
expert, ACT and Astra alike. The buffered writer uses XZ3/CRC64 for that explicit
suffix and retains gzip6 for legacy `.gz` callers. Its 240-row buffering and
original event construction are unchanged. The reader dispatches strictly among
XZ, gzip and legacy plain JSONL; the original bound compressed-file SHA, row,
clock, pose, q/command, complete/error and physical checks still apply. This adds
no acceptance gate or exception. The whole-job cap remains 100 MB.

Against base `730cb83`, AST comparison confirms that the only changed functions
are writer construction/attachment, matched analytic attachment, and the reader's
transport opener. All 109 other neutral-evaluation functions, including
`NativeAudit._append`, `_physical_native` and `_command_gates`, are unchanged.
The selected-model numerical/protocol source files and original dataset physical
acceptance file remain byte identical. Neither changed transport module belongs
to the selected checkpoint's pinned numerical source list. A fresh source10
inventory must honestly bind the new transport files; original source7 weights,
selection, scope and receipts must remain untouched. Root separately verifies
actual selected-policy CPU admission before any fresh native launch.

Actual Linux-only evidence, with source9 originals rehashed unchanged:

- `diagnostics/rebuilt-native-recording-size-audit-20261106-v1/SUMMARY.json`,
  5,591 bytes, SHA256 `96dd4078e02bab42dcdf4ac6d838488c2e79f0b100cd6b6e386c00d753c45b5f`.
- `diagnostics/rebuilt-native-recording-size-audit-20261106-v2/SUMMARY.json`,
  2,820 bytes, SHA256 `b42e81e34154e713eeeb23e7c3ad210dec6902af279dbf269744537a0e5b0157`.
- `diagnostics/rebuilt-native-recording-xz-reader-20261106-v1/SUMMARY.json`,
  3,101 bytes, SHA256 `1254c26d0ecb5fd7e7c6266a8e597b2dbdeba69207150e955aaf7ef820cffbb1`.

These are logical Linux evidence names; original receipts and raw recordings
remain with the experiment owner. The last fixture
used the actual patched buffered writer and reader on all 26,943 closed rows.
Its 1,183,032,328 decompressed bytes retain SHA256
`b6fcd058a14284af08cecbbd7eb547b61b6c0e5a9a3f5ebd8da2bf1608778ea8`.
The reader exercised the original complete/error, full tick-clock, q/command and
pose coverage checks. It took 109.5 seconds and peaked at 3.60 GB RSS while
holding all original rows. This is a labelled transport fixture, not a repaired
source9 outcome, new physical qualification or native execution. No model was
called; raw files and the fixture stayed on Linux.

The original private validation also included per-run authentication and
evaluation-operator tests. Those operator programs/tests are not distributed
in this public repository. Use the distributed library tests here:

```sh
python3 -B -m pytest tests/test_ausc_native_audit_transport.py tests/test_ausc_given_contact_pilot_native_audit.py tests/test_ausc_rebuild_paired_eval.py -q
```

Root owns review, source10 freezing/admission and the independent fixed eight-case,
24-arm rerun with unchanged checkpoint, scenarios, physics and authoritative
qualifiers. No source9 result is promoted or reused as a fresh benchmark outcome.

For parameterized preparation, execution, qualification and summarization through
the original owner, see [the paired evaluation guide](m2_rebuild_paired_eval_v1.md).
