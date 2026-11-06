# Tighter current-interior L2 jerk request

The pose2/L3 retained failure has a second CPU counterexample: continuing its
recorded current waypoint after the rejected STOP proposal yields J4 commands
`[2.109328622332521,2.1093349904499203,2.1093402376660033,2.1093444724877157]`
at row7446. Their unchanged third finite difference is1.5000000166764949rad/s³,
above the existing1.5+1e-8 acceptance limit. No command from that hypothetical
continuation was sent to physics, and no physical acceptance was issued.

Current interior-v3 teacher and ACT now request L2 jerk1.4999, using an explicit
1e-4rad/s³ planning reserve (about0.0067% of1.5). STOP still requests1.49.
The physical FD acceptance, all velocity/acceleration/jerk gates, and the64-ULP
Finished check are unchanged. This is conservative planning, not a larger
error allowance. The matching handoff input snapshots, cloned/live profiles
and run result record the actual request. The dataset reader requires the
declared jerk reserve and exact matching current profile. The retired route
without the interior policy selector retains its old behavior; it is not a
rebuild/training route.

Linux CPU hypotheses replayed the actual old20Hz setter schedules under the
new requested profile, explicitly without claiming command identity. Maximum
command change versus old captures was1.944562970881969e-6rad. Original FD
gates passed throughout both schedules and bounded continuations:

| Schedule | Max jerk | Numerically accepted legal STOP candidate |
| --- | ---: | --- |
| R3 |1.4999000109128247|row7164, after24ticks; one STOP command |
| L3 |1.4999000231910031|row7452, after12ticks; nine STOP commands |

The R3 STOP hypotheses at7140/7152 and L3 at7440 still fail the unchanged
64-ULP check. The separate typed-preview rejection handles those failures
without normalizing their residuals. All wait/STOP/settle commands still fit
the same original1200-tick clock. Future measured geometry/contact/settling
remain unproven and require new native collection.

The first Linux hypothesis summary is
`/opt/wam-runtime/fast-wam/diagnostics/stop-preview-numeric-20261105-v1/JERK_14999_HYPOTHESIS.json`,
SHA`a41a07fce6f46614f95d94da372d9c2b195d35aaa28e0e00c05aee4de728fa37`.
The executable diagnostic supports an explicitly separate current-profile
hypothesis:

```sh
CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.stop_preview_numeric_diagnostic \
  --episode /ORIGINAL_FAILED_EPISODE --requested-jerk 1.4999 \
  --output /FRESH_HYPOTHESIS_OUTPUT
```

Its receipt sets `counterfactual_profile:true` and
`all_issued_commands_bit_identical:false`; arrays distinguish counterfactual
schedules from actual prefix replay. Actual old-prefix replay uses the
matching original-source version, as documented in the preceding narrow fix.
Do not reinterpret or merge old1.5 captures into the new1.4999 collection.
The rebuild must collect its complete declared cohort normally under the new
source, then independently export, train and evaluate it.

Focused current-interior/native CPU tests pass. The broader archival velocity
fixture also exposes the same preexisting legacy STOP Finished failure under
unchanged `ded0100` code and old1.5 request; that retired-route failure is
preserved and is not used to block or qualify the current rebuild.
