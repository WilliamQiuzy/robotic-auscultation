# Reject a numerical STOP hypothesis without relaxing its bound

The fresh `p001120-p000476_a001231_f0023-r3` capture failed in the cloned
interior STOP preview at physical row7140, before any STOP setter or contact.
CPU Ruckig0.15.3 replay matched all6900 interpolated commands after the240-row
settle exactly. The first rejected clone returned Finished at step31, output
time0.12916666666666665s, trajectory duration0.1264993177084053s.

J7 had a +3.196354292356318e-11rad position residual against its unchanged
64-ULP bound1.424586002620553e-14rad. Other q residuals were at most2.22e-16;
maximum |v| was1.3923e-18 and |a|1.3878e-17. The same J7 discrepancy was
already present in the native synchronized `UP_VEL_UDDU` profile endpoint.
Independent70-digit integration of its seven segments gave3.19635569915e-11,
so this is a native profile solution residual, not the old1e-22 joint-limit
roundoff or post-end clock drift. The tagged upstream
[profile validation](https://github.com/pantor/ruckig/blob/v0.15.3/include/ruckig/profile.hpp)
admits position residuals below1e-8; its synchronized
[step2 solver](https://github.com/pantor/ruckig/blob/v0.15.3/src/ruckig/position_third_step2.cpp)
therefore can report Finished without satisfying our much smaller bound.

The64-ULP rule stays unchanged. `FinishedStationaryRoundoffError` names this
exact existing rejection and records its q/v/a/target. Only a read-only STOP
clone may catch it. That hypothesis is retained in `parking_numeric_rejections`,
with unchanged live input/output/history/object IDs and no STOP setter. The
same row cannot retry. Ordinary checked L2 continues to the next fresh20Hz
grid, which must pass the original current observed/commanded interior trigger.
Teacher resumes its original waypoint; ACT uses its existing L2 query path.
No extrapolated observation or off-grid candidate is accepted.

All waiting and actual STOP/settle now share the original1200-tick budget
starting at the first rejected candidate. Rejected and accepted preview work
shares the existing30s CPU budget. Other exceptions, including native live
execution, source, derivative, joint, geometry, contact or force errors, remain
fatal. The dataset reader independently reconstructs each claimed numerical
rejection from its actual input, rejects forged events, and checks the shared
clock. Successful old captures without rejection events retain their contract.
Observation/action labels, force/contact/camera/parking/physical bounds and
the actual failed episode verdict are unchanged.

The held-original-waypoint R3 CPU hypothesis rejected rows7140 and7152, then
accepted row7164 with one exact stationary STOP command. Waiting24ticks,
STOP1tick and24constant-tail intervals total49ticks (0.20417s). This proves a
bounded numerical candidate exists; future measured pose, collision and
settling still require a new native run. The running frozen5 source was not
modified and retained execution SHA
`42d98ee5aee6534b7e8577688c4d5ecb2f870384b392d4952fc71e870b22f01d`.

Linux-only actual replay receipt:
`/opt/wam-runtime/fast-wam/diagnostics/stop-preview-numeric-20261105-v1/replay/COMPLETE.json`,
SHA`472f24dffb432ae19c9923c818b3205ecded77593b12b2586b86a1438d2603ec`.
Its NPZ retains the bit-identical issued prefix and hypothetical future commands.
The independent proposed code bytes are retained beside it; raw captures were
not mirrored to Mac. Reproduce from a new code version on Linux with:

```sh
CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.stop_preview_numeric_diagnostic \
  --episode /ACTUAL_FAILED_EPISODE --output /FRESH_DIAGNOSTIC_DIRECTORY
```

The analogous L3 case matched all7440physical/7200interpolated prefix commands,
then exposed a separate limitation: after rejecting its row7440 STOP profile,
held-waypoint continuation hit FD jerk1.5000000166764949 at row7446, exceeding
the unchanged1.5+1e-8 limit. This fix correctly leaves that continuation fatal.
The Linux negative receipt is `.../stop-preview-numeric-20261105-v1/L3_NEGATIVE.json`,
SHA`6fc353660c7f51df45984f2004f5a8020d0530dd7c9401f9915e1cb0f1717ccf`.
No whole-case or physical success is claimed for either diagnostic.
