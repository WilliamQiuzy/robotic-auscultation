# What remains before a real-robot study

This project currently evaluates simulated motion and contact. Passing its
physics gates would not establish acoustic quality or clinical safety. The
3.5–4.5 N hold band is the fixed simulation benchmark condition, not a force
recommendation for use on people.

The rebuilt comparison uses two held-out body identities, two already-seen
seated pose templates and the R3/L3 targets: eight independent, home-start
scenarios per system. It does not test the complete ten-point sequence on
these bodies, unseen pose families, moving patients or clothed contact. The
historical nominal ten-point expert run belongs to a separate cohort.

All three systems receive prescribed 3D target poses from simulation plans.
This benchmark does not train or evaluate an RGB-D target localizer. ACT state17
is q7 + target-relative position3 + rotation6D + confidence1 (exactly1); H2
produces34 model features, with no RGB, depth or Hall model input.

ACT controls the learned approach from state input. Its contact, hold,
release and return skills are analytic. Astra receives RGB, depth summaries,
simulated Hall observations and joint state, and decides whether a local
analytic expert may proceed at skill boundaries. It does not produce joint
trajectories. Results compare these complete systems; they do not isolate
language-model motor skill or prove useful dependence on each sensor.

## Measurements the hardware team can collect in parallel

| Missing evidence | What to measure and retain | How it changes the simulator or evaluation |
| --- | --- | --- |
| Actual tool geometry | Installed C protrusion, B recess, ring diameter, tool-to-flange transform and their measurement uncertainty. The current approved nominal model has C protruding 5 mm and B recessed about 1 mm. | Replace nominal dimensions and propagate their uncertainty into reachability, contact frames and collision checks. |
| Hall response | Time-aligned raw voltage/ADC, independent axial load and slider displacement through loading and unloading, including the unloaded baseline and repeated cycles. | Fit the voltage–stroke–axial-force conversion and its hysteresis/noise. The simulated preload is metadata, not a measured patient force. |
| Axial versus surface-normal force | Measure axial sensor load and an independent contact-force reference at several tool tilts and sliding conditions. | Check the projection and friction discrepancy already observed at R2; Hall axial force alone is not a direct normal-force label. |
| Spring and mounting behavior | Force–displacement loops, preload, travel limits and settling after a change in load. | Identify stiffness, damping and mechanical stop behavior instead of treating CAD or nominal spring values as measurements. |
| Robot and command timing | Synchronized commanded/measured joints, joint sign and zero checks, timestamps, control frequency, latency, jitter and stopping response. | Validate the joint/action contract and add measured delay and tracking error to training and evaluation. |
| RGB-D calibration and visibility | Intrinsics, depth scale, camera-to-base transform, frame synchronization and recordings with the arm/tool occluding the back. | Check target localization and its uncertainty without using privileged simulator surface coordinates as a perception result. |
| Contact surface | Bench measurements on appropriate tissue surrogates, separately for bare-surface and thin-cloth conditions, including compression and friction. | Add surface variation supported by measurements; the current rigid bare-SMPL benchmark does not establish either real condition. |
| Audio quality | Synchronized stethoscope audio, force, tool pose and contact-state annotations in a separately authorized acquisition study. | Define an acoustic endpoint. Current simulation rewards and mechanical contact verdicts do not supply an acoustic training signal. |

Keep raw synchronized logs, units, hardware identifiers, calibration procedures
and uncertainty estimates. A voltage offset or a preload correction must not
be silently relabeled as patient force. These measurements should use a bench
and suitable surrogates before any separately approved work with participants.

## Next evidence for the learning system

The next training cohort needs more body identities and seated poses,
arbitrary-state expert queries, noisy demonstrations and recovery labels.
Choose model changes using TRAIN and selection-VAL only. Once a held-out
benchmark has been inspected, it becomes a reported test cohort; a later model
needs a new untouched identity split for a new generalization claim.

Test sensor dependence by perturbing or withholding each observation while
holding the scenario and execution contract fixed. Report approach, actual
contact, continuous hold, release and return outcomes, as well as aborts and
infrastructure coverage. A valid input packet, a fitted loss curve, an expert
replay or a language-model response is not a successful robot episode.
