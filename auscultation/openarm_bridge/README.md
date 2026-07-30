# OpenArm left exact-control telemetry candidate v4

This directory contains a versioned candidate for the physical collection pair:

- side: `left_arm`
- leader: `can1`
- follower: `can3`
- control: OpenArm unilateral leader-to-follower at 1 kHz
- transport: nominal 100 Hz `AF_UNIX/SOCK_DGRAM` envelopes

## What v4 records

The v1 AdminThread snapshot was only a copied reference and was not proof of
the action consumed by the follower controller. v4 instruments the follower
branch of `Control::unilateral_step` itself. Immediately around the two
`mit_control_all` calls, every 1 kHz record captures:

- exact arm and gripper `MITParam` values: `kp`, `kd`, `q`, `dq`, `tau`;
- canonical follower joint observation plus raw gripper motor observation;
- atomic reference generation and reference-source `CLOCK_MONOTONIC` time;
- arm/gripper call begin and normal-return times; and
- per-arm and per-gripper expected, full-frame-write and failed-write counts
  returned from the underlying `CANSocket` bool writes; and
- controller, source-manifest, YAML, OpenArm SDK, pair, URDF, mapper and binary
  hashes.

The candidate patches OpenArm CAN 1.2.2 so `mit_control_all` aggregates every
underlying `write_can_frame`/`write_canfd_frame` bool. A control tick succeeds
only when all seven arm frames and the one gripper frame returned full size.
This status means **Linux SocketCAN TX enqueue only**. It is not a motor receipt
or execution acknowledgement; v4 always records `motor_ack_observable=false`.
The separate receive-side SDK hook increments each of the seven arm counters
and the gripper counter only after the CAN ID matches, STATE decoding succeeds,
and `Motor::update_state` runs. Every control record carries those counters,
generations, monotonic timestamps, last gaps, and cumulative gap violations.
Joint motion is never used as a substitute for receive liveness.

Capture binds the last clean preflight receipt as the session Start baseline,
then checks counter progress, freshness, and new gap violations through End.
Training also requires a non-degenerate whole-session 7D observation trajectory
and command/response consistency. An all-zero, bit-identical, or near-zero 7D
trajectory is rejected; one held joint is allowed when the remaining joints
move. A materially changing command with the corresponding observation frozen
is rejected. Stored v2/v3 packets remain diagnostic-only and cannot be promoted.

## Real-time isolation

The 1 kHz control thread only copies a fixed-size trivially-copyable POD into a
preallocated lock-free SPSC ring. It performs no JSON serialization, socket
operation, allocation, logging or retry for telemetry. A separate lowered
priority thread serializes up to 16 consecutive exact records into each
envelope and uses non-blocking `sendto`.

Ring overflow, publisher rejection, socket-send failure, envelope loss, tick
loss, provenance drift and reference overlap are all explicit fail-closed
conditions. Observation interpolation is permitted for camera geometry.
Action interpolation is forbidden; actions are available only by exact
`control_tick_seq`.

## Candidate application

The audited Lingbox tree is dirty, so do not edit it in place. First copy the
official OpenArm CAN 1.2.2 source at commit
`3363f9879f8947f79db179e4e91f6dbf78f44a6f` into a candidate directory:

```bash
python3 apply_openarm_can_tx_status_patch.py \
  --openarm-can-root /path/to/openarm_can_1.2.2_candidate \
  --check

python3 apply_openarm_can_tx_status_patch.py \
  --openarm-can-root /path/to/openarm_can_1.2.2_candidate

python3 apply_openarm_can_feedback_patch.py \
  --openarm-can-root /path/to/openarm_can_1.2.2_candidate \
  --check

python3 apply_openarm_can_feedback_patch.py \
  --openarm-can-root /path/to/openarm_can_1.2.2_candidate
```

Build/install that SDK only into an isolated candidate prefix. Both added SDK
headers are included in the versioned CMake install set. Then copy the exact
unilateral teleop tree, apply the complete initialization reset, and stage its
controller patch:

```bash
python3 apply_openarm_init_reset_patch.py \
  --teleop-root /path/to/openarm_teleop_candidate \
  --check

python3 apply_openarm_init_reset_patch.py \
  --teleop-root /path/to/openarm_teleop_candidate

python3 apply_openarm_telemetry_patch.py \
  --teleop-root /path/to/openarm_teleop_candidate \
  --check

python3 apply_openarm_telemetry_patch.py \
  --teleop-root /path/to/openarm_teleop_candidate
```

The guards verify the OpenArm CAN 1.2.2 source plus unilateral entrypoint, `Control` source/header,
CMake, `follower.yaml`, state/mapper headers, MIT API headers and static
OpenArm SDK library before any candidate write. A mismatch causes zero writes.
The initialization patch is hash-locked to the exact
`IGNORE -> refresh -> receive -> enable -> PARAM -> MST_ID query -> receive ->
STATE -> receive` recovery sequence. The candidate still requires a
non-actuating `vcan`/shadow test, direct safe-stop proof, and an authorized HIL
test on Lingbox before promotion. No script in this directory activates a
controller.

## Runtime provenance

The candidate launcher must compute and export:

```text
WAM_OPENARM_TELEMETRY_SOCKET
WAM_OPENARM_URDF_SHA256
WAM_OPENARM_JOINT_MAPPER_SHA256
WAM_OPENARM_PAIR_CONTRACT_SHA256
WAM_OPENARM_CONTROLLER_BINARY_SHA256
WAM_OPENARM_CONTROLLER_SOURCE_MANIFEST_SHA256
WAM_OPENARM_FOLLOWER_CONFIG_SHA256
WAM_OPENARM_SDK_SHA256
```

Telemetry is only instantiated for the exact `left_arm can1 can3` arguments.
Missing/invalid telemetry never changes robot control, but collection readiness
must stay false.

The current pair activation pins a controller binary hash but not an immutable
binary path or argv. A trusted launcher therefore cannot yet be constructed
from activation alone. Add a no-symlink absolute binary path plus exact argv
and environment allowlist to the signed activation, then attest the final exec
before attaching the already-bound telemetry receiver.
