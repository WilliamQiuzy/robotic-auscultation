# Astra inference bridge and local skill integration

**Release scope:** the actual standard library inference service and HTTP client are now included. No native Astra qualification result is added. The historical expert and ACT evidence is unchanged. A validated model proposal is not a skill execution acknowledgment or a physical success result.

The new bridge is [`demo/astra_service.py`](../demo/astra_service.py) plus [`demo/astra_supervisor.py`](../demo/astra_supervisor.py). These are byte-for-byte exports of the service/protocol used by the development native supervisor. The older [`demo/astra_policy.py`](../demo/astra_policy.py) is a separate Responses API proposal prototype; it does not dispatch simulator skills and is not the native bridge described below.

## What runs where

| Inference host | Native Linux simulator owner |
| --- | --- |
| An already authenticated Codex CLI, exact model `gpt-6-astra` | Original scene, assets, physics, camera and local executor |
| Token-gated `POST /decision` listener bound to `127.0.0.1` | HTTP client through an explicit loopback SSH tunnel |
| RGB preview and structured depth/Hall/native-state summary for one call | Exact raw depth/masks, Hall ledger and authoritative episode artifacts |
| Strict proposed skill, timing and filtered CLI turn events | Decision admission, actual dispatch, execution acknowledgment and independent physical audit |

All motors, alignment, parking, contact/HOLD/release, collision checks and checked return home belong to **analytic `GivenContactExpert`**. Astra selects one bounded skill at a safe boundary. It does not generate joint actions, change force thresholds, identify an anatomical target independently from depth, or supply a physical success label. Simulation state must remain frozen during observation and inference; there is no remote inference inside an ongoing contact skill.

## Start the actual inference service

Requirements are Python 3 with its standard library, a compatible Codex CLI already authenticated **on the inference host**, and access to `gpt-6-astra`. The service does not install/authenticate Codex or establish account/model availability. The CLI must support every pinned flag shown below. An unsupported flag or unavailable model fails the call; there is no fallback model or retry.

```sh
# Generate a new bridge capability for this run; do not print or commit it.
export AUSC_ASTRA_RUN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 -m demo.astra_service --codex codex --port 8123
```

Use `--codex` to select your authenticated executable. Explicit `--codex codex` resolves from PATH and avoids the original bundled desktop-app default. Supply the same ephemeral `AUSC_ASTRA_RUN_TOKEN` securely to the native owner's environment. It is a bridge capability of at least 24 characters, **not** an API/account token. Do not place it in config JSON or logs, and do not copy account authentication files to the simulator host.

From the inference host, a reverse tunnel can expose the service only on the native host's loopback address. Replace the SSH alias with your own configured host; no project network address is distributed here:

```sh
ssh -N -R 127.0.0.1:8123:127.0.0.1:8123 <native-host-alias>
```

`ASTRA_LOOPBACK_READY` only reports that the local HTTP socket was bound. It does not prove authentication, model access, a successful inference call, sensor availability, simulator readiness or native acceptance.

For each request, the actual service runs:

```text
codex exec --model gpt-6-astra --ignore-user-config --ignore-rules
  --ephemeral --skip-git-repo-check --sandbox read-only --json
  --output-schema <temporary-schema> -o <temporary-response>
  -i <temporary-rgb.png> -
```

The prompt is sent on stdin. The default subprocess deadline is 120 seconds. The service stores RGB/schema/last-response files only in a temporary directory removed on success or failure; its HTTP logger does not print prompts, images or bearer tokens. Codex account state remains local. See the official [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) and [CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli); the exact invocation above is pinned by the released implementation and must be supported by the selected CLI.

## Call it with a current native observation

The public HTTP client is callable and uses only the standard library. It accepts a **current native observation**, not the old synthetic `examples/astra_observation.json`, which has a different schema. The following is an integration fragment for a native owner that already captured and retained the observation and its exact RGB bytes:

```python
import json
import os
from pathlib import Path
from demo.astra_supervisor import HTTPClient

observation = json.loads(Path("/data/current-observation.json").read_text())
rgb = Path("/data/current-rgb.png").read_bytes()
proposal = HTTPClient(
    "http://127.0.0.1:8123/decision",
    os.environ["AUSC_ASTRA_RUN_TOKEN"],
)(observation, rgb)
# Retain proposal separately; only the guarded native owner can admit/dispatch it.
```

Both ends require identical `astra_service.py` and `astra_supervisor.py` bytes: responses include SHA256 values that the client checks against its local copies. Restart the service after updating either module. The bridge uses a 2 MiB request limit, a 1 MiB decoded RGB limit, a 128 KiB HTTP response limit and a bounded HTTP timeout. The client disables environment HTTP proxies and accepts only an HTTP `127.0.0.1` `/decision` endpoint without query credentials.

## Native integration contract

The public bridge does **not** contain the new native sensor/dispatch adapters or a complete runnable asset/runtime package. The development `astra_sensors.py` and `astra_native.py` implementations follow this contract; their presence and actual native output must be verified in a separately prepared simulator environment.

1. **Capture and bind.** Build `ausc_astra_native_observation_v1` with episode/observation IDs, physics row and time, safe boundary and phase, measured `q_rad`/`dq_rad_s`, measured slider compression, a hash-bound prescribed full contact pose, RGB SHA256, and `local_executor: GivenContactExpert` / `local_motor_assistance: true`. Preserve a single approved `/World/RobotCam`. Do not advance world time, physics rows or native state to capture or infer.
2. **Preserve sensor provenance.** Native RGB and depth come from the same actual render product. Depth uses `distance_to_image_plane`: metric optical Z in meters. Keep exact 640×360 arrays and validity masks on the native host. Send a 32×18 valid-pixel median grid, per-cell validity counts, units, calibration/camera metadata and hashes. Invalid values stay masked; empty cells are JSON null. A probe chosen by projecting the prescribed pose is a depth measurement, not an independent anatomical target estimate.
3. **Keep Hall simulated.** The passive observer runs the existing Hall simulator from measured slider displacement and decodes its quantized packet through pinned calibration `ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054`. `hall_voltage_v` and `hall_force_N` are decoded **simulated ADC** voltage and spring-load estimates. Preserve `actual_adc_measurement: false`, `force_label_used: false` and `bench_verified: false`. They are neither hardware ADC readings nor physical normal-force/patient-pressure labels. The original local force controller retains its original native feedback.
4. **Admit, then dispatch.** Validate the exact observation ID, target, allowed skill, image digest, source hashes and frozen native state before invoking one local operation. Missing images, transport/model errors, stale state, forbidden fields and illegal skills must stop the native arm without another motion skill or model fallback. The portable protocol supports absent RGB for interface checks; **the native owner must require a current image** for native admission.
5. **Acknowledge, then audit.** Retain observation, image, request, proposal, actual execution acknowledgment and a record binding their hashes. Include native physics row ranges and failure reasons. Preserve original motor `command_source_id`; add separate Astra decision/model IDs and `motor_actor: analytic_GivenContactExpert`. Exclude these supervised motions from expert training capture and learned-policy labels. Skill completion does not set physical success; use the original independent native audit and sensor teardown receipt.

| Boundary | Permitted motion request | Native local operation |
| --- | --- | --- |
| `HOME` | `approach_target` | Existing strict measured approach and parking |
| `PARKED` | `continue_local_control` | Complete local alignment, contact, HOLD and release |
| `RELEASED` | `retract` | Existing checked return home |

Every boundary also permits `abort`. The strict decision has exactly `observation_id`, `skill`, `target_id`, `reason`. The full sensor packet is validated by `validate_sensor_fields`; the scalar Hall summary alone is insufficient for a native sensor-provenance claim. The original contact, force, tilt, joint, collision and return requirements remain authoritative and cannot be changed by the supervisor.

## Reproduction and evidence limits

A fresh public checkout can run the transport tests and, with separately authenticated compatible Codex, serve actual inference for a valid supplied observation. It cannot reproduce the native experiment by starting the HTTP listener alone. Licensed body/scene assets, human raw data, training arrays, weights, account state and the complete original simulator environment are not included. Neither are the new native sensor/dispatch adapter files or a prepared native case configuration in this patch. Existing simulation source in the repository does not substitute for those prerequisites.

```sh
python3 -m unittest demo_tests.test_astra_bridge -v
```

The bridge tests use explicitly synthetic sensor fixtures and controlled local transport/subprocess responses. They check authentication, payload/source/image identity, simulated Hall/depth provenance, bounded decisions, exact CLI model/flags, failure without fallback, and temporary-file cleanup. They make no model inference or native physics call and establish no Astra physical score, sensor usefulness, generalization, learned motor control or clinical performance.

The older API proposal dry run remains available via `python3 -m demo.astra_policy --observation examples/astra_observation.json --output /tmp/astra-request.json`. It has a different schema and provides no native dispatch. The [benchmark protocol](BENCHMARK.md) and frozen historical evidence remain unchanged.
