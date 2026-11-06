# Replacement-Pod resource identity for expanded diagnostic ACT

The deleted Pod's formal runtime resource constants remain unchanged:
`/workspace/wam/locks/isaac_gpu.lock`, device/inode `(47, 288098408)`.
Only `m2_given_contact_expanded_pilot_train` accepts an explicit optional
`replacement_pod_gpu_binding` config field. The ordinary `owned_gpu()` path and
`process_guard` are unchanged. This is a resource identity migration, with no
change to a physical threshold, dataset gate, model, training schedule or
checkpoint selection rule.

The new field is the issuer's exact JSON object:

```json
{
  "schema": "m2_rebuild_expanded_gpu_resource_binding_v1",
  "base": "/opt/wam-runtime/fast-wam",
  "path": "/opt/wam-runtime/fast-wam/locks/isaac_gpu.lock",
  "device": 137,
  "inode": 2706858966,
  "gpu_uuid": "GPU-26e2eb9c-c0f5-9870-687c-28cdec1a68ea"
}
```

Those device/inode/UUID values were observed by the read-only issuer on the
replacement Pod on 2026-11-05; they are not new global defaults. Reissue for any
resource change. No lock acquisition or training was performed by that read.

Admission requires an actual bound diagnostic dataset COMPLETE with
`rebuild_epoch: "observed-reconstruction-20261105"`, `original_cases: 0`,
`complete: true`, `diagnostic_only: true`, `formal_acceptance: false`,
`training_eligible: false`, `test_payloads_opened: false`, the original task
contract, and exact TRAIN4 identities p001120/p000758/p001031/p000833 plus
selection VALIDATION1 p000923. Frozen-evaluation and TEST identities cannot
enter this route. The original expanded dataset loader still checks all episode
scopes, arrays, causal indices and TRAIN-only normalization before resource
acquisition. Later CPU checkpoint admission validates binding metadata without
opening the original lock or querying/acquiring a GPU.

## Install and issue

Keep the existing capture and exported v5 source/data immutable. Install the new
source into a separate v7 archive/freeze using the normal source installation
workflow. Do not overwrite an active capture archive, recreate a lock, use a
symlink, or change its inode. From that new source, issue a fresh resource JSON
using the CPU interpreter:

```sh
CUDA_VISIBLE_DEVICES='' /CPU/PYTHON -B -m sim.ausc_policy.rebuild_gpu_binding \
  --issue --base /opt/wam-runtime/fast-wam \
  --output /FRESH_CONFIG_DIR/replacement-pod-gpu-binding.json
```

Issuance opens the **existing** lock read-only, inspects its regular-file,
single-link, canonical-path/ancestor and FD/path identity, and reads physical
GPU 0's UUID with a bounded `nvidia-smi` identity query. It does not call flock,
create/replace/write the lock, initialize CUDA, start compute, or create an owner.
It may run while the original capture holds the lock. Only the explicitly named
new output JSON is written; existing output is refused.

Copy the actual v5 exported ACT config into a fresh v7 config and add the exact
issued object, without rewriting the exported dataset, arrays or provenance:

```python
import json
from pathlib import Path
config = json.loads(Path('/V5_EXPORT/ACT_CONFIG.json').read_text())
config['replacement_pod_gpu_binding'] = json.loads(
    Path('/FRESH_CONFIG_DIR/replacement-pod-gpu-binding.json').read_text())
with Path('/FRESH_CONFIG_DIR/ACT_CONFIG-v7.json').open('x') as stream:
    json.dump(config, stream, indent=2)
    stream.write('\n')
```

Use the actual export's config filename. Its `base` must exactly match the issued
canonical base; all original training/data/schedule fields remain intact. Root
owns stopping the waiting continuation, waiting for the current capture to
finish and drain, and launching the original guarded owner with the new config:

```sh
/ACT/PYTHON -B -m sim.ausc_policy.run \
  --config /FRESH_CONFIG_DIR/ACT_CONFIG-v7.json --run-id rebuilt-act-v7
```

## Acquisition evidence

The new runtime path requires Linux, the original `AUSC_OWNED_RUN=1` marker,
`CUDA_VISIBLE_DEVICES=0`, idle-GPU checks, and the exact issued GPU-0 UUID. It
opens only the existing regular single-link lock, rejects symlink ancestors and
aliases, verifies issued device/inode and FD/path equality before and after
nonblocking exclusive flock, and repeats UUID/idle checks while holding the
lease. A busy capture lock fails; there is no retry, fallback owner, alternate
lock, environment override or bypass. The lease is released on normal exit or
failure. Expanded STARTED/COMPLETE ownership records retain the exact binding and
actual acquisition checks. Existing guard session checks remain mandatory.

CPU tests exercise read-only issuance while a local lock is held, refusal of
active acquisition, wrong inode/device/UUID, changed file identity after flock,
symlinks/hardlinks/missing/nonregular paths, unowned/CUDA-hidden/busy processes,
non-rebuild/formal/TEST/scope changes, and explicit versus legacy trainer lease
dispatch before CUDA initialization. Actual training acquisition, CUDA work,
checkpoint equivalence and physical qualification require their normal live
receipts; the read-only issuer establishes none of those outcomes.
