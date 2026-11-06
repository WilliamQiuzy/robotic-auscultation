# Training a robot to approach an auscultation point

The current reproducible numerical baseline is **state-only official LeRobot ACT**. The public simulation code already contains the robot, expert and control interfaces; the new `demo` runner provides a portable numerical reference. Licensed SMPL/HuMMan assets, exported training arrays, original native run bundles and learned weights are obtained separately. This release is a recorded demo and source release, not a one-command download of the original complete simulator environment.

The November 6 [handoff and recovery extension](act_handoff_recovery_v1.md)
adds an explicitly typed seventh `action_is_pad` array and actual recovery data
through the native library workflow. The six-array portable example below
remains a separate legacy contract. Neither prepared data nor offline training
establishes successful Isaac contact; the [experiment notebook](drafts/act-recovery-20261106.html)
records actual phase status and native results.

## Task and data flow

```
licensed seated body + ten anatomical anchors + OpenArm / stethoscope scene
  → analytic expert with collision-checked paths and local force control
  → actual Isaac recording / independent physical qualification
  → same-phase, same-owner causal history + H20 raw target windows
  → identity-disjoint TRAIN / validation
  → TRAIN-only normalization → official ACT → full-validation checkpoint selection
  → actual selected weights in Isaac → contact / hold / withdrawal / home scoring
```

Keep failed attempts in the outcome ledger. Only physically qualified expert episodes contribute positive imitation labels. Windows spanning phase switches, parking, contact, STOP, perturbations or actor-owner changes are excluded from the L2 approach labels. DART/recovery windows require an explicitly correct same-state expert label; commanded noise itself is not a positive label.

## Observation and action contract

| Item | Contract |
|---|---|
| State | state17: 7 measured joints + 10 target/geometry features |
| History | Two real 20 Hz observations, current history frame exactly equal to current state |
| ACT input | 7 current joint features + 27 features containing current geometry and previous full state |
| Output | H20 × 7 raw joint targets relative to **one fixed block origin** |
| Execution | Convert to absolute targets once; consume E10, get new observations, query again |
| Normalization | Shared mean/std across TRAIN history axes (0,1); standard deviation floor 1e-6 |
| Architecture | Official ACT, 2,658,823 parameters, width256, 4 heads, encoder2/decoder1, FFN1024, dropout0.1, no VAE |
| Optimizer | AdamW, learning rate1e-4, weight decay1e-4, clipped gradient norm1 |
| Selection | Earliest minimum full-validation H20 joint-target MAE; no test-based selection |

The runner's default has no q-input augmentation. The original augmentation-on pilot used q noise0.002 / dropout0.05; the no-augmentation ablation was worse on the single VAL body. The portable runner does not relabel itself as either historical experiment.

The ACT input is currently state and prescribed target geometry. The dataset can retain RGB-D and Hall without this baseline consuming them. Local Hall-based force control is separate. No acoustic reward, acoustic training or clinical decision is part of this baseline.

## Install on Linux

Use a separate environment, with a compatible CUDA PyTorch installation. The recorded experiment used the pinned official LeRobot source below:

```sh
python3.12 -m venv .venv-act
source .venv-act/bin/activate
pip install numpy
pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git@eb7a54c6ff4983fc86075442bf4fa61503e7b5d1'
```

Installation on a different driver/PyTorch stack is not numerical equivalence to the original environment. The original run used FP32, deterministic algorithms and TF32 disabled; retain software and array/checkpoint hashes when rerunning. This publication did not reinstall Isaac or migrate the old Pod's datasets into the new Pod.

## Array manifest

Export six finite, non-object arrays into each NPZ file:

```
states             [N,17]
histories          [N,2,17]
actions            [N,20,7]  # raw targets minus the fixed observation-block origin
origins            [N,7]     # exactly states[:,:7]
velocities         [N,7]     # (current_q - previous_q) / 0.05
observation_times  [N]
```

The manifest is JSON, schema `ausc-act-arrays-v1`:

```json
{
  "schema": "ausc-act-arrays-v1",
  "action_semantics": "raw_joint_target_minus_block_origin",
  "splits": {
    "train": {
      "file": "train.npz", "sha256": "<actual file hash>",
      "subject_ids": ["train-body-a"],
      "window_subject_ids": ["train-body-a"]
    },
    "validation": {
      "file": "validation.npz", "sha256": "<actual file hash>",
      "subject_ids": ["val-body-b"],
      "window_subject_ids": ["val-body-b"]
    }
  }
}
```

`window_subject_ids` has **one entry per actual window**, not one per episode. The one-entry lists above illustrate the schema only. Hashes and identity metadata must be produced from your real export. The portable reader validates arrays, checksums, current-history alignment, causal velocity, fixed origins and identity isolation. It trusts the upstream physics/phase qualification of those arrays; it does not create that qualification.

## Fit and inspect

Coordinate exclusive GPU use with other jobs and provide the shared lock agreed for that GPU:

```sh
mkdir -p /data/locks
python -m demo.train_act \
  --manifest /data/arrays.json --output /data/runs/act-v1 \
  --steps 5000 --eval-every 500 --device cuda \
  --gpu-lock /data/locks/gpu.lock
```

The lock coordinates processes using that same lock; it does not prevent unrelated jobs from using the GPU. A fresh output directory is required. The runner writes `metrics.jsonl`, `best.pt` and `COMPLETE.json` with checkpoint/input digests. Full validation is evaluated at each selection step; the last step is evaluated even when not divisible by the interval. `closed_loop_evaluated` remains false. A successful training exit does not imply a successful robot episode.

Use the existing native simulator executor to run the selected policy under the original scene, handoff and physical gates. This portable runner deliberately has no connection to a physical robot. Connecting its checkpoint to the exact original owned native deployment requires a source-bound adapter; it is not automatically accepted by the historical deployment receipt format.

## Released experiment

Seven TRAIN bodies / 3,908 windows and one selection VAL body / 500 windows produced two actual 5,000-update ACT trials. In the augmentation-on trial, selected TRAIN/VAL MAE was 7.396 / 126.991 mrad. The actual selected model then failed both released Isaac episodes before contact. See [evaluation](BENCHMARK.md) and [original evidence](data/act_summary.json).

CPU checks for this publication cover array/action contracts, identity isolation and the Astra request boundary. They do not establish that the portable entry point recreates the original 5,000-step weights or passes Isaac. The full original data and weights are not bundled.
