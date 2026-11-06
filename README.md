# Robotic Auscultation

**[Watch the interactive simulation demo](https://williamqiuzy.github.io/robotic-auscultation/)** · [中文](README.zh-CN.md) · [Train ACT](docs/TRAINING.md) · [Evaluation & benchmark](docs/BENCHMARK.md) · [Astra integration](docs/ASTRA.md)

<p align="center"><a href="https://williamqiuzy.github.io/robotic-auscultation/"><img src="docs/assets/expert-poster.png" width="760" alt="Actual Isaac Sim analytic expert episode; this is not a learned-policy success"></a></p>

### Rebuilt matched cohort · November 6, 2026 · partial comparison

The new cohort has **two held-out identities × two already-seen seated poses × R3/L3**, eight independent home-start scenarios per system. Original independent qualifications are closed for **expert 8/8** and **selected state-only ACT 0/8**; **all eight Astra verdicts remain pending**. ACT stopped at the original posture envelope in six cases and original mesh-clearance gate in two. Actual model invocation is authenticated in all eight. This is not a completed three-system result or a continuous ten-point validation.

The new ACT fit completed **20,000 updates**, selecting **step 4,500**, with TRAIN/selection-VAL H20 target MAE **5.664018/26.522571 mrad**. Training used 18 positive expert episodes in 20 planned data slots (four TRAIN identities / one selection identity). These offline target errors are not measured tracking or contact success. ACT receives q7 + given-target-relative pose9 + confidence1, H2 (34 features), with no RGB-D/Hall input. Astra supervises three fixed skill boundaries; GivenContactExpert executes local motor/contact control. All use prescribed 3D targets, not learned visual target localization. Modality causality is untested; planned worker parallelism 3/3/1 prevents a controlled speed claim.

[Actual same-case expert/ACT videos and experiment article](docs/drafts/rebuild-20261105.html) · [Partial closed-arm evidence](docs/drafts/evidence/rebuilt-matched-20261106-v3.partial.json) · [Hardware and real-world gaps](docs/real_world_gap_20261105.md). The final canonical cohort remains unpublished and incomplete while Astra runs. The historical November 5 release below keeps its original evidence, denominators and assets.

### Simulation demo release · November 5, 2026

This release makes the training task, actual simulator videos and evaluation evidence available together. **The analytic expert is qualified; the published ACT pilot is not yet usable.** The task has ten posterior targets, but this small release does not validate all ten with a learned model.

| Published evidence | Result | Interpretation |
|---|---|---|
| Analytic expert, fixed eight-case physical qualification | **8/8** | Actual approach, contact, ≥3 s valid hold, unload and home |
| Official state-only ACT, two 5,000-step trials | TRAIN/VAL MAE **7.396/126.991 mrad** for the selected augmentation-on model | Seven TRAIN bodies / one selection VAL body; offline action error is not task success |
| Selected ACT weights driving Isaac | **0/2** | One TRAIN + one selection VAL scene; both stopped before contact |
| Independent-body benchmark | **Not reported** | A protocol is published; no generalization score is inferred |
| Astra + local skills | **Interface prototype** | Dry-run request and local validation code; no live API or Isaac result |

These rows are different diagnostic cohorts, **not a controlled leaderboard**. [Original JSON evidence and media checksums](docs/data/assets_manifest.json) are included. Videos are actual simulation recordings, not real patients or generated motion. Evidence dates are October 28–29; publication date is November 5.

```sh
# Inspect results and verify all released summary/media hashes; no GPU needed.
python -m demo.benchmark --verify-assets --output /tmp/ausc-results.json
# Test the Astra request interface without an API call or robot motion.
python -m demo.astra_policy --observation examples/astra_observation.json --output /tmp/astra-request.json
```

The [portable official-ACT training entry point](demo/train_act.py) uses state17/history2, H20/E10, identity-disjoint data, TRAIN-only normalization and full-validation selection. Install NumPy for CPU interface checks. Training also requires the pinned LeRobot/PyTorch environment and your qualified exported arrays. Licensed body assets, original raw arrays, checkpoints and the complete original native environment are not bundled; [reproduction boundaries](docs/TRAINING.md) are explicit.

Astra is proposed as a supervisor above local skills. It consumes RGB (when supplied), depth-derived target geometry, joints and Hall voltage; it cannot change the local contact controller's thresholds. The [integration note](docs/ASTRA.md) separates this prototype from future direct-action and actual simulator evaluations.

---

[![tests](https://github.com/WilliamQiuzy/robotic-auscultation/actions/workflows/tests.yml/badge.svg)](https://github.com/WilliamQiuzy/robotic-auscultation/actions/workflows/tests.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

A robot that performs posterior lung auscultation. It finds the ten standard listening points on a
seated patient's back, places a spring-loaded digital stethoscope on each one with controlled force,
and judges whether the recording is good enough to use.

Research project at the Rajpurkar Lab, Harvard University (May 2026 to present).
**This is active, unfinished work.** The perception, audio, hardware and simulation pieces below are
working; a learned closed-loop policy is still being developed, and this repository is updated as the
project moves forward.

<p align="center">
  <img src="auscultation/docs/figures/model_architecture.png" width="860"
       alt="Architecture: perception produces target points, an arm-control policy executes contact, and an audio loop refines the target">
</p>

## Approach

Existing robotic auscultation systems reduce the task to "locate a 3D point, then move the stethoscope
there", with the point coming from an anatomical atlas, a registered body mesh or a network. Clinicians
work differently: they listen while they move, compare left and right, and adjust pressure until the
sound is clear. This project treats auscultation as a sensorimotor task with sound in the loop:

1. **Perception** turns camera frames into the ten auscultation points and keeps them locked to the
   same anatomy while the patient breathes and shifts.
2. **An arm-control policy** approaches, makes contact, holds a target force and records.
3. **A recording-quality model** decides whether the clip is usable, which tells the policy whether to
   stay or search the neighbourhood.

## What is in place

| Component | What it does | Code |
|---|---|---|
| Point detection and tracking | DINOv2 ViT-B with LoRA and a 10-landmark head that reads a depth-derived surface-normal map, so clothing colour and texture never enter the input. A model-based tracker fits one similarity pose to all ten points each frame (weighted Umeyama with IRLS outlier rejection) and flags, per point, whether it was actually observed. Held-out real frames: PCK@5 = 98%; per-frame jitter cut from 2.8 mm to 1.0 mm at about 33 ms lag. | [`auscultation/ausc/perception`](auscultation/ausc/perception/tracking) |
| Recording-quality model | AST embeddings with a gradient-boosted head flag unusable lung-sound clips. On SPRSound (4,960 clips, 382 patients) it reaches AUC 0.94 on a patient-grouped split, against 0.86 for WavLM. Training with hum and noise augmentation keeps AUC at 0.96 or higher under those corruptions. | [`auscultation/ausc/audio`](auscultation/ausc/audio), [results](auscultation/docs/sprsound_quality_loop_RESULTS.md) |
| Force-sensing end effector | Parametric CadQuery models of a passive spring mount around a Thinklabs One digital stethoscope. A coaxial magnet and an SS49E Hall sensor measure spring compression, read by an Arduino. | [`auscultation/mechanical_sensor`](auscultation/mechanical_sensor) |
| Real-arm data collection | Leader-follower teleoperation of an OpenArm at 1 kHz over CAN, a C++ telemetry publisher on a preallocated lock-free ring patched into `openarm_can`, hand-eye calibration, and a force-limited safety state machine. | [`auscultation/openarm_bridge`](auscultation/openarm_bridge), [`auscultation/scripts`](auscultation/scripts) |
| Simulation and expert | NVIDIA Isaac Sim digital twin (5.1 development; released qualification uses native 6.0) of the seated-patient task: the OpenArm with its gripper replaced by a stethoscope on a preloaded prismatic spring joint, clearance, joint-limit and contact-force gates, and an analytic expert (collision-checked motion planning with Ruckig trajectories, plus force feedback) that labels demonstrations for imitation learning. | [`sim/ausc_policy`](sim/ausc_policy), [`configs/ausc_policy`](configs/ausc_policy) |

Earlier policy experiments (π0.5 fine-tuning, ACT and other baselines) live in the versioned
`sim/auscultation_*` directories. They are kept read-only because the current code imports parts of
them and their results are referenced by content digest.

## Recording-quality results

<p align="center">
  <img src="auscultation/docs/figures/fig_generalization.png" width="49%"
       alt="Patient-grouped versus random cross-validation AUC for four audio encoders">
  <img src="auscultation/docs/figures/fig_robustness.png" width="49%"
       alt="AUC under 60 Hz hum and added noise, with and without augmentation">
</p>

Random cross-validation inflates AUC because clips from the same patient land in both train and test.
On a patient-grouped split, AST generalizes best (0.94, leakage gap 0.05). Models trained on clean audio
collapse under 60 Hz hum and noise; augmenting with the corruptions expected at the bedside restores them.

## Repository layout

```
auscultation/
  ausc/               perception, audio, datasets and training (Python package)
  mechanical_sensor/  end-effector CAD, Hall-sensor firmware, contact simulations
  openarm_bridge/     C++ telemetry patches and test harnesses for the OpenArm CAN stack
  scripts/            data collection, deployment to the robot PC, audio studies
  docs/               design notes and experiment write-ups
sim/
  ausc_policy/        current simulation, expert and policy code
  auscultation_*/     earlier iterations, kept read-only
configs/ausc_policy/  experiment configurations
reports/ausc_policy/  experiment reports
tests/                tests for sim/ausc_policy
```

Some design notes and experiment reports are written in Chinese.

## Running the tests

The simulation, expert, policy, point-tracking and robot-safety tests run on CPU with no robot, camera
or GPU:

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

python -m pytest tests                         # simulation, expert and policy logic
cd auscultation && python -m pytest ausc/perception/tracking/tests ausc/perception/robot/tests
```

CI skips the few test files that need simulator assets kept out of git;
[the workflow](.github/workflows/tests.yml) lists them.
Isaac Sim runs need Linux, a compatible NVIDIA GPU and the simulator version matching the experiment. The new portable demo interfaces do not require Isaac for CPU checks.

## What is not included

- No real-person images, real-patient audio/data or model weights. The demo includes rendered Isaac simulation videos and small public result summaries.
- Datasets (HuMMan, SMPL body models, SPRSound and others) must be obtained from their owners under
  their own licences; the scripts expect them under `auscultation/data/`.
- Server and lab-network configuration.

## Acknowledgements

The OpenArm robot description and CAN stack are by Enactic, Inc. (Apache License 2.0); the patches in
`auscultation/openarm_bridge` apply to `openarm_can` 1.2.2. The project also builds on DINOv2, the Audio
Spectrogram Transformer, LeRobot, openpi, cuRobo and Ruckig, and uses the SPRSound and HuMMan datasets.

## Author

Ziyue (William) Qiu, Harvard University
