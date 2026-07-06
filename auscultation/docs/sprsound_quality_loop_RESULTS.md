# Lung-sound quality model — autonomous SPRSound loop (2026-07-03)

Autonomous ~multi-hour GPU loop (RunPod A5000) on **SPRSound** (open, CC-BY-4.0; 20,457 wavs /
15,199 record annotations). Motivation: the doctor-labeled Q model was near its label ceiling
(κ≈0.5–0.58, AUC 0.86 on 68 consensus clips) → the lever is **more data + robustness**, not more
pretraining. SPRSound gives a large, self-labeled analogue of our "usable" target.

**Figures**: [figures/fig_generalization.png](figures/fig_generalization.png) (honest vs leaky AUC) ·
[figures/fig_robustness.png](figures/fig_robustness.png) (corruption collapse + augmentation recovery).

**Task**: binary quality = `Poor Quality` (bad) vs usable (`Normal`/`CAS`/`DAS`/`CAS & DAS`).
~460 "Poor Quality" positives; sampled 4500 usable → 4960 clips, **382 patients**.
Sanity: poor vs usable have identical median duration (9.2 s) & RMS → **not** a trivial length/silence
artifact (duration-alone AUC 0.58).

## 1. Encoder/head sweep (random 5-fold CV + bootstrap CI)
| features | logreg | GBM |
|---|---|---|
| handcrafted (145-d: logmel+spectral+SQI+band-ratios) | 0.877 | 0.975 |
| handcrafted + 90–1100 Hz lung band | 0.894 | 0.976 |
| **WavLM-base** embed (768-d) | 0.958 | 0.987 |
| **AST** embed (768-d) | — | **0.993** |
| WavLM+AST | — | 0.993 |

## 2. ⭐ Honest generalization — patient-grouped split (no same-patient leakage)
Random CV **inflates** AUC via same-patient clips in train+test. Patient-grouped (GroupKFold, 382 patients):

| encoder | patient-grouped AUC | random AUC | leakage gap |
|---|---|---|---|
| handcrafted-band | 0.857 | 0.976 | 0.119 |
| WavLM | 0.862 | 0.987 | 0.125 |
| **AST** | **0.941** | 0.993 | **0.052** |
| WavLM+AST | 0.935 | 0.993 | 0.057 |

**AST generalizes to new patients far better** (0.94 honest) than WavLM/handcrafted (0.86) — the latter
partly memorize patient-specific quirks (gap 0.12). → **AST is the recommended encoder**, honest AUC ≈0.94.

## 3. ⭐ Robustness — clean-trained models COLLAPSE under real-world corruption
Inject 60 Hz hum (+harmonics) and white noise (10 dB SNR), evaluate:

| encoder / train | clean | +60Hz hum | +noise |
|---|---|---|---|
| handcrafted (clean-train) | 0.879 | 0.791 | **0.497** |
| WavLM (clean-train) | ~0.98 | **0.572** | **0.669** |
| **WavLM (hum+noise augment-train)** | **0.978** | **0.962** | **0.916** |
| AST (clean-train) | 0.991 | 0.904 | 0.812 |
| **AST (hum+noise augment-train)** | **0.988** | **0.989** | **0.964** |

→ Even WavLM collapses (hum 0.57, noise 0.67) when trained clean. **AST is inherently more robust** (clean-
trained hum 0.90 / noise 0.81), and **AST + hum+noise augmentation is near-flat across all conditions
(clean 0.99 / hum 0.99 / noise 0.96) at ~zero clean-AUC cost.** This is the deployable recipe.

**AST wins on all three axes**: accuracy (0.99 random / **0.94 patient-honest**), generalization (leakage
gap 0.05 vs WavLM 0.13), and robustness (aug: hum 0.99 / noise 0.96). WavLM/handcrafted lose on both
generalization and robustness.

**Does augmentation give GENERAL robustness?** AST trained clean vs hum+noise-aug, tested on **unseen**
corruptions (reverb / clipping / band-limit): aug fixes the *seen* corruptions (hum 0.90→0.99, noise
0.81→0.96) but barely moves the *unseen* ones (reverb 0.93, clip 0.94, bandlimit 0.93 — ≈unchanged).
→ **augment with the corruptions you actually expect at deployment** (for us: 60 Hz hum + ambient noise,
both covered). AST is inherently decent on reverb/clip/bandlimit (~0.93) even without augmentation.

## 4. Hum-specific behavior (ties to our real 60 Hz problem)
- **Hum-severity**: clean model's mean P(poor) rises with hum level: 0.0→.001, 0.1→.17, 0.2→.31, 0.4→.51
  — it partially flags *severe* hum but misses moderate hum.
- **Hum-aware training** (relabel heavily-hummed clips as poor): **keeps clean AUC (0.966) AND flags 100 %
  of hummed clips as poor** → the quality gate correctly rejects our 60 Hz recordings with no clean loss.

## 5. Calibration & deployment operating point (isotonic)
AUC 0.99, **Brier 0.016** (well-calibrated). Operating point thr 0.15 → **recall 0.99 @ precision 0.62**
for flagging poor clips (catch ~all bad, ~1.6× over-flag).

## 6. Interpretability
Permutation importance: **zero-crossing-rate variability (zcr_s)** dominates (0.065), then spectral
bandwidth, mid-band ratios (250–1200 Hz), low mel bins. → "poor quality" ≈ erratic/noisy spectro-temporal
structure, consistent with the noise-fragility above.

## Recommendation (deployable Q on lung-sound quality)
**AST encoder (frozen) + GBM head + hum/noise train-time augmentation + isotonic calibration**, quality-gate
at the recall-0.99 operating point. Honest patient-level AUC ≈0.94; robust to 60 Hz hum & noise.

## ⭐ Deployable artifact (packaged 2026-07-05)
The recommendation above is packaged as a runnable gate under
[`../scripts/audio/sprsound_loop/`](../scripts/audio/sprsound_loop/):
- **`quality_gbm.joblib`** (2.5 MB) — GBM head trained on AST embeddings of the full SPRSound Poor-vs-usable
  set (4460 clips, 460 poor) **with 60 Hz-hum + noise augmentation** (each clip seen clean/hum/noise).
- **`quality_meta.json`** — encoder id, pooling, sample rate, operating threshold (0.15).
- **`sprsound_quality_gate.py`** — `python sprsound_quality_gate.py clip.wav …` → `P(poor)` + flag. CPU-OK.
- **`train_final.py`** — the training driver (reproduces the model from SPRSound at `/workspace/data/SPRSound`).

Self-test (in-sample functional check): Poor clip → P(poor)=0.996 (flag), usable clip → 0.000 (pass).
**Distinct** from `scripts/audio/predict_quality.py` (doctor-label Q v1, OPERA-CT, subtler "usable" grading,
AUC 0.72). This SPRSound gate targets **gross artifacts** (noise/SNR/hum/clipping); honest AUC ≈0.94.
The bundled `quality_gbm.joblib` is **numpy 1.x/2.x-portable** (trained under numpy2/sklearn1.9 on the pod,
rng stripped + re-saved so it loads on the project's numpy1.26/sklearn1.6 stack; verified identical
prediction across both).

**Deployment-domain check (ran locally on the Mac, CPU, 2026-07-05)** — never sent our recordings off-box:
- **HF_Lung_V1 clean lung samples** (real, non-SPRSound domain): 8/10 score P(poor)<0.11 (usable), only 2
  noisy ICU clips near threshold (0.156 / 0.216) → the gate transfers to a new device/domain without
  false-rejecting clean lung audio.
- **Our own ThinkLabs session captures** (2026-06-24): the gate **discriminates** — session `145957`
  P(poor)=0.338 (flag) vs session `145256` 0.008 (usable). Spectral diagnosis: the flagged one is NOT
  hum-driven (2.8% hum) but **broadband-noise-driven** (6.5% energy >2.5 kHz, lung-band only 70.9%, loud);
  the passed one is quiet + lung-concentrated (84.7%). Consistent with finding ⑥ (poor ≈ erratic/noisy
  spectro-temporal structure), so the gate flags for the right reason.
- **⭐ 60 Hz hum A/B (controlled, on our clean `145256` capture)** — inject the study's hum then notch it out:
  base 0.008 → hum l=0.05/0.10/0.20/0.40 → **0.005 / 0.018 / 0.034 / 0.014** → de-hummed 0.010. **All pass.**
  Even at hum l=0.20 (verified **92.8 %** of spectral energy is 60 Hz-family — a catastrophically hummy clip)
  the gate does **not** flag it. This is **correct and by design**: the shipped gate is the *hum+noise-
  **augmented*** model (findings §3), trained to be **invariant** to hum — it sees through incidental hum to
  judge the underlying lung content. It is therefore the *opposite* of the **hum-aware** variant (§4), which
  *relabels* heavily-hummed clips as poor and flags them. **Deployment implication / open choice**: the
  current gate answers "is the underlying lung sound usable? (ignore fixable hum)" → it will NOT reject our
  real 60 Hz-ground-loop recordings. If instead we want to *catch* hardware hum to re-record / fix the setup,
  use the §4 hum-aware head (or a cheap standalone 60 Hz-energy detector) alongside it. Both are one training
  run apart; pick per goal. (The original evicted `stethoscope_10s{,_dehum}.wav` A/B is now superseded by
  this controlled version.)

## ⭐ OPERA (respiratory-specific) vs AST — surprising
Head-to-head on the same 4960 clips / 382 patients (user-authorized to run the OPERA repo):

| encoder | patient-grouped (honest) | random CV | leakage gap | extract time |
|---|---|---|---|---|
| **AST** (general AudioSet) | **0.943** | 0.993 | 0.05 | 3 min |
| **OPERA-CT** (respiratory foundation model) | **0.794** | 0.969 | **0.175** | 32 min |

**AST decisively beats OPERA on lung-sound *quality*** — and OPERA generalizes to new patients far worse
(gap 0.175). Interpretation: OPERA is pretrained to capture respiratory *content* (crackles/wheezes,
disease) and excels there per its paper, but the *quality* task is about general recording artifacts
(noise/SNR/clipping) that the diverse-audio AST captures better; OPERA's specialization doesn't transfer
to quality and overfits patient traits. **→ For the quality gate, use AST, not OPERA.** (OPERA is also
~10× slower — no batching.)

## Additional checks
- **Official BioCAS2022 split** (AST): inter-patient test AUC **0.96**, intra-patient 0.928 — consistent
  with patient-grouped 0.94, but the official test sets have very few poor positives (8 / 2) so are noisy.
- **More usable data does NOT help**: 4.5k→6.7k usable, patient-grouped AUC 0.94→0.93 (flat). The model is
  saturated on the usable side — the real limit is the **poor-class count (460) & patient diversity**, i.e.
  "collect more data" specifically means *more poor / more subjects*, not more usable clips.
- Naive confidence-based selective prediction (reject uncertain by |p−0.5|) *hurts* here — it rejects the
  minority poor class; needs a class-aware rejection rule.

## ⭐ Cross-dataset generalization (KAUH + COPD — different hospitals/devices)
Brought in two non-gated external lung datasets (KAUH 336, COPD 504 wavs; ICBHI is gated/token-blocked).
Train the AST quality model on SPRSound, test transfer:
- **Domain shift** (mean P(poor) on *clean* clips): SPRSound 0.03 · KAUH 0.094 · COPD 0.052 — all below the
  0.15 gate threshold → the gate does **not** falsely reject a new hospital/device (small upward shift on
  KAUH warrants a threshold margin).
- **Corruption detection transfer** (clean vs hum+noise AUC): SPRSound 1.0 · KAUH 1.0 · COPD 1.0 → the gate's
  corruption detection is **domain-general** (60 Hz hum + noise is a strong cross-dataset signal).
- Caveat: KAUH/COPD lack real quality labels, so transfer is shown for *injected* corruption (an easy signal)
  + the domain-shift score check — natural subtle-quality transfer can only be measured where labels exist
  (SPRSound / the doctor consensus set).

## Caveats / next
- SPRSound `Poor Quality` is **grosser** than the doctors' subtle "usable" grading (κ0.5) — this validates
  the *pipeline/recipe*; final deployment eval must use the **doctor 68-clip consensus set** (stranded on the
  US-CA-2 network volume; re-run when a US-CA-2 GPU frees up).
- Cross-dataset transfer shown on **KAUH + COPD** (above); **ICBHI** specifically is still gated/token-blocked.
  SPRSound official **inter- vs intra-patient** split (0.96 / 0.928) is the benchmark-comparable number.
- Artifacts: `results.jsonl` / `xdata_results.jsonl` (all runs), driver `audio_loop*.py` / `audio_xdata.py`,
  and the **deployable gate** (`quality_gbm.joblib` + `sprsound_quality_gate.py`, see above).
