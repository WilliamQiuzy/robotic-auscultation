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

## Caveats / next
- SPRSound `Poor Quality` is **grosser** than the doctors' subtle "usable" grading (κ0.5) — this validates
  the *pipeline/recipe*; final deployment eval must use the **doctor 68-clip consensus set** (stranded on the
  US-CA-2 network volume; re-run when a US-CA-2 GPU frees up).
- No cross-dataset (ICBHI) test yet; SPRSound official **inter- vs intra-patient** test split is the next
  benchmark-comparable number.
- Artifacts: `results.jsonl` (all runs), feature cache, driver `audio_loop*.py`.
