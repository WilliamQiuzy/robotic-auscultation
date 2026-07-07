# Audio modality & reward design (LOCKED 2026-07-07)

How the **stethoscope audio** is designed across the policy, the reward, the reward model, and the
data schema — so the imitation-learning data we collect is trainable *and* supports a reward model
that can keep improving. Decisions here are **locked** (approved 2026-07-07); open items are marked.

> Builds on [realtime_multimodal_pipeline.md](realtime_multimodal_pipeline.md) (acquisition, sync,
> LeRobot frames, contact-gated Q) and [acoustic_quality_strategy.md](acoustic_quality_strategy.md)
> (the Q model). Scope: posterior **lung** auscultation with the stethoscope on the follower OpenArm.

---

## 0. Core reframe — audio is the task's *success measurement*, not just another observation

In ordinary manipulation, success is geometric/visual and observable at the moment of action. In
auscultation, vision and contact force can look perfect while the recorded lung sound is still
unusable (subject moved, ambient noise, acoustically wrong spot). **Audio quality is the true
success criterion, and it arrives delayed (needs 2–3 respiratory cycles) and noisy (doctor κ≈0.5).**

A delayed, noisy measurement is awkward as a per-step *observation* (partial observability; the
policy must integrate it over time) but natural as a *reward*. Analogy: a dart policy does not
observe the score mid-throw — the score is the reward. The stethoscope-placement policy acts on
anatomy + contact (observable); audio quality is the "score".

**→ Decision (LOCKED): audio feeds the REWARD, not the v1 policy observation.** This also removes the
sim blocker — audio cannot be physically simulated, but if it is not in the motor-decision loop,
sim-to-real does not depend on it.

---

## 1. One encoder, three heads (do not build three audio stacks)

A single shared audio encoder — the **frozen AST encoder, 8 s window / 1 s hop** (same as the Q
pipeline) — produces one embedding stream. Everything hangs off it:

```
stethoscope ─> AST encode (8s win, 1s hop, frozen) ─> audio embedding ─┬─> Head-Q      : P(usable)  (gate / annotation / eval)
                                                                        ├─> Head-Reward : segment reward (§3–4)
                                                                        └─> (optional) Head-Policy : acoustic-servo feature (v2 only)
```

One embedding → consistent signals, no recomputation, no drift between Q and reward.

---

## 2. Policy staging — v1 "listens via reward only", v2 adds an audio residual

- **v1 policy (LOCKED first target):** `(vision, proprioception, contact_force) → action`. **Does not
  hear audio.** Rationale: (a) sim-to-real tractable; (b) in BC the human operator's audio-driven
  corrections are recorded as *motion*, so the policy learns them from the vision/force context
  without hearing, as long as that context is predictive; (c) coarse placement (reach the anatomical
  point with good contact) is mostly vision + contact anyway.
- **v2 audio-conditioned residual (later, real-data only):** a small residual policy that, once in
  contact, makes fine moves to maximize Q — the doctor's "listen and slide". This **isolates the
  only un-simulatable dependence** into a small module; the coarse policy still gets the sim + vision
  benefit. Only build it once real data shows coarse placement needs acoustic fine-tuning.

---

## 3. Reward — segment-level anchor + dense shaping; filtered-BC is the first use

Reward's time unit matches the task: **a segment = 2–3 respiratory cycles (~10–15 s)**, cycle-aligned
(boundaries from the Hall respiratory oscillation, see realtime_multimodal_pipeline §6–7). NOT per-frame.

- **Segment anchor reward:** `R_seg = gate(force in-band & stable) · Q(audio over the segment)`.
  This is realtime_multimodal_pipeline §4's `P_usable = gate · Q_audio`. It is **clinically anchored**
  — we optimize "capture a usable lung sound", not a proxy that can be gamed.
- **Dense shaping (optional, for RL):** streaming Q (~1 Hz sliding window) + force-in-band + low
  heart-contamination as dense proxies; the segment anchor is the terminal reward.
- **First use = reward-weighted / filtered BC (LOCKED first landing):** weight or filter demo
  *segments* by Q → imitate only high-quality auscultation. No RL needed; it directly makes the
  collected data trainable. Online RL on a real patient (each rollout needs a real recording; sparse;
  safety) is out of near-term scope; the path is offline / reward-weighted BC, later offline RL.

---

## 4. Reward model — Q-as-reward → learned, preference-based, audio-anchored

- **Start:** no separate model. `R = gate · Q` directly (interpretable, hard to hack).
- **Upgrade to a learned reward model `R_φ`:** input = audio embedding (+ optional force/position),
  trained on **doctor labels** (the 68-clip consensus set now; in-domain labels from real collection
  later). **Preference-based (LOCKED):** doctors *rank pairs* (Bradley–Terry) rather than absolute
  usable/unusable — because absolute labels are noisy (κ0.5) but relative preferences are more
  consistent. This changes the clinician annotation tool to pairwise comparison.
- **Anti reward-hacking guardrails:** (a) keep the reward **audio-anchored** (it truly measures the
  recording — hard to fool); (b) periodic doctor re-validation; (c) reward-model **ensemble /
  uncertainty**, conservative on OOD; (d) use it in filtered-BC / offline RL, not unconstrained
  online RL. Multimodal (force/position) terms are **shaping/early-prediction only** — never let the
  policy bypass audio by optimizing a vision-force proxy.

---

## 5. Data design — store *ingredients*, never freeze the reward

The reward model *will* improve, so at collection time we do **not** bake the reward into the
dataset. We store enough to **recompute reward offline over all past data**. Schema lives in
[`../ausc/audio/reward_schema.py`](../ausc/audio/reward_schema.py); the session container
(realtime_multimodal_pipeline §8) gains a **segment layer**:

```
session_YYYYMMDD_HHMMSS/
  ...                       # audio.wav, force.parquet, robot.parquet, rgb.mp4, depth/  (raw, kept)
  segments.jsonl            # one Segment per 2-3 cycles: boundaries, audio pointer, contact,
                            #   reward INGREDIENTS (q_usable@collect, q_stream, heart_contam, snr),
                            #   doctor_label?, DERIVED reward/weight (versioned, recomputable)
  preferences.jsonl         # pairwise doctor rankings over segment_ids (reward-model training)
  lerobot/                  # LeRobot frames; each frame carries segment_index only
```

Principles:
- **Raw audio + force + vision are retained** so any future reward model can relabel the whole
  history — no re-collection.
- **Reward & weight are DERIVED fields**, tagged with `reward_model_version`. Recomputing reward
  rewrites `segments.jsonl` only; the LeRobot frames (which carry just `segment_index`) stay stable.
- Filtered-BC joins frame → segment via `segment_index`, reads the segment's `weight`.

---

## 6. Sim validation of the reward loop (validate the pipes, not the acoustics)

Injected audio in sim does not respond to the sim arm's position, so sim cannot validate the true
reward↔motion causality. It *can* validate the **reward plumbing**: define a synthetic reward that
**does** depend on sim state (force in-band + EE distance to a "good acoustic spot" landmark) and
**degrade the injected audio consistently** with it (bad contact → add noise to the injected clip).
Then the full `reward → filtered-BC / offline-RL` machinery runs end-to-end against a known-correct
reward; the real audio Q model swaps in at deployment.

---

## 7. Phased plan

1. Fix the encoder contract: AST 8 s/1 s-hop embedding as the shared base for Q + reward (mostly wiring).
2. **Filtered-BC first:** segment layer + per-segment Q weight in the session→LeRobot converter.
3. Lock the data schema (segment boundaries / Q / doctor label / dense quality track / raw modalities)
   **before** large-scale collection — avoids rework. (This doc + `reward_schema.py`.)
4. Q-as-reward → offline weighted BC; then swap Q for the learned preference reward model (68-set start).
5. Validate the reward loop in sim with a synthetic state-dependent reward (merge with sim Phase 1/2).
6. v2 audio-residual fine-tuning policy — only once real data shows coarse placement needs it.

---

## 8. Locked decisions & open items

**Locked (2026-07-07):**
- Audio → reward, **not** the v1 policy observation.
- v1 policy = vision + proprio + force → action; v2 = small audio residual, later, real-data only.
- One shared frozen AST encoder → Q / reward / (optional) policy heads.
- Reward = segment-level `gate · Q`; **filtered/weighted BC is the first use**.
- Reward model: Q-as-reward first → **preference-based** learned model, **audio-anchored**.
- Data stores **ingredients + raw**, reward/weight are derived & versioned; recompute, don't re-collect.

**Open (decide before the relevant phase):**
- Action space: joint-position targets (ALOHA/ACT, matches leader–follower) vs EE-pose deltas — leaning joint targets.
- Episode success definition for offline-RL (good contact + high Q sustained over 2–3 cycles).
- Exact filtered-BC rule: hard threshold on Q vs soft reward-weighting (start soft-weight, keep both).
