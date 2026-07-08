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

✅ **Reward model + shared encoder done (2026-07-08):** [`../ausc/audio/q_head.py`](../ausc/audio/q_head.py)
now exposes the shared AST encoder (`make_embed_fn`, the design's one-encoder) feeding both Head-Q and
Head-Reward. [`../ausc/audio/reward_model.py`](../ausc/audio/reward_model.py) is the Bradley-Terry
`R_φ(embedding)→scalar` (`P(a≻b)=σ(R_a−R_b)`, BCE loss). Validated **on the H200** with synthetic
preferences (prefer the cleaner clip): trained on some base clips, it ranks **unseen** clips by
quality — spearman(score, quality) **0.976, pairwise acc 100%**. Ready to swap synthetic prefs for
real doctor pairwise rankings (`Preference` rows in reward_schema.py).

✅ **Pairwise annotation tool + preference training done (2026-07-08):**
[`../scripts/audio/preference_annotator.py`](../scripts/audio/preference_annotator.py) — a stdlib
browser tool: a doctor hears two clips and picks the more usable one; choices append `Preference` rows
to `preferences.jsonl`. **Active-learning pair selection** (`--smart`): score every clip with the Q
head and prefer SIMILAR-Q pairs (the fine calls the model is unsure about) + a few wide-gap anchors —
so scarce doctor time is spent informatively. **Data-agnostic** (open data / our recordings now, robot
flight-data segments later): the audio reward is decoupled from flight data — quality is a property of
the clip, not who held the stethoscope — so this trains the reward model *without* waiting on teleop.
`reward_model.train_from_preferences(prefs, clip_dir)` closes the loop; validated end-to-end (server
records rows; model trained from them ranks clips by quality, spearman 0.86).

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

✅ **Phase-0 procedural dry-run done (2026-07-07):** [`../scripts/gen_synthetic_session.py`](../scripts/gen_synthetic_session.py)
writes a record_session-format session with the three missing pieces filled in — 7-DOF OpenArm-left
`robot.csv` (leader targets = action, follower = obs), a scripted contact-force profile (in-band 4 N
during the hold, breathing-modulated), and **state-dependent audio** (a real lung clip degraded
outside contact). Through `session_to_lerobot.py` it yields contact-aware segments (gate=True /
reward>0 only in the good-contact window, gate=False gaps around it) and a `frames.parquet` with
7-DOF state+action + per-frame `reward.weight`. [`../scripts/train_bc_smoke.py`](../scripts/train_bc_smoke.py)
then trains a reward-weighted BC MLP on it — loss converges (~330×), confirming the data loads,
weights apply, and the loop trains. This validates pipeline + structure + multimodality + trainability
end-to-end with **no hardware**; MuJoCo physics (real contact/kinematics) is the Phase-1 upgrade.

✅ **Phase-1 MuJoCo physics done (2026-07-07):** [`../mechanical_sensor/sim/openarm_mjcf.py`](../mechanical_sensor/sim/openarm_mjcf.py)
builds a **self-contained MJCF of the OpenArm left arm from the URDF's concrete kinematics + inertials**
(no external meshes — they aren't in-repo; primitives suffice for contact physics), plus a soft
breathing 'back' surface. [`../scripts/gen_mujoco_session.py`](../scripts/gen_mujoco_session.py)
IK-drives the stethoscope tip into the surface under a **force-targeting loop** (press until ~4 N,
the design setpoint) and records a session with the **contact force EMERGENT from the MuJoCo solver**
(not scripted) + physically-achieved follower joints + a real offscreen-rendered camera. Runs in a
native **arm64 MuJoCo venv** (`~/envs/mjx`; MuJoCo refuses x86_64/Rosetta Python — the anaconda base
is x86_64, so the sim uses brew python3.11).

Quality eval (converted through the same pipeline): emergent contact **3.77 N, 100% in-band [3-5] N**
during the hold, peak 5.0 N (< 8 N limit); **follower≠leader mean|q−a| = 0.027 rad (≈7× Phase-0's
0.004)** — contact resistance genuinely pushes back on the arm; gate=True on 752/1000 frames with
reward>0 concentrated in contact; reward-weighted BC converges (117×). The force controller (slow,
by design) lets breathing show as force ripple that the contact-aware segmenter tiles.

✅ **Vision sim-to-real done (2026-07-08):** the back surface is now an **anatomical curved ellipsoid**
(≈torso; force-targeting adapts — hold 3.9 N / 100% in-band, peak 5.99 N < 8 N limit), and
`gen_mujoco_session.py --seed` applies **per-session domain randomization** (light position+color,
skin tone, arm color, background, camera pose+fovy) — physics untouched. Renders vary across seeds
(cross-seed per-pixel std 9.3/255) while contact stays identical, so a vision encoder trained across
seeds is robust to the sim→real gap. Curved+DR sessions still convert (real-Q contact reward 1.0) and
train. `--no-randomize` disables it. The gap to a real camera/clothed back remains large — this is the
sim-side robustness lever, not a replacement for real vision data.

---

## 7. Phased plan

1. Fix the encoder contract: AST 8 s/1 s-hop embedding as the shared base for Q + reward (mostly wiring).
   ✅ **Q head done (2026-07-07):** [`../ausc/audio/q_head.py`](../ausc/audio/q_head.py) — AST + the
   SPRSound GBM → `P(usable) = 1 − P(poor)`, lazy-loaded; `session_to_lerobot.py --real-q` uses it as
   `q_fn`. Validated on the MuJoCo session: the good-contact clean-audio segment scores **Q 0.98 →
   reward 0.98** (the cheap proxy underrated it at 0.15); noisy no-contact gaps score low. This is the
   real learned reward, replacing the placeholder proxy.
2. **Filtered-BC first:** segment layer + per-segment Q weight in the session→LeRobot converter.
   ✅ implemented: [`../scripts/session_to_lerobot.py`](../scripts/session_to_lerobot.py) — Hall
   respiratory-cycle splitting → `segments.jsonl` (reward_schema) + `frames.parquet` with per-frame
   `segment_index` + `reward.weight`. Validated on a real session; it also **reports** the still-missing
   `observation.state`/`action` in `info.json` (teleop unwired) and uses a **proxy Q** until the real
   AST Q head is plugged via `q_fn` on the 5090.
3. Lock the data schema (segment boundaries / Q / doctor label / dense quality track / raw modalities)
   **before** large-scale collection — avoids rework. (This doc + `reward_schema.py`.)
4. Q-as-reward → offline weighted BC; then swap Q for the learned preference reward model (68-set start).
5. Validate the reward loop in sim with a synthetic state-dependent reward (merge with sim Phase 1/2).
6. v2 audio-residual fine-tuning policy — only once real data shows coarse placement needs it.

---

## 7b. Reproduce (the sim data pipeline)

Two envs (MuJoCo needs native arm64 — anaconda base is x86_64/Rosetta): **anaconda base**
(pandas/torch/AST, runs conversion+training) and **`~/envs/mjx`** (brew py3.11 arm64, runs MuJoCo).

```bash
# Phase-0 procedural session (pure numpy) → LeRobot → reward-weighted BC
python scripts/gen_synthetic_session.py --out data/sessions/session_sim0001
python scripts/session_to_lerobot.py data/sessions/session_sim0001 --out data/lerobot/sim0001 [--real-q]
python scripts/train_bc_smoke.py data/lerobot/sim0001

# Phase-1 MuJoCo physics session (emergent contact) — generation runs in the arm64 venv
~/envs/mjx/bin/python mechanical_sensor/sim/openarm_mjcf.py            # (re)generate the MJCF
~/envs/mjx/bin/python scripts/gen_mujoco_session.py --out data/sessions/session_mj0001
python scripts/session_to_lerobot.py data/sessions/session_mj0001 --out data/lerobot/mj0001 --real-q

# End-to-end acceptance test (locks the Phase-0/1 invariants; exit 0/1)
python scripts/test_il_pipeline.py

# RL-readiness: mixed-outcome episodes (good contact vs under-press) -> offline-RL value/Q fit
for i in 0 1 2; do python scripts/gen_mujoco_session.py --seed $i --f-target 4.0 --out data/sessions/g$i; done  # good
for i in 3 4 5; do python scripts/gen_mujoco_session.py --seed $i --f-target 1.5 --out data/sessions/b$i; done  # under-press
# convert each --real-q, then:  python scripts/rl_readiness_check.py --dir <dir of lerobot_*>
```

**✅ RL-readiness verified (2026-07-08):** 6 mixed-outcome MuJoCo episodes (3 good-contact return ~386,
3 under-press return 0) → 3600 `(s,a,r,s',done)` transitions. State fuses the modalities: **7 arm
joints + magnetic contact force + gate + audio-Q** (10-D; video is the `rgb.mp4` sidecar a vision
encoder adds). `scripts/rl_readiness_check.py`: **V(s) regresses the return R²=0.97 and separates
good/bad (V 75 vs 9.5); Q(s,a) R²=0.95** — so offline RL (IQL/CQL/AWAC/TD3+BC) can consume the format.
(Bootstrapped TD oscillates — the deadly triad — which is exactly why conservative offline RL exists;
an algorithm choice, not a data issue.) All 4 modalities present + timestamp-aligned per episode.

Pieces: `gen_synthetic_session.py` / `gen_mujoco_session.py` (sessions) · `openarm_mjcf.py` (URDF→MJCF)
· `session_to_lerobot.py` (segment/reward conversion) · `ausc/audio/reward_schema.py` (schema) ·
`ausc/audio/q_head.py` (real Q) · `train_bc_smoke.py` (trainability) · `test_il_pipeline.py` (CI gate).

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
