# Design: Humanizing Drum Dynamics — Velocity Prediction

**Date:** 2026-08-11
**Status:** Approved design; ready for implementation planning
**Dataset:** [Expanded Groove MIDI Dataset (E-GMD)](https://magenta.tensorflow.org/datasets/e-gmd)

---

## 1. Problem framing

**Task.** Given a MIDI drum track where *timing* and *which drum is hit* are known, predict each
note's velocity — i.e. restore expressive dynamics to a flat/de-humanized performance. This is the
dynamics analogue of tempo/timing humanization.

**Deterministic first, generative later.** Phase 1 targets a *deterministic* mapping
(note-context → one velocity). Humanization is genuinely one-to-many — the same groove played twice
has different velocities, and that variance *is* the humanity — so a plain MSE regressor will tend to
regress toward the mean and flatten dynamics. We accept that as a known limitation of the baseline and
plan a probabilistic extension (Section 6) that keeps the same backbone and swaps only the output head.

### 1.1 Hard constraint — no velocity features (no leakage)

At inference the input is a **flat MIDI**: velocities are exactly what we are generating. Therefore
**no note's velocity may be used as a feature for any other note.** All context must be built from
*structure and timing only* (which drums, when). This rule is absolute and rules out an entire tempting
feature category (neighbouring velocities, running velocity averages, etc.).

A direct consequence: the transformer is **non-autoregressive** — it sees the full structural sequence
(all onsets + drum parts) and emits *all* velocities in a single parallel pass. No error accumulation,
no ordering hacks for generation.

---

## 2. Representation — why no grid

The canonical E-GMD/GrooVAE representation is a fixed 16th-note **grid** (timesteps × instruments) with
a micro-timing offset channel. We deliberately **do not use it**, for two reasons:

1. **Triplets / irregular feels.** A 16th grid is a *duple* subdivision. E-GMD is full of triple/irregular
   feels (`jazz/swing`, `funk/purdieshuffle`, `afrocuban/*`, `latin/*`, `blues/shuffle`). Triplet onsets
   sit ~1/12-beat off the nearest gridline, forcing large offset residuals and, worse, **collisions**
   where two triplet notes snap to the same cell (Magenta drops/merges these) — corrupting exactly the
   genres where dynamics are most expressive.
2. **The grid exists for *generation* of timing; we don't generate timing.** Our timing is *conditioning
   input*. So we never quantize. Instead we feed each note's **continuous metrical phase** as a feature,
   which represents any subdivision (triplet at phase 0.333, quintuplet at 0.2/0.4/…, swing wherever it
   actually is) exactly.

This is a defensible thesis point: *because timing is conditioning input rather than a generation target,
we avoid the quantization artifacts of grid-based groove models.*

### 2.1 Metrical phase and its encoding

For each note, using `bpm` and time signature (both in metadata):

```
beat_dur   = 60 / bpm
phase_beat = (onset_secs mod beat_dur) / beat_dur      # in [0, 1)
phase_bar  = (onset_secs mod bar_dur)  / bar_dur       # in [0, 1)
```

**Circular encoding for phase.** Phase is *circular*: 0.99 and 0.01 are musically adjacent but numerically
far apart, and a raw scalar has a hard discontinuity at the beat boundary. We therefore encode phase on the
unit circle: `[sin(2π·phase), cos(2π·phase)]`. Optional higher harmonics `sin/cos(2π·k·phase)`, k = 1,2,…
sharpen within-beat resolution; start at k = 1.

**Rule of thumb used throughout:** *phase (circular) → sin/cos; duration/delta (magnitude) → log-scaled
scalar in beats.* A time-delta is a distance, not a position on a circle, so it must **not** be sin/cos
encoded.

---

## 3. Phase 0 — exploratory validation (train split only)

Feature definitions that would otherwise be guessed are made **data-driven**. Phase 0 runs on the
**train split only** and its outputs feed the final feature spec. E-GMD is recorded on a Roland electronic
kit, so most GM percussion pitches never appear and some toms/cymbals are sparse — hand-defining voice
groups first would be guessing.

**Deliverables:**

1. **Pitch usage & velocity distributions.** Per GM pitch: count and velocity histogram. Decides the final
   **voice grouping** — merge voices that are rare or have indistinguishable velocity distributions; split
   ones that are common and distinct.
2. **Simultaneity tolerance.** Histogram of very-small inter-onset gaps to find the natural valley
   separating "meant to be simultaneous" from "genuinely sequential." That valley sets the `SIMULTANEITY_TOL`
   (a musical fraction of a beat), recorded as a documented hyperparameter.
3. **Categorical cardinality / skew.** Distributions of `style`, `time_signature`, `bpm`, `beat_type` —
   how many rare classes, how skewed.
4. **Time-signature audit** — see Section 5 (gates a transformer feature decision).

**Candidate voice groups (liberal; Phase 0 prunes/merges):** kick, snare, side-stick, closed-HH, open-HH,
pedal-HH, ride, ride-bell, crash-1, crash-2, china, splash, high-tom, hi-mid-tom, low-mid-tom, low-tom,
high-floor-tom, low-floor-tom, cowbell, tambourine.

---

## 4. Section A — tabular baseline features (LightGBM/XGBoost)

Per-note rows; **all structural** (no velocity features). Trees do implicit feature selection, so we are
liberal here — redundant structural features are low-risk. The real dangers are leakage (forbidden by §1.1)
and collinearity muddying feature-importance *interpretation* (we read importances with that caveat).

- **Drum part** — canonical voice from Phase 0 grouping. One-hot (or LightGBM native categorical).
- **Metrical position**
  - `sin/cos(2π·phase_beat)`, `sin/cos(2π·phase_bar)`.
  - Raw scalars `phase_beat`, `phase_bar` (trees handle raw phase fine; this is the "position within the
    bar" number). *Note: this is position, not swing.*
  - **Swing ratio** — a *separate* derived feature: how far an offbeat is pushed toward the triplet
    position (0 = straight, ~1 = hard shuffle).
  - **Nearest-subdivision hint** — category over {8th, 16th, 32nd, 8th-triplet, quintuplet, triple-meter
    variant}: the candidate grid whose nearest gridline the onset is closest to.
- **Conditioning** — `style` (top-level genre *and* full style string), `bpm` (numeric), `time_signature`,
  `beat_type`. Native categorical features (not embeddings). No feature scaling needed for trees.
- **Structural context (hand-engineered window)**
  - `time_to_prev` / `time_to_next` onset (any voice), in **beats**, `log1p`-scaled and clipped.
  - Same-voice inter-onset intervals (prev / next), same units.
  - **Simultaneity** — multi-hot vector over canonical voices for everything firing at this instant (the
    full "chord", self included), plus a scalar count. "This instant" = within `SIMULTANEITY_TOL` (Phase 0).
  - **Local density** — onset count within ±1 beat.

---

## 5. Section B — transformer features (event-sequence, non-autoregressive)

Same *base* per-note vector, but **drop the hand-engineered window features** — self-attention learns
context. Sequence = one track, chunked into fixed windows (e.g. a few bars) if length is an issue.
Simultaneous hits ordered by pitch, with a `same_onset_as_previous` flag.

Per-token features:

- **Drum-part embedding** — `nn.Embedding` table (~8-dim), learned jointly by backprop.
- **Genre embedding** — `nn.Embedding` (~8–16-dim), learned jointly. (Visualizing the learned genre
  geometry — e.g. funk near soul — is a nice thesis artifact.)
- **Metrical phase** — `sin/cos` of `phase_beat` and `phase_bar` (harmonics optional).
- **BPM** and any other numeric features — **standardized (z-score) using train-split statistics only**;
  scaler fit on train, applied unchanged to val/test, persisted.
- **Delta-time to previous event** — in beats, `log1p`-scaled scalar (not sin/cos).

**Time-signature — Phase 0-conditional (OPEN):** if Phase 0 shows E-GMD is effectively all-4/4 (with at
most trivial 3/4), **drop time-signature from the transformer feature set** (it is low-variance and
partly redundant with meter-relative phase). If odd meters (5/4, 7/8, 7/4, …) appear in meaningful
numbers, **keep it** and revisit its encoding (candidate: small time-sig embedding). Tabular keeps it
regardless (cheap for trees). *This decision is deferred to Phase 0 results.*

---

## 6. Model progression

One data pipeline, three modelling stages:

1. **Tabular baseline** — LightGBM/XGBoost on Section A features. Fast, strong, interpretable; the honest
   "does the transformer earn its complexity?" reference.
2. **Event-sequence Transformer** — Section B features, non-autoregressive, deterministic scalar head
   (`Linear(d, 1)`), MSE/MAE loss.
3. **Probabilistic heads (future)** — *same backbone*, swap only head + loss:

| Head | Outputs | Loss | Velocity readout |
|------|---------|------|------------------|
| Deterministic (Phase 1) | 1 scalar μ | MSE/MAE | μ |
| Gaussian | (μ, σ) | Gaussian NLL | sample N(μ,σ), or μ |
| Mixture density (MDN) | K weights + K·(μ,σ) | mixture NLL | sample mixture (captures ghost-vs-accent multimodality) |
| Categorical | 128 (or coarser) bin logits | cross-entropy | sample softmax; argmax = deterministic |

The categorical head is **independent of the input representation** (it is a statement about the *target*,
0–127 velocity as classes) — it works for the event-sequence model, no grid required. It makes zero
assumptions about the distribution shape, so it can represent arbitrary multimodal/skewed velocity
distributions.

---

## 7. Evaluation

- **Splits.** Primary: E-GMD's provided `train/validation/test` column. Secondary robustness check:
  held-out drummer.
- **Baselines to beat.** Global-mean velocity; and a **lookup table** of mean velocity per
  (drum-part × genre × metrical-position bin) — the "dumb humanizer." Beating this is the real bar.
- **Metrics (MSE alone is insufficient):**
  - MAE / RMSE — absolute accuracy.
  - **Per-track Pearson / Spearman** of predicted vs. true velocity curve — is the *dynamic shape* right?
  - **Std / histogram match** — does the model reproduce the natural spread or flatten it? (Directly
    measures the mean-regression failure.)
  - **Accent / ghost-note placement** — within-bar ranking correctness.
  - **Per-genre breakdown** — does funk get more dynamic range than pop, confirming the hypothesis?
  - Optional qualitative A/B listening test.

---

## 8. Implementation staging (separate plans / sessions)

The spec is the shared source of truth; implementation is split so each session stays focused:

- **Plan A:** data pipeline + Phase 0 exploratory validation + tabular baseline.
- **Plan B:** event-sequence transformer + deterministic head + evaluation harness.
- **Plan C (future):** probabilistic heads (Gaussian / MDN / categorical).

---

## 9. Open / deferred decisions

- **Time-signature in transformer** — gated on Phase 0 meter audit (§5).
- **Final voice grouping** — output of Phase 0 (§3).
- **`SIMULTANEITY_TOL`** — output of Phase 0 (§3).
- **Probabilistic heads** — full design deferred to Plan C (§6).
- **Odd-meter handling** — revisited only if Phase 0 surfaces 5/4, 7/8, 7/4, etc.
