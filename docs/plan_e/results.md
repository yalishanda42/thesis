# Plan E — Interpretability & Error Analysis

Does the model learn *musical* structure, or just fit numbers? And where does it fail?
We analyze the recommended **categorical (32-bin) transformer** on the in-distribution
E-GMD **test** split (1.56 M notes), using both the **point** readout (argmax-style
conditional mean) and one **sampled** draw per note. All logic lives in the tested
`ml/src/drum_dynamics/research/analysis.py`; figures are produced by `scripts/analyze_model.py`.

## I1 — What the embeddings encode

**Voice embedding (8-d) recovers instrument families.** Nearest neighbors in the learned
space are musically coherent, especially the hi-hat articulations:

- `closed-hh-edge` ↔ `open-hh` (cos 0.68), `pedal-hh` → `closed-hh-edge` (0.56),
  `closed-hh` → `pedal-hh` (0.48) — the whole **hi-hat family clusters**.
- `aux-cymbal` ↔ `aux-perc` (0.63); `ride-bell` → `ride` (0.33).

Colored by mean velocity, the 2D projection separates the *loud* voices
(`snare-accent` 112.6, cymbals ~90–96) from the *soft* ones (`snare` 52, `side-stick` 49,
`kick` 46). The model has learned a **dynamic identity per instrument**.

**Genre embedding (16-d) is weaker and encodes dynamics, not a taxonomy.** Nearest
neighbors are only moderately similar (cos 0.2–0.5) but several are musically sensible —
`country`↔`soul` (0.49), `blues`↔`funk` (0.36), `pop`↔`dance` (0.41). It is optimized for
velocity prediction, so genres with *similar dynamic behavior* end up near each other
rather than forming a clean genre map. Honest read: real but modest structure.

Figures: `fig_voice_embedding.png`, `fig_genre_embedding.png`.

## I2 — Learned drumming phenomena

**Metrical hierarchy — reproduced almost exactly.** On-beat notes (on the quarter-note
pulse) are far louder than off-beat notes, and the model tracks it precisely:

| | true | pred |
|---|---:|---:|
| on-beat | 75.9 | 75.0 |
| off-beat | 58.4 | 57.4 |

Notably, the emphasis is **pulse vs subdivision, not beat position**: mean velocity is
essentially *flat across beats 1–4* (64–66) in this dataset. So E-GMD grooves show **no
textbook backbeat** (2 & 4 louder) or downbeat accent when measured by velocity — a
useful negative result. (`fig_metrical.png`)

**Accents are voice-encoded.** The loud backbeat "accent" is not a velocity spike on the
snare; it is a **distinct articulation voice** — `snare-accent` (mean 112.6) vs plain
`snare` (52.0). The model learns this cleanly (per-voice ranges in `fig_voice_range.png`),
which is why accents come out right.

**Ghost notes — and why sampling matters.** Plain snare is strongly **bimodal**: 50 % of
hits fall below velocity 40 (soft ghost notes), the rest are normal/loud. The **point**
readout collapses this — only **27 %** of its snare predictions land below 40 — while
**sampling restores the ghost population to 41 %**, close to the true 50 %. This is a
concrete, voice-level demonstration of the Plan C thesis: a point estimate erases ghost
notes; sampling brings them back. (`fig_ghost_notes.png`)

## E1 — Where the error is

**By voice** (`fig_residual_by_voice.png`). Error tracks how *variable* a voice is, not how
often it occurs:

| easiest (MAE) | hardest (MAE) |
|---|---|
| snare-accent 11.1, kick 15.1, open-hh 18.8 | ride-bell 28.4, side-stick 26.4, aux-perc 23.8, closed-hh 22.2 |

Consistent-velocity voices (accent always loud, kick steady) are easy; expressive
timekeeping voices (ride-bell, side-stick, hats) are hard. Two **systematic per-voice
biases** stand out as concrete fix targets: **closed-hh is predicted too soft (bias
−13.8)** and **side-stick too loud (+14.5)**.

**By metrical position — flat.** MAE is identical on- vs off-beat (19.13 vs 19.19) and
varies little by subdivision (17.3–20.4). The model is **uniformly competent in time**;
its skill varies by *instrument*, not by *where in the bar* a note falls.
(`fig_residual_by_metrical.png`)

**Regression to the mean — the core limitation, quantified.** Binning by true velocity,
the point readout's bias is monotone and large at the extremes:

| true velocity | 0–11 | 21–32 | 53–64 | 85–95 | 116–127 |
|---|---:|---:|---:|---:|---:|
| point bias | **+32.6** | +17.7 | +0.7 | −13.7 | **−27.8** |

The point estimator pulls soft notes up and loud notes down toward the conditional mean
(crossing zero near velocity ~55). This *is* why point predictions flatten dynamics
(std_ratio < 1) and why Plan C samples instead. (`fig_dynamic_regression.png`)

## Findings

1. **The model learned real drumming, not just numbers**: it reproduces the metrical
   pulse hierarchy, per-voice dynamic identities (incl. accents via articulation), and —
   when sampled — ghost-note structure.
2. **Point estimation demonstrably regresses to the mean** (soft +33, loud −28 at the
   extremes); **sampling restores the tails** (ghosts 27 %→41 %). This is the strongest
   in-model evidence for the sampling design.
3. **Error is instrument-driven, not time-driven**: hardest on intrinsically variable
   voices (ride-bell, side-stick, hats), flat across metrical position.
4. **Per-voice biases are largely a data artifact**: closed-hh (too soft) and side-stick
   (too loud) have large systematic biases — but these trace to E-GMD's multi-kit rendering,
   which remaps pads to different voices per kit and scrambles the `voice` label. See
   `docs/methodology/kit-remapping-artifact.md`. The fix is at the data level (single-kit
   rebuild), not per-voice recalibration.
5. **Embeddings are interpretable**: voice space recovers instrument families (esp.
   hi-hats); genre space captures dynamic-behavior similarity more than a genre taxonomy.

## Reproduce

```bash
.venv/bin/python ml/scripts/analyze_model.py            # uses cached preds if present
.venv/bin/python ml/scripts/analyze_model.py --refresh  # recompute predictions
```
