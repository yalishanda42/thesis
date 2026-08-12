# Phase 0 — Exploratory Validation Results

**Date:** 2026-08-11
**Scope:** E-GMD **train split only** (35,217 files, 11,070,345 notes, 0 parse failures).
**Design ref:** [`../superpowers/specs/2026-08-11-drum-velocity-humanization-design.md`](../superpowers/specs/2026-08-11-drum-velocity-humanization-design.md) §3.
**Status:** Complete. Feeds the final feature spec; a few grouping/threshold choices are
recommendations flagged **[decision]** for sign-off.

## How to reproduce

```bash
.venv/bin/python ml/scripts/phase0_analysis.py --workers 9      # ~40 s on 9 workers
.venv/bin/python -m pytest ml/tests/test_phase0.py -q           # pure-function unit tests
```

Logic lives in `ml/src/drum_dynamics/research/phase0.py` (pure, unit-tested aggregation) driven by
`ml/scripts/phase0_analysis.py` (parallel scan → artifacts). All artifacts are in this folder:
`aggregates.npz` (raw mergeable counts), `pitch_velocity.csv`, `summary.json`, and four figures.

---

## Deliverable 1 — Pitch usage & velocity distributions → voice grouping

25 GM pitches appear. Two are **outside General MIDI** — the Roland kit's hi-hat *edge*
articulations (pitch 22, 26), now named via `drum_dynamics.core.midi.drum_name`. Full table:
`pitch_velocity.csv`; see `fig_pitch_counts.png` and `fig_velocity_dists.png`.

| pitch | voice | count | share % | vel μ | vel σ | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 38 | Acoustic Snare | 2,603,554 | 23.52 | 54.9 | **36.3** | 43 | 14 | 125 |
| 36 | Bass Drum 1 | 2,180,435 | 19.70 | 54.2 | 32.2 | 48 | 13 | 108 |
| 44 | Pedal Hi-Hat | 1,227,540 | 11.09 | 64.0 | 22.5 | 65 | 30 | 92 |
| 51 | Ride Cymbal 1 | 1,200,689 | 10.85 | 66.3 | 28.4 | 60 | 35 | 113 |
| 42 | Closed Hi-Hat (bow) | 709,957 | 6.41 | 48.6 | 27.1 | 43 | 20 | 87 |
| 22 | Hi-Hat Closed (**edge**) | 691,178 | 6.24 | **83.6** | 31.1 | 82 | 43 | 127 |
| 40 | Electric Snare | 600,033 | 5.42 | **112.6** | 20.3 | 122 | 92 | 127 |
| 48 | Hi-Mid Tom | 303,451 | 2.74 | 88.6 | 28.5 | 88 | 51 | 127 |
| 43 | High Floor Tom | 293,303 | 2.65 | 88.8 | 29.5 | 90 | 48 | 127 |
| 37 | Side Stick | 264,018 | 2.38 | 61.2 | 20.9 | 64 | 33 | 86 |
| 26 | Hi-Hat Open (**edge**) | 214,760 | 1.94 | **99.0** | 29.2 | 106 | 56 | 127 |
| 53 | Ride Bell | 186,964 | 1.69 | 90.0 | 32.1 | 90 | 46 | 127 |
| 54 | Tambourine | 141,473 | 1.28 | 69.5 | 31.4 | 67 | 29 | 124 |
| 45 | Low Tom | 105,780 | 0.96 | 89.7 | 27.5 | 89 | 54 | 127 |
| 55 | Splash Cymbal | 70,090 | 0.63 | 94.3 | 23.4 | 94 | 63 | 127 |
| 59 | Ride Cymbal 2 | 58,953 | 0.53 | 83.1 | 29.1 | 82 | 43 | 127 |
| 46 | Open Hi-Hat (bow) | 52,640 | 0.48 | 57.7 | 33.3 | 51 | 17 | 116 |
| 50 | High Tom | 45,960 | 0.42 | 98.8 | 27.1 | 105 | 56 | 127 |
| 52 | Chinese Cymbal | 30,272 | 0.27 | 97.2 | 27.8 | 102 | 55 | 127 |
| 47 | Low-Mid Tom | 29,946 | 0.27 | 96.6 | 26.3 | 100 | 58 | 127 |
| 58 | Vibra Slap | 26,754 | 0.24 | 100.5 | 32.2 | 112 | 45 | 127 |
| 49 | Crash Cymbal 1 | 14,706 | 0.13 | 36.0 | 34.6 | 16 | 6 | 92 |
| 57 | Crash Cymbal 2 | 6,966 | 0.06 | 67.1 | 31.2 | 60 | 33 | 119 |
| 39 | Hand Clap | 6,373 | 0.06 | 97.9 | 27.9 | 103 | 54 | 127 |
| 56 | Cowbell | 4,550 | 0.04 | 97.2 | 27.3 | 102 | 55 | 127 |

### Key findings

- **Snare is strongly bimodal (ghost vs accent).** Acoustic snare (38) has σ = 36 and spans
  p10 = 14 to p90 = 125 — the ghost-note/accent multimodality the design's §6 MDN/categorical
  head is meant to capture. Direct motivation for the probabilistic extension.
- **Hi-hat *edge* hits are distinct accents, not duplicates.** Closed-edge (22, μ 84) and
  open-edge (26, μ 99) sit far above their bow counterparts (42, μ 49; 46, μ 58). They are both
  common *and* distributionally distinct → **keep separate** (spec's "split common & distinct").
- **Electric snare (40)** is a near-always-loud articulation (μ 113, p10 92) — effectively the
  rimshot/accent snare; distinct from acoustic snare → keep separate.
- **Toms are mutually indistinguishable** in velocity (all μ ≈ 89–99, σ ≈ 26–29) → safe to
  **merge** (spec's "merge indistinguishable").
- **Crash 1 (49) is oddly quiet** (μ 36, median 16) — likely soft bow/choke triggers on the
  e-kit rather than full crashes; rare (0.13%).

### Recommended voice grouping **[decision]**

Applying the spec's rule (merge rare/indistinguishable; split common/distinct). ~14 voices:

| canonical voice | pitches | rationale |
|---|---|---|
| kick | 36 | only kick present |
| snare | 38 | main snare (bimodal) |
| snare-accent | 40 | distinct hot articulation |
| side-stick | 37 | distinct cross-stick |
| closed-hh | 42 | bow |
| closed-hh-edge | 22 | common & distinct (accent) |
| pedal-hh | 44 | distinct |
| open-hh | 46, 26 | merge bow+edge (46 rare) |
| ride | 51, 59 | 59 rare → into ride |
| ride-bell | 53 | distinct accent |
| tom | 43, 45, 47, 48, 50 | velocity-indistinguishable → merge |
| crash | 49, 57 | merge crashes |
| aux-cymbal | 52, 55 | china + splash (rare) |
| aux-perc | 54, 58, 39, 56 | tambourine, vibraslap, clap, cowbell |

Open sub-choices to confirm: (a) fold `closed-hh-edge` into `closed-hh`? (recommend **no** — 6.2%
and +35 velocity distinct); (b) split `tom` into high/low? (recommend **no** — indistinguishable);
(c) keep `tambourine` (1.3%) as its own voice rather than aux-perc? (defensible either way).

---

## Deliverable 2 — Simultaneity tolerance

Histogram of consecutive inter-onset gaps (any voice), in **beats** and **milliseconds**
(`fig_gap_hist.png`, `fig_gap_hist_ms.png`). The distribution has **two distinct boundaries**,
and the original design assumption (§3) conflated them:

1. **Near-zero region (< ~0.05 beat / < ~25 ms): no valley.** True "struck-together"
   simultaneity decays *monotonically* into tight ornaments (flams, drags, grace notes). There
   is **no gap-desert** separating chords from ornaments, so `SIMULTANEITY_TOL` **cannot** be
   read off a valley — it must be a documented fixed hyperparameter. (The ~1 ms comb in the ms
   plot is 480-PPQ MIDI tick quantization, not musical structure.)
2. **A clear valley at ≈ 0.149 beat (≈ 48 ms at median tempo)** separates the sub-16th ornament
   mass from the **16th-note grid** (peaks follow at 0.25 = 16th, 0.33 = 8th-triplet, 0.5 = 8th).

**`SIMULTANEITY_TOL = 0.02 beat` (fixed) [decision].** Chosen in the flat near-zero decay, well
below the smallest musical subdivision. Expressed in *beats* (not ms) for tempo-robustness:
≈ 11 ms at the median 110 bpm, and it stays below a 16th note even at the fastest tempo (a 16th
at 290 bpm is 0.25 beat ≫ 0.02). This is a **documented hyperparameter to sensitivity-test**, not
a data-derived valley. `±` a bit is unlikely to matter (the region is a smooth slope, not a knife-edge).

**Bonus finding (supports the no-grid design, §2).** ~40% of all inter-onset gaps fall *below*
the 16th-note grid — the ornament/triplet/swing mass that a 16th grid would quantize, collide, or
drop. The clean ≈ 0.149-beat valley is exactly the "same-cell" tolerance a grid representation
implicitly assumes; keeping timing continuous avoids corrupting that 40%.

---

## Deliverable 3 — Categorical cardinality / skew (train split)

- **style:** 61 full style strings across **17 top-level genres**. Long-tailed: top styles are
  `rock` (8,686), `hiphop` (3,182), `funk` (2,451), `punk` (1,935), `neworleans/funk` (1,505),
  `jazz` (1,462) … → embedding (transformer) / native categorical (trees) both fine; expect rare
  classes in the tail.
- **beat_type:** fill 21,629 / beat 13,588.
- **bpm:** 50–290, mean 110.0, std 24.3 (median 110). Standardize on train stats for the transformer.
- **time_signature:** see Deliverable 4.

---

## Deliverable 4 — Time-signature audit (gates §5)

| time sig | train files | % |
|---|---:|---:|
| 4-4 | 34,787 | 98.78 |
| 3-4 | 172 | 0.49 |
| 6-8 | 172 | 0.49 |
| 5-4 | 43 | 0.12 |
| 5-8 | 43 | 0.12 |

The train split is **98.78% 4/4**; genuinely odd meters (5/4, 5/8) are 0.24% and 3/4+6/8 are
0.98%. **[decision] Recommend: drop time-signature from the transformer feature set** (§5) — it is
near-constant and largely redundant with meter-relative phase (`phase_bar`) — while **keeping it in
the tabular model** (free for trees). Odd-meter handling (§9) is **not** worth special treatment at
this prevalence. Revisit only if a held-out-drummer or genre slice over-indexes on odd meters.

---

## Decisions summary

| item | outcome | status |
|---|---|---|
| Voice grouping | ~14 voices (table above); edge-HH & electric-snare kept separate; toms merged | **[decision]** recommend |
| `SIMULTANEITY_TOL` | **0.02 beat** fixed (no near-zero valley); documented hyperparameter | **[decision]** recommend |
| Subdivision valley | ≈ 0.149 beat / 48 ms — evidence for no-grid design | finding |
| Time-signature in transformer | **drop** (98.78% 4/4); keep in tabular | **[decision]** recommend |
| Odd-meter special handling | none needed (0.24% truly odd) | resolved |
| Snare/hi-hat multimodality | confirmed → motivates §6 probabilistic head | finding |
