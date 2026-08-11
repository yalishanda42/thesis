# Methodology note — E-GMD multi-kit rendering remaps voices (data artifact)

**TL;DR.** E-GMD renders each drum performance through **43 Roland-module kits**, and each
kit **remaps pads to different MIDI pitches** (→ different canonical *voices*) while keeping
timing and (mostly) velocity. Our `build_dataset` pooled **all 43 kit-renderings under one
`file_id`**, so every canonical voice is a 43-kit mixture of the same underlying gestures.
This injects **label noise into the `voice` input feature** and is the likely cause of the
systematic per-voice prediction biases seen in Plan E (e.g. closed-hh −13.8, side-stick
+14.5). It is a **data artifact, not a model deficiency**. Velocity — the prediction target
— is essentially kit-invariant, so the humanization objective is unaffected, and because
splits are per-performance there is **no train/test leakage**. Prior results are valid but
rest on a 43× redundant, voice-noisy dataset.

## How E-GMD is structured

`e-gmd-v1.0.0.csv`: **45,537 rows = 1,059 unique performances (`id`) × 43 kits (`kit_name`)**.
Each performance was played back through the drum module under all 43 kit presets and the
resulting MIDI re-captured; there is one `.midi` file per (performance × kit).

`scripts/build_dataset.py` iterates every CSV row (all 45,537 MIDI files) and sets
`file_id = meta["id"]` (`drumhumanizer/features.py:122`). Since `id` is the performance id —
shared by all 43 kit renderings — **each `file_id` aggregates 43 renderings of the same
groove**.

## Evidence

Verified directly from the raw MIDI (see *Reproduce* below):

1. **43× duplication.** For `drummer1/eval_session/1`, every `(onset_sec, voice)` pair recurs
   **exactly 43 times** (median = max = 43); 406 unique onsets → 17,630 notes ≈ 406 × 43.

2. **Velocity/timing preserved, pitch remapped.** Comparing two kits of the same
   performance: identical onsets, **identical velocity multiset**, but **different pitch
   multiset** — "60s Rock" emits pitch 37 (side-stick); "808 Simple" does not.

3. **Voices trade membership by kit** (`drummer1/eval_session/1`, count across 43 kits):

   | voice | present in | note count (min–max) | remaps with |
   |---|---:|---:|---|
   | pedal-hh | 41/43 | 123–123 | ↔ aux-perc (present in the other 2/43) |
   | side-stick | 36/43 | 1 | ↔ snare (snare count rises 94→100 when side-stick absent) |
   | snare-accent | 42/43 | 6–7 | — |

4. **The remapping is a kit property, consistent across drummers** (answers "do other kits do
   the same?"): side-stick is present in **36–38/43 kits for every performance tested**
   (drummers 1/3/5/8) — the *same* kits drop it, independent of who plays. pedal-hh swings
   hugely by kit (e.g. drummer5: some kits 1,132 pedal-hh notes, others 0).

5. **Velocity mostly — but not perfectly — kit-invariant.** A few "Layered" kits ("Big Room
   (Layered)", "Super Boom (Layered)", "Raw Dnb (Layered Hybrid)") **add doubled/layer
   notes**, so their note set (and velocity multiset) differs. The core timing/velocity of
   the original hits is preserved; layering adds extra notes.

## Why this explains the Plan E per-voice biases

Because the same physical hit (fixed velocity) is labeled `side-stick` in some kits and
`snare` in others (and `pedal-hh` vs `aux-perc`, etc.), each canonical voice pools
genuinely-that-voice hits with hits remapped in from other pads. The `voice` embedding
therefore conditions on a **kit-scrambled label**, and the model learns the *mixture* mean —
producing systematic per-voice offsets (Plan E: closed-hh predicted too soft, side-stick too
loud). These biases are an artifact of pooling 43 kits, so **per-voice recalibration would
treat a symptom**; the correct fix is at the data level (below).

## Implications

- **No leakage, but heavy redundancy.** All 43 kits of a performance share its split, so
  train/test integrity holds. But the dataset is ~43× larger than the number of distinct
  grooves, and test metrics are computed over 43 kit-copies of each groove.
- **Target is safe.** Velocity is kit-invariant (barring added layer notes), so the
  humanization target and its distribution are well-defined regardless of this artifact.
- **Voice feature is noisy.** The single most kit-contaminated model input is `voice`.

## Recommended remedy (deferred)

Rebuild on a **single canonical kit** — "Acoustic Kit" carries the full standard mapping
(side-stick 37, snare-accent 40, closed/edge/pedal/open hats, ride/bell). Add a
`--kit "Acoustic Kit"` filter in `build_dataset` (`df = df[df.kit_name == kit]`). Expected
effects: voice-label noise removed, dataset ~43× smaller (much faster training), and a clean
test of whether the per-voice biases **disappear**. Multi-kit rendering could alternatively be
kept deliberately as pitch-remap *augmentation*, but that is a separate design choice and
should be explicit, not accidental.

## Reproduce

```python
import pandas as pd, os, warnings; warnings.simplefilter("ignore")
from drumhumanizer.midi import load_note_array
from drumhumanizer.voicemap import voice_of
m = pd.read_csv("data/e-gmd/e-gmd-v1.0.0/e-gmd-v1.0.0.csv")
base = "data/e-gmd/e-gmd-v1.0.0"
sub = m[m["id"] == "drummer1/eval_session/1"]          # 43 kit renderings
vel = set()
for r in sub.itertuples():
    na = load_note_array(os.path.join(base, r.midi_filename))
    vel.add(tuple(sorted(na["velocity"].tolist())))    # identical across kits
    # voice counts differ across kits -> pad remapping
print("velocity identical across kits:", len(vel) == 1)
```
