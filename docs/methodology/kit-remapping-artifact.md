# Methodology note — E-GMD multi-kit rendering remaps voices (data artifact)

**TL;DR.** E-GMD renders each drum performance through **43 Roland-module kits**, and each
kit **remaps pads to different MIDI pitches** (→ different canonical *voices*) while keeping
timing and (mostly) velocity. Our `build_dataset` pooled **all 43 kit-renderings under one
`file_id`**, so every canonical voice is a 43-kit mixture of the same underlying gestures.
This injects **label noise into the `voice` input feature** and is the likely cause of the
systematic per-voice prediction biases seen in Plan E (e.g. closed-hh −13.8, side-stick
+14.5). It is a **data artifact, not a model deficiency**. Notably, this **contradicts the
E-GMD paper's** description that the 43 kits change only audio timbre — the released MIDI
files show per-kit note remapping. Velocity — the prediction target
— is essentially kit-invariant, so the humanization objective is unaffected, and because
splits are per-performance there is **no train/test leakage**. Prior results are valid but
rest on a 43× redundant, voice-noisy dataset.

## How E-GMD is structured

`e-gmd-v1.0.0.csv`: **45,537 rows = 1,059 unique performances (`id`) × 43 kits (`kit_name`)**.
Each performance was played back through the drum module under all 43 kit presets and the
resulting MIDI re-captured; there is one `.midi` file per (performance × kit).

`ml/scripts/build_dataset.py` iterates every CSV row (all 45,537 MIDI files) and sets
`file_id = meta["id"]` (`ml/src/drum_dynamics/data/features.py:122`). Since `id` is the performance id —
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

## Cross-module context: this is general e-drum behavior, not a Roland quirk

Web research into how e-drum modules encode hits confirms the E-GMD artifact is a specific
case of a general reality across brands.

1. **One physical pad → many MIDI notes is universal.** General MIDI defines only ~12
   percussion slots (notes 35–81) with no room for ride bell, cross-stick, 4th tom, or
   hi-hat edge, so every module extends beyond GM and emits a **different MIDI note per
   zone/articulation**:
   - **Roland TD-17** (corroborates what we see in E-GMD): snare pad → 38 head / 40 rim /
     37 cross-stick; hi-hat → 42·22 closed bow·edge / 46·26 open bow·edge / 44 pedal; ride →
     51 bow / 59 edge / 53 bell.
   - **Yamaha DTX** (entry DTX402): kick 36, snare 38, rim 40, x-stick 37, closed-HH 42,
     open-HH 46, foot-HH 44 — GM-aligned for core pads, but cymbal slots diverge; higher-end
     DTX uses a proprietary map that "often conflicts with GM."
   - **Alesis**: same principle, "fewer zone options than Roland or Yamaha."
   The multi-note-per-pad behavior is inherent to e-drums, but the **specific
   note↔articulation assignment differs by brand, model, and kit preset, and is
   user-remappable.**

2. **Velocity-switched pitch (note layering) is real.** Roland modules support "MIDI note
   layering, where one pad triggers different notes at different velocities" (dynamic sample
   switching). This is the extreme form of the original hypothesis — the *pitch* itself can
   depend on velocity — and it is exactly what E-GMD's "Layered" kits (Big Room, Super Boom,
   Raw Dnb) do, which is why their note/velocity multisets differ from the other kits.

3. **Velocity is a shaped signal, not raw force.** Strike force → MIDI velocity passes
   through the module's configurable pipeline: **sensitivity** (gain 1–32) → **velocity
   curve** (LINEAR default; EXP1/2 emphasize hard hits; LOG1/2 emphasize soft; SPLINE
   extreme; LOUD1/2 compressed) → **threshold** (floor gate). So the recorded velocity is
   force *after* that curve — module- and setting-dependent.

**Implications for this project.**

- Our canonical `ml/src/drum_dynamics/core/voicemap.py` is effectively a **brand-normalization layer**,
  and it is necessary rather than optional. It works well because Roland (and entry Yamaha)
  are GM-adjacent for core voices (kick/snare/hats); the messy parts are auxiliary/cymbal
  zones and velocity-layered pads.
- **Transferability is bounded.** A humanizer trained on E-GMD predicts velocities in the
  **TD-17 note scheme + its velocity curve**. Applying it to another module needs (a) a
  note→canonical-voice remap for that brand and (b) awareness that "loud" is defined relative
  to a velocity curve. State this as an explicit assumption/limitation.
- **But velocity *ordering* is curve-invariant.** Any monotonic velocity curve preserves the
  rank order of hits, so *relative*-dynamics humanization transfers across modules even when
  absolute calibration does not — consistent with the Plan D finding that relative dynamics
  generalize but absolute level does not.

Sources: General MIDI percussion map (en.wikipedia.org/wiki/General_MIDI); Roland TD-17
mapping (github.com/mcfredrick/drum-transcription-training,
`drum_transcription/ROLAND_MAPPING.md`); cross-brand differences, pad note-layering, and
velocity curves (drumdash.com "MIDI Mapping 101"); Roland velocity-curve settings
(edrums.github.io/en/roland/trigger_settings); Yamaha DTX402 default note map
(manualzz/paperzz "DTX402K/432K/452K MIDI Reference"). The official Roland TD-17 MIDI
Implementation PDF (static.roland.com) is AES-encrypted and could not be extracted directly;
the GitHub reference corroborates its assignments.

## The TD-17 patches, characterized empirically

E-GMD re-recorded the original GMD performances on a **Roland TD-17** across **43 kits**
(`kit_name` = the TD-17 patch name; 40 factory patches + `Custom1/2/3`), and is "the first
[dataset] with human-performed velocity annotations" (Magenta E-GMD page; Callender et al.,
*Improving Perceptual Quality of Drum Transcription…*, arXiv:2004.00188).

**Paper vs. data discrepancy.** The paper frames the 43 kits as timbre variation over the
*same* MIDI ("recorded 43 drumkits… aligned within 2ms of the original MIDI files"), implying
fixed note assignments. **The released MIDI contradicts this** — patches remap pads, so the
per-kit MIDI note streams genuinely differ. `kit_name` is therefore a real MIDI-content
variable, not just an audio label, and our 43× pooling under one `file_id` mixes genuinely
different note streams for the same performance.

Profiling the raw MIDI of 6 performances (one per drummer) across all 43 patches:

- **Electronic and Custom patches never emit side-stick.** `808 Simple`, `909 Simple`, and
  `Custom1/2/3` produced pitch 37 in **0/6** performances — they remap cross-stick to the
  snare pitch (or omit it). Acoustic patches emit it whenever the groove uses cross-stick.
- **Some patches add auxiliary voices, matching their names.** `Compact Lite (w/ Tambourine
  HH)` and `Dark Hybrid` emit **tambourine (pitch 54)** and remap **pedal-hh → aux-perc** in
  4/6 performances; `Deep Daft` and `Big Room (Layered)` show the pedal-hh→aux-perc swap in
  2/6. Patch names telegraph the remapping (`2nd Hi-Hat`, `w/ Tambourine HH`, `More Cowbell`,
  `808/909 Simple`).
- **Velocity is preserved across most patches**, but a few performances on the "Layered"
  patches (`Big Room (Layered)`, `Super Boom (Layered)`, `Raw Dnb (Layered Hybrid)`) yield
  differing note/velocity multisets, consistent with Roland's velocity-switched note layering.

The upshot: the TD-17 patch set spans not just timbres but **different trigger→MIDI-note
maps**, so treating `kit_name` as a pure audio choice (as the paper does) understates its
effect on the symbolic data we train on.

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
from drum_dynamics.core.midi import load_note_array
from drum_dynamics.core.voicemap import voice_of
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
