---
license: cc-by-4.0
pretty_name: Expanded Groove MIDI Dataset (E-GMD v1.0.0) — MIDI-only mirror
tags:
  - music
  - midi
  - drums
  - percussion
  - groove
size_categories:
  - 10K<n<100K
---

# Expanded Groove MIDI Dataset (E-GMD v1.0.0) — MIDI-only mirror

> **Unofficial mirror.** This is a convenience mirror of the **MIDI-only** portion
> of the Expanded Groove MIDI Dataset (E-GMD), redistributed under its CC BY 4.0
> license (see below). It is **not affiliated with or endorsed by Google / Magenta**.
> The canonical source is <https://g.co/magenta/e-gmd>.

E-GMD is a large dataset of human drum performances (~444 hours) recorded on a
Roland TD-17 electronic kit, captured as MIDI with expressive timing and velocity.
It extends the original Groove MIDI Dataset (GMD) to many more performances and kits.

## What's in this mirror

- `e-gmd-v1.0.0-midi.zip` — the original Magenta MIDI archive, byte-for-byte
  (45,537 `.midi` performances organized by drummer/session).
- `e-gmd-v1.0.0.csv` — the metadata table (also inside the zip), exposed at the top
  level for quick previewing. Columns: `drummer, session, id, style, bpm, beat_type,
  time_signature, duration, split, midi_filename, audio_filename, kit_name`.
- `LICENSE` — the original CC BY 4.0 license file.

**Not included:** the audio renderings (the full E-GMD audio is ~100 GB). The
`audio_filename` column therefore references files not present in this mirror. For
audio, use the official source.

## Splits

The `split` column carries E-GMD's official train / validation / test partition
(shared across drummers). Use it directly for reproducibility.

## License

Provided by Google LLC under a **Creative Commons Attribution 4.0 International
(CC BY 4.0)** license — <http://creativecommons.org/licenses/by/4.0/>. You may share
and adapt with attribution.

## Citation

Please cite the E-GMD paper and specify the dataset version (v1.0.0):

```bibtex
@misc{callender2020improving,
    title={Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset},
    author={Lee Callender and Curtis Hawthorne and Jesse Engel},
    year={2020},
    eprint={2004.00188},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```

## Provenance

Mirrored for the *Dynamics Needed* thesis project (drum-velocity / dynamics
prediction). A processed, tabular feature version derived from this data is
published separately as `yalishanda/dynamics-needed-egmd-tabular`.
