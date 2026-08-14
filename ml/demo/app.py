"""Dynamics Needed - Gradio demo for structure-driven drum-velocity prediction.

Upload a drum-MIDI groove (flat or existing velocities), pick a model and the
groove's musical context, and get back a version whose note velocities are
predicted from *structure and timing alone* - plus a before/after velocity plot
and an audio preview rendered through a General MIDI SoundFont.

Runtime notes
-------------
* ``import spaces`` MUST precede any torch import so its ZeroGPU hijack lands.
  The models here are tiny and CPU-only, so a single no-op ``@spaces.GPU``
  function satisfies the ZeroGPU requirement without ever requesting a GPU
  (nothing in the hot path is decorated, so no visitor quota is burned).
* The whole ``drum_dynamics`` package is vendored next to this file at deploy
  time, and the six ready-to-load model files live under ``models/``.
"""
from __future__ import annotations

import spaces  # noqa: F401  (must come before torch; see module docstring)

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SF2 = os.path.join(HERE, "sf", "FluidR3_GM.sf2")
SR = 44100


def _preseed_partitura_soundfont() -> None:
    """Stop partitura from FTP-downloading a soundfont at import time.

    partitura's top-level ``__init__`` imports its audio-export module, which -
    when fluidsynth is importable and its bundled asset is missing - fetches a
    MuseScore soundfont over FTP. HF Spaces block FTP egress, so that call hangs
    and the app crashes on ``import partitura``. Pre-seed the asset (symlink our
    bundled GM soundfont) so partitura finds it and skips the download. Must run
    BEFORE ``import partitura``.
    """
    import importlib.util

    spec = importlib.util.find_spec("partitura")  # locate without importing
    if not spec or not spec.origin:
        return
    assets = os.path.join(os.path.dirname(spec.origin), "assets")
    target = os.path.join(assets, "MuseScore_General.sf3")
    if os.path.exists(target):
        return
    try:
        os.makedirs(assets, exist_ok=True)
        if os.path.exists(SF2):
            os.symlink(SF2, target)
        else:
            open(target, "wb").close()  # placeholder; partitura's synth is unused
    except OSError:
        pass


_preseed_partitura_soundfont()

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import partitura

from drum_dynamics.serve.models import Engine
from drum_dynamics.viz.playback import set_soundfont, _ensure_fluidsynth_discoverable
from drum_dynamics.core.midi import drum_name

# --- load once at module scope ------------------------------------------------
ENGINE = Engine.load(os.path.join(HERE, "models"))
LEVELS = ENGINE.levels()
STYLES = LEVELS["styles"]
if os.path.exists(SF2):
    set_soundfont(SF2)

MODEL_LABELS = {
    "Transformer - MDN (temperature)": "mdn",
    "Transformer - Categorical": "categorical",
    "LightGBM (deterministic, less dynamic)": "lgbm",
}


# --- ZeroGPU registration hook (never called on the hot path) ------------------
@spaces.GPU(duration=1)
def _zerogpu_noop():  # pragma: no cover - satisfies the "needs one GPU fn" rule
    """No-op so the Space qualifies as ZeroGPU; real inference runs on CPU."""
    return "ok"


# --- helpers ------------------------------------------------------------------
def _load_notes(midi_path: str):
    """Return the partitura performance + its editable note-dict list."""
    perf = partitura.load_performance_midi(midi_path)
    pp = perf.performedparts[0]
    return perf, pp


def _note_events(pp):
    """Snapshot the current notes as (on_sec, off_sec, pitch, velocity) tuples."""
    return [(float(nd["note_on"]), float(nd["note_off"]),
             int(nd["midi_pitch"]), int(nd["velocity"])) for nd in pp.notes]


def _render_events(events) -> tuple[int, np.ndarray]:
    """Render (on, off, pitch, velocity) drum events to a mono int16 waveform."""
    _ensure_fluidsynth_discoverable()  # no-op off macOS; helps on local dev
    import fluidsynth  # lazy: keeps the app importable without the native lib

    fl = fluidsynth.Synth(samplerate=SR)
    sfid = fl.sfload(SF2)
    fl.program_select(9, sfid, 128, 0)  # GM percussion set on channel 9

    timeline = []  # (time_sec, is_on, pitch, velocity)
    for on, off, pitch, vel in events:
        timeline.append((on, True, pitch, vel))
        timeline.append((off, False, pitch, 0))
    timeline.sort(key=lambda e: e[0])

    audio = []
    for (t0, is_on, pitch, vel), (t1, *_rest) in zip(timeline[:-1], timeline[1:]):
        if is_on:
            fl.noteon(9, pitch, vel)
        else:
            fl.noteoff(9, pitch)
        n = int(max(0, (t1 - t0)) * SR)
        if n:
            audio.extend(fl.get_samples(n)[::2])  # interleaved stereo -> mono
    audio.extend(fl.get_samples(SR)[::2])  # let the last hits ring out
    fl.delete()
    return SR, np.asarray(audio, dtype=np.int16)


def _drumroll_figure(na_before, na_after):
    """Stacked drum-rolls (velocity as colour) for original vs humanized.

    Only the drum pieces actually present are shown, on a shared 0-127 colour
    scale so the change in dynamics is directly comparable between the two.
    """
    pitches = sorted({int(p) for p in na_before["pitch"]}, reverse=True)  # high on top
    row_of = {p: i for i, p in enumerate(pitches)}
    n_ticks = int(max(na_before["onset_tick"] + na_before["duration_tick"])) + 1
    min_w = max(1, n_ticks // 200)  # widen hits so short drums stay visible

    def build(na):
        roll = np.zeros((len(pitches), n_ticks))
        for pt, on, dur, vel in zip(na["pitch"], na["onset_tick"],
                                    na["duration_tick"], na["velocity"]):
            r = row_of[int(pt)]
            s = int(on)
            e = min(n_ticks, s + max(int(dur), min_w))
            roll[r, s:e] = np.maximum(roll[r, s:e], vel / 127.0)
        return roll

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    im = None
    for ax, na, title in ((axes[0], na_before, "Original"),
                          (axes[1], na_after, "Humanized")):
        im = ax.imshow(build(na), aspect="auto", origin="lower", interpolation="none",
                       cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_yticks(range(len(pitches)))
        ax.set_yticklabels([drum_name(p) for p in pitches], fontsize=7)
        ax.set_ylabel("drum piece", fontsize=8)
    axes[1].set_xlabel("time (MIDI ticks)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("velocity")
    cbar.set_ticks(np.linspace(0, 1, 8))
    cbar.set_ticklabels(np.linspace(0, 127, 8, dtype=int))
    return fig


# --- main callback ------------------------------------------------------------
def humanize(midi_file, model_label, style, bpm, time_signature, beat_type,
             temperature, blend, seed, want_audio):
    if midi_file is None:
        raise gr.Error("Upload a drum-MIDI file first (or click an example).")
    model = MODEL_LABELS[model_label]

    perf, pp = _load_notes(midi_file)
    if not pp.notes:
        raise gr.Error("No notes found in that MIDI file.")

    na_before = pp.note_array().copy()          # original, for the roll
    events_before = _note_events(pp)            # original, for the audio
    notes = [
        {"index": i, "onset_sec": float(nd["note_on"]), "pitch": int(nd["midi_pitch"]),
         "velocity": int(nd["velocity"]), "selected": True}
        for i, nd in enumerate(pp.notes)
    ]
    request = {
        "model": model, "seed": int(seed), "temperature": float(temperature),
        "blend": float(blend), "bpm": float(bpm), "time_signature": time_signature,
        "style": style, "beat_type": beat_type, "notes": notes,
    }
    pred = ENGINE.predict(request)  # {note_index: new_velocity}

    for i, nd in enumerate(pp.notes):
        if i in pred:
            nd["velocity"] = int(pred[i])
    na_after = pp.note_array().copy()           # humanized, for the roll
    events_after = _note_events(pp)             # humanized, for the audio

    out_midi = os.path.join(tempfile.mkdtemp(), "humanized.mid")
    partitura.save_performance_midi(perf, out_midi)

    fig = _drumroll_figure(na_before, na_after)
    do_audio = bool(want_audio) and os.path.exists(SF2)
    audio_before = _render_events(events_before) if do_audio else None
    audio_after = _render_events(events_after) if do_audio else None
    return out_midi, fig, audio_before, audio_after


GENRES = LEVELS["genres"]
_EX_DIR = os.path.join(HERE, "examples")
EXAMPLES = [
    [os.path.join(_EX_DIR, "funk_138_beat_4-4.midi")],
    [os.path.join(_EX_DIR, "jazz-funk_116_beat_4-4.midi")],
    [os.path.join(_EX_DIR, "soul_102_beat_4-4.midi")],
    [os.path.join(_EX_DIR, "soul_105_beat_4-4.midi")],
]


def _autofill_from_file(path):
    """Best-effort: read musical context from an E-GMD-style filename.

    Fills style / bpm / time-signature / beat-type when the name carries them
    (e.g. ``rock_65_beat_4-4.midi``); leaves any field it can't parse untouched.
    Returns gr.update()s in the order (style, bpm, time_signature, beat_type).
    """
    import re

    keep = (gr.update(), gr.update(), gr.update(), gr.update())
    if not path:
        return keep
    tokens = re.split(r"[_\s]+", os.path.splitext(os.path.basename(path))[0].lower())

    bpm = next((int(t) for t in tokens if t.isdigit() and 30 <= int(t) <= 300), None)
    ts = next((t for t in tokens if re.fullmatch(r"\d-\d", t)), None)
    beat = next((t for t in tokens if t in ("beat", "fill")), None)
    style = None
    for t in tokens:
        if t in STYLES:
            style = t
            break
        if t.split("-")[0] in GENRES:
            style = t.split("-")[0]
            break

    return (
        gr.update(value=style) if style else gr.update(),
        gr.update(value=bpm) if bpm else gr.update(),
        gr.update(value=ts) if ts else gr.update(),
        gr.update(value=beat) if beat else gr.update(),
    )

DESC = """\
# 🥁 Dynamics Needed

Drum programming often lands notes at a **flat, robotic velocity**. This demo
predicts a human-sounding velocity for every note from **structure and timing
alone** - never from the note's own loudness - using models trained on the
[E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) dataset of real
performances.

Upload a drum-MIDI groove **or** pick an example, set its musical context, and
compare the result **by ear** (original vs humanized audio) and **by eye**
(a drum-roll where colour is velocity). Same engine that drives the REAPER
plugin, behind a web UI.
"""


with gr.Blocks(title="Dynamics Needed") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            midi_in = gr.File(label="① Upload a drum MIDI", file_types=[".mid", ".midi"],
                              type="filepath")
            gr.Markdown("<div style='text-align:center;opacity:.7'>— or —</div>")
            gr.Examples(examples=EXAMPLES, inputs=[midi_in],
                        label="Pick an example groove (flat velocities · E-GMD, CC-BY 4.0)")
            model_in = gr.Dropdown(list(MODEL_LABELS), value=list(MODEL_LABELS)[0],
                                   label="② Model")
            style_in = gr.Dropdown(STYLES, value="rock", label="Genre / style",
                                   info="Conditions the transformer; the genre is the part before '/'.")
            with gr.Row():
                bpm_in = gr.Number(value=120, label="BPM", precision=0)
                ts_in = gr.Dropdown(["3-4", "4-4", "5-4", "5-8", "6-8"], value="4-4",
                                    label="Time signature")
                beat_in = gr.Dropdown(["beat", "fill"], value="beat", label="Beat type")
            with gr.Accordion("Sampling & blend", open=False):
                temp_in = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature",
                                    info="MDN only; ignored by LightGBM and Categorical.")
                blend_in = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Blend",
                                     info="1.0 = fully predicted, 0.0 = keep original velocities.")
                seed_in = gr.Number(value=42, label="Seed", precision=0)
            audio_chk = gr.Checkbox(value=True, label="Render audio previews")
            run_btn = gr.Button("③ Predict velocities", variant="primary")
        with gr.Column(scale=1):
            with gr.Row():
                audio_before_out = gr.Audio(label="Original (as uploaded)", type="numpy")
                audio_after_out = gr.Audio(label="Humanized", type="numpy")
            midi_out = gr.File(label="Humanized MIDI (download)")

    # Full-width so the two stacked rolls stay legible.
    plot_out = gr.Plot(label="Drum-roll · colour = velocity (original vs humanized)")

    # Auto-fill musical context from the filename (examples and named uploads).
    ctx_out = [style_in, bpm_in, ts_in, beat_in]
    midi_in.change(_autofill_from_file, inputs=[midi_in], outputs=ctx_out)

    inputs = [midi_in, model_in, style_in, bpm_in, ts_in, beat_in,
              temp_in, blend_in, seed_in, audio_chk]
    outputs = [midi_out, plot_out, audio_before_out, audio_after_out]
    run_btn.click(humanize, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(mcp_server=True, theme=gr.themes.Soft())
