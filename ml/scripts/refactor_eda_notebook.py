"""Rewrite notebooks/eda.ipynb to import helpers from the drum_dynamics package.

Replaces the inline helper-definition cells (Idx/MidiNote, DRUM_MIDI_NAME,
drums_roll, play_midi_*) with imports, points paths at the local dataset, and
clears stale outputs. Matches cells by a source substring so it is robust to
minor index drift.
"""

import json
from pathlib import Path

NB = Path("notebooks/eda.ipynb")


def code(src: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


# (substring to find in a cell's joined source) -> new source string
REPLACEMENTS = {
    "# uncomment this when running outside of Kaggle": (
        "# The E-GMD dataset is expected under ../data/e-gmd (see repo README / requirements.txt).\n"
        "# To (re)download it locally, run from the repo root:\n"
        "#   mkdir -p data && curl -fL -o data/e-gmd-v1.0.0-midi.zip \\\n"
        "#     https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip\n"
        "#   unzip -nq data/e-gmd-v1.0.0-midi.zip -d data/e-gmd"
    ),
    "DATASET_BASE_PATH = os.path.join('..', 'input'": (
        "DATASET_BASE_PATH = os.path.join('..', 'data', 'e-gmd', 'e-gmd-v1.0.0')\n"
        "DATASET_METADATA_PATH = os.path.join(DATASET_BASE_PATH, 'e-gmd-v1.0.0.csv')"
    ),
    "!pip3 install music21 partitura scikit-learn torch matplotlib": (
        "# Dependencies are managed via ../requirements.txt in a venv.\n"
        "# On a fresh machine / Kaggle, uncomment:\n"
        "# %pip install -r ../requirements.txt"
    ),
    "import pandas as pd\nimport partitura": (
        "import sys\n"
        "sys.path.insert(0, '..')  # make the drum_dynamics package importable from notebooks/\n\n"
        "import pandas as pd\n"
        "import partitura\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n\n"
        "# Helpers previously defined inline are now in the drum_dynamics package:\n"
        "from drum_dynamics import (\n"
        "    Idx, MidiNote, DRUM_MIDI_NAME, midi_number_to_tone,\n"
        "    load_note_array, piano_roll, drums_roll,\n"
        "    play_midi_file, play_midi_notes, set_soundfont,\n"
        ")"
    ),
    "from enum import Enum": (
        "# `Idx` and `MidiNote` are now imported from drum_dynamics.midi (see imports cell above).\n"
        "# They wrap a partitura note tuple:\n"
        "#   [0]=onset_secs [1]=duration_secs [2]=onset_ticks [3]=duration_ticks\n"
        "#   [4]=pitch [5]=velocity [6]=track [7]=channel [8]=id"
    ),
    "DRUM_MIDI_NAME = {": (
        "# DRUM_MIDI_NAME (General MIDI percussion map) is imported from drum_dynamics.midi."
    ),
    "def drums_roll(notes):": (
        "# drums_roll() / piano_roll() are imported from drum_dynamics.viz."
    ),
    "!brew install fluidsynth || apt-get install fluidsynth": (
        "# Playback needs the native FluidSynth library:\n"
        "#   macOS:  brew install fluid-synth\n"
        "#   Debian: apt-get install fluidsynth"
    ),
    "!pip3 install pyfluidsynth": (
        "# pyfluidsynth is installed via ../requirements.txt."
    ),
    "!wget -nc -P sf/big/": (
        "# A General MIDI soundfont (with a drum kit) is expected at ../sf/big/FluidR3_GM.sf2.\n"
        "# To download it locally, run from the repo root:\n"
        "#   mkdir -p sf/big && curl -fL -o sf/big/FluidR3_GM.sf2 \\\n"
        "#     'https://raw.githubusercontent.com/urish/cinto/master/media/FluidR3%20GM.sf2'"
    ),
    "SF_PATH = os.path.join('sf', 'big'": (
        "set_soundfont(os.path.join('..', 'sf', 'big', 'FluidR3_GM.sf2'))"
    ),
    "import fluidsynth\nfrom IPython.display import Audio": (
        "# play_midi_file() / play_midi_notes() are imported from drum_dynamics.playback.\n"
        "# NOTE: for drum tracks pass is_drums=True so playback routes to the GM percussion set."
    ),
    "play_midi_notes(example_notes)": (
        "play_midi_notes(example_notes, is_drums=True)"
    ),
}

nb = json.loads(NB.read_text())
applied = set()

for cell in nb["cells"]:
    src = "".join(cell["source"])
    # clear outputs on every code cell so we start from a clean, re-runnable state
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    for needle, new_src in REPLACEMENTS.items():
        if needle in src:
            cell["source"] = new_src.splitlines(keepends=True)
            applied.add(needle)
            break

missing = set(REPLACEMENTS) - applied
if missing:
    raise SystemExit(f"ERROR: these replacements did not match any cell:\n" + "\n".join(missing))

NB.write_text(json.dumps(nb, indent=1) + "\n")
print(f"Rewrote {NB} — applied {len(applied)} replacements across {len(nb['cells'])} cells.")
