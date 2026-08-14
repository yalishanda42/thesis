"""Fetch model weights from the Hugging Face Hub into a local proc-dir.

Each of the 3 model repos holds its combined checkpoint + metadata JSON; the
binaries are renamed to the names Engine.load() expects. Reads HF_TOKEN from the
environment for private repos.
"""
from __future__ import annotations

import os
import shutil

# repo -> [(name_in_repo, local_name), ...]
REPOS = {
    "yalishanda/dynamics-needed-lgbm": [
        ("model.joblib", "lightgbm_model.joblib"),
        ("lightgbm_features.json", "lightgbm_features.json"),
    ],
    "yalishanda/dynamics-needed-mdn": [
        ("mdn_head.pt", "head_mdn.pt"),
        ("mdn_meta.json", "mdn_meta.json"),
    ],
    "yalishanda/dynamics-needed-categorical": [
        ("categorical_head.pt", "head_categorical.pt"),
        ("transformer_meta.json", "transformer_meta.json"),
    ],
}

REQUIRED_FILES = [
    "lightgbm_model.joblib", "lightgbm_features.json",
    "mdn_meta.json", "head_mdn.pt",
    "transformer_meta.json", "head_categorical.pt",
]


def missing_files(proc_dir):
    """Return the REQUIRED_FILES not present in proc_dir."""
    return [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(proc_dir, f))]


def download_models(proc_dir, revision=None, only_missing=True, log=lambda m: None):
    """Download the model files from HF into proc_dir (renaming to local names).

    only_missing=True skips files already present. Returns the sorted list of
    local filenames that were downloaded this call.
    """
    from huggingface_hub import hf_hub_download  # local import: keep module light
    os.makedirs(proc_dir, exist_ok=True)
    fetched = []
    for repo, files in REPOS.items():
        for src_name, local_name in files:
            dst = os.path.join(proc_dir, local_name)
            if only_missing and os.path.isfile(dst):
                continue
            log("downloading {} -> {}".format(repo, local_name))
            path = hf_hub_download(repo_id=repo, filename=src_name, revision=revision)
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.copy(path, dst)
            fetched.append(local_name)
    return sorted(fetched)
