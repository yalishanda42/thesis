"""Test download.py: HF model fetching with mocked network calls."""
import os
import sys

import drum_dynamics.serve.download as dl


def test_missing_files(tmp_path):
    """Verify missing_files reports correctly for empty and partially filled dirs."""
    # empty dir: all missing
    assert set(dl.missing_files(str(tmp_path))) == set(dl.REQUIRED_FILES)
    # create one file: it should not be in missing list
    open(os.path.join(tmp_path, "head_mdn.pt"), "w").close()
    missing = dl.missing_files(str(tmp_path))
    assert "head_mdn.pt" not in missing
    assert len(missing) == len(dl.REQUIRED_FILES) - 1


def test_download_models_renames_and_skips(tmp_path, monkeypatch):
    """Verify download_models fetches, renames, and skips present files when only_missing=True."""
    calls = []

    def fake_hf_hub_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename))
        # fake a download by creating a temp file with the source name
        src = tmp_path / ("src_" + filename)
        src.write_text("x")
        return str(src)

    # patch huggingface_hub.hf_hub_download (the symbol imported inside download_models)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    out = str(tmp_path / "proc")
    got = dl.download_models(out, only_missing=True)

    # all 6 files fetched and renamed to local names
    assert set(got) == set(dl.REQUIRED_FILES)
    for f in dl.REQUIRED_FILES:
        assert os.path.isfile(os.path.join(out, f)), f"Expected {f} in {out}"

    # verify the calls matched the REPOS mapping
    assert len(calls) == 6
    # check a few spot checks on the repo->file mapping
    assert ("yalishanda/dynamics-needed-lgbm", "model.joblib") in calls
    assert ("yalishanda/dynamics-needed-mdn", "mdn_head.pt") in calls
    assert ("yalishanda/dynamics-needed-categorical", "categorical_head.pt") in calls

    # second call with only_missing=True: should skip everything
    calls.clear()
    got2 = dl.download_models(out, only_missing=True)
    assert got2 == []
    assert calls == []  # no network calls


def test_download_models_only_missing_false(tmp_path, monkeypatch):
    """Verify only_missing=False re-downloads even if present."""
    def fake_hf_hub_download(repo_id, filename, revision=None):
        src = tmp_path / ("src_" + filename)
        src.write_text("y")
        return str(src)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    out = str(tmp_path / "proc")
    # first download
    dl.download_models(out, only_missing=True)
    # second with only_missing=False should re-fetch all
    got = dl.download_models(out, only_missing=False)
    assert len(got) == 6


def test_no_torch_import():
    """Verify importing download.py does not pull in torch (keep it light)."""
    # This test is more of a documentation check: the module itself doesn't import torch.
    # If torch is already in sys.modules (from other tests), that's fine; what matters is
    # that download.py itself doesn't require it.
    # We already imported dl at the top, so just verify the module doesn't have `import torch` at top.
    import inspect
    source = inspect.getsource(dl)
    # Check module-level imports don't include torch
    lines = [line.strip() for line in source.split("\n") if line.strip() and not line.strip().startswith("#")]
    # Find the imports before the first function def
    imports = []
    for line in lines:
        if line.startswith("def "):
            break
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    # torch should not be in module-level imports
    assert not any("torch" in imp for imp in imports), "download.py must not import torch at module level"
