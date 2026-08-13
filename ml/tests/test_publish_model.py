import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "publish_model.py"


def _load_publish_module():
    spec = importlib.util.spec_from_file_location("publish_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_card_substitutes_nested_and_formats_floats():
    render_card = _load_publish_module().render_card
    template = "MAE {{lightgbm.mae}} | NLL {{native_nll}} | rho {{lightgbm.per_track_pearson}}"
    metrics = {"lightgbm": {"mae": 12.34567, "per_track_pearson": 0.5}, "native_nll": 2.0}
    rendered, unresolved = render_card(template, metrics)
    assert rendered == "MAE 12.346 | NLL 2.000 | rho 0.500"
    assert unresolved == []


def test_render_card_leaves_unknown_placeholder_intact():
    render_card = _load_publish_module().render_card
    template = "known {{a.b}} unknown {{missing.key}}"
    rendered, unresolved = render_card(template, {"a": {"b": 1.0}})
    assert "{{missing.key}}" in rendered
    assert "1.000" in rendered
    assert unresolved == ["missing.key"]


def test_help_runs():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--repo" in r.stdout


def test_missing_artifact_exits_nonzero(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", "x/y",
         "--artifact", str(tmp_path / "nope.pt"), "--path-in-repo", "model.pt"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "artifact not found" in (r.stderr + r.stdout)
