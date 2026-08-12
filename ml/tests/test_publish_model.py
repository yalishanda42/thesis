import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "publish_model.py"


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
