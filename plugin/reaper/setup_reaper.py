#!/usr/bin/env python
"""One-time setup for the Dynamics Needed Reaper tool.

Writes plugin/reaper/config.local.json (the venv python + repo root the ReaScript
needs to auto-start the engine) and prints how to register the action in Reaper.

Usage: .venv/bin/python plugin/reaper/setup_reaper.py [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    cfg = {"venv_python": sys.executable, "repo_root": REPO_ROOT, "port": args.port}
    cfg_path = os.path.join(HERE, "config.local.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    print("Wrote {}".format(cfg_path))
    print("  venv_python: {}".format(cfg["venv_python"]))
    print("  repo_root  : {}".format(cfg["repo_root"]))
    print()
    print("Register the action in Reaper (one time):")
    print("  Actions -> Show action list -> New action -> Load ReaScript...")
    print("  Select: {}".format(os.path.join(HERE, "dynamics_needed.py")))
    print("  Then bind it to a key or add it to a toolbar / MIDI editor menu.")


if __name__ == "__main__":
    main()
