#!/usr/bin/env python
"""Development-mode config writer for the Dynamics Needed Reaper plugin.

Writes the **development** config (venv python + repo root + port) to the Reaper
resource path as ``dynamics_needed_config.json``. This is used by developers running
from the repo; end users never run this script. When installed via ReaPack, the
ReaScript's first-run bootstrap automatically downloads the engine and writes the
frozen config ({engine_path, proc_dir, engine_version, port}) instead.

The ReaScript reads the config via ``RPR_GetResourcePath()`` because Reaper defines
neither ``__file__`` nor a Python ``get_action_context()`` for it.

Usage: .venv/bin/python plugin/reaper/setup_reaper.py [--port 8765]
                                                      [--reaper-resource-path DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# Default Reaper resource path on macOS; override with --reaper-resource-path.
DEFAULT_RESOURCE_PATH = os.path.expanduser("~/Library/Application Support/REAPER")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Configure the Dynamics Needed Reaper tool (one-time)."
    )
    p.add_argument("--port", type=int, default=8765,
                   help="localhost port the inference engine listens on")
    p.add_argument("--reaper-resource-path", default=DEFAULT_RESOURCE_PATH,
                   help="Reaper resource dir (see Reaper: Options -> Show REAPER "
                        "resource path). Defaults to the macOS location.")
    args = p.parse_args()

    cfg = {"venv_python": sys.executable, "repo_root": REPO_ROOT, "port": args.port}

    resource_path = os.path.abspath(os.path.expanduser(args.reaper_resource_path))
    if not os.path.isdir(resource_path):
        raise SystemExit(
            "Reaper resource path not found: {}\n"
            "Pass the correct one with --reaper-resource-path (Reaper: Options -> "
            "Show REAPER resource path).".format(resource_path))
    primary = os.path.join(resource_path, "dynamics_needed_config.json")
    with open(primary, "w") as fh:
        json.dump(cfg, fh, indent=2)

    local_copy = os.path.join(HERE, "config.local.json")
    with open(local_copy, "w") as fh:
        json.dump(cfg, fh, indent=2)

    print("Wrote config the ReaScript will read:")
    print("  {}".format(primary))
    print("Reference copy:")
    print("  {}".format(local_copy))
    print("  venv_python: {}".format(cfg["venv_python"]))
    print("  repo_root  : {}".format(cfg["repo_root"]))
    print("  port       : {}".format(cfg["port"]))
    print()
    print("Register the action in Reaper (one time):")
    print("  Actions -> Show action list -> New action -> Load ReaScript...")
    print("  Select: {}".format(os.path.join(HERE, "dynamics_needed.py")))
    print("  Then bind it to a key or add it to a toolbar / MIDI editor menu.")


if __name__ == "__main__":
    main()
