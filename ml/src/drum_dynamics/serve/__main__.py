"""Entry point: python -m drum_dynamics.serve [options]."""
from __future__ import annotations

import argparse
import os

from .models import Engine
from .server import run


def main() -> None:
    p = argparse.ArgumentParser(prog="drum_dynamics.serve")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--parent-pid", type=int, default=None)
    p.add_argument("--idle-timeout", type=int, default=1800)
    p.add_argument("--proc-dir", default=os.path.join("data", "processed"))
    p.add_argument("--download", action="store_true",
                   help="download missing model files from the HF Hub before serving")
    args = p.parse_args()

    from .download import missing_files, download_models
    if args.download or missing_files(args.proc_dir):
        download_models(args.proc_dir, log=lambda m: print(m, flush=True))

    engine = Engine.load(args.proc_dir)
    run(engine, port=args.port, parent_pid=args.parent_pid, idle_timeout=args.idle_timeout)


if __name__ == "__main__":
    main()
