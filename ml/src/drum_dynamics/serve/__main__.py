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
    args = p.parse_args()

    engine = Engine.load(args.proc_dir)
    run(engine, port=args.port, parent_pid=args.parent_pid, idle_timeout=args.idle_timeout)


if __name__ == "__main__":
    main()
