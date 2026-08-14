"""Download the engine's model files into --out (default data/processed) for CI.

Thin wrapper over drum_dynamics.serve.download so the mapping lives in one place.
"""
from __future__ import annotations

import argparse
import os

from drum_dynamics.serve.download import download_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("data", "processed"))
    p.add_argument("--revision", default=None, help="e.g. v0.1.0 to pin a tagged model")
    args = p.parse_args()
    got = download_models(args.out, revision=args.revision, only_missing=False, log=print)
    print("weights ready in {} ({} files)".format(args.out, len(got)))


if __name__ == "__main__":
    main()
