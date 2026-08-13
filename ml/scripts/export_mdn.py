#!/usr/bin/env python
"""Export MDN inference artifacts (genre vocab + bpm stats) to mdn_meta.json.

The MDN checkpoint (head_mdn.pt["best_model"]) is a full model state dict, but the
genre vocabulary and bpm normalization are recomputed from the train parquet at
train time and never persisted. The service needs them to reproduce inference.

Usage: .venv/bin/python ml/scripts/export_mdn.py
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from drum_dynamics.serve.export import build_mdn_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join("data", "processed", "egmd_tabular_train.parquet"))
    p.add_argument("--out", default=os.path.join("data", "processed", "mdn_meta.json"))
    args = p.parse_args()

    meta = build_mdn_meta(pd.read_parquet(args.train))
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.out}  (genres={len(meta['genre_vocab'])}, "
          f"bpm_mean={meta['bpm_mean']:.2f}, bpm_std={meta['bpm_std']:.2f})")


if __name__ == "__main__":
    main()
