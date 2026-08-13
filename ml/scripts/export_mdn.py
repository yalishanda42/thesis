#!/usr/bin/env python
"""Export MDN inference artifacts (genre vocab + bpm stats) to mdn_meta.json.

The MDN checkpoint (head_mdn.pt["best_model"]) is a full model state dict, but the
genre vocabulary and bpm normalization are recomputed from the train parquet at
train time and never persisted. The service needs them to reproduce inference.

Usage:
    .venv/bin/python ml/scripts/export_mdn.py --train data/processed/egmd_tabular_train.parquet --out data/processed/mdn_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from drum_dynamics.serve.export import build_mdn_meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export MDN inference artifacts (genre vocab + bpm stats) to mdn_meta.json."
    )
    p.add_argument("--train", default="data/processed/egmd_tabular_train.parquet",
                   help="train parquet file (required to extract genre vocab and bpm stats)")
    p.add_argument("--out", default="data/processed/mdn_meta.json",
                   help="output JSON path for metadata (genre_vocab, bpm_mean, bpm_std, head)")
    args = p.parse_args()

    train_path = Path(args.train)
    if not train_path.is_file():
        raise SystemExit(f"train parquet not found: {train_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = build_mdn_meta(pd.read_parquet(train_path))
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}  (genres={len(meta['genre_vocab'])}, "
          f"bpm_mean={meta['bpm_mean']:.2f}, bpm_std={meta['bpm_std']:.2f})")


if __name__ == "__main__":
    main()
