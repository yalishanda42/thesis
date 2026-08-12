"""Build a drummer-disjoint split (Plan D robustness) from the existing
featurized tabular parquets — no re-extraction, just re-partition by drummer.

The held-out drummers become the test split; the rest are split into
train/val by file (genre-stratified). Writes egmd_holdout_{split}.parquet.

Usage: .venv/bin/python scripts/build_holdout_split.py [--holdout d3 d8] [--val-frac 0.1]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from drum_dynamics.holdout import build_drummer_holdout

PROC = os.path.join("data", "processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", nargs="+", default=["drummer3", "drummer8"],
                    help="drummer ids to hold out as the test split")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = [pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{s}.parquet"))
              for s in ("train", "validation", "test")]
    total = sum(len(f) for f in frames)

    train, val, test = build_drummer_holdout(frames, set(args.holdout),
                                             val_frac=args.val_frac, seed=args.seed)

    for name, part in (("train", train), ("validation", val), ("test", test)):
        path = os.path.join(PROC, f"egmd_holdout_{name}.parquet")
        part.to_parquet(path, index=False)
        genres = part["genre"].nunique()
        drummers = sorted(part["drummer"].unique())
        print(f"[{name:10}] rows={len(part):>9,}  files={part['file_id'].nunique():>4}  "
              f"genres={genres:2d}  drummers={drummers}")

    assert len(train) + len(val) + len(test) == total, "partition dropped rows"
    missing = set(pd.concat(frames)["genre"]) - set(train["genre"])
    print(f"\nheld out: {sorted(args.holdout)}")
    print(f"genres missing from train: {missing or 'NONE — all retained'}")
    print(f"total rows preserved: {total:,}")
    print(f"wrote egmd_holdout_*.parquet to {PROC}/")


if __name__ == "__main__":
    main()
