"""Build the Section A tabular dataset for every E-GMD split -> parquet.

Usage: .venv/bin/python scripts/build_dataset.py [--limit N] [--workers K]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer.features import build_note_features   # noqa: E402
from drumhumanizer.midi import load_note_array           # noqa: E402

EGMD_BASE = os.path.join("data", "e-gmd", "e-gmd-v1.0.0")
EGMD_CSV = os.path.join(EGMD_BASE, "e-gmd-v1.0.0.csv")
OUT_DIR = os.path.join("data", "processed")


def _worker(record):
    path, meta = record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            na = load_note_array(path)
        if len(na) == 0:
            return None
        return build_note_features(na, meta)
    except Exception:
        return None


def build_split(df, split, workers):
    sub = df[df["split"] == split]
    records = [(os.path.join(EGMD_BASE, r.midi_filename), r._asdict())
               for r in sub.itertuples(index=False)]
    print(f"[{split}] {len(records)} files on {workers} workers ...")
    t0 = time.time()
    frames, failed = [], 0
    with Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(_worker, records, chunksize=32), 1):
            if out is None:
                failed += 1
            else:
                frames.append(out)
            if i % 5000 == 0 or i == len(records):
                print(f"  [{split}] {i}/{len(records)}  ({i/(time.time()-t0):.0f} files/s)")
    result = pd.concat(frames, ignore_index=True)
    print(f"[{split}] {len(result):,} rows, {failed} files failed")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(EGMD_CSV)
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)

    for split in ("train", "validation", "test"):
        out = build_split(df, split, args.workers)
        path = os.path.join(OUT_DIR, f"egmd_tabular_{split}.parquet")
        out.to_parquet(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
