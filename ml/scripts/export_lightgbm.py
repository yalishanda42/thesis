#!/usr/bin/env python
"""Export the fitted LightGBM model to native, cross-language formats (for C++/DAW).

The training script (train_tabular.py) saves a joblib pickle of the sklearn
``LGBMRegressor`` — Python-only. C++ needs LightGBM's *native* text model, loaded
via ``LGBM_BoosterCreateFromModelfile``. This script reads the joblib bundle and
writes, next to it:

  - ``lightgbm_model.txt``     native LightGBM text model, baked to best_iteration.
  - ``lightgbm_features.json`` the exact training feature order + each categorical
                               feature's ordered level list. A C++ caller builds
                               the feature vector in ``feature_names`` order and
                               passes each categorical as its **index** in
                               ``categorical_levels[col]`` (== the integer code the
                               model splits on). Without this, categorical splits
                               are meaningless on the C++ side.

Usage:
    .venv/bin/python ml/scripts/export_lightgbm.py \\
        --model data/processed/lightgbm_model.joblib --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export a fitted LightGBM joblib to native .txt + feature spec."
    )
    p.add_argument("--model", default="data/processed/lightgbm_model.joblib",
                   help="joblib bundle saved by train_tabular.py")
    p.add_argument("--out-dir", default="data/processed",
                   help="directory to write lightgbm_model.txt + lightgbm_features.json")
    args = p.parse_args()

    bundle_path = Path(args.model)
    if not bundle_path.is_file():
        raise SystemExit(f"model not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]                      # lgb.LGBMRegressor
    best_it = int(bundle.get("best_iteration") or model.best_iteration_ or 0)
    booster = model.booster_

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "lightgbm_model.txt"
    booster.save_model(str(txt_path), num_iteration=best_it)

    cats = list(bundle["cat"])
    spec = {
        "feature_names": list(booster.feature_name()),
        "categorical_features": cats,
        # index in each list == the integer category code the model splits on
        "categorical_levels": {c: [str(v) for v in bundle["cat_categories"][c]] for c in cats},
        "best_iteration": best_it,
        "note": ("Build the feature vector in feature_names order; pass each "
                 "categorical as its index in categorical_levels[col]."),
    }
    spec_path = out_dir / "lightgbm_features.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    print(f"wrote {txt_path} ({txt_path.stat().st_size} bytes, best_iteration={best_it})")
    print(f"wrote {spec_path} ({len(spec['feature_names'])} features, "
          f"{len(cats)} categorical)")


if __name__ == "__main__":
    main()
