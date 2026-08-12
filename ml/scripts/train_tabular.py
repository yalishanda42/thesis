"""Fit baselines + LightGBM on the Section A dataset and evaluate on the test split.

Usage: .venv/bin/python scripts/train_tabular.py
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from drum_dynamics.baselines import GlobalMeanBaseline, LookupTableBaseline
from drum_dynamics.metrics import evaluate

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_a")
MODEL_PATH = os.path.join(PROC, "lightgbm_model.joblib")   # gitignored (under data/)
CAT = ["voice", "genre", "style", "time_signature", "beat_type", "nearest_subdiv"]
DROP = ["file_id", "drummer", "split", "onset_sec", "bar_index", "velocity"]
SEED = 42


def _load(split):
    return pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{split}.parquet"))


def _xy(df, cat_dtypes=None):
    X = df.drop(columns=DROP)
    for c in CAT:
        X[c] = X[c].astype("category")
        if cat_dtypes is not None:      # align val/test categories to train's
            X[c] = X[c].cat.set_categories(cat_dtypes[c])
    return X, df["velocity"].astype(float)


def main():
    os.makedirs(OUT, exist_ok=True)
    train, val, test = _load("train"), _load("validation"), _load("test")

    # baselines (fit on train, evaluate on test)
    gm = GlobalMeanBaseline().fit(train)
    lut = LookupTableBaseline().fit(train)
    results = {
        "global_mean": evaluate(test, gm.predict(test)),
        "lookup_table": evaluate(test, lut.predict(test)),
    }

    # LightGBM
    Xtr, ytr = _xy(train)
    cat_dtypes = {c: Xtr[c].cat.categories for c in CAT}
    Xval, yval = _xy(val, cat_dtypes)
    Xte, yte = _xy(test, cat_dtypes)
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=2000, learning_rate=0.05,
        num_leaves=255, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="l1",
              categorical_feature=CAT,
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    pred = model.predict(Xte, num_iteration=model.best_iteration_)
    results["lightgbm"] = evaluate(test, pred)
    results["lightgbm_best_iteration"] = int(model.best_iteration_)
    results["lightgbm_feature_importance"] = (
        pd.Series(model.feature_importances_, index=Xtr.columns)
        .sort_values(ascending=False).astype(int).to_dict()
    )

    # persist the fitted model + train category levels so downstream inference
    # (e.g. the audition notebook) can align categoricals exactly.
    joblib.dump({"model": model, "cat_categories": cat_dtypes,
                 "cat": CAT, "drop": DROP, "best_iteration": int(model.best_iteration_)},
                MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # comparison table
    print(f"\n{'model':14} {'MAE':>7} {'RMSE':>7} {'trk_r':>7} {'wbar_rho':>9} {'std_ratio':>9}")
    for name in ("global_mean", "lookup_table", "lightgbm"):
        m = results[name]
        print(f"{name:14} {m['mae']:7.3f} {m['rmse']:7.3f} {m['per_track_pearson']:7.3f} "
              f"{m['within_bar_spearman']:9.3f} {m['global_std_ratio']:9.3f}")

    # figures
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values()
    ax = imp.tail(20).plot.barh(figsize=(8, 7)); ax.set_title("LightGBM gain importance (top 20)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_importance.png"), dpi=120); plt.close()

    idx = np.random.RandomState(SEED).choice(len(yte), size=min(20000, len(yte)), replace=False)
    plt.figure(figsize=(5, 5)); plt.hexbin(yte.to_numpy()[idx], pred[idx], gridsize=60, cmap="viridis")
    plt.plot([0, 127], [0, 127], "r--", lw=1); plt.xlabel("true"); plt.ylabel("pred")
    plt.title("LightGBM: predicted vs true velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pred_vs_true.png"), dpi=120); plt.close()

    genres = results["lightgbm"]["per_genre_mae"]
    gs = pd.Series(genres).sort_values()
    gs.plot.barh(figsize=(7, 8)); plt.xlabel("MAE"); plt.title("LightGBM per-genre MAE")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_per_genre.png"), dpi=120); plt.close()

    print(f"\nwrote results to {OUT}/")


if __name__ == "__main__":
    main()
