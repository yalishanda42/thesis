"""Plan E — interpretability & error analysis for the categorical model.

I1 embedding geometry (genre + voice), I2 learned drumming phenomena (backbeat,
metrical hierarchy, ghost notes, per-voice dynamic range), E1 residual
decomposition (voice, dynamic level, metrical position).

Runs inference once on the in-distribution test split (point + one sampled draw
per note), caches the merged predictions to parquet, then writes figures + a
results summary. Re-run with --refresh to recompute predictions.

Usage: .venv/bin/python scripts/analyze_model.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import torch                      # noqa: E402
from torch.utils.data import DataLoader, TensorDataset   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer import analysis, heads                                          # noqa: E402
from drumhumanizer.model import VelocityTransformer                                # noqa: E402
from drumhumanizer.seqdata import build_split_tensors, scatter_predictions         # noqa: E402
from drumhumanizer.voicemap import CANONICAL_VOICES                                # noqa: E402

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_e")
CACHE = os.path.join(PROC, "preds_categorical_test.parquet")
HEAD_CKPT = os.path.join(PROC, "head_categorical.pt")
META = os.path.join(PROC, "transformer_meta.json")
SEED = 42
KEYS = ["voice_idx", "genre_idx", "num_feats", "target", "pad_mask", "row_idx"]
KEEP = ["file_id", "voice", "genre", "velocity", "phase_beat", "phase_bar",
        "nearest_subdiv", "bar_index"]


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _load_model(meta, device):
    model = VelocityTransformer(n_genres=len(meta["genre_vocab"]) + 1, head="categorical").to(device)
    model.load_state_dict(torch.load(HEAD_CKPT, map_location=device)["best_model"])
    model.eval()
    return model


def _predict(model, df, meta, device):
    te = build_split_tensors(df, meta["genre_vocab"], meta["bpm_mean"], meta["bpm_std"])
    dl = DataLoader(TensorDataset(*[te[k] for k in KEYS]), batch_size=64)
    n = len(df)
    point = np.zeros(n)
    samp = np.zeros(n)
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for v, g, num, t, p, r in dl:
            raw = model(v.to(device), g.to(device), num.to(device), p.to(device)).cpu()
            point += scatter_predictions(r, heads.point("categorical", raw), p, n)
            samp += scatter_predictions(r, heads.sample("categorical", raw, generator=gen), p, n)
    return point, samp


def _get_predictions(args, meta, device):
    if os.path.exists(CACHE) and not args.refresh:
        print(f"loading cached predictions from {CACHE}")
        return pd.read_parquet(CACHE)
    print("running inference on test split ...")
    test = pd.read_parquet(os.path.join(PROC, "egmd_tabular_test.parquet"))
    model = _load_model(meta, device)
    point, samp = _predict(model, test, meta, device)
    df = test[KEEP].copy()
    df["pred"] = point
    df["pred_sampled"] = samp
    df = analysis.add_metrical_cols(df)
    df.to_parquet(CACHE, index=False)
    print(f"cached {len(df):,} rows to {CACHE}")
    return df


# ── figures ───────────────────────────────────────────────────────────────────
def _fig_embeddings(meta, device):
    model = _load_model(meta, device)
    inv_genre = {i: g for g, i in meta["genre_vocab"].items()}     # row idx -> genre
    df = pd.read_parquet(CACHE, columns=["genre", "voice", "velocity"])
    gvel = df.groupby("genre")["velocity"].mean()
    vvel = df.groupby("voice")["velocity"].mean()

    # genre embedding (drop row 0 = <unk>)
    gw = model.genre_emb.weight.detach().cpu().numpy()
    rows = [i for i in range(1, gw.shape[0]) if i in inv_genre]
    labels = [inv_genre[i] for i in rows]
    xy = analysis.embedding_2d(gw[rows])
    col = np.array([gvel.get(g, np.nan) for g in labels])
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=col, cmap="coolwarm", s=90, edgecolor="k")
    for (x, y), lab in zip(xy, labels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(sc, label="mean true velocity")
    plt.title("Genre embedding (PCA 2D), colored by mean velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_genre_embedding.png"), dpi=120); plt.close()

    # voice embedding (all 14 canonical voices)
    vw = model.voice_emb.weight.detach().cpu().numpy()
    vlabels = CANONICAL_VOICES
    xy = analysis.embedding_2d(vw)
    col = np.array([vvel.get(v, np.nan) for v in vlabels])
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=col, cmap="coolwarm", s=90, edgecolor="k")
    for (x, y), lab in zip(xy, vlabels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(sc, label="mean true velocity")
    plt.title("Voice embedding (PCA 2D), colored by mean velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_voice_embedding.png"), dpi=120); plt.close()


def _fig_metrical(df):
    tab = df.groupby("metrical_class").agg(
        true_mean=("velocity", "mean"), pred_mean=("pred", "mean"), n=("velocity", "size"))
    plt.figure(figsize=(5, 4))
    x = np.arange(len(tab)); w = 0.4
    plt.bar(x - w / 2, tab["true_mean"], w, label="true")
    plt.bar(x + w / 2, tab["pred_mean"], w, label="pred")
    plt.xticks(x, tab.index); plt.ylabel("mean velocity"); plt.legend()
    plt.title("Metrical hierarchy: on-beat louder than off-beat")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_metrical.png"), dpi=120); plt.close()
    return tab


def _fig_ghost(df, thresh=40):
    """Plain snare is bimodal (soft ghosts + normal hits). Show that sampling
    restores the ghost population that the point readout washes out."""
    snare = df[df["voice"] == "snare"]
    frac = {"true": float((snare["velocity"] < thresh).mean()),
            "sampled": float((snare["pred_sampled"] < thresh).mean()),
            "point": float((snare["pred"] < thresh).mean())}
    plt.figure(figsize=(7, 4))
    b = np.linspace(0, 127, 40)
    plt.hist(snare["velocity"], bins=b, density=True, alpha=0.5, label=f"true (ghosts {frac['true']*100:.0f}%)")
    plt.hist(snare["pred_sampled"], bins=b, density=True, alpha=0.5, label=f"pred sampled ({frac['sampled']*100:.0f}%)")
    plt.hist(snare["pred"], bins=b, density=True, histtype="step", color="k", lw=1.5, label=f"pred point ({frac['point']*100:.0f}%)")
    plt.axvline(thresh, color="gray", ls=":", lw=1)
    plt.xlabel("snare velocity"); plt.ylabel("density"); plt.legend()
    plt.title(f"Ghost notes: point readout washes them out, sampling restores them\n(% = share below velocity {thresh})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_ghost_notes.png"), dpi=120); plt.close()
    return frac


def _fig_voice_range(df):
    r = analysis.residual_table(df, "voice").sort_values("true_mean")
    y = np.arange(len(r))
    plt.figure(figsize=(8, 7))
    plt.errorbar(r["true_mean"], y - 0.15, xerr=r["true_std"], fmt="o", label="true", capsize=3)
    plt.errorbar(r["pred_mean"], y + 0.15, xerr=r["pred_std"], fmt="s", label="pred", capsize=3)
    plt.yticks(y, r.index); plt.xlabel("velocity (mean ± std)"); plt.legend()
    plt.title("Per-voice dynamic range (true vs pred)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_voice_range.png"), dpi=120); plt.close()
    return r


def _fig_residual_by_voice(r):
    r = r.sort_values("mae")
    plt.figure(figsize=(8, 7))
    plt.barh(np.arange(len(r)), r["mae"])
    plt.yticks(np.arange(len(r)), r.index); plt.xlabel("MAE")
    plt.title("Error by voice (in-distribution test)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_residual_by_voice.png"), dpi=120); plt.close()


def _fig_dynamic_regression(df):
    tp = analysis.dynamic_level_table(df, pred_col="pred", n_bins=12)
    ts = analysis.dynamic_level_table(df, pred_col="pred_sampled", n_bins=12)
    ctr = (tp["bin_lo"] + tp["bin_hi"]) / 2
    plt.figure(figsize=(6.5, 4.5))
    plt.axhline(0, color="gray", lw=1)
    plt.plot(ctr, tp["bias"], "o-", label="point readout")
    plt.plot((ts["bin_lo"] + ts["bin_hi"]) / 2, ts["bias"], "s-", label="sampled readout")
    plt.xlabel("true velocity bin center"); plt.ylabel("bias (pred − true)")
    plt.title("Regression to the mean: soft over-, loud under-predicted")
    plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_dynamic_regression.png"), dpi=120); plt.close()
    return tp


def _fig_residual_by_metrical(df):
    """Error by metrical position: on/off-beat and by nearest subdivision."""
    mc = analysis.residual_table(df, "metrical_class")
    sub = analysis.residual_table(df, "nearest_subdiv").sort_values("mae")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(np.arange(len(mc)), mc["mae"]); ax[0].set_xticks(np.arange(len(mc)))
    ax[0].set_xticklabels(mc.index); ax[0].set_ylabel("MAE"); ax[0].set_title("Error by on/off-beat")
    ax[1].barh(np.arange(len(sub)), sub["mae"]); ax[1].set_yticks(np.arange(len(sub)))
    ax[1].set_yticklabels(sub.index); ax[1].set_xlabel("MAE"); ax[1].set_title("Error by nearest subdivision")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_residual_by_metrical.png"), dpi=120); plt.close()
    return mc, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute predictions (ignore cache)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = _device()
    meta = json.load(open(META))
    print(f"device: {device}")

    df = _get_predictions(args, meta, device)

    print("I1: embeddings")
    _fig_embeddings(meta, device)
    print("I2: learned phenomena")
    metr = _fig_metrical(df)
    ghost = _fig_ghost(df)
    vrange = _fig_voice_range(df)
    print("E1: residual decomposition")
    _fig_residual_by_voice(vrange)
    dyn = _fig_dynamic_regression(df)
    mc, sub = _fig_residual_by_metrical(df)

    summary = {
        "n_test_notes": int(len(df)),
        "metrical_class": metr.round(2).to_dict(),
        "ghost_note_fraction_below_40": {k: round(v, 3) for k, v in ghost.items()},
        "voice_residuals": vrange[["n", "true_mean", "pred_mean", "true_std", "pred_std", "mae", "bias"]].round(2).to_dict(orient="index"),
        "dynamic_bias_point": dyn[["bin_lo", "bin_hi", "bias", "mae", "n"]].round(2).to_dict(orient="index"),
        "mae_by_metrical_class": mc["mae"].round(2).to_dict(),
        "mae_by_subdivision": sub["mae"].round(2).to_dict(),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nmetrical: on {metr.loc['on-beat','true_mean']:.1f}/{metr.loc['on-beat','pred_mean']:.1f}  "
          f"off {metr.loc['off-beat','true_mean']:.1f}/{metr.loc['off-beat','pred_mean']:.1f}")
    print(f"ghosts (<40): true {ghost['true']:.2f}  sampled {ghost['sampled']:.2f}  point {ghost['point']:.2f}")
    print(f"wrote figures + metrics to {OUT}/")


if __name__ == "__main__":
    main()
