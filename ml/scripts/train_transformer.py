"""Train the event-sequence Transformer on the Section B features and evaluate
on the test split against Plan A's baselines.

Usage: .venv/bin/python ml/scripts/train_transformer.py [--epochs N] [--batch-size B]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from drum_dynamics.eval.metrics import evaluate
from drum_dynamics.models.model import VelocityTransformer
from drum_dynamics.data.seqdata import (
    build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions,
)

PROC = os.path.join("data", "processed")
PLAN_A = os.path.join("docs", "plan_a", "metrics.json")
SEED = 42
KEYS = ["voice_idx", "genre_idx", "num_feats", "target", "pad_mask", "row_idx"]


def _free(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(split, stem="tabular"):
    return pd.read_parquet(os.path.join(PROC, f"egmd_{stem}_{split}.parquet"))


def _loader(tensors, batch_size, shuffle):
    ds = TensorDataset(*[tensors[k] for k in KEYS])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _masked_mae(pred, target, pad_mask):
    m = ~pad_mask
    return (pred[m] - target[m]).abs().mean()


def _predict(model, loader, device, n_rows):
    model.eval()
    preds = np.zeros(n_rows, dtype=np.float64)
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            preds += scatter_predictions(row, y.cpu(), pad, n_rows)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15, help="total target epochs")
    ap.add_argument("--run-epochs", type=int, default=15,
                    help="max epochs to run in THIS invocation (for chunked/foreground runs)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="resume model+optimizer+progress from the checkpoint")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load the best checkpoint and evaluate")
    ap.add_argument("--tag", default="",
                    help="dataset stem + ckpt/meta suffix, e.g. 'holdout' reads "
                         "egmd_holdout_*.parquet and writes transformer_best_holdout.pt")
    ap.add_argument("--out", default="",
                    help="results dir (default docs/plan_b, or docs/plan_b_<tag>)")
    args = ap.parse_args()

    stem = args.tag or "tabular"
    suffix = f"_{args.tag}" if args.tag else ""
    OUT = args.out or (os.path.join("docs", "plan_b") if not args.tag
                       else os.path.join("docs", f"plan_b_{args.tag}"))
    CKPT = os.path.join(PROC, f"transformer_best{suffix}.pt")   # gitignored (under data/)
    META = os.path.join(PROC, f"transformer_meta{suffix}.json")

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    print(f"device: {device}  tag: {args.tag or '(default)'}  data stem: egmd_{stem}_*")

    train, val, test = _load("train", stem), _load("validation", stem), _load("test", stem)
    genre_vocab = build_genre_vocab(train)
    bpm_mean, bpm_std = bpm_stats(train)
    print(f"genres: {len(genre_vocab)}  bpm mean/std: {bpm_mean:.1f}/{bpm_std:.1f}")

    # sidecar so downstream inference (e.g. the audition notebook) can rebuild the
    # exact train-fit genre vocab + bpm normalization without re-reading the parquet.
    with open(META, "w") as fh:
        json.dump({"genre_vocab": genre_vocab, "bpm_mean": bpm_mean, "bpm_std": bpm_std}, fh, indent=2)

    t0 = time.time()
    tr_t = build_split_tensors(train, genre_vocab, bpm_mean, bpm_std)
    va_t = build_split_tensors(val, genre_vocab, bpm_mean, bpm_std)
    te_t = build_split_tensors(test, genre_vocab, bpm_mean, bpm_std)
    print(f"windows train/val/test: {tr_t['target'].shape[0]}/"
          f"{va_t['target'].shape[0]}/{te_t['target'].shape[0]}  "
          f"({time.time()-t0:.0f}s to build)")

    tr_loader = _loader(tr_t, args.batch_size, shuffle=True)
    va_loader = _loader(va_t, args.batch_size, shuffle=False)
    te_loader = _loader(te_t, args.batch_size, shuffle=False)

    model = VelocityTransformer(n_genres=len(genre_vocab) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def _cpu_state(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    best_epoch, curve = -1, []
    if args.eval_only:
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["best_model"])
        best_epoch, curve = ck.get("best_epoch", -1), list(ck.get("curve", []))
        print(f"eval-only: loaded best_model from {CKPT} (best_epoch {best_epoch})")
    else:
        best_val, bad, start = float("inf"), 0, 0
        best_state = None
        if args.resume and os.path.exists(CKPT):
            ck = torch.load(CKPT, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            best_val, best_epoch = ck["best_val"], ck["best_epoch"]
            start, curve, bad, best_state = ck["epochs_done"], list(ck["curve"]), ck["bad"], ck["best_model"]
            print(f"resumed at epoch {start} (best_val {best_val:.4f}, best_epoch {best_epoch})")

        ran = 0
        for epoch in range(start + 1, args.epochs + 1):
            if ran >= args.run_epochs:
                print(f"run-epochs budget ({args.run_epochs}) reached; stopping this invocation")
                break
            model.train()
            te0 = time.time()
            for voice, genre, num, target, pad, row in tr_loader:
                opt.zero_grad()
                y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
                loss = _masked_mae(y, target.to(device), pad.to(device))
                loss.backward()
                opt.step()
            _free(device)
            # validation MAE
            model.eval()
            with torch.no_grad():
                num_sum, den = 0.0, 0
                for voice, genre, num, target, pad, row in va_loader:
                    y = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
                    m = ~pad.to(device)
                    num_sum += (y[m] - target.to(device)[m]).abs().sum().item()
                    den += int(m.sum().item())
                val_mae = num_sum / den
            _free(device)
            curve.append(val_mae)
            ran += 1
            print(f"epoch {epoch:2d}  val_mae {val_mae:.4f}  ({time.time()-te0:.0f}s)")
            if val_mae < best_val - 1e-4:
                best_val, best_epoch, bad = val_mae, epoch, 0
                best_state = _cpu_state(model)
            else:
                bad += 1
            # full checkpoint every epoch (survives external kills)
            torch.save({
                "model": _cpu_state(model), "opt": opt.state_dict(),
                "best_model": best_state, "best_val": best_val, "best_epoch": best_epoch,
                "epochs_done": epoch, "curve": curve, "bad": bad,
            }, CKPT)
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

    # evaluate on test with the shared harness
    pred = _predict(model, te_loader, device, len(test))
    results = {
        "transformer": evaluate(test, pred),
        "transformer_best_epoch": best_epoch,
        "transformer_val_mae_curve": curve,
    }
    if os.path.exists(PLAN_A):
        with open(PLAN_A) as fh:
            a = json.load(fh)
        results["baselines"] = {k: a[k] for k in ("global_mean", "lookup_table", "lightgbm") if k in a}

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # comparison table
    rows = []
    if "baselines" in results:
        for name in ("global_mean", "lookup_table", "lightgbm"):
            if name in results["baselines"]:
                rows.append((name, results["baselines"][name]))
    rows.append(("transformer", results["transformer"]))
    print(f"\n{'model':14} {'MAE':>7} {'RMSE':>7} {'trk_r':>7} {'wbar_rho':>9} {'std_ratio':>9}")
    for name, m in rows:
        print(f"{name:14} {m['mae']:7.3f} {m['rmse']:7.3f} {m['per_track_pearson']:7.3f} "
              f"{m['within_bar_spearman']:9.3f} {m['global_std_ratio']:9.3f}")

    # figures
    if curve:
        plt.figure(figsize=(6, 4)); plt.plot(range(1, len(curve) + 1), curve, marker="o")
        plt.xlabel("epoch"); plt.ylabel("val MAE"); plt.title("Transformer validation MAE")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_training_curve.png"), dpi=120); plt.close()

    yte = test["velocity"].to_numpy()
    idx = np.random.RandomState(SEED).choice(len(yte), size=min(20000, len(yte)), replace=False)
    plt.figure(figsize=(5, 5)); plt.hexbin(yte[idx], pred[idx], gridsize=60, cmap="viridis")
    plt.plot([0, 127], [0, 127], "r--", lw=1); plt.xlabel("true"); plt.ylabel("pred")
    plt.title("Transformer: predicted vs true velocity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pred_vs_true.png"), dpi=120); plt.close()

    gs = pd.Series(results["transformer"]["per_genre_mae"]).sort_values()
    gs.plot.barh(figsize=(7, 8)); plt.xlabel("MAE"); plt.title("Transformer per-genre MAE")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_per_genre.png"), dpi=120); plt.close()

    print(f"\nwrote results to {OUT}/")


if __name__ == "__main__":
    main()
