"""Train one probabilistic head on the Plan B backbone (warm-start, resumable)
and evaluate: NLL, deterministic-readout metrics, and sampled-distribution match.

Usage: .venv/bin/python scripts/train_head.py --head {gaussian,mdn,categorical}
       [--epochs N] [--run-epochs N] [--resume] [--eval-only] [--no-warm-start]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import torch                      # noqa: E402
from torch.utils.data import DataLoader, TensorDataset   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drumhumanizer import heads                                                   # noqa: E402
from drumhumanizer.metrics import evaluate, wasserstein1d, hist_intersection      # noqa: E402
from drumhumanizer.model import VelocityTransformer, warm_start_backbone          # noqa: E402
from drumhumanizer.seqdata import (                                               # noqa: E402
    build_genre_vocab, bpm_stats, build_split_tensors, scatter_predictions,
)

PROC = os.path.join("data", "processed")
OUT = os.path.join("docs", "plan_c")
BACKBONE = os.path.join(PROC, "transformer_best.pt")
SEED = 42
KEYS = ["voice_idx", "genre_idx", "num_feats", "target", "pad_mask", "row_idx"]


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _free(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def _load(split):
    return pd.read_parquet(os.path.join(PROC, f"egmd_tabular_{split}.parquet"))


def _loader(t, bs, shuffle):
    return DataLoader(TensorDataset(*[t[k] for k in KEYS]), batch_size=bs, shuffle=shuffle)


def _val_nll(model, loader, head, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            keep = ~pad.to(device)
            lp = heads.logprob(head, raw, target.to(device))
            tot += (-lp[keep]).sum().item()
            n += int(keep.sum().item())
    return tot / n


def _predict(model, loader, head, device, n_rows, seed):
    """Return (point_pred, sampled_pred) aligned to df rows."""
    model.eval()
    point = np.zeros(n_rows)
    samp = np.zeros(n_rows)
    gen = torch.Generator().manual_seed(seed)      # CPU generator for reproducible sampling
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device)).cpu()
            point += scatter_predictions(row, heads.point(head, raw), pad, n_rows)
            samp += scatter_predictions(row, heads.sample(head, raw, generator=gen), pad, n_rows)
    return point, samp


def _test_nlls(model, loader, head, device):
    model.eval()
    nat_tot = dis_tot = 0.0
    n = 0
    with torch.no_grad():
        for voice, genre, num, target, pad, row in loader:
            raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
            keep = ~pad.to(device)
            y = target.to(device)
            nat_tot += (-heads.logprob(head, raw, y)[keep]).sum().item()
            dis_tot += (-heads.bin_logprob(head, raw, y)[keep]).sum().item()
            n += int(keep.sum().item())
    return nat_tot / n, dis_tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=["gaussian", "mdn", "categorical"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--run-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--load-balance", type=float, default=0.0,
                    help="MDN only: weight on the load-balancing (anti-collapse) penalty")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-warm-start", action="store_true")
    args = ap.parse_args()

    head = args.head
    out_dir = os.path.join(OUT, head)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(PROC, f"head_{head}.pt")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    print(f"device: {device}  head: {head}")

    train, val, test = _load("train"), _load("validation"), _load("test")
    genre_vocab = build_genre_vocab(train)
    bpm_mean, bpm_std = bpm_stats(train)
    t0 = time.time()
    tr = build_split_tensors(train, genre_vocab, bpm_mean, bpm_std)
    va = build_split_tensors(val, genre_vocab, bpm_mean, bpm_std)
    te = build_split_tensors(test, genre_vocab, bpm_mean, bpm_std)
    print(f"windows tr/va/te: {tr['target'].shape[0]}/{va['target'].shape[0]}/{te['target'].shape[0]} "
          f"({time.time()-t0:.0f}s)")
    tr_l = _loader(tr, args.batch_size, True)
    va_l = _loader(va, args.batch_size, False)
    te_l = _loader(te, args.batch_size, False)

    model = VelocityTransformer(n_genres=len(genre_vocab) + 1, head=head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_epoch, curve = -1, []
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device)["best_model"])
    else:
        best_val, bad, start, best_state = float("inf"), 0, 0, None
        if args.resume and os.path.exists(ckpt):
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            best_val, best_epoch = ck["best_val"], ck["best_epoch"]
            start, curve, bad, best_state = ck["epochs_done"], list(ck["curve"]), ck["bad"], ck["best_model"]
            print(f"resumed at epoch {start} (best_val {best_val:.4f})")
        elif not args.no_warm_start and os.path.exists(BACKBONE):
            missing, unexpected = warm_start_backbone(model, BACKBONE)
            print(f"warm-started backbone from {BACKBONE} (fresh: {list(missing)})")

        ran = 0
        for epoch in range(start + 1, args.epochs + 1):
            if ran >= args.run_epochs:
                print("run-epochs budget reached")
                break
            model.train()
            te0 = time.time()
            for voice, genre, num, target, pad, row in tr_l:
                opt.zero_grad()
                raw = model(voice.to(device), genre.to(device), num.to(device), pad.to(device))
                pad_d = pad.to(device)
                loss = heads.nll(head, raw, target.to(device), pad_d)
                if head == "mdn" and args.load_balance > 0:
                    loss = loss + args.load_balance * heads.mdn_load_balance(raw, pad_d)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # MDN stability
                opt.step()
            _free(device)
            val_nll = _val_nll(model, va_l, head, device)
            _free(device)
            curve.append(val_nll)
            ran += 1
            print(f"epoch {epoch:2d}  val_nll {val_nll:.4f}  ({time.time()-te0:.0f}s)")
            if val_nll < best_val - 1e-4:
                best_val, best_epoch, bad = val_nll, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(), "best_model": best_state, "best_val": best_val,
                        "best_epoch": best_epoch, "epochs_done": epoch, "curve": curve, "bad": bad}, ckpt)
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
        if best_state is not None:
            model.load_state_dict(best_state)

    # ── evaluate ──────────────────────────────────────────────────────────
    native_nll, discretized_nll = _test_nlls(model, te_l, head, device)
    point, samp = _predict(model, te_l, head, device, len(test), SEED)
    true = test["velocity"].to_numpy(float)
    results = {
        "native_nll": native_nll,
        "discretized_nll": discretized_nll,
        "deterministic_readout": evaluate(test, point),
        "sampled": evaluate(test, samp),
        "sampled_wasserstein": wasserstein1d(samp, true),
        "sampled_hist_intersection": hist_intersection(samp, true),
        "best_epoch": best_epoch,
        "val_nll_curve": curve,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[{head}] native_nll {native_nll:.4f}  discretized_nll {discretized_nll:.4f}")
    print(f"  point   : MAE {results['deterministic_readout']['mae']:.3f}  "
          f"std_ratio {results['deterministic_readout']['global_std_ratio']:.3f}")
    print(f"  sampled : MAE {results['sampled']['mae']:.3f}  "
          f"std_ratio {results['sampled']['global_std_ratio']:.3f}  "
          f"W1 {results['sampled_wasserstein']:.3f}  histI {results['sampled_hist_intersection']:.3f}")

    # figures
    plt.figure(figsize=(7, 4))
    plt.hist(true, bins=32, range=(0, 128), alpha=0.5, density=True, label="true")
    plt.hist(samp, bins=32, range=(0, 128), alpha=0.5, density=True, label=f"{head} sampled")
    plt.xlabel("velocity"); plt.ylabel("density"); plt.legend()
    plt.title(f"{head}: sampled vs true velocity distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_hist.png"), dpi=120); plt.close()

    if curve:
        plt.figure(figsize=(6, 4)); plt.plot(range(1, len(curve) + 1), curve, marker="o")
        plt.xlabel("epoch"); plt.ylabel("val NLL"); plt.title(f"{head}: validation NLL")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_val_nll.png"), dpi=120); plt.close()

    print(f"wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
