"""Model wrappers (LightGBM, MDN) + the Engine that routes requests to them."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch

from .core import predict_velocities
from ..models import heads
from ..models.model import VelocityTransformer
from ..data.seqdata import build_split_tensors, scatter_predictions


class LgbmModel:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.cat = list(bundle["cat"])
        self.drop = list(bundle["drop"])
        self.cat_categories = bundle["cat_categories"]
        self.best_iteration = int(bundle["best_iteration"])

    def predict_all(self, df):
        X = df.drop(columns=self.drop)
        for c in self.cat:
            X[c] = X[c].astype("category").cat.set_categories(self.cat_categories[c])
        return np.asarray(self.model.predict(X, num_iteration=self.best_iteration), dtype=float)


class TransformerModel:
    """Serves the shared transformer backbone with a sampling head.

    head_type selects the output head + sampler: "mdn" (temperature-controlled
    mixture) or "categorical" (softmax over velocity bins; temperature ignored
    by heads.sample). Both condition on genre only (not style).
    """
    def __init__(self, transformer, genre_vocab, bpm_mean, bpm_std, device="cpu", head_type="mdn"):
        self.m = transformer.to(device).eval()
        self.genre_vocab = genre_vocab
        self.bpm_mean = float(bpm_mean)
        self.bpm_std = float(bpm_std)
        self.device = device
        self.head_type = head_type

    @classmethod
    def load(cls, meta_path, ckpt_path, device="cpu", head_type="mdn"):
        with open(meta_path) as fh:
            meta = json.load(fh)
        gv = meta["genre_vocab"]
        m = VelocityTransformer(n_genres=len(gv) + 1, head=head_type)
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["best_model"])
        return cls(m, gv, meta["bpm_mean"], meta["bpm_std"], device, head_type)

    def predict_all(self, df, temperature=1.0, seed=42):
        t = build_split_tensors(df, self.genre_vocab, self.bpm_mean, self.bpm_std)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            raw = self.m(t["voice_idx"].to(self.device), t["genre_idx"].to(self.device),
                         t["num_feats"].to(self.device), t["pad_mask"].to(self.device)).cpu()
        s = heads.sample(self.head_type, raw, generator=gen, temperature=float(temperature))
        return scatter_predictions(t["row_idx"], s, t["pad_mask"], len(df))


MdnModel = TransformerModel   # backward-compatible alias (head_type defaults to "mdn")


class Engine:
    def __init__(self, lgbm, mdn, cat, styles, genres):
        self.lgbm = lgbm
        self.mdn = mdn
        self.cat = cat
        self.styles = list(styles)
        self.genres = list(genres)

    @classmethod
    def load(cls, proc_dir=os.path.join("data", "processed")):
        lgbm = LgbmModel(joblib.load(os.path.join(proc_dir, "lightgbm_model.joblib")))
        mdn = TransformerModel.load(os.path.join(proc_dir, "mdn_meta.json"),
                                    os.path.join(proc_dir, "head_mdn.pt"), head_type="mdn")
        cat = TransformerModel.load(os.path.join(proc_dir, "transformer_meta.json"),
                                    os.path.join(proc_dir, "head_categorical.pt"),
                                    head_type="categorical")
        with open(os.path.join(proc_dir, "lightgbm_features.json")) as fh:
            feats = json.load(fh)
        lv = feats["categorical_levels"]
        return cls(lgbm, mdn, cat, lv["style"], lv["genre"])

    def predict(self, request):
        model = request["model"]
        seed = int(request.get("seed", 42))
        if model == "lgbm":
            predict_all = self.lgbm.predict_all
        elif model == "mdn":
            temp = float(request.get("temperature", 1.0))
            predict_all = lambda df: self.mdn.predict_all(df, temperature=temp, seed=seed)  # noqa: E731
        elif model == "categorical":
            predict_all = lambda df: self.cat.predict_all(df, seed=seed)  # noqa: E731 (temperature n/a)
        else:
            raise ValueError(f"unknown model {model!r}")
        return predict_velocities(request, predict_all)

    def levels(self):
        return {"models": ["lgbm", "mdn", "categorical"], "styles": self.styles, "genres": self.genres}
