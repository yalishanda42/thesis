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


class MdnModel:
    def __init__(self, transformer, genre_vocab, bpm_mean, bpm_std, device="cpu"):
        self.m = transformer.to(device).eval()
        self.genre_vocab = genre_vocab
        self.bpm_mean = float(bpm_mean)
        self.bpm_std = float(bpm_std)
        self.device = device

    @classmethod
    def load(cls, meta_path, ckpt_path, device="cpu"):
        with open(meta_path) as fh:
            meta = json.load(fh)
        gv = meta["genre_vocab"]
        m = VelocityTransformer(n_genres=len(gv) + 1, head="mdn")
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["best_model"])
        return cls(m, gv, meta["bpm_mean"], meta["bpm_std"], device)

    def predict_all(self, df, temperature=1.0, seed=42):
        t = build_split_tensors(df, self.genre_vocab, self.bpm_mean, self.bpm_std)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            raw = self.m(t["voice_idx"].to(self.device), t["genre_idx"].to(self.device),
                         t["num_feats"].to(self.device), t["pad_mask"].to(self.device)).cpu()
        s = heads.sample("mdn", raw, generator=gen, temperature=float(temperature))
        return scatter_predictions(t["row_idx"], s, t["pad_mask"], len(df))


class Engine:
    def __init__(self, lgbm, mdn, styles, genres):
        self.lgbm = lgbm
        self.mdn = mdn
        self.styles = list(styles)
        self.genres = list(genres)

    @classmethod
    def load(cls, proc_dir=os.path.join("data", "processed")):
        lgbm = LgbmModel(joblib.load(os.path.join(proc_dir, "lightgbm_model.joblib")))
        mdn = MdnModel.load(os.path.join(proc_dir, "mdn_meta.json"),
                            os.path.join(proc_dir, "head_mdn.pt"))
        with open(os.path.join(proc_dir, "lightgbm_features.json")) as fh:
            feats = json.load(fh)
        lv = feats["categorical_levels"]
        return cls(lgbm, mdn, lv["style"], lv["genre"])

    def predict(self, request):
        model = request["model"]
        if model == "lgbm":
            predict_all = self.lgbm.predict_all
        elif model == "mdn":
            temp = float(request.get("temperature", 1.0))
            seed = int(request.get("seed", 42))
            predict_all = lambda df: self.mdn.predict_all(df, temperature=temp, seed=seed)  # noqa: E731
        else:
            raise ValueError(f"unknown model {model!r}")
        return predict_velocities(request, predict_all)

    def levels(self):
        return {"models": ["lgbm", "mdn"], "styles": self.styles, "genres": self.genres}
