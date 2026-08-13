#!/usr/bin/env python
"""Publish a trained drum_dynamics model artifact to the Hugging Face Hub.

One model per repo. The model card is uploaded as the repo README; when a
``--metrics`` JSON file is given, ``{{dotted.key}}`` placeholders in the card
are auto-filled from it (nested keys resolved by dotted path).

Examples:
    # LightGBM baseline (auto-filled card)
    python ml/scripts/publish_model.py \\
        --repo <namespace>/dynamics-needed-lgbm \\
        --artifact data/processed/lightgbm_model.joblib \\
        --path-in-repo model.joblib \\
        --card ml/model_cards/lightgbm.md \\
        --metrics <lgbm_out>/metrics.json

    # MDN transformer (auto-filled card)
    python ml/scripts/publish_model.py \\
        --repo <namespace>/dynamics-needed-mdn \\
        --artifact data/processed/transformer_best.pt \\
        --path-in-repo model.pt \\
        --card ml/model_cards/mdn.md \\
        --metrics <mdn_out>/metrics.json

Auth: run `hf auth login` first, or set HF_TOKEN. No namespace is hardcoded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

REPO_TYPE = "model"
# Generic default card; per-model cards live under ml/model_cards/.
MODEL_CARD = Path(__file__).resolve().parent.parent / "model_card.md"

_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _resolve(dotted: str, metrics: dict):
    """Resolve a dotted key (e.g. 'lightgbm.mae') into a nested dict, or None."""
    cur = metrics
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _fmt(value) -> str:
    """Format a metric value for the card (floats to 3 decimals)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_card(template: str, metrics: dict) -> tuple[str, list[str]]:
    """Substitute ``{{dotted.key}}`` placeholders from a (possibly nested)
    metrics dict. Placeholders with no matching key are left intact.

    Returns (rendered_text, unresolved_keys).
    """
    unresolved: list[str] = []

    def _repl(m: re.Match) -> str:
        key = m.group(1)
        value = _resolve(key, metrics)
        if value is None:
            unresolved.append(key)
            return m.group(0)
        return _fmt(value)

    return _PLACEHOLDER.sub(_repl, template), unresolved


def publish(repo_id: str, artifact: str, path_in_repo: str, *,
            private: bool, upload_card: bool, card_path: str,
            metrics_path: str | None) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=artifact,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
    )
    if upload_card:
        text = Path(card_path).read_text()
        if metrics_path:
            metrics = json.loads(Path(metrics_path).read_text())
            text, unresolved = render_card(text, metrics)
            if unresolved:
                print(f"warning: unresolved card placeholders: {sorted(set(unresolved))}")
        api.upload_file(
            path_or_fileobj=text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Publish a drum_dynamics model artifact to the Hugging Face Hub."
    )
    p.add_argument("--repo", required=True, help="HF repo id, e.g. user/dynamics-needed-mdn")
    p.add_argument("--artifact", required=True, help="local path to the weight file")
    p.add_argument("--path-in-repo", required=True, help="destination filename in the repo")
    p.add_argument("--card", default=str(MODEL_CARD),
                   help="model-card template to upload as README (default: ml/model_card.md)")
    p.add_argument("--metrics", default=None,
                   help="metrics JSON whose values fill {{dotted.key}} placeholders in the card")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--no-card", action="store_true",
                   help="skip uploading the model card as the repo README")
    args = p.parse_args()

    if not Path(args.artifact).is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")
    if not args.no_card and not Path(args.card).is_file():
        raise SystemExit(f"card not found: {args.card}")
    if args.metrics and not Path(args.metrics).is_file():
        raise SystemExit(f"metrics not found: {args.metrics}")

    publish(args.repo, args.artifact, args.path_in_repo,
            private=args.private, upload_card=not args.no_card,
            card_path=args.card, metrics_path=args.metrics)
    print(f"published {args.artifact} -> {args.repo}:{args.path_in_repo}")


if __name__ == "__main__":
    main()
