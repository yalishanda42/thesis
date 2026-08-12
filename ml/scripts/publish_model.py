#!/usr/bin/env python
"""Publish a trained drum_dynamics model artifact to the Hugging Face Hub.

Example:
    python ml/scripts/publish_model.py \\
        --repo <namespace>/dynamics-needed \\
        --artifact data/processed/transformer_best.pt \\
        --path-in-repo model.pt

Auth: run `hf auth login` first, or set HF_TOKEN. No namespace is hardcoded.
On the first run the model card (ml/model_card.md) is uploaded as the repo README.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_TYPE = "model"
MODEL_CARD = Path(__file__).resolve().parent.parent / "model_card.md"


def publish(repo_id: str, artifact: str, path_in_repo: str, *,
            private: bool, upload_card: bool) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=artifact,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
    )
    if upload_card and MODEL_CARD.is_file():
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Publish a drum_dynamics model artifact to the Hugging Face Hub."
    )
    p.add_argument("--repo", required=True, help="HF repo id, e.g. user/dynamics-needed")
    p.add_argument("--artifact", required=True, help="local path to the weight file")
    p.add_argument("--path-in-repo", required=True, help="destination filename in the repo")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--no-card", action="store_true",
                   help="skip uploading model_card.md as the repo README")
    args = p.parse_args()

    if not Path(args.artifact).is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")

    publish(args.repo, args.artifact, args.path_in_repo,
            private=args.private, upload_card=not args.no_card)
    print(f"published {args.artifact} -> {args.repo}:{args.path_in_repo}")


if __name__ == "__main__":
    main()
