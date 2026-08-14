"""Freeze the engine, stage weights, (mac) ad-hoc sign, zip, checksum.

Run from repo root: .venv/bin/python ml/packaging/build_engine.py --version 0.1.0
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "plugin", "reaper"))
import bootstrap  # noqa: E402

DIST = os.path.join(HERE, "dist")
WEIGHT_FILES = ["lightgbm_model.joblib", "lightgbm_features.json", "mdn_meta.json",
                "head_mdn.pt", "transformer_meta.json", "head_categorical.pt"]


def platform_key():
    system = platform.system(); mach = platform.machine().lower()
    if system == "Darwin":
        return "macos-arm64" if mach in ("arm64", "aarch64") else "macos-x86_64"
    if system == "Windows":
        return "windows-x64"
    return "linux-x64"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--weights-dir", default=os.path.join("data", "processed"))
    p.add_argument("--sign", dest="sign", action="store_true", default=(platform.system() == "Darwin"))
    p.add_argument("--no-sign", dest="sign", action="store_false")
    args = p.parse_args()

    shutil.rmtree(DIST, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--distpath", DIST, "--workpath", os.path.join(HERE, "build"),
                    os.path.join(HERE, "dn_engine.spec")], check=True)

    engine = os.path.join(DIST, "dn-engine")
    weights_out = os.path.join(engine, "weights")
    os.makedirs(weights_out, exist_ok=True)
    for f in WEIGHT_FILES:
        shutil.copy(os.path.join(args.weights_dir, f), os.path.join(weights_out, f))

    if args.sign and platform.system() == "Darwin":
        subprocess.run(["codesign", "-s", "-", "--deep", "--force",
                        os.path.join(engine, "dn-engine")], check=True)

    if platform.system() == "Darwin":
        bootstrap.unify_libomp(engine)
        for orig in glob.glob(os.path.join(engine, "_internal", "**", "*.dylib.orig"), recursive=True):
            os.remove(orig)

    pk = platform_key()
    zip_name = "dn-engine-{}-{}.zip".format(args.version, pk)
    zip_path = os.path.join(DIST, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(engine):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, engine))

    h = hashlib.sha256()
    with open(zip_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with open(os.path.join(DIST, "SHA256SUMS"), "a") as fh:
        fh.write("{}  {}\n".format(h.hexdigest(), zip_name))
    print("built", zip_path)


if __name__ == "__main__":
    main()
