"""First-run engine bootstrap: resolve, download, verify, unpack. Pure stdlib.

ASCII-only, no Reaper API, no __file__ -> unit-tested under pytest and runnable
under Reaper's embedded Python. The network fetcher is injectable so the whole
flow is testable offline (see install()).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import urllib.request
import zipfile

import dn_paths

PINNED_ENGINE_VERSION = "0.1.0"
GITHUB_OWNER_REPO = "yalishanda42/dynamics-needed"


def unify_libomp(engine_dir):
    """macOS: point the redundant libomp.dylib copies at sklearn's single copy.

    Idempotent. No-op if the canonical copy is absent. Uses relative symlinks so
    the tree stays relocatable. Safe to call on any OS (only acts on files present).
    """
    internal = os.path.join(engine_dir, "_internal")
    canonical = os.path.join(internal, "sklearn", ".dylibs", "libomp.dylib")
    if not os.path.isfile(canonical):
        return
    redundant = [
        os.path.join(internal, "libomp.dylib"),
        os.path.join(internal, "torch", "lib", "libomp.dylib"),
    ]
    for path in redundant:
        if not os.path.exists(path) and not os.path.islink(path):
            continue
        rel = os.path.relpath(canonical, os.path.dirname(path))
        if os.path.islink(path) and os.readlink(path) == rel:
            continue  # already unified
        if os.path.lexists(path):
            os.remove(path)
        os.symlink(rel, path)


def release_base_url():
    override = os.environ.get("DN_RELEASE_BASE_URL")
    if override:
        return override
    return "https://github.com/{}/releases/download".format(GITHUB_OWNER_REPO)


def asset_name(version, platform_key):
    return "dn-engine-{}-{}.zip".format(version, platform_key)


def asset_url(base, version, platform_key):
    return "{}/engine-v{}/{}".format(base.rstrip("/"), version, asset_name(version, platform_key))


def sums_url(base, version):
    return "{}/engine-v{}/SHA256SUMS".format(base.rstrip("/"), version)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sum(sums_text, name):
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0]
    return None


def verify(path, sums_text, name):
    want = expected_sum(sums_text, name)
    return want is not None and want == sha256_of(path)


class BootstrapError(Exception):
    pass


def default_fetch(url):
    with urllib.request.urlopen(url) as r:  # nosec - fixed release host / file://
        return r.read()


def is_installed(resource_path, version):
    return os.path.isfile(dn_paths.engine_exe(resource_path, version))


def write_config(resource_path, version, port=8765):
    cfg = {
        "engine_path": dn_paths.engine_exe(resource_path, version),
        "proc_dir": dn_paths.weights_dir(resource_path, version),
        "engine_version": version,
        "port": int(port),
    }
    with open(dn_paths.config_path(resource_path), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def _prune_other_versions(resource_path, keep_version):
    root = dn_paths.engine_root(resource_path)
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        if name != keep_version:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def install(resource_path, version, platform_key, fetch=default_fetch, log=lambda m: None):
    base = release_base_url()
    name = asset_name(version, platform_key)
    dest = dn_paths.engine_dir(resource_path, version)
    tmp = dest + ".part"
    try:
        log("Fetching checksums...")
        sums_text = fetch(sums_url(base, version)).decode()
        log("Downloading engine (~250 MB)...")
        blob = fetch(asset_url(base, version, platform_key))
    except Exception as e:  # network / URL errors
        raise BootstrapError("download failed: {}".format(e))

    zpath = os.path.join(os.path.dirname(dest) or ".", name)
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    with open(zpath, "wb") as fh:
        fh.write(blob)
    if not verify(zpath, sums_text, name):
        os.remove(zpath)
        raise BootstrapError("checksum mismatch for {}".format(name))

    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        os.remove(zpath)
        raise BootstrapError("unpack failed: {}".format(e))
    os.remove(zpath)

    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)

    exe = dn_paths.engine_exe(resource_path, version)
    if os.name != "nt" and os.path.isfile(exe):
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if platform_key.startswith("macos"):
        unify_libomp(dest)

    _prune_other_versions(resource_path, version)
    log("Engine installed.")
    return write_config(resource_path, version)
