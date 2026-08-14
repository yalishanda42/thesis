"""First-run engine bootstrap: resolve, download, verify, unpack. Pure stdlib.

ASCII-only, no Reaper API, no __file__ -> unit-tested under pytest and runnable
under Reaper's embedded Python. The network fetcher is injectable so the whole
flow is testable offline (see install()).
"""
from __future__ import annotations

import hashlib
import os

PINNED_ENGINE_VERSION = "0.1.0"
GITHUB_OWNER_REPO = "yalishanda42/dynamics-needed"


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
