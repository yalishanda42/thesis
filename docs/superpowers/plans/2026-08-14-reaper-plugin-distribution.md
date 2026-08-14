# Reaper Plugin Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship "Dynamics Needed" as a public Reaper tool installable via ReaPack, with a bundled per-OS inference engine downloaded on first run from GitHub Releases.

**Architecture:** A tiny ReaScript package (installed by ReaPack) that, on first launch, downloads a matching PyInstaller-frozen engine (torch + LightGBM + all 3 models + weights) from a GitHub Release, verifies its SHA-256, unpacks it under Reaper's resource path, and spawns it exactly like the current dev engine (`127.0.0.1:<port>`, `/health` + `/predict`). All new client-side logic is pure-stdlib so it is unit-tested under pytest and runs under Reaper's embedded Python. The freeze + publish is automated by a GitHub Actions matrix; distribution rides on GitHub **Releases** (not Actions artifacts).

**Tech Stack:** Python 3.12, stdlib (`urllib`, `zipfile`, `hashlib`, `subprocess`), PyInstaller (one-folder), LightGBM, PyTorch, ReaImGui (ReaScript UI, unchanged), ReaPack (`index.xml`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-14-reaper-plugin-distribution-design.md`

## Global Constraints

- **Python:** `>=3.12` (matches `ml/pyproject.toml`).
- **Client modules are pure-stdlib** (`engine_client.py`, `dn_core.py`, `predict_worker.py`, and the new `dn_paths.py`, `bootstrap.py`): no third-party imports, no Reaper API — so they import cleanly under pytest and under Reaper's embedded Python.
- **ReaScript files are ASCII-only** and must not rely on `__file__` (Reaper defines neither `__file__` nor `get_action_context()` for Python). Self-location is derived from `RPR_GetResourcePath()` + fixed repo/category constants.
- **Models shipped:** all three (`lgbm`, `mdn`, `categorical`) — torch stays in the freeze.
- **Weights bundled (6 files, ~26 MB):** `lightgbm_model.joblib`, `lightgbm_features.json`, `mdn_meta.json`, `head_mdn.pt`, `transformer_meta.json`, `head_categorical.pt` (the exact set `Engine.load()` reads). Source of truth: `data/processed/`.
- **Freeze:** PyInstaller **one-folder**, name `dn-engine`, per platform `macos-arm64 | macos-x86_64 | windows-x64 | linux-x64`. Exclude `partitura`, `music21`, `pyfluidsynth`, `matplotlib`, `pyarrow`, `jupyter`, `ipykernel`, `nbconvert`.
- **Distribution channel:** GitHub **Release** assets (≤2 GiB/file, uncapped total, no expiry). Never Actions artifacts for user downloads.
- **macOS:** ad-hoc sign the freeze in CI (`codesign -s - --deep --force`); bootstrap downloads programmatically via `urllib` (no quarantine). Restore the exec bit after unzip on POSIX.
- **Repo/release identity (fixed once, hard to rename later):** GitHub `OWNER/REPO`, ReaPack repo name `Dynamics Needed`, category `MIDI Editor`, engine release tag pattern `engine-vX.Y.Z`, engine zip name `dn-engine-<version>-<platform_key>.zip`. **Confirm `OWNER/REPO` before Task 3** (used as a literal). This plan uses the placeholder `OWNER/REPO` — replace globally at Task 3.
- **Run tests from repo root:** `.venv/bin/python -m pytest ml/tests/ -v`.

---

## File Structure

**New (client, pure-stdlib):**
- `plugin/reaper/dn_paths.py` — platform detection + all resource-path-derived locations.
- `plugin/reaper/bootstrap.py` — pinned version, URL/asset naming, SHA-256 verify, download→verify→unpack→prune orchestration, config writer.

**Modified (client):**
- `plugin/reaper/engine_client.py` — dual-mode `start_engine` (frozen vs dev).
- `plugin/reaper/dynamics_needed.py` — self-location + first-run bootstrap wiring + new config; ASCII-only.
- `plugin/reaper/setup_reaper.py` — becomes the dev-mode config writer (legacy `{venv_python, repo_root, port}` shape).

**New (packaging):**
- `ml/packaging/run_engine.py` — PyInstaller entry (calls `drum_dynamics.serve.__main__.main`).
- `ml/packaging/dn_engine.spec` — PyInstaller spec (hiddenimports + excludes).
- `ml/packaging/build_engine.py` — local build wrapper: freeze → stage weights → ad-hoc sign (mac) → zip → SHA256SUMS.
- `ml/packaging/smoke_test.py` — launch an unpacked engine, hit `/health` + `/predict`, assert (reused by CI and locally).

**New (distribution):**
- `plugin/reaper/reapack/make_index.py` — generate ReaPack `index.xml`.
- `.github/workflows/engine-release.yml` — build/sign/smoke/publish matrix + index publish.

**New (tests):**
- `ml/tests/test_dn_paths.py`, `ml/tests/test_bootstrap.py`, `ml/tests/test_make_index.py`.
- `ml/tests/test_engine_client.py` — update the existing spawn test for dual-mode.

**Modified (docs):**
- `README.md` — user install (ReaPack) + dev/build sections.

---

## Task 1: `dn_paths` — platform + path resolution

**Files:**
- Create: `plugin/reaper/dn_paths.py`
- Test: `ml/tests/test_dn_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `platform_key() -> str` in `{"macos-arm64","macos-x86_64","windows-x64","linux-x64"}`
  - `app_root(resource_path) -> str`, `engine_root(resource_path) -> str`
  - `engine_dir(resource_path, version) -> str`
  - `engine_exe(resource_path, version) -> str` (adds `.exe` on Windows)
  - `weights_dir(resource_path, version) -> str`
  - `config_path(resource_path) -> str`
  - `script_dir(resource_path) -> str`
  - Constants: `APP_DIR_NAME="DynamicsNeeded"`, `REAPACK_REPO_NAME="Dynamics Needed"`, `REAPACK_CATEGORY="MIDI Editor"`, `ENGINE_EXE_STEM="dn-engine"`

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/test_dn_paths.py
import os, sys
sys.path.insert(0, os.path.join("plugin", "reaper"))
import dn_paths


def test_platform_key_macos_arm(monkeypatch):
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dn_paths.platform, "machine", lambda: "arm64")
    assert dn_paths.platform_key() == "macos-arm64"


def test_platform_key_macos_intel(monkeypatch):
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dn_paths.platform, "machine", lambda: "x86_64")
    assert dn_paths.platform_key() == "macos-x86_64"


def test_platform_key_windows(monkeypatch):
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dn_paths.platform, "machine", lambda: "AMD64")
    assert dn_paths.platform_key() == "windows-x64"


def test_platform_key_linux(monkeypatch):
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dn_paths.platform, "machine", lambda: "x86_64")
    assert dn_paths.platform_key() == "linux-x64"


def test_engine_paths_layout():
    rp = os.path.join("RES")
    assert dn_paths.engine_dir(rp, "0.1.0") == os.path.join("RES", "DynamicsNeeded", "engine", "0.1.0")
    assert dn_paths.weights_dir(rp, "0.1.0") == os.path.join("RES", "DynamicsNeeded", "engine", "0.1.0", "weights")
    assert dn_paths.config_path(rp) == os.path.join("RES", "dynamics_needed_config.json")
    assert dn_paths.script_dir(rp) == os.path.join("RES", "Scripts", "Dynamics Needed", "MIDI Editor")


def test_engine_exe_extension(monkeypatch):
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Windows")
    assert dn_paths.engine_exe("RES", "0.1.0").endswith(os.path.join("0.1.0", "dn-engine.exe"))
    monkeypatch.setattr(dn_paths.platform, "system", lambda: "Darwin")
    assert dn_paths.engine_exe("RES", "0.1.0").endswith(os.path.join("0.1.0", "dn-engine"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_dn_paths.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'dn_paths'`).

- [ ] **Step 3: Write minimal implementation**

```python
# plugin/reaper/dn_paths.py
"""Resource-path-derived locations + platform key. Pure stdlib, ASCII-only.

No Reaper API and no __file__, so it imports under pytest and under Reaper's
embedded Python. The resource path is passed in by the caller (the ReaScript
gets it from RPR_GetResourcePath()).
"""
from __future__ import annotations

import os
import platform

APP_DIR_NAME = "DynamicsNeeded"
REAPACK_REPO_NAME = "Dynamics Needed"
REAPACK_CATEGORY = "MIDI Editor"
ENGINE_EXE_STEM = "dn-engine"


def platform_key():
    system = platform.system()
    mach = platform.machine().lower()
    if system == "Darwin":
        return "macos-arm64" if mach in ("arm64", "aarch64") else "macos-x86_64"
    if system == "Windows":
        return "windows-x64"
    return "linux-x64"


def app_root(resource_path):
    return os.path.join(resource_path, APP_DIR_NAME)


def engine_root(resource_path):
    return os.path.join(app_root(resource_path), "engine")


def engine_dir(resource_path, version):
    return os.path.join(engine_root(resource_path), version)


def weights_dir(resource_path, version):
    return os.path.join(engine_dir(resource_path, version), "weights")


def engine_exe(resource_path, version):
    name = ENGINE_EXE_STEM + (".exe" if platform.system() == "Windows" else "")
    return os.path.join(engine_dir(resource_path, version), name)


def config_path(resource_path):
    return os.path.join(resource_path, "dynamics_needed_config.json")


def script_dir(resource_path):
    return os.path.join(resource_path, "Scripts", REAPACK_REPO_NAME, REAPACK_CATEGORY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest ml/tests/test_dn_paths.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/dn_paths.py ml/tests/test_dn_paths.py
git commit -m "feat(plugin): dn_paths — platform key + resource-path locations"
```

---

## Task 2: `engine_client.start_engine` — dual-mode spawn

**Files:**
- Modify: `plugin/reaper/engine_client.py`
- Test: `ml/tests/test_engine_client.py` (update the existing spawn test, add a frozen-mode test)

**Interfaces:**
- Consumes: cfg dict — either dev `{venv_python, repo_root, port}` or frozen `{engine_path, proc_dir, port}`.
- Produces: `start_engine(cfg)` spawns the engine; frozen mode spawns `[engine_path, --port, --parent-pid, --proc-dir]`, dev mode unchanged. `base_url`, `health`, `predict`, `ensure_engine` unchanged.

- [ ] **Step 1: Update/replace the failing tests**

Replace `test_start_engine_spawns_expected_argv` in `ml/tests/test_engine_client.py` with two tests:

```python
def test_start_engine_dev_mode_argv(monkeypatch, tmp_path):
    seen = {}
    def fake_popen(argv, **kwargs):
        seen["argv"] = argv; seen["cwd"] = kwargs.get("cwd")
        class P: pass
        return P()
    monkeypatch.setattr(engine_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_client.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(engine_client, "open", lambda *a, **k: open(os.devnull, "a"), raising=False)
    cfg = {"venv_python": "/venv/py", "repo_root": str(tmp_path), "port": 8765}
    engine_client.start_engine(cfg)
    assert seen["argv"][:4] == ["/venv/py", "-m", "drum_dynamics.serve", "--port"]
    assert "8765" in seen["argv"] and "--parent-pid" in seen["argv"]
    assert seen["cwd"] == str(tmp_path)


def test_start_engine_frozen_mode_argv(monkeypatch, tmp_path):
    seen = {}
    def fake_popen(argv, **kwargs):
        seen["argv"] = argv; seen["cwd"] = kwargs.get("cwd")
        class P: pass
        return P()
    monkeypatch.setattr(engine_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_client.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(engine_client, "open", lambda *a, **k: open(os.devnull, "a"), raising=False)
    exe = str(tmp_path / "dn-engine")
    weights = str(tmp_path / "weights")
    cfg = {"engine_path": exe, "proc_dir": weights, "port": 8770}
    engine_client.start_engine(cfg)
    assert seen["argv"][0] == exe
    assert "--proc-dir" in seen["argv"] and weights in seen["argv"]
    assert "8770" in seen["argv"] and "--parent-pid" in seen["argv"]
    assert seen["cwd"] == str(tmp_path)  # dirname(engine_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_engine_client.py -v`
Expected: `test_start_engine_frozen_mode_argv` FAILS (frozen branch not implemented yet; argv[0] would be wrong or KeyError).

- [ ] **Step 3: Write minimal implementation**

Replace `start_engine` in `plugin/reaper/engine_client.py`:

```python
def _runtime_dir(cfg):
    if cfg.get("engine_path"):
        return os.path.join(os.path.dirname(cfg["engine_path"]), ".runtime")
    return os.path.join(cfg["repo_root"], "plugin", "reaper", ".runtime")


def start_engine(cfg):
    runtime = _runtime_dir(cfg)
    os.makedirs(runtime, exist_ok=True)
    log = open(os.path.join(runtime, "engine.log"), "a")
    port = str(cfg.get("port", 8765))
    if cfg.get("engine_path"):
        argv = [cfg["engine_path"], "--port", port, "--parent-pid", str(os.getpid())]
        if cfg.get("proc_dir"):
            argv += ["--proc-dir", cfg["proc_dir"]]
        cwd = os.path.dirname(cfg["engine_path"])
    else:
        argv = [cfg["venv_python"], "-m", "drum_dynamics.serve",
                "--port", port, "--parent-pid", str(os.getpid())]
        cwd = cfg["repo_root"]
    subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=log,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest ml/tests/test_engine_client.py -v`
Expected: PASS (all, including both spawn tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/engine_client.py ml/tests/test_engine_client.py
git commit -m "feat(plugin): engine_client dual-mode start_engine (frozen vs dev)"
```

---

## Task 3: `bootstrap` — versioning, URLs, SHA-256 verify (pure)

**Files:**
- Create: `plugin/reaper/bootstrap.py`
- Test: `ml/tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `dn_paths` (Task 1).
- Produces:
  - `PINNED_ENGINE_VERSION` (str, e.g. `"0.1.0"`), `GITHUB_OWNER_REPO` (str literal `OWNER/REPO`)
  - `release_base_url() -> str` (honors `DN_RELEASE_BASE_URL`)
  - `asset_name(version, platform_key) -> str` = `dn-engine-<version>-<platform_key>.zip`
  - `asset_url(base, version, platform_key) -> str`, `sums_url(base, version) -> str`
  - `sha256_of(path) -> str`, `expected_sum(sums_text, name) -> str | None`, `verify(path, sums_text, name) -> bool`

> **Before starting:** replace the `OWNER/REPO` placeholder in this file with the real GitHub repo (see Global Constraints).

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/test_bootstrap.py
import hashlib
import os, sys
sys.path.insert(0, os.path.join("plugin", "reaper"))
import bootstrap


def test_asset_name_and_urls():
    assert bootstrap.asset_name("0.1.0", "macos-arm64") == "dn-engine-0.1.0-macos-arm64.zip"
    base = "https://example/releases/download"
    assert bootstrap.asset_url(base, "0.1.0", "linux-x64") == \
        "https://example/releases/download/engine-v0.1.0/dn-engine-0.1.0-linux-x64.zip"
    assert bootstrap.sums_url(base, "0.1.0") == \
        "https://example/releases/download/engine-v0.1.0/SHA256SUMS"


def test_release_base_url_env_override(monkeypatch):
    monkeypatch.setenv("DN_RELEASE_BASE_URL", "file:///tmp/rel")
    assert bootstrap.release_base_url() == "file:///tmp/rel"
    monkeypatch.delenv("DN_RELEASE_BASE_URL", raising=False)
    assert bootstrap.GITHUB_OWNER_REPO in bootstrap.release_base_url()


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "a.zip"
    p.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert bootstrap.sha256_of(str(p)) == digest
    sums = "{}  a.zip\ndeadbeef  other.zip\n".format(digest)
    assert bootstrap.expected_sum(sums, "a.zip") == digest
    assert bootstrap.expected_sum(sums, "missing.zip") is None
    assert bootstrap.verify(str(p), sums, "a.zip") is True
    assert bootstrap.verify(str(p), sums, "other.zip") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_bootstrap.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bootstrap'`).

- [ ] **Step 3: Write minimal implementation**

```python
# plugin/reaper/bootstrap.py
"""First-run engine bootstrap: resolve, download, verify, unpack. Pure stdlib.

ASCII-only, no Reaper API, no __file__ -> unit-tested under pytest and runnable
under Reaper's embedded Python. The network fetcher is injectable so the whole
flow is testable offline (see install()).
"""
from __future__ import annotations

import hashlib
import os

PINNED_ENGINE_VERSION = "0.1.0"
GITHUB_OWNER_REPO = "OWNER/REPO"  # TODO replace with the real repo before use


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest ml/tests/test_bootstrap.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/bootstrap.py ml/tests/test_bootstrap.py
git commit -m "feat(plugin): bootstrap URL/asset naming + SHA-256 verification"
```

---

## Task 4: `bootstrap.install` — download → verify → unpack → prune → config

**Files:**
- Modify: `plugin/reaper/bootstrap.py`
- Test: `ml/tests/test_bootstrap.py` (add cases)

**Interfaces:**
- Consumes: Task 3 functions + `dn_paths`.
- Produces:
  - `is_installed(resource_path, version) -> bool`
  - `default_fetch(url) -> bytes` (urllib; not unit-tested — injected out)
  - `install(resource_path, version, platform_key, fetch, log=lambda m: None) -> dict` — downloads the zip + `SHA256SUMS`, verifies, unpacks into `engine_dir`, restores exec bit on the exe (POSIX), prunes other version dirs, writes config, returns the config dict. Raises `BootstrapError` on any failure.
  - `write_config(resource_path, version, port=8765) -> dict`
  - `BootstrapError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# add to ml/tests/test_bootstrap.py
import hashlib, io, json, zipfile
sys.path.insert(0, os.path.join("plugin", "reaper"))
import dn_paths


def _make_engine_zip(platform_key):
    buf = io.BytesIO()
    exe = "dn-engine.exe" if platform_key == "windows-x64" else "dn-engine"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(exe, b"#!binary\n")
        z.writestr("weights/mdn_meta.json", b"{}")
        z.writestr("_internal/marker", b"x")
    return buf.getvalue()


def test_install_happy_path(tmp_path, monkeypatch):
    rp = str(tmp_path / "RES")
    os.makedirs(rp)
    pk = "linux-x64"
    zip_bytes = _make_engine_zip(pk)
    name = bootstrap.asset_name("0.1.0", pk)
    sums = "{}  {}\n".format(hashlib.sha256(zip_bytes).hexdigest(), name)

    def fetch(url):
        if url.endswith("SHA256SUMS"):
            return sums.encode()
        if url.endswith(name):
            return zip_bytes
        raise AssertionError("unexpected url " + url)

    cfg = bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
    assert bootstrap.is_installed(rp, "0.1.0") is True
    assert cfg["engine_path"] == dn_paths.engine_exe(rp, "0.1.0")
    assert cfg["proc_dir"] == dn_paths.weights_dir(rp, "0.1.0")
    assert cfg["engine_version"] == "0.1.0"
    on_disk = json.load(open(dn_paths.config_path(rp)))
    assert on_disk == cfg
    assert os.path.isfile(os.path.join(dn_paths.weights_dir(rp, "0.1.0"), "mdn_meta.json"))


def test_install_checksum_mismatch_raises(tmp_path):
    rp = str(tmp_path / "RES"); os.makedirs(rp)
    pk = "linux-x64"; name = bootstrap.asset_name("0.1.0", pk)
    def fetch(url):
        if url.endswith("SHA256SUMS"):
            return ("deadbeef  " + name + "\n").encode()
        return b"corrupt-bytes"
    try:
        bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
        assert False, "expected BootstrapError"
    except bootstrap.BootstrapError:
        pass
    assert bootstrap.is_installed(rp, "0.1.0") is False


def test_install_prunes_old_versions(tmp_path):
    rp = str(tmp_path / "RES"); os.makedirs(rp)
    old = dn_paths.engine_dir(rp, "0.0.9"); os.makedirs(old)
    open(os.path.join(old, "stale"), "w").close()
    pk = "linux-x64"; zip_bytes = _make_engine_zip(pk); name = bootstrap.asset_name("0.1.0", pk)
    sums = "{}  {}\n".format(hashlib.sha256(zip_bytes).hexdigest(), name)
    fetch = lambda u: sums.encode() if u.endswith("SHA256SUMS") else zip_bytes
    bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
    assert not os.path.isdir(old)
    assert bootstrap.is_installed(rp, "0.1.0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_bootstrap.py -v`
Expected: FAIL (`AttributeError: module 'bootstrap' has no attribute 'install'`).

- [ ] **Step 3: Write minimal implementation**

Append to `plugin/reaper/bootstrap.py` (add `import io, shutil, stat, zipfile, urllib.request` at top, and `import dn_paths`):

```python
import io
import shutil
import stat
import urllib.request
import zipfile

import dn_paths


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

    _prune_other_versions(resource_path, version)
    log("Engine installed.")
    return write_config(resource_path, version)
```

Note: `json` is already needed — add `import json` to the top imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest ml/tests/test_bootstrap.py -v`
Expected: PASS (all bootstrap tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/bootstrap.py ml/tests/test_bootstrap.py
git commit -m "feat(plugin): bootstrap install — download, verify, unpack, prune, config"
```

---

## Task 5: PyInstaller entry + spec + local build script

**Files:**
- Create: `ml/packaging/run_engine.py`, `ml/packaging/dn_engine.spec`, `ml/packaging/build_engine.py`
- Modify: `ml/pyproject.toml` (add `pyinstaller` to a `packaging` optional-dependencies group)

**Interfaces:**
- Consumes: `drum_dynamics.serve` (existing), weights in `data/processed/`.
- Produces: on your OS, `ml/packaging/dist/dn-engine/` (one-folder freeze with `weights/`) and `ml/packaging/dist/dn-engine-<version>-<platform_key>.zip` + a `SHA256SUMS` line. `build_engine.py` CLI: `--version`, `--weights-dir` (default `data/processed`), `--sign/--no-sign`.

This task has no pytest cycle (it produces a binary); it is verified by building and by Task 6's smoke test.

- [ ] **Step 1: Add the packaging dependency**

Edit `ml/pyproject.toml`, under `[project.optional-dependencies]`:

```toml
packaging = ["pyinstaller>=6.0"]
```

Install it: `.venv/bin/pip install -e "ml[packaging]"`

- [ ] **Step 2: Create the PyInstaller entry**

```python
# ml/packaging/run_engine.py
"""PyInstaller entry point: freezes to the `dn-engine` executable."""
from drum_dynamics.serve.__main__ import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create the spec**

```python
# ml/packaging/dn_engine.spec
# PyInstaller one-folder spec for the Dynamics Needed inference engine.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("lightgbm", "sklearn", "torch"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

EXCLUDES = ["partitura", "music21", "fluidsynth", "pyfluidsynth",
            "matplotlib", "pyarrow", "jupyter", "ipykernel", "nbconvert",
            "notebook", "IPython"]

a = Analysis(
    ["run_engine.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["drum_dynamics.serve.__main__"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="dn-engine",
          console=True, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, name="dn-engine")
```

- [ ] **Step 4: Create the build wrapper**

```python
# ml/packaging/build_engine.py
"""Freeze the engine, stage weights, (mac) ad-hoc sign, zip, checksum.

Run from repo root: .venv/bin/python ml/packaging/build_engine.py --version 0.1.0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
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
```

- [ ] **Step 5: Build locally and verify the artifact exists**

Run: `.venv/bin/python ml/packaging/build_engine.py --version 0.1.0`
Expected: `ml/packaging/dist/dn-engine/dn-engine` exists, `dist/dn-engine/weights/` has 6 files, `dist/dn-engine-0.1.0-<platform>.zip` and `dist/SHA256SUMS` exist. (Hidden-import gaps surface at the next task's smoke test — if so, add the missing module name to `hiddenimports` in the spec and rebuild.)

- [ ] **Step 6: Commit**

```bash
git add ml/packaging/run_engine.py ml/packaging/dn_engine.spec ml/packaging/build_engine.py ml/pyproject.toml
git commit -m "feat(packaging): PyInstaller one-folder freeze + build script"
```

Note: `ml/packaging/dist/` and `ml/packaging/build/` are build outputs — ensure they are gitignored (Task 11 covers `.gitignore`; if committing before then, add them now).

---

## Task 6: `smoke_test.py` — launch freeze, assert `/health` + `/predict`

**Files:**
- Create: `ml/packaging/smoke_test.py`

**Interfaces:**
- Consumes: an unpacked engine dir (from Task 5's `dist/dn-engine/`, or a CI-unzipped dir).
- Produces: CLI `smoke_test.py <engine_dir> [--port N]`; exit 0 on success, non-zero + message on failure. Reused by CI (Task 10).

This task's own execution is its verification (there is no separate unit test — it *is* the integration test).

- [ ] **Step 1: Write the smoke test script**

```python
# ml/packaging/smoke_test.py
"""Launch a frozen dn-engine, hit /health and /predict, assert a valid result.

Usage: python ml/packaging/smoke_test.py ml/packaging/dist/dn-engine
Exit 0 on success; prints and exits non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request


def _get(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url, payload, timeout=60.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    engine_dir = sys.argv[1]
    port = 8799
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    exe = os.path.join(engine_dir, "dn-engine.exe" if os.name == "nt" else "dn-engine")
    weights = os.path.join(engine_dir, "weights")
    proc = subprocess.Popen([exe, "--port", str(port), "--proc-dir", weights],
                            cwd=engine_dir)
    try:
        base = "http://127.0.0.1:{}".format(port)
        health = None
        for _ in range(60):
            time.sleep(1.0)
            try:
                health = _get(base + "/health")
                break
            except Exception:
                continue
        assert health and health.get("status") == "ok", "health failed: {}".format(health)
        assert set(["lgbm", "mdn", "categorical"]).issubset(set(health.get("models", []))), \
            "models missing: {}".format(health.get("models"))

        req = {
            "model": "mdn", "style": (health.get("styles") or ["rock"])[0],
            "temperature": 1.0, "blend": 1.0, "beat_type": "beat",
            "bpm": 120.0, "time_signature": "4-4",
            "notes": [
                {"index": 0, "pitch": 38, "onset_sec": 0.0, "velocity": 80, "selected": True},
                {"index": 1, "pitch": 38, "onset_sec": 0.5, "velocity": 80, "selected": True},
            ],
        }
        res = _post(base + "/predict", req)
        vels = res.get("velocities", {})
        assert vels, "no velocities returned"
        assert all(1 <= int(v) <= 127 for v in vels.values()), "velocity out of range: {}".format(vels)
        print("SMOKE OK:", health.get("models"), "->", vels)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test against the local freeze**

Run: `.venv/bin/python ml/packaging/smoke_test.py ml/packaging/dist/dn-engine`
Expected: prints `SMOKE OK: ['lgbm', 'mdn', 'categorical'] -> {...}` and exits 0. If it fails with an import error inside the engine (e.g. `ModuleNotFoundError`), add that module to `hiddenimports` in `dn_engine.spec`, rebuild (Task 5 Step 5), and re-run.

- [ ] **Step 3: Commit**

```bash
git add ml/packaging/smoke_test.py
git commit -m "feat(packaging): engine smoke test (/health + /predict)"
```

---

## Task 7: Wire the ReaScript to self-locate + bootstrap on first run

**Files:**
- Modify: `plugin/reaper/dynamics_needed.py`
- Modify: `plugin/reaper/setup_reaper.py`

**Interfaces:**
- Consumes: `dn_paths`, `bootstrap`, `engine_client` (all prior).
- Produces: a ReaScript that (a) computes its own module dir from `RPR_GetResourcePath()` when no dev config exists, (b) on first run downloads+installs the engine and writes the frozen config, (c) falls back to dev config `{venv_python, repo_root, port}` when present. `setup_reaper.py` writes only the dev config.

This task is verified in Reaper (Task 9), not pytest — `dynamics_needed.py` imports `reaper_python` at module load and cannot be imported under pytest. Keep all testable logic in `dn_paths`/`bootstrap` (already tested); this task is thin wiring only.

- [ ] **Step 1: Replace config load + module path bootstrap in `dynamics_needed.py`**

Replace `_load_config()` and the `sys.path.insert(...)` block at the top of `_init()` with resource-path-based self-location and first-run install. New helpers near the top (ASCII-only):

```python
def _resource_path():
    return RPR_GetResourcePath()


def _module_dir(cfg, resource_path):
    # Dev config points at the repo; installed build ships helpers beside the script.
    if cfg and cfg.get("repo_root"):
        return os.path.join(cfg["repo_root"], "plugin", "reaper")
    return os.path.join(resource_path, "Scripts", "Dynamics Needed", "MIDI Editor")


def _read_config_if_any(resource_path):
    cfg_path = os.path.join(resource_path, "dynamics_needed_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None
```

- [ ] **Step 2: Rewrite `_init()` to self-locate, then bootstrap if needed**

```python
def _init():
    resource_path = _resource_path()
    cfg = _read_config_if_any(resource_path)

    # Make helper modules importable (installed location, or the dev repo).
    sys.path.insert(0, _module_dir(cfg, resource_path))
    import dn_paths
    import bootstrap
    import engine_client
    import dn_core
    import predict_worker
    import threading

    # First run (installed build, no config yet): download + install the engine.
    if cfg is None or (not cfg.get("engine_path") and not cfg.get("repo_root")):
        version = bootstrap.PINNED_ENGINE_VERSION
        if not bootstrap.is_installed(resource_path, version):
            ok = RPR_ShowMessageBox(
                "Dynamics Needed needs to download its engine (~250 MB). Download now?",
                "Dynamics Needed", 4)  # 4 = Yes/No; 6 = Yes
            if ok != 6:
                raise RuntimeError("engine download declined")
            try:
                cfg = bootstrap.install(resource_path, version, dn_paths.platform_key())
            except bootstrap.BootstrapError as e:
                raise RuntimeError("engine install failed: {}".format(e))
        else:
            cfg = bootstrap.write_config(resource_path, version)

    state = {
        "cfg": cfg,
        "engine_client": engine_client,
        "dn_core": dn_core,
        "ctx": imgui.CreateContext("Dynamics Needed"),
        "engine_lock": threading.Lock(),
        "engine_ready": False,
        "health": None,
        "open": True,
    }
    _start_engine_async(state)
    take = None
    editor = RPR_MIDIEditor_GetActive()
    if editor:
        take = RPR_MIDIEditor_GetTake(editor)
    state["last_params"] = _load_last(_track_key(take)) if take else {}
    state["params"] = None
    state["live"] = True
    ec, cfgd = state["engine_client"], state["cfg"]
    state["worker"] = predict_worker.PredictWorker(
        lambda req: dn_core.parse_velocities(ec.predict(cfgd, req)))
    state["worker"].start()
    state["seq"] = 0
    state["preview"] = {}
    return state
```

(Delete the old `_load_config()` and the old inline `sys.path.insert(0, os.path.join(cfg["repo_root"], ...))`/imports it replaced.)

- [ ] **Step 3: Rewrite `setup_reaper.py` as the dev-config writer**

Change the config dict it writes to the dev shape and update the printed guidance. The core write becomes:

```python
    cfg = {"venv_python": sys.executable, "repo_root": REPO_ROOT, "port": args.port}
```

(This is already what it writes — verify it still writes `venv_python` + `repo_root` + `port`, update the module docstring to say it is the **development-mode** config writer, and note that end users never run it — the ReaScript's first-run bootstrap writes the frozen config instead.)

- [ ] **Step 4: ASCII + syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('plugin/reaper/dynamics_needed.py').read()); ast.parse(open('plugin/reaper/setup_reaper.py').read()); print('ok')"`
Run: `.venv/bin/python -c "b=open('plugin/reaper/dynamics_needed.py','rb').read(); assert all(c<128 for c in b), 'non-ASCII'; print('ascii ok')"`
Expected: both print ok.

- [ ] **Step 5: Confirm the existing client tests still pass**

Run: `.venv/bin/python -m pytest ml/tests/ -v`
Expected: PASS (dn_paths, bootstrap, engine_client, reaper_core, predict_worker, serve).

- [ ] **Step 6: Commit**

```bash
git add plugin/reaper/dynamics_needed.py plugin/reaper/setup_reaper.py
git commit -m "feat(plugin): ReaScript self-location + first-run engine bootstrap"
```

---

## Task 8: `make_index.py` — generate the ReaPack index

**Files:**
- Create: `plugin/reaper/reapack/make_index.py`
- Test: `ml/tests/test_make_index.py`

**Interfaces:**
- Consumes: nothing (stdlib `xml.etree`).
- Produces: `build_index(script_version, time_iso, raw_base_url) -> str` (the `index.xml` text) and a CLI writing it to `plugin/reaper/reapack/index.xml`. The index declares one package (`dynamics_needed.py`, type `script`) under category `MIDI Editor`, with `<source>` entries for the ReaScript + its helper `.py` files, and a `<metadata>` description. `raw_base_url` is where ReaPack fetches the raw files (e.g. `https://raw.githubusercontent.com/OWNER/REPO/main/plugin/reaper`).

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/test_make_index.py
import os, sys
from xml.etree import ElementTree as ET
sys.path.insert(0, os.path.join("plugin", "reaper", "reapack"))
import make_index


def test_index_has_package_and_sources():
    xml = make_index.build_index("0.1.0", "2026-08-14T00:00:00Z",
                                 "https://raw.example/OWNER/REPO/main/plugin/reaper")
    root = ET.fromstring(xml)
    assert root.tag == "index"
    cat = root.find("category")
    assert cat.get("name") == "MIDI Editor"
    pkg = cat.find("reapack")
    assert pkg.get("name") == "dynamics_needed.py"
    assert pkg.get("type") == "script"
    ver = pkg.find("version")
    assert ver.get("name") == "0.1.0"
    srcs = [s.get("file") for s in ver.findall("source")]
    # main script has no file attr (installs under package name); helpers do
    files = set(f for f in srcs if f)
    assert {"dn_paths.py", "bootstrap.py", "engine_client.py", "dn_core.py",
            "predict_worker.py"}.issubset(files)
    assert any(s.text and s.text.endswith("dynamics_needed.py") for s in ver.findall("source"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest ml/tests/test_make_index.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'make_index'`).

- [ ] **Step 3: Write minimal implementation**

```python
# plugin/reaper/reapack/make_index.py
"""Generate the ReaPack index.xml for the Dynamics Needed package.

Run: python plugin/reaper/reapack/make_index.py --version 0.1.0 \
        --time 2026-08-14T00:00:00Z \
        --raw-base https://raw.githubusercontent.com/OWNER/REPO/main/plugin/reaper
"""
from __future__ import annotations

import argparse
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom

INDEX_NAME = "Dynamics Needed"
CATEGORY = "MIDI Editor"
MAIN = "dynamics_needed.py"
HELPERS = ["dn_paths.py", "bootstrap.py", "engine_client.py", "dn_core.py", "predict_worker.py"]
DESCRIPTION = ("Dynamics Needed - restore humanized drum-note velocities in the "
               "MIDI editor. On first run it downloads its inference engine.")


def build_index(script_version, time_iso, raw_base_url):
    base = raw_base_url.rstrip("/")
    index = ET.Element("index", {"version": "1", "name": INDEX_NAME})
    cat = ET.SubElement(index, "category", {"name": CATEGORY})
    pkg = ET.SubElement(cat, "reapack", {"name": MAIN, "type": "script"})
    meta = ET.SubElement(pkg, "metadata")
    desc = ET.SubElement(meta, "description")
    desc.text = DESCRIPTION
    ver = ET.SubElement(pkg, "version", {"name": script_version, "author": "Dynamics Needed", "time": time_iso})
    # Main file: main="true", installs under the package name (no file attr).
    main_src = ET.SubElement(ver, "source", {"main": "true"})
    main_src.text = "{}/{}".format(base, MAIN)
    for h in HELPERS:
        s = ET.SubElement(ver, "source", {"file": h})
        s.text = "{}/{}".format(base, h)
    rough = ET.tostring(index, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--raw-base", required=True)
    p.add_argument("--out", default=os.path.join("plugin", "reaper", "reapack", "index.xml"))
    args = p.parse_args()
    xml = build_index(args.version, args.time, args.raw_base)
    with open(args.out, "w") as fh:
        fh.write(xml)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest ml/tests/test_make_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/reaper/reapack/make_index.py ml/tests/test_make_index.py
git commit -m "feat(reapack): index.xml generator"
```

---

## Task 9: Local end-to-end dry run (freeze + `file://` bootstrap + Reaper)

**Files:** none (verification task). Uses artifacts from Tasks 5–8.

This task ties the pieces together on your own OS with no GitHub release. No pytest; it is a manual checklist that must pass before wiring CI.

- [ ] **Step 1: Stage a local `file://` release**

```bash
VER=0.1.0
PK=$(.venv/bin/python -c "import sys;sys.path.insert(0,'plugin/reaper');import dn_paths;print(dn_paths.platform_key())")
mkdir -p /tmp/dn-rel/engine-v$VER
cp ml/packaging/dist/dn-engine-$VER-$PK.zip /tmp/dn-rel/engine-v$VER/
cp ml/packaging/dist/SHA256SUMS /tmp/dn-rel/engine-v$VER/
```

- [ ] **Step 2: Exercise the bootstrap end-to-end against `file://` (no Reaper)**

```bash
DN_RELEASE_BASE_URL="file:///tmp/dn-rel" .venv/bin/python -c "
import sys, tempfile, os
sys.path.insert(0,'plugin/reaper')
import dn_paths, bootstrap
rp = tempfile.mkdtemp()
cfg = bootstrap.install(rp, bootstrap.PINNED_ENGINE_VERSION, dn_paths.platform_key(), log=print)
print('config:', cfg)
assert bootstrap.is_installed(rp, bootstrap.PINNED_ENGINE_VERSION)
print('OK installed at', dn_paths.engine_dir(rp, bootstrap.PINNED_ENGINE_VERSION))
"
```
Expected: prints `Engine installed.` and `OK installed at ...`. Then smoke-test the installed copy:
Run: `.venv/bin/python ml/packaging/smoke_test.py <the engine_dir printed above>`
Expected: `SMOKE OK`.

- [ ] **Step 3: Set up a local ReaPack repo and install in Reaper**

1. Generate a local index pointing at your working tree:
   `.venv/bin/python plugin/reaper/reapack/make_index.py --version 0.1.0 --time 2026-08-14T00:00:00Z --raw-base "file://$(pwd)/plugin/reaper"`
2. In Reaper: Extensions -> ReaPack -> Import repositories -> paste `file://<repo>/plugin/reaper/reapack/index.xml`.
3. ReaPack -> Browse packages -> install "dynamics_needed.py".
4. Confirm the package installed the main script + all 5 helpers under `<resource>/Scripts/Dynamics Needed/MIDI Editor/`.

- [ ] **Step 4: Run the action in Reaper (first-run bootstrap)**

1. Actions -> find "Custom: dynamics_needed.py" (or Load ReaScript for the installed path), run it with `DN_RELEASE_BASE_URL=file:///tmp/dn-rel` exported in the environment Reaper was launched from.
2. Accept the "download engine" prompt. Confirm the panel reaches "ready" and a MIDI item preview + Apply works.
3. Confirm `<resource>/dynamics_needed_config.json` now contains `engine_path`/`proc_dir`/`engine_version`.

- [ ] **Step 5: Record results**

Note any issues (hidden-import gaps → fix in Task 5 spec; path issues → fix Task 1/7). Re-run until Steps 2–4 are clean. No commit (verification only) unless fixes were made in earlier tasks' files.

---

## Task 10: CI — build/sign/smoke/publish matrix

**Files:**
- Create: `.github/workflows/engine-release.yml`

**Interfaces:**
- Consumes: `build_engine.py`, `smoke_test.py`, weights (fetched in CI), `make_index.py`.
- Produces: on tag `engine-vX.Y.Z`, a GitHub Release with 4 platform zips + `SHA256SUMS`, and an updated `index.xml` on `gh-pages`.

> **Weights in CI:** `data/processed/` is gitignored, so CI must fetch the 6 weight files before freezing. Add a `fetch-weights` step that downloads them (from the project's Hugging Face model repo — see `ml/scripts/publish_model.py` for the repo id — via `huggingface_hub.hf_hub_download`, or from a secure release asset). Set `--weights-dir` to that download dir. Confirm the weights source before implementing this step.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/engine-release.yml
name: engine-release
on:
  push:
    tags: ["engine-v*"]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: macos-14
            key: macos-arm64
          - os: macos-13
            key: macos-x86_64
          - os: windows-latest
            key: windows-x64
          - os: ubuntu-latest
            key: linux-x64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: |
          python -m pip install -e "ml[packaging]"
      - name: Fetch weights
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python ml/packaging/fetch_weights.py --out data/processed   # see note; implement to pull the 6 files
      - name: Version from tag
        id: ver
        shell: bash
        run: echo "version=${GITHUB_REF_NAME#engine-v}" >> "$GITHUB_OUTPUT"
      - name: Build
        run: python ml/packaging/build_engine.py --version ${{ steps.ver.outputs.version }}
      - name: Smoke test
        run: python ml/packaging/smoke_test.py ml/packaging/dist/dn-engine
      - name: Upload to release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ github.ref_name }}
          files: |
            ml/packaging/dist/dn-engine-*-${{ matrix.key }}.zip

  checksums:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: robinraju/release-downloader@v1
        with:
          tag: ${{ github.ref_name }}
          fileName: "dn-engine-*.zip"
      - name: Build SHA256SUMS
        run: sha256sum dn-engine-*.zip > SHA256SUMS
      - uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ github.ref_name }}
          files: SHA256SUMS

  reapack-index:
    needs: checksums
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Generate index
        run: |
          python plugin/reaper/reapack/make_index.py \
            --version "${GITHUB_REF_NAME#engine-v}" \
            --time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --raw-base "https://raw.githubusercontent.com/${{ github.repository }}/main/plugin/reaper" \
            --out index.xml
      - name: Publish to gh-pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: .
          keep_files: true
          publish_branch: gh-pages
```

> Each job's `SHA256SUMS` must aggregate **all four** zips (per the checksums job), because the bootstrap fetches one `SHA256SUMS` for all platforms. The per-platform `build_engine.py` `SHA256SUMS` is local-only; CI's `checksums` job produces the authoritative combined file.

- [ ] **Step 2: Create the weights-fetch helper**

```python
# ml/packaging/fetch_weights.py
"""Download the 6 engine weight files into --out (default data/processed).

Uses the project's Hugging Face model repo. Confirm repo id vs ml/scripts/publish_model.py.
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "OWNER/dynamics-needed-models"  # TODO confirm against publish_model.py
FILES = ["lightgbm_model.joblib", "lightgbm_features.json", "mdn_meta.json",
         "head_mdn.pt", "transformer_meta.json", "head_categorical.pt"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("data", "processed"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=f)
        dst = os.path.join(args.out, f)
        if os.path.abspath(path) != os.path.abspath(dst):
            import shutil
            shutil.copy(path, dst)
    print("weights ready in", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Validate the workflow YAML**

Run: `.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/engine-release.yml')); print('yaml ok')"`
(Install pyyaml if missing: `.venv/bin/pip install pyyaml`.)
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/engine-release.yml ml/packaging/fetch_weights.py
git commit -m "ci(engine): build/sign/smoke/publish matrix + ReaPack index"
```

- [ ] **Step 5: Real dry-run with a pre-release tag**

Push a throwaway tag (e.g. `engine-v0.0.1-rc1`) and watch the run. Confirm 4 zips + `SHA256SUMS` land on the Release and `index.xml` publishes to `gh-pages`. Fix any hidden-import/signing issues, then delete the pre-release. (This validates the one thing not testable locally: the other three OS freezes.)

---

## Task 11: Ignore build outputs + runtime; docs

**Files:**
- Modify: `.gitignore`, `README.md`

- [ ] **Step 1: Ignore build/runtime outputs**

Append to `.gitignore`:

```gitignore
ml/packaging/build/
ml/packaging/dist/
plugin/reaper/.runtime/
plugin/reaper/reapack/index.xml
```

- [ ] **Step 2: Rewrite the plugin install section of `README.md`**

Replace the current developer-setup instructions for the Reaper tool with a user section and a dev section:

```markdown
### Install the Reaper tool (users)

1. Install [ReaPack](https://reapack.com/) and [ReaImGui](https://github.com/cfillion/reaimgui) (ReaPack: Browse packages -> "ReaImGui").
2. ReaPack -> Import repositories -> paste:
   `https://OWNER.github.io/REPO/index.xml`
3. ReaPack -> Browse packages -> install **Dynamics Needed**.
4. Run the action **Custom: dynamics_needed.py**. On first launch it downloads
   the inference engine (~250 MB) and starts automatically.
5. macOS note: if you obtained the engine manually and macOS blocks it, run
   `xattr -dr com.apple.quarantine "~/Library/Application Support/REAPER/DynamicsNeeded"`.

### Develop the Reaper tool

`.venv/bin/python plugin/reaper/setup_reaper.py` writes a dev config pointing at
your venv + repo, so the ReaScript runs `python -m drum_dynamics.serve` directly
(no freeze needed). Build the frozen engine with
`.venv/bin/python ml/packaging/build_engine.py --version <v>` and smoke-test it
with `ml/packaging/smoke_test.py`.
```

- [ ] **Step 3: Verify tests still green + commit**

Run: `.venv/bin/python -m pytest ml/tests/ -v`
Expected: PASS.

```bash
git add .gitignore README.md
git commit -m "docs: ReaPack install + dev/build instructions; ignore build outputs"
```

---

## Self-Review

**Spec coverage:**
- Two artifacts (engine / ReaScript) → Tasks 5–6 (engine), 7–8/11 (ReaScript+index). ✓
- ReaScript self-location (no `__file__`) → Task 1 (`script_dir`), Task 7 (`_module_dir`). ✓
- Dual-mode `start_engine` → Task 2. ✓
- Config schema `{engine_path, proc_dir, engine_version, port}` + dev shape → Task 4 (`write_config`), Task 7 (`setup_reaper.py`). ✓
- First-run bootstrap (prompt, download via `urllib`, checksum, unpack, prune, failure states) + `DN_RELEASE_BASE_URL` → Tasks 3–4, wired in 7, exercised in 9. ✓
- Signing/quarantine (ad-hoc sign, exec-bit restore, `xattr` doc) → Task 5 (sign), Task 4 (chmod), Task 11 (README). ✓
- Distribution via Releases not artifacts → Task 10 (release upload; combined `SHA256SUMS`). ✓
- CI matrix + per-platform smoke test → Task 10 (build+smoke), Task 6 (smoke script). ✓
- ReaPack index on gh-pages → Task 8 + Task 10 `reapack-index`. ✓
- Updates (version bump → re-download, prune) → Task 4 (`_prune_other_versions`), Task 7 (version check). ✓
- Local testing ladder → Tasks 1–4 (pytest), 5–6 (local freeze+smoke), 9 (`file://` + local ReaPack), 10 Step 5 (draft tag). ✓
- Action-identity preservation → documented behavior; no code needed (ReaPack overwrites same path). ✓
- Excludes (partitura/music21/pyfluidsynth/matplotlib/pyarrow/jupyter) → Task 5 spec `EXCLUDES`. ✓
- Weights set (6 files) → Task 5 `WEIGHT_FILES`, Task 10 `fetch_weights.FILES`, Task 6 usage. ✓

**Placeholder scan:** The only intentional placeholders are the identity literals `OWNER/REPO` and the HF `REPO_ID`, flagged in Global Constraints and Tasks 3/10 to be confirmed before those tasks. No `TBD`/"handle edge cases"/uncoded steps remain.

**Type consistency:** `platform_key()` values, `engine_exe`/`weights_dir`/`config_path` names, and the config keys `engine_path`/`proc_dir`/`engine_version`/`port` are used identically across dn_paths, bootstrap, engine_client, smoke_test, and the ReaScript. `asset_name`/`asset_url`/`sums_url`/`SHA256SUMS` names match between bootstrap, build_engine, and CI.

**Open confirmations before execution:** (1) GitHub `OWNER/REPO`; (2) the HF model repo id for `fetch_weights.py`; (3) whether the release tag `engine-v0.1.0` matches the pinned `PINNED_ENGINE_VERSION` = `0.1.0`.
