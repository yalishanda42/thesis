# Dynamics Needed — Reaper Plugin Distribution Design

**Date:** 2026-08-14
**Status:** Approved for planning
**Scope:** How the "Dynamics Needed" Reaper tool reaches real musicians — packaging, install, the inference engine, and CI/CD.

## Goal

Turn the current developer-only setup into a **public Reaper community release**: a stranger on macOS, Windows, or Linux installs the tool with zero hand-holding, no Python/venv, no `brew`, no repo clone, and gets working humanized-velocity prediction offline.

## Decisions (locked)

| Axis | Decision |
|------|----------|
| Audience | Public Reaper community release (unattended installs, cross-platform) |
| Plugin delivery channel | **ReaPack** (add-repo URL → browse → one-click install/update) |
| Inference location | **Bundled local** engine (offline, private, no hosting cost/uptime) |
| Engine delivery | Frozen per-OS binary, **downloaded on first run** from GitHub Releases (Approach A) |
| Models shipped | **All three** (`lgbm`, `mdn`, `categorical`) — transformer is the headline model, so torch stays in the freeze |
| Code signing | **No paid signing.** Free ad-hoc sign (macOS) + programmatic download to avoid quarantine/MOTW; documented `xattr` fallback |

## Current state (what we're replacing)

- `plugin/reaper/dynamics_needed.py` is a ReaImGui Python ReaScript. On launch it reads `dynamics_needed_config.json` from `RPR_GetResourcePath()`, then `sys.path.insert(0, <repo_root>/plugin/reaper)` to import its helper modules (`engine_client`, `dn_core`, `predict_worker`).
- `engine_client.start_engine` spawns `<venv_python> -m drum_dynamics.serve --port … --parent-pid …` and talks to it over `http://127.0.0.1:<port>`.
- `setup_reaper.py` hand-writes the config pointing at the developer's venv + repo root.
- The engine (`drum_dynamics.serve`) needs Python 3.12 + torch + lightgbm + numpy/pandas/joblib + the model weights (~26 MB: two 8.8 MB `.pt` heads + 4.7 MB LightGBM `.joblib`).

This requires a repo clone, a venv, an editable install, native deps (libomp for LightGBM), and a manual setup script — a non-starter for musicians.

## Runtime dependency footprint (verified)

The inference path imports only: `numpy`, `pandas`, `torch`, `joblib`, `lightgbm` (loaded from the joblib bundle), plus the `drum_dynamics` package and the stdlib HTTP server. Confirmed by tracing `serve/__main__` → `serve/models` → `serve/core` → `data/features`, `data/seqdata`, `models/model`, `models/heads`.

**Excluded from the freeze:** `partitura`, `music21`, `pyfluidsynth` (+ native FluidSynth), `matplotlib`, `pyarrow`, `jupyter`. None are on the inference path. This keeps the freeze to torch + LightGBM as the only heavy pieces.

## Architecture: two artifacts, versioned independently

### Artifact 1 — `dn-engine` (the frozen inference engine)

- **What:** PyInstaller **one-folder** freeze of `python -m drum_dynamics.serve`, one per platform: macOS arm64, macOS x86_64, Windows x64, Linux x64.
- **Contents:** embedded CPython + torch + lightgbm + numpy/pandas/joblib + the `drum_dynamics` package + the model weights (`lightgbm_model.joblib`, `lightgbm_features.json`, `mdn_meta.json`, `head_mdn.pt`, `transformer_meta.json`, `head_categorical.pt`).
- **One-folder, not one-file:** faster cold start (no per-launch self-extract) and materially fewer Windows AV false positives than one-file self-extractors.
- **Size:** ~250–400 MB per platform (torch dominates). Acceptable for a one-time download; unacceptable as a ReaPack payload or an Actions artifact (see §"Distribution channel").
- **Entry point unchanged:** it still serves `/health` and `/predict` on `127.0.0.1:<port>` and honors `--port`, `--parent-pid`, `--idle-timeout`. The freeze changes *how* it's launched, not its wire contract — so the whole HTTP client stays as-is.

### Artifact 2 — `dn-reascript` (the ReaPack package)

- **What:** the ReaScript and its stdlib helpers only — `dynamics_needed.py`, `engine_client.py`, `dn_core.py`, `predict_worker.py`. No weights, no binaries. Tiny.
- **Installed by ReaPack** into the deterministic path `<resource>/Scripts/<RepoName>/<category>/`.
- Depends on ReaImGui (already a ReaPack dependency, declared in the index).

## Component changes

### ReaScript self-location (replaces `repo_root` in config)

Reaper defines no `__file__` for Python ReaScripts, and there is no repo anymore. Because ReaPack installs a package to a **deterministic** location, the ReaScript computes its own directory from `RPR_GetResourcePath()` + the known repo/category name (chosen by us in `index.xml`) and `sys.path.insert`s that, instead of relying on `repo_root`. This is `__file__`-free and survives updates.

### `engine_client.start_engine` — dual-mode spawn (dev vs frozen)

`start_engine` gains two modes, chosen from the config:

- **Frozen (default, what users get):** spawn `[cfg["engine_path"], "--port", …, "--parent-pid", …]` where `engine_path` points at the frozen executable inside the installed engine folder. `cwd` becomes the engine folder; weights are found via `--proc-dir` pointing at the bundled weights dir.
- **Dev:** if the config carries `venv_python` + `repo_root` (or `DN_DEV=1` is set), spawn `[cfg["venv_python"], "-m", "drum_dynamics.serve", …]` as today. This preserves the fast inner loop — edit code, re-run, no re-freeze — so the freeze is only exercised when packaging is under test.

Everything else (`health`, `predict`, `ensure_engine`) is unchanged and stays pure-stdlib + unit-tested.

### Config schema shrinks and is auto-derived

The user-facing `dynamics_needed_config.json` becomes `{ "engine_path": <path to frozen exe>, "engine_version": <str>, "port": <int> }`, written by the first-run bootstrap (not by a human). For development, `setup_reaper.py` is kept as a dev convenience that writes the legacy `{ "venv_python", "repo_root", "port" }` shape, which `start_engine` recognizes as dev mode (see above). So the same config file drives both worlds; users never see the dev fields.

### First-run bootstrap (new, inside the ReaScript, pure stdlib)

On launch the ReaScript checks for an installed engine at `<resource>/DynamicsNeeded/engine/<pinned-version>/` matching the current platform.

- **If present and version matches:** proceed normally.
- **If absent or version differs:** show a dialog — *"Dynamics Needed needs to download its engine (~250 MB). Download now?"* On confirm:
  1. Resolve the platform-specific asset URL for the **pinned engine version** (the version string is a constant in the shipped ReaScript, so ReaScript and engine releases stay decoupled but coordinated). The base URL is overridable via `DN_RELEASE_BASE_URL` (a `file://` dir or `localhost` server) so the full flow can be exercised offline against a locally built freeze; unset, it defaults to the public GitHub Releases URL.
  2. Download the archive **and** its `SHA256SUMS` from the GitHub **Release** via Python `urllib` (programmatic download avoids `com.apple.quarantine` on macOS and Mark-of-the-Web on Windows).
  3. Verify SHA-256 against `SHA256SUMS`; abort with a clear message on mismatch.
  4. Unpack into `<resource>/DynamicsNeeded/engine/<version>/`, write the config, prune older engine version dirs on success.
- **Failure states** (no network, checksum mismatch, insufficient disk, unpack error) each show an actionable message with a manual-download link and the `xattr` fallback note.
- Idempotent and re-runnable; the existing async engine-start UI ("Starting engine…" / "Engine unreachable / Retry") absorbs the extra latency.

### Signing / quarantine handling

- **macOS:** CI **ad-hoc signs** the freeze (`codesign -s - --deep --force`). Ad-hoc signing is free and is *required* for any binary to execute at all on Apple Silicon. Because the bootstrap downloads programmatically (no quarantine attribute), Gatekeeper's notarization gate does not fire in the normal path.
- **Windows:** unsigned; the one-folder layout and programmatic download minimize SmartScreen/AV friction. README documents the "More info → Run anyway" path.
- **Fallback (documented, not the default path):** users who obtain the archive by browser get `xattr -dr com.apple.quarantine <dir>` (macOS) instructions.
- Design so paid signing/notarization can slot into CI later as a drop-in step without changing the download flow.

## Distribution channel: Releases, not Actions artifacts

Confirmed against GitHub docs (2026-08):

- **Actions artifacts** — 500 MB storage on GitHub Free, *shared* with Packages; default/max retention 90 days on public repos (auto-expire). Four engines at ~250–400 MB ≈ 1–1.6 GB total **exceeds the free quota and would expire.** → **Not** the distribution channel.
- **Release assets** — ≤ 2 GiB per file, up to 1000 assets per release, **no total-size limit, no bandwidth cap, do not count against the Actions/Packages quota.** Each engine is well under 2 GiB and never expires. → **This is the distribution channel.**

CI may still pass a build between jobs as a short-lived Actions *artifact* (transient, per-job), but the **user-facing engine is published as a Release asset**, which is what the bootstrap fetches.

## CI/CD (GitHub Actions)

- **Trigger:** tag `engine-vX.Y.Z`.
- **Build matrix:** `macos-14` (arm64), `macos-13` (x86_64), `windows-latest` (x64), `ubuntu-latest` (x64). Each job: install deps → PyInstaller freeze (with the exclude list) → (macOS) ad-hoc sign → **smoke test** (§Testing) → zip → emit `SHA256SUMS`.
- **Publish:** attach all platform archives + `SHA256SUMS` to the GitHub Release for the tag.
- **ReaPack index:** a separate step regenerates `index.xml` and publishes it to the existing `gh-pages` branch (same place the landing page deploys). ReaScript-only changes ship via a normal ReaPack release; engine-version bumps ship by editing the pinned version constant in the ReaScript and cutting a matching `engine-vX.Y.Z`.

## Updates

- **ReaScript:** standard ReaPack "Synchronize."
- **Engine:** when a ReaScript update raises the pinned engine version, the bootstrap detects the installed version differs and offers to download the new one; the previous version dir is pruned after a successful install.

## Action identity / binding preservation

Reaper derives a script's command ID from its **file path** (`_RS…` hash). Implications:

- **Across ReaPack updates:** the file stays at the same installed path, so the command ID is stable — users' keybinding / toolbar / MIDI-editor-menu bindings survive "Synchronize" untouched (bind once, never re-bind). A core reason to ship via ReaPack.
- **Between the dev repo copy and the ReaPack-installed copy:** different paths → different command IDs, so the installed build must be registered/bound once, separately from the dev action. Stable thereafter.
- Editing the dev `.py` needs no re-register (deferred script, re-read each launch; same path, same ID).

## Local testing & dev loop

The two hooks above (dual-mode `start_engine`, `DN_RELEASE_BASE_URL`) make almost the entire pipeline testable without cutting a real release. The ladder:

| Layer | What it proves | Needs |
|-------|----------------|-------|
| `pytest` | client + bootstrap logic — URL build, SHA-256 verify, path/idempotency, failure branches (network mocked) | nothing |
| Local PyInstaller freeze + smoke | hidden-imports OK **for the dev's own OS**; `/health` + `/predict` work | dev platform only |
| Bootstrap against a `file://` release | real resolve → download → checksum → unpack → spawn path, offline | `DN_RELEASE_BASE_URL` |
| **Local ReaPack repo** | full "browse → install → Synchronize → run", action registers/binds correctly | import a `file://` `index.xml` into ReaPack |
| Draft / pre-release GitHub tag | the real GitHub Releases download path end-to-end | one throwaway tag |

The one thing not doable locally is building the **other** platforms' freezes (a Windows/Linux binary can't be reliably produced from macOS) — that is exactly what the CI matrix + per-platform smoke test cover. The dev validates their own OS locally; CI validates the other three.

## Testing (CI)

- `engine_client` + bootstrap stay pure-stdlib and unit-tested: cover version resolution, asset-URL construction, SHA-256 verification, path/idempotency logic, and failure branches (mock the network).
- **Per-platform CI smoke test** (the key de-risker): unzip the freeze, launch it, poll `/health`, POST one `/predict`, assert a valid velocity map. This catches **PyInstaller hidden-import gaps** — torch and lightgbm are notorious for these — *before* a release ships.
- Existing `dn_core` / `predict_worker` / serve tests remain green.

## Repo additions

- `ml/packaging/` — PyInstaller spec, build script, and the explicit include/exclude list.
- `plugin/reaper/reapack/` — `index.xml` generator + package metadata.
- `.github/workflows/engine-release.yml` — the build/sign/smoke/publish matrix + ReaPack index publish.
- `setup_reaper.py` — demoted to dev-only or removed.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| PyInstaller hidden-import gaps (torch/lightgbm) | Per-platform CI smoke test gates every release |
| Freeze size (~250–400 MB × 4) | One-time download via Releases (free, uncapped); clear size prompt in the bootstrap |
| macOS Gatekeeper / arm64 must-be-signed | Free ad-hoc sign + programmatic (non-quarantined) download; `xattr` fallback documented |
| Windows SmartScreen / AV false positives | One-folder build + programmatic download; documented click-through; signing slot for later |
| ReaScript can't self-locate (no `__file__`) | Deterministic ReaPack install path derived from `RPR_GetResourcePath()` + known repo/category name |
| ReaScript/engine version drift | Pinned engine-version constant in the ReaScript ties the two together per release |

## Explicitly out of scope (YAGNI)

- Hosted/remote inference and any hybrid path (rejected: hosting cost/uptime/privacy).
- Paid Apple notarization / Windows EV signing (design leaves a drop-in slot; not built now).
- Delta/partial engine updates (full re-download on version bump is fine at this scale).
- Auto-update of the engine without user consent (always prompt).
