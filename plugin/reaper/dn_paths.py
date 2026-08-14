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
