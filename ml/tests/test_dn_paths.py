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
