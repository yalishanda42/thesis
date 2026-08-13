import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
import engine_client


def test_base_url_uses_port():
    assert engine_client.base_url({"port": 9000}) == "http://127.0.0.1:9000"


def test_base_url_defaults_to_8765():
    assert engine_client.base_url({}) == "http://127.0.0.1:8765"


def test_health_returns_none_on_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(engine_client.urllib.request, "urlopen", boom)
    assert engine_client.health({"port": 8765}) is None


def test_ensure_engine_returns_immediately_when_up(monkeypatch):
    monkeypatch.setattr(engine_client, "health", lambda cfg, **k: {"status": "ok"})
    started = []
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: started.append(True))
    out = engine_client.ensure_engine({"port": 8765}, sleep=lambda s: None)
    assert out == {"status": "ok"}
    assert started == []  # never started, it was already up


def test_ensure_engine_starts_then_polls_until_ready(monkeypatch):
    calls = {"n": 0}
    def fake_health(cfg, **k):
        calls["n"] += 1
        return {"status": "ok"} if calls["n"] >= 3 else None
    started = []
    monkeypatch.setattr(engine_client, "health", fake_health)
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: started.append(True))
    out = engine_client.ensure_engine({"port": 8765}, tries=5, sleep=lambda s: None)
    assert out == {"status": "ok"}
    assert started == [True]  # started exactly once


def test_ensure_engine_gives_up_and_returns_none(monkeypatch):
    monkeypatch.setattr(engine_client, "health", lambda cfg, **k: None)
    monkeypatch.setattr(engine_client, "start_engine", lambda cfg: None)
    assert engine_client.ensure_engine({"port": 8765}, tries=3, sleep=lambda s: None) is None


def test_start_engine_spawns_expected_argv(monkeypatch, tmp_path):
    seen = {}
    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        class P: pass
        return P()
    monkeypatch.setattr(engine_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_client.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(engine_client, "open", lambda *a, **k: open(os.devnull, "a"), raising=False)
    cfg = {"venv_python": "/venv/py", "repo_root": str(tmp_path), "port": 8765}
    engine_client.start_engine(cfg)
    assert seen["argv"][:4] == ["/venv/py", "-m", "drum_dynamics.serve", "--port"]
    assert "8765" in seen["argv"]
    assert seen["cwd"] == str(tmp_path)


def test_predict_returns_response_dict(monkeypatch):
    """Test predict function with mocked urlopen."""
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"velocities": {"0": 97}}'

    def fake_urlopen(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(engine_client.urllib.request, "urlopen", fake_urlopen)
    result = engine_client.predict({"port": 8765}, {"model": "mdn"})
    assert result == {"velocities": {"0": 97}}
