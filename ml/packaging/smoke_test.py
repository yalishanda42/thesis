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
    engine_dir = os.path.abspath(sys.argv[1])
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
