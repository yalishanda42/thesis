"""Engine lifecycle + HTTP for the Dynamics Needed panel.

Pure stdlib, no Reaper API, so it runs under normal pytest. Extracted from the
old ReaScript so the network/subprocess logic is unit-tested.
"""
import json
import os
import subprocess
import time
import urllib.request


def base_url(cfg):
    return "http://127.0.0.1:{}".format(cfg.get("port", 8765))


def health(cfg, timeout=1.0):
    try:
        with urllib.request.urlopen(base_url(cfg) + "/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


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


def predict(cfg, request, timeout=60.0):
    data = json.dumps(request).encode()
    req = urllib.request.Request(base_url(cfg) + "/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_engine(cfg, tries=360, delay=0.5, sleep=time.sleep):
    # tries*delay = 180s: the FIRST launch of a freshly downloaded frozen bundle
    # triggers a macOS Gatekeeper/XProtect scan of its many bundled dylibs, which
    # can take well over a minute on a cold start (subsequent starts are ~2s).
    # The panel shows "Starting engine..." while this poll runs on a worker thread.
    h = health(cfg)
    if h:
        return h
    start_engine(cfg)
    for _ in range(tries):
        sleep(delay)
        h = health(cfg)
        if h:
            return h
    return None
