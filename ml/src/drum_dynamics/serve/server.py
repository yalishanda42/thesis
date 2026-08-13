"""Stdlib HTTP wrapper + lifecycle watchdog for the inference engine."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def route(engine, method, path, body):
    """Pure request router over a loaded engine. Returns (status, json-able dict)."""
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", **engine.levels()}
    if method == "POST" and path == "/predict":
        vel = engine.predict(body)
        return 200, {"velocities": {str(k): v for k, v in vel.items()}}
    return 404, {"error": "not found"}


def pid_alive(pid):
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists but not owned by us
    return True


def should_exit(now, last_request, idle_timeout, parent_alive):
    if not parent_alive:
        return True
    if idle_timeout and (now - last_request) > idle_timeout:
        return True
    return False


def _make_handler(engine, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # silence default stderr logging
            pass

        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            status, body = route(engine, "GET", self.path, None)
            self._send(status, body)

        def do_POST(self):
            state["last_request"] = time.monotonic()
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                status, out = route(engine, "POST", self.path, body)
            except Exception as e:                       # noqa: BLE001 - report to client
                status, out = 500, {"error": str(e)}
            self._send(status, out)

    return Handler


def run(engine, port=8765, parent_pid=None, idle_timeout=1800):
    state = {"last_request": time.monotonic()}
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(engine, state))

    def watchdog():
        while True:
            time.sleep(5)
            if should_exit(time.monotonic(), state["last_request"], idle_timeout, pid_alive(parent_pid)):
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"Dynamics Needed engine on http://127.0.0.1:{port} (parent={parent_pid})")
    httpd.serve_forever()
