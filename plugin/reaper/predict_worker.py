"""Background predict worker: debounced, coalescing, pure (no Reaper API).

The UI thread calls submit() on every change; the worker thread runs the most
recent request after a short debounce, discarding superseded ones, and stashes
the result for the UI thread to read. Deterministic via _process_once().
"""
import threading
import time


class PredictWorker:
    def __init__(self, predict_fn, debounce=0.15):
        self._predict_fn = predict_fn
        self._debounce = debounce
        self._lock = threading.Lock()
        self._pending = None          # (seq, request) or None
        self._result = None           # (seq, velocities) or None
        self._error = None            # str or None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def submit(self, seq, request):
        with self._lock:
            self._pending = (seq, request)
        self._wake.set()

    def _take_pending(self):
        with self._lock:
            p = self._pending
            self._pending = None
            return p

    def _process_once(self):
        p = self._take_pending()
        if p is None:
            return
        seq, request = p
        try:
            velocities = self._predict_fn(request)
        except Exception as e:            # keep last good result on failure
            with self._lock:
                self._error = str(e)
            return
        with self._lock:
            self._result = (seq, velocities)
            self._error = None

    def result(self):
        with self._lock:
            return self._result

    def last_error(self):
        with self._lock:
            return self._error

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            time.sleep(self._debounce)     # debounce; later submits coalesce
            self._process_once()
