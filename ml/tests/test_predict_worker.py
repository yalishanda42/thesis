import os
import sys

sys.path.insert(0, os.path.join("plugin", "reaper"))
from predict_worker import PredictWorker


def test_no_pending_is_noop():
    w = PredictWorker(lambda req: {0: 1})
    w._process_once()
    assert w.result() is None


def test_processes_latest_and_coalesces():
    calls = []
    w = PredictWorker(lambda req: (calls.append(req), {0: req["t"]})[1])
    w.submit(1, {"t": 1.0})
    w.submit(2, {"t": 1.5})   # supersedes seq 1 before processing
    w._process_once()
    assert len(calls) == 1 and calls[0]["t"] == 1.5
    assert w.result() == (2, {0: 1.5})


def test_error_keeps_last_good_result():
    w = PredictWorker(lambda req: {0: 42})
    w.submit(1, {"t": 1.0})
    w._process_once()
    assert w.result() == (1, {0: 42})

    def boom(req):
        raise RuntimeError("predict failed")
    w._predict_fn = boom
    w.submit(2, {"t": 2.0})
    w._process_once()
    assert w.result() == (1, {0: 42})          # unchanged
    assert "predict failed" in w.last_error()


def test_success_clears_previous_error():
    w = PredictWorker(lambda req: {0: 7})
    w._predict_fn = lambda req: (_ for _ in ()).throw(RuntimeError("x"))
    w.submit(1, {"t": 1.0}); w._process_once()
    assert w.last_error() is not None
    w._predict_fn = lambda req: {0: 7}
    w.submit(2, {"t": 1.0}); w._process_once()
    assert w.last_error() is None
    assert w.result() == (2, {0: 7})
