from drum_dynamics.serve.server import route, pid_alive, should_exit


class _Eng:
    def levels(self):
        return {"models": ["lgbm", "mdn"], "styles": ["funk"], "genres": ["funk"]}

    def predict(self, request):
        return {0: 99}


def test_route_health():
    status, body = route(_Eng(), "GET", "/health", None)
    assert status == 200 and body["status"] == "ok" and body["models"] == ["lgbm", "mdn"]


def test_route_predict_stringifies_keys():
    status, body = route(_Eng(), "POST", "/predict", {"model": "lgbm", "notes": []})
    assert status == 200 and body == {"velocities": {"0": 99}}


def test_route_unknown_path_404():
    status, _ = route(_Eng(), "GET", "/nope", None)
    assert status == 404


def test_pid_alive_current_process():
    import os
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_147_483_000) is False   # implausibly-high PID


def test_should_exit_conditions():
    assert should_exit(now=100.0, last_request=0.0, idle_timeout=30, parent_alive=True) is True
    assert should_exit(now=10.0, last_request=0.0, idle_timeout=30, parent_alive=True) is False
    assert should_exit(now=10.0, last_request=0.0, idle_timeout=30, parent_alive=False) is True
