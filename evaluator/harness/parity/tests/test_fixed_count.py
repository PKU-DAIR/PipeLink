from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluator.harness.parity import fixed_count as fixed
from evaluator.harness.parity.example import _ORDERS


def pair(speedup=0.5):
    blocks = []
    for order in _ORDERS:
        blocks.append({"order": order, "milliseconds": [0.02 if arm == "A" else 0.02 * speedup for arm in order]})
    return {"protocol": fixed.PROTOCOL, "status": "PASS", "repetitions": 10,
            "blocks": blocks, "ratio_onetu_over_linked": speedup}


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    directory = tmp_path_factory.mktemp("fixed-cpp")
    source = Path(__file__).with_name("fixed_count_probe.cpp")
    output = directory / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(fixed.REPO), str(source), "-o", str(output)], check=True)
    return output


@pytest.mark.parametrize("mode", ["normal", "short"])
def test_native_protocol_cancels_instance_bias_without_erasing_image_cost(native_probe, mode):
    completed = subprocess.run([str(native_probe), mode], text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["scored"] == 320
    assert result["warmup"] == 32
    assert result["reloads"] == 1
    assert fixed._valid_payload(result)
    assert result["pair"]["ratio_onetu_over_linked"] == pytest.approx(0.5)
    assert fixed.arm_ms(result["pair"], "A") > 0


@pytest.mark.parametrize("mode,scored,warmup", [("reload-error", 160, 16), ("invalid", 170, 32)])
def test_native_protocol_cannot_publish_after_reload_or_elapsed_failure(native_probe, mode, scored, warmup):
    completed = subprocess.run([str(native_probe), mode], text=True, capture_output=True)
    assert completed.returncode != 0
    result = json.loads(completed.stdout)
    assert "pair" not in result
    assert result["scored"] == scored
    assert result["warmup"] == warmup


@pytest.mark.parametrize("defect", ["budget", "order", "empty", "zero", "nan", "boolean"])
def test_native_payload_rejects_incomplete_or_nonphysical_measurements(defect):
    value = pair()
    if defect == "budget": value["repetitions"] = 20
    elif defect == "order": value["blocks"][4]["order"] = "AABB"
    elif defect == "empty": value["blocks"].pop()
    elif defect == "zero": value["blocks"][7]["milliseconds"][3] = 0
    elif defect == "nan": value["ratio_onetu_over_linked"] = float("nan")
    else: value["blocks"][0]["milliseconds"][0] = True
    assert not fixed._valid_payload({"pair": value})


@pytest.mark.parametrize("contended,healthy,stopped,code,admitted", [
    (False, True, True, 0, True),
    (True, True, True, 0, False),
    (False, False, True, 0, False),
    (False, True, False, 0, False),
    (False, True, True, 7, False),
])
def test_execution_admission_is_independent_of_performance_miss(monkeypatch, tmp_path,
        contended, healthy, stopped, code, admitted):
    def start(command, **kwargs):
        root = Path(command[-1]).parent
        request = json.loads(Path(command[-1]).read_text())
        assert request["gpus"] == [3]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        fixed._save(root / "worker-result.json", {
            "payload": {"pair": pair(0.5)}, "command_returncode": code,
            "clock": {"healthy": healthy, "stopped": stopped}, "timed_out": False,
        })
        return SimpleNamespace(returncode=0, communicate=lambda: (
            "", f"gpu_window: waited=0s ran=1s contended={int(contended)} rc=0\n"))
    monkeypatch.setattr(fixed.subprocess, "Popen", start)
    record = fixed.run_native_pair(["native"], env={}, gpus=[3], clock_mhz=1230, out_dir=tmp_path/"case")
    assert record["admitted"] is admitted
    assert record["payload"]["pair"]["ratio_onetu_over_linked"] == 0.5


@pytest.mark.parametrize("clock_mhz", [None, 0])
@pytest.mark.parametrize("contended,admitted", [(False, True), (True, False)])
def test_unlocked_clock_is_admitted_unless_contended(monkeypatch, tmp_path, clock_mhz, contended, admitted):
    def start(command, **kwargs):
        assert kwargs["env"]["PIPELINK_LOCK_SM_MHZ"] == ""
        fixed._save(Path(command[-1]).parent / "worker-result.json", {
            "payload": {"pair": pair(0.5)}, "command_returncode": 0, "timed_out": False,
            "clock": {"locked": False, "sm_mhz": None, "healthy": True, "stopped": True},
        })
        return SimpleNamespace(returncode=0, communicate=lambda: (
            "", f"gpu_window: waited=0s ran=1s contended={int(contended)} rc=0\n"))
    monkeypatch.setattr(fixed.subprocess, "Popen", start)
    record = fixed.run_native_pair(["native"], env={}, gpus=[3], clock_mhz=clock_mhz, out_dir=tmp_path/"case")
    assert record["clock_locked"] is False
    assert record["admitted"] is admitted


def test_unlocked_clock_lease_never_calls_gpulockd(monkeypatch, tmp_path):
    from evaluator.harness.parity import example
    monkeypatch.setattr(example, "_run", lambda *args, **kwargs: pytest.fail("gpulockd called"))
    with example.clock_lease("1", env={"PIPELINK_LOCK_SM_MHZ": ""}, log_dir=tmp_path) as record:
        assert record["locked"] is False and record["sm_mhz"] is None
    assert record["healthy"] and record["stopped"]


def worker_setup(monkeypatch, tmp_path, command, *, timeout=2, input=None):
    from evaluator.harness.parity import example
    record = {"healthy": False, "stopped": False}
    @contextmanager
    def lease(*args, **kwargs):
        try:
            yield record
            record["healthy"] = True
        finally:
            record["stopped"] = True
    monkeypatch.setattr(example, "clock_lease", lease)
    monkeypatch.setattr(fixed.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0))
    monkeypatch.setenv("PIPELINK_GPU_WINDOW", f"{os.getppid()}.test")
    request = tmp_path / "request.json"
    fixed._save(request, {"command": command, "gpus": [1], "timeout": timeout, "input": input})
    return request, record


def test_worker_delivers_eof_request_and_releases_lease_on_command_failure(monkeypatch, tmp_path):
    code = "import json,sys; value=json.load(sys.stdin); sys.exit(7 if value['action']=='paired' else 9)"
    request, lease = worker_setup(monkeypatch, tmp_path, [sys.executable, "-c", code],
                                  input=json.dumps({"action": "paired"}))
    assert fixed._worker(request) == 1
    result = json.loads((tmp_path / "worker-result.json").read_text())
    assert result["command_returncode"] == 7
    assert not result["timed_out"]
    assert lease["stopped"]


def test_worker_timeout_terminates_owned_native_before_releasing_lease(monkeypatch, tmp_path):
    request, lease = worker_setup(monkeypatch, tmp_path,
        [sys.executable, "-c", "import time; time.sleep(120)"], timeout=0.1)
    assert fixed._worker(request) == 1
    result = json.loads((tmp_path / "worker-result.json").read_text())
    owned = json.loads((tmp_path / "owned.json").read_text())
    assert result["timed_out"]
    assert not fixed._matches(owned["native"])
    assert lease["stopped"]


@pytest.mark.parametrize("when", ["acquire", "release"])
def test_interruption_cannot_escape_owned_lease_acquisition_or_release(monkeypatch, tmp_path, when):
    from evaluator.harness.parity import example
    request, record = worker_setup(monkeypatch, tmp_path, [sys.executable, "-c", "print('{}')"])

    @contextmanager
    def interrupted_lease(*args, **kwargs):
        if when == "acquire":
            os.kill(os.getpid(), signal.SIGTERM)
        try:
            yield record
            record["healthy"] = True
        finally:
            if when == "release":
                os.kill(os.getpid(), signal.SIGTERM)
            record["stopped"] = True

    monkeypatch.setattr(example, "clock_lease", interrupted_lease)
    assert fixed._worker(request) == 1
    result = json.loads((tmp_path / "worker-result.json").read_text())
    assert result["clock"]["stopped"]
    assert result["command_returncode"] == (None if when == "acquire" else 0)


def test_cancellation_does_not_signal_an_unrelated_identity(tmp_path):
    window = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        fixed._save(tmp_path / "owned.json", {"worker": fixed._identity(unrelated.pid),
                                             "window_token": "different-window.token"})
        fixed._cancel_window(window, tmp_path)
        assert window.poll() is not None
        assert unrelated.poll() is None
    finally:
        fixed._stop_native(window)
        fixed._stop_native(unrelated)


def test_existing_window_is_rejected_before_any_nested_lock(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        fixed.run_native_pair(["native"], env={"PIPELINK_GPU_WINDOW":"outer"},
                              gpus=[1], clock_mhz=1155, out_dir=tmp_path/"never-started")
    assert not (tmp_path/"never-started").exists()
