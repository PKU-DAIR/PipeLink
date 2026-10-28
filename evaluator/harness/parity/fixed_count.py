"""Run native fixed-count campaign commands inside their own GPU window and clock lease."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import median
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "paired-reload-v2"


def arm_ms(pair, arm):
    return median(sum(value for label, value in zip(block["order"], block["milliseconds"])
                      if label == arm) / (2 * pair["repetitions"])
                  for block in pair["blocks"])


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return {"pid": pid, "pgid": os.getpgid(pid), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _matches(identity):
    return bool(identity and _identity(identity["pid"]) == identity)


def _stop_native(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def _interrupts():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    depth = 0
    pending = None

    def interrupt(sig, frame):
        nonlocal pending
        if depth:
            pending = sig
            return
        # Ignore further signals so a second one cannot skip native termination and lease cleanup.
        for item in previous:
            signal.signal(item, signal.SIG_IGN)
        raise InterruptedError(f"interrupted by signal {sig}")

    @contextmanager
    def defer():
        nonlocal depth, pending
        depth += 1
        try:
            yield
        finally:
            depth -= 1
            if depth == 0 and pending is not None:
                sig, pending = pending, None
                if sys.exc_info()[0] is None:
                    interrupt(sig, None)
                else:
                    for item in previous:
                        signal.signal(item, signal.SIG_IGN)

    for sig in previous:
        signal.signal(sig, interrupt)
    try:
        yield defer
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _valid_payload(payload):
    from .example import _ORDERS
    pair = payload.get("pair", {}) if isinstance(payload, dict) else {}
    if pair.get("protocol") != PROTOCOL or pair.get("status") != "PASS" or pair.get("repetitions") != 10:
        return False
    blocks = pair.get("blocks", [])
    if len(blocks) != 8:
        return False
    for order, block in zip(_ORDERS, blocks):
        values = block.get("milliseconds", [])
        if block.get("order") != order or len(values) != 4:
            return False
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
            return False
    ratio = pair.get("ratio_onetu_over_linked")
    return not isinstance(ratio, bool) and isinstance(ratio, (int, float)) and math.isfinite(ratio) and ratio > 0


def _worker(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    root = request_path.parent
    native = None
    stdout = stderr = ""
    record = {"command_returncode": None, "clock": {}, "timed_out": False, "payload": None}
    with _interrupts() as defer:
        token = os.environ.get("PIPELINK_GPU_WINDOW", "")
        if not token:
            raise RuntimeError("fixed-count worker requires its GPU window")
        parent_pid = int(token.split(".", 1)[0])
        # Die with the window supervisor so an orphan never acquires a lease.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
        if os.getppid() != parent_pid:
            raise RuntimeError("GPU-window supervisor exited before worker startup")
        owned = {"worker": _identity(os.getpid()), "window_token": token}
        _save(root / "owned.json", owned)
        try:
            from .example import clock_lease
            leases = ExitStack()
            try:
                # Defer signals until the lease is registered for cleanup:
                # clock_lease starts the daemon before its own try/finally.
                with defer():
                    record["clock"] = leases.enter_context(clock_lease(
                        ",".join(map(str, request["gpus"])), env=os.environ, log_dir=root / "clock"))
                try:
                    with defer():
                        native = subprocess.Popen(request["command"], stdin=subprocess.PIPE,
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                  text=True, start_new_session=True)
                        owned["native"] = _identity(native.pid)
                        _save(root / "owned.json", owned)
                    try:
                        stdout, stderr = native.communicate(input=request.get("input"), timeout=request["timeout"])
                    except subprocess.TimeoutExpired:
                        record["timed_out"] = True
                        _stop_native(native)
                        stdout, stderr = native.communicate()
                    record["command_returncode"] = native.returncode
                finally:
                    with defer():
                        _stop_native(native)
            finally:
                with defer():
                    leases.close()
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            (root / "native.stdout").write_text(stdout)
            (root / "native.stderr").write_text(stderr)
            if stdout.strip() and request.get("parse_json", True):
                try:
                    record["payload"] = json.loads(stdout)
                except ValueError as error:
                    record["parse_error"] = str(error)
            _save(root / "worker-result.json", record)
    return 0 if (record["command_returncode"] == 0 and not record["timed_out"]
                 and record["clock"].get("healthy") and record["clock"].get("stopped")
                 and not record.get("error")) else 1


def _cancel_window(window, root):
    path = root / "owned.json"
    owned = json.loads(path.read_text()) if path.exists() else {}
    token_matches = owned.get("window_token", "").startswith(f"{window.pid}.")
    worker = owned.get("worker") if token_matches else None
    if _matches(worker):
        os.kill(worker["pid"], signal.SIGTERM)
    else:
        # No worker yet, so killing the window is safe: PDEATHSIG and the
        # worker's parent check stop a late orphan from acquiring a lease.
        if window.poll() is None:
            os.killpg(window.pid, signal.SIGTERM)
    try:
        return window.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        owned = json.loads(path.read_text()) if path.exists() else {}
        if owned.get("window_token", "").startswith(f"{window.pid}."):
            for name in ("native", "worker"):
                identity = owned.get(name)
                if _matches(identity):
                    os.killpg(identity["pgid"], signal.SIGKILL)
        if window.poll() is None:
            os.killpg(window.pid, signal.SIGTERM)
        return window.communicate(timeout=10)


def _run_windowed(command, *, env, gpus, clock_mhz, out_dir, timeout, input, parse_json):
    if env.get("PIPELINK_GPU_WINDOW"):
        raise ValueError("the execution controller owns its GPU window; remove the outer gpu_window.sh")
    gpus = list(gpus)
    # clock_mhz None or 0 selects unlocked clocks: no lease, timings are relative only.
    locked = bool(clock_mhz)
    if not gpus or len(set(gpus)) != len(gpus) or (locked and clock_mhz < 0):
        raise ValueError("fixed-count timing requires distinct GPUs and a qualified clock or 0 (unlocked)")
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    child_env = {**env, "PIPELINK_GPUS": ",".join(map(str, gpus)),
                 "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                 "PIPELINK_LOCK_SM_MHZ": str(clock_mhz) if locked else ""}
    request = {"command": [str(value) for value in command], "gpus": gpus,
               "clock_mhz": clock_mhz, "timeout": timeout, "input": input, "parse_json": parse_json}
    request_path = root / "request.json"
    _save(request_path, request)
    window = None
    stdout = stderr = ""
    with _interrupts() as defer:
        try:
            with defer():
                window = subprocess.Popen([str(REPO / "tools/gpu_window.sh"), sys.executable,
                                           "-m", __name__, "--worker", str(request_path)],
                                          cwd=REPO, env=child_env, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, start_new_session=True)
            stdout, stderr = window.communicate()
        except BaseException:
            if window is not None:
                with defer():
                    stdout, stderr = _cancel_window(window, root)
            (root / "window.stdout").write_text(stdout)
            (root / "window.stderr").write_text(stderr)
            raise
    (root / "window.stdout").write_text(stdout)
    (root / "window.stderr").write_text(stderr)
    match = re.search(r"gpu_window: waited=(\d+)s ran=(\d+)s contended=([01]) rc=(-?\d+)", stderr)
    window_record = {"returncode": window.returncode, "contended": True}
    if match:
        window_record.update(waited_seconds=int(match[1]), ran_seconds=int(match[2]), contended=match[3] == "1")
    worker_path = root / "worker-result.json"
    worker = json.loads(worker_path.read_text()) if worker_path.exists() else {"payload": None, "clock": {}}
    admitted = (window.returncode == 0 and not window_record["contended"]
                and worker.get("command_returncode") == 0 and not worker.get("timed_out")
                and worker["clock"].get("healthy") and worker["clock"].get("stopped"))
    record = {**worker, "clock_locked": locked, "execution_policy": "qualified-lease-window", "admitted": bool(admitted),
              "gpus": gpus, "cuda_visible_devices": child_env["CUDA_VISIBLE_DEVICES"],
              "window": window_record, "logs": {name: str(root / name) for name in
                  ("native.stdout", "native.stderr", "window.stdout", "window.stderr")}}
    return record


def run_native_pair(command, *, env, gpus, clock_mhz, out_dir, timeout=300, input=None):
    """Run one fixed-count comparison and record whether its execution was healthy.

    Call from the main thread; this function owns the GPU window and clock lease.
    The command sees the selected GPUs as ordinals 0..N-1 and reads ``input``
    until EOF. ``admitted`` reflects execution, lease and window health and a
    valid payload, never the measured speedup. Numerical and sanitizer checks
    and the performance threshold belong to the campaign. There are no retries.
    """
    record = _run_windowed(command, env=env, gpus=gpus, clock_mhz=clock_mhz,
                           out_dir=out_dir, timeout=timeout, input=input, parse_json=True)
    record["protocol"] = PROTOCOL
    record["admitted"] = record["admitted"] and _valid_payload(record.get("payload"))
    _save(Path(out_dir) / "execution.json", record)
    return record


def run_locked_command(command, *, env, gpus, clock_mhz, out_dir, timeout=1800, input=None):
    """Run a numerical or sanitizer check under the same window, lease and cleanup.

    ``admitted`` reflects execution, lease and window health only; the caller
    reads the command's own result files.
    """
    record = _run_windowed(command, env=env, gpus=gpus, clock_mhz=clock_mhz,
                           out_dir=out_dir, timeout=timeout, input=input, parse_json=False)
    _save(Path(out_dir) / "execution.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    return _worker(parser.parse_args().worker)


if __name__ == "__main__":
    raise SystemExit(main())
