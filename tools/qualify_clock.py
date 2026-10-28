#!/usr/bin/env python3
"""Find one sustainable SM clock for a machine using simultaneous dense GEMMs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

if __package__:
    from . import gpulockd, pipelink_env
else:
    import gpulockd
    import pipelink_env

ROOT = Path(__file__).resolve().parents[1]
MATRIX_SIZE = 8192
WARMUP_SECONDS = 2.0
EXCLUDE_SECONDS = 5.0
SAMPLE_SECONDS = 0.1
CONTENTION_POLL_SECONDS = 30.0


def supported_clocks(xml: str) -> list[int]:
    # nvidia-smi forbids combining -d SUPPORTED_CLOCKS with -x, so parse the full XML query.
    groups = ET.fromstring(xml).findall(".//supported_clocks/supported_mem_clock")
    highest = max(groups, key=lambda group: int(group.findtext("value").split()[0]))
    return sorted({int(node.text.split()[0]) for node in highest.findall("supported_graphics_clock")},
                  reverse=True)


def highest_passing(clocks, trial):
    """Binary-search descending ``clocks`` for the highest passing one; return its index or None.

    Assumes pass/fail is monotonic: if a clock passes, every lower clock passes.
    """
    low, high, best = 0, len(clocks) - 1, None
    while low <= high:
        middle = (low + high) // 2
        if trial(clocks[middle]):
            best, high = middle, middle - 1
        else:
            low = middle + 1
    return best


def sample_violation(sample, clock, tolerance):
    return bool(sample["reasons"] & gpulockd.SLOWDOWN_REASON_MASK) or (
        sample["utilization"] > 0 and abs(sample["sm_mhz"] - clock) > tolerance)


def summarize(samples, clock, tolerance):
    measured = [s for s in samples if s["seconds"] >= EXCLUDE_SECONDS]
    loaded = [s for s in measured if s["utilization"] > 0]
    violations = [s for s in measured if sample_violation(s, clock, tolerance)]
    peak = max(measured, key=lambda s: s["power_w"])
    return {
        "samples": len(measured), "loaded_samples": len(loaded),
        "violations": len(violations), "first_violation": next(iter(violations), None),
        "sw_power_cap_fraction": sum(bool(s["reasons"] & 0x4) for s in measured) / len(measured),
        "max_power_w": peak["power_w"], "power_limit_w": peak["power_limit_w"],
        "max_power_fraction_of_limit": max(s["power_w"] / s["power_limit_w"] for s in measured),
        "max_temperature_c": max(s["temperature_c"] for s in measured),
        "min_loaded_sm_mhz": min((s["sm_mhz"] for s in loaded), default=None),
        "max_loaded_sm_mhz": max((s["sm_mhz"] for s in loaded), default=None),
    }


def write_env(path: Path, clock: int, timestamp: str):
    original = path.read_bytes() if path.exists() else b""
    if path.exists():
        shutil.copy2(path, path.with_name(f".env.backup-{timestamp}"))
    replacement = f"PIPELINK_LOCK_SM_MHZ={clock}".encode()
    pattern = rb"(?m)^[ \t]*PIPELINK_LOCK_SM_MHZ[ \t]*=[^\r\n]*"
    if re.search(pattern, original):
        updated = re.sub(pattern, replacement, original)
    else:
        updated = original + (b"\n" if original and not original.endswith(b"\n") else b"") + replacement + b"\n"
    path.write_bytes(updated)


def _worker(gpu_uuid, connection):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    try:
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        capability = torch.cuda.get_device_capability()
        connection.send({"torch": torch.__version__, "cuda": torch.version.cuda,
                         "capability": list(capability)})
        while True:
            phase = connection.recv()
            if phase == "exit":
                return
            if phase == "fp8" and capability < (8, 9):
                connection.send({"skipped": f"FP8 tensor cores unsupported on sm{capability[0]}{capability[1]}"})
                continue
            dtype = torch.bfloat16 if phase == "bf16" else torch.float8_e4m3fn
            a = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device="cuda", dtype=torch.bfloat16).to(dtype)
            b = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device="cuda", dtype=torch.bfloat16).to(dtype).t()
            out = torch.empty((MATRIX_SIZE, MATRIX_SIZE), device="cuda", dtype=torch.bfloat16)
            scale = torch.ones((), device="cuda", dtype=torch.float32)

            def batch():
                for _ in range(16):
                    if phase == "bf16":
                        torch.matmul(a, b, out=out)
                    else:
                        torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                         out_dtype=torch.bfloat16, out=out)
                torch.cuda.synchronize()

            until = time.monotonic() + WARMUP_SECONDS
            while time.monotonic() < until:
                batch()
            connection.send({"ready": True})
            start, seconds = connection.recv()
            time.sleep(max(0, start - time.monotonic()))
            until = start + seconds
            while time.monotonic() < until:
                batch()
            connection.send({"done": True})
            del a, b, out, scale
    except BaseException as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


class InvalidTrial(RuntimeError):
    """A trial voided by contention or an operational error; says nothing about the clock."""


class Qualification:
    def __init__(self, args, config, record):
        import pynvml
        self.nvml = pynvml
        self.args, self.config, self.record = args, config, record
        self.gpus = args.gpus
        self.gpu_csv = ",".join(map(str, self.gpus))
        self.workers = {}
        self.leased = False
        self.stop = threading.Event()
        self.problem = None
        self.watcher = None
        self.token = f"PIPELINK_CLOCK_QUALIFICATION={uuid.uuid4()}"
        key, value = self.token.split("=", 1)
        os.environ[key] = value
        self.env = {**os.environ, **config}
        self.nvml.nvmlInit()
        self.handles = {gpu: self.nvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in self.gpus}

    def check(self):
        if self.problem:
            raise InvalidTrial(self.problem)
        if self.stop.is_set():
            raise InvalidTrial("interrupted by signal")

    def foreign_processes(self):
        result = subprocess.run(["nvidia-smi", "-i", self.gpu_csv,
                                 "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        foreign = []
        for line in result.stdout.splitlines():
            pid = int(line.strip())
            try:
                environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            except PermissionError:
                environment = []
            if self.token.encode() not in environment:
                foreign.append(pid)
        return sorted(set(foreign))

    def watch(self):
        while not self.stop.is_set():
            try:
                foreign = self.foreign_processes()
                if foreign and self.problem is None:
                    self.problem = f"foreign compute processes on selected GPUs: {foreign}"
            except Exception as exc:
                self.problem = f"compute process monitoring failed: {exc}"
                return
            self.stop.wait(0.5)

    def wait_until_free(self):
        """Wait until no foreign compute process uses the selected GPUs, then clear the contention flag."""
        while self.foreign_processes():
            if self.stop.wait(CONTENTION_POLL_SECONDS):
                raise InvalidTrial("interrupted by signal")
        self.problem = None

    def receive(self, gpu, timeout=120):
        worker, connection = self.workers[gpu]
        deadline = time.monotonic() + timeout
        while not connection.poll(0.1):
            self.check()
            if not worker.is_alive() or time.monotonic() >= deadline:
                raise InvalidTrial(f"GPU {gpu} worker exited or timed out")
        message = connection.recv()
        if "error" in message:
            raise InvalidTrial(f"GPU {gpu}: {message['error']}")
        return message

    def start(self):
        foreign = self.foreign_processes()
        if foreign:
            raise InvalidTrial(f"refusing occupied GPUs; compute PIDs: {foreign}")
        state = Path(self.config["PIPELINK_LOCK_STATE_DIR"])
        if any((state / f"gpu{gpu}.json").exists() for gpu in self.gpus):
            raise InvalidTrial("selected GPUs have existing gpulockd markers; refusing to take their leases")
        self.watcher = threading.Thread(target=self.watch, daemon=True)
        self.watcher.start()
        for gpu, versions in self.spawn_workers().items():
            handle = self.handles[gpu]
            self.record["gpus"].append({"index": gpu, "uuid": self.nvml.nvmlDeviceGetUUID(handle),
                                        "name": self.nvml.nvmlDeviceGetName(handle), **versions})

    def spawn_workers(self):
        context = mp.get_context("spawn")
        for gpu, handle in self.handles.items():
            parent, child = context.Pipe()
            gpu_uuid = self.nvml.nvmlDeviceGetUUID(handle)
            process = context.Process(target=_worker, args=(gpu_uuid, child))
            process.start()
            child.close()
            self.workers[gpu] = (process, parent)
        return {gpu: self.receive(gpu) for gpu in self.handles}

    def stop_workers(self):
        for process, connection in self.workers.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()
        self.workers = {}

    def lock(self, clock):
        self.check()
        result = subprocess.run(["python3", str(ROOT / "tools/gpulockd.py"), "start",
                                 "--gpus", self.gpu_csv, "--sm-mhz", str(clock)],
                                env=self.env, capture_output=True, text=True)
        self.leased = result.returncode == 0
        if not self.leased:
            raise InvalidTrial(f"gpulockd start failed: {result.stdout.strip()} {result.stderr.strip()}")
        self.check()

    def unlock(self):
        if self.leased:
            command = ["python3", str(ROOT / "tools/gpulockd.py"), "stop", "--gpus", self.gpu_csv]
            result = subprocess.run(command, env=self.env, capture_output=True, text=True)
            if result.returncode:  # the daemon can be slow to exit under load; retry once after a pause
                time.sleep(10)
                result = subprocess.run(command, env=self.env, capture_output=True, text=True)
            if result.returncode:
                raise InvalidTrial(f"gpulockd stop failed: {result.stdout.strip()} {result.stderr.strip()}")
            self.leased = False

    def phase(self, name, seconds, clock, result):
        phase = {"name": name, "gpus": {}}
        result["phases"].append(phase)
        for _, connection in self.workers.values():
            connection.send(name)
        active = []
        for gpu in self.gpus:
            ready = self.receive(gpu)
            if "skipped" in ready:
                phase["gpus"][str(gpu)] = ready
            else:
                active.append(gpu)
        if not active:
            return
        samples = {gpu: [] for gpu in active}
        start = time.monotonic() + 0.2
        for gpu in active:
            self.workers[gpu][1].send((start, seconds))
        next_sample = start
        try:
            while next_sample < start + seconds:
                time.sleep(max(0, next_sample - time.monotonic()))
                self.check()
                for gpu in active:
                    handle = self.handles[gpu]
                    n = self.nvml
                    samples[gpu].append({
                        "seconds": time.monotonic() - start,
                        "sm_mhz": n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_SM),
                        "power_w": n.nvmlDeviceGetPowerUsage(handle) / 1000,
                        "power_limit_w": n.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
                        "temperature_c": n.nvmlDeviceGetTemperature(handle, n.NVML_TEMPERATURE_GPU),
                        "utilization": n.nvmlDeviceGetUtilizationRates(handle).gpu,
                        "reasons": n.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                    })
                next_sample += SAMPLE_SECONDS
            for gpu in active:
                self.receive(gpu)
        finally:
            for gpu in active:
                if any(s["seconds"] >= EXCLUDE_SECONDS for s in samples[gpu]):
                    phase["gpus"][str(gpu)] = summarize(samples[gpu], clock, self.args.tolerance)
        if any(not phase["gpus"][str(gpu)]["loaded_samples"] for gpu in active):
            raise InvalidTrial(f"{name} produced no loaded samples")

    def trial(self, clock, seconds, kind):
        """Run one trial at `clock`; if foreign processes void it, wait for the GPUs to free up and retry."""
        for attempt in range(self.args.contention_retries + 1):
            try:
                return self.attempt(clock, seconds, kind)
            except InvalidTrial as exc:
                if not str(exc).startswith("foreign compute processes") or attempt == self.args.contention_retries:
                    raise
                print(f"{kind}: {clock} MHz: contended, waiting for the GPUs to free up", flush=True)
                self.stop_workers()  # restart workers that may be mid-phase
                self.wait_until_free()
                self.spawn_workers()

    def attempt(self, clock, seconds, kind):
        result = {"clock_mhz": clock, "seconds_per_phase": seconds, "kind": kind,
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "invalid", "phases": []}
        self.record["trials"].append(result)
        try:
            self.lock(clock)
            for phase in ("bf16", "fp8"):
                self.phase(phase, seconds, clock, result)
            self.check()
            failed = any(stats.get("violations", 0) for phase in result["phases"]
                         for stats in phase["gpus"].values())
            result["status"] = "failed" if failed else "passed"
        except BaseException as exc:
            result["status"], result["error"] = "invalid", str(exc)
            raise
        finally:
            self.unlock()
            print(f"{kind}: {clock} MHz: {result['status']}", flush=True)
        return result["status"] == "passed"

    def close(self):
        self.stop.set()
        self.stop_workers()
        try:
            self.unlock()
        finally:
            if self.watcher:
                self.watcher.join()
            self.nvml.nvmlShutdown()


def main(argv=None):
    config = pipelink_env.load(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=config["PIPELINK_LOCKED_GPUS"])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--confirm-seconds", type=float, default=180)
    parser.add_argument("--floor-mhz", type=int)
    parser.add_argument("--contention-retries", type=int, default=30,
                        help="retries of a trial voided by foreign GPU processes, each after they exit")
    parser.add_argument("--write-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.gpus = gpulockd._parse_gpu_csv(args.gpus, label="--gpus")
        allowed = gpulockd._parse_gpu_csv(config["PIPELINK_LOCKED_GPUS"], label="PIPELINK_LOCKED_GPUS")
        if not set(args.gpus) <= set(allowed):
            raise ValueError("--gpus must be a subset of PIPELINK_LOCKED_GPUS")
        args.tolerance = gpulockd._clock_tolerance(config)
        if any(not math.isfinite(s) or s <= EXCLUDE_SECONDS for s in (args.seconds, args.confirm_seconds)):
            raise ValueError("phase durations must exceed the excluded first 5 seconds")
        if args.floor_mhz is not None and args.floor_mhz <= 0:
            raise ValueError("--floor-mhz must be positive")
    except (ValueError, gpulockd.GpuLockError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    hostname = socket.gethostname()
    output = ROOT / "out/clock" / f"{hostname}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"hostname": hostname, "started_at": stamp, "gpus": [], "trials": [],
              "chosen_clock_mhz": None, "status": "invalid", "protocol": {
                  "matrix_size": MATRIX_SIZE, "seed": 0, "warmup_seconds": WARMUP_SECONDS,
                  "excluded_seconds": EXCLUDE_SECONDS, "sample_seconds": SAMPLE_SECONDS,
                  "seconds": args.seconds, "confirm_seconds": args.confirm_seconds,
                  "confirmation_runs": 2, "tolerance_mhz": args.tolerance}}
    qualification = None
    previous = {}
    try:
        qualification = Qualification(args, config, record)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: qualification.stop.set())
        qualification.start()
        record["driver"] = qualification.nvml.nvmlSystemGetDriverVersion()
        sets = []
        for gpu in args.gpus:
            xml = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "-q", "-x"], text=True, timeout=10)
            clocks = supported_clocks(xml)
            record["gpus"][len(sets)]["supported_sm_clocks_mhz"] = clocks
            sets.append(set(clocks))
        common = sorted(set.intersection(*sets), reverse=True)
        floor = args.floor_mhz if args.floor_mhz is not None else math.ceil(common[0] * 0.5)
        candidates = [clock for clock in common if clock >= floor]
        if not candidates:
            raise InvalidTrial("no common supported SM clocks at or above the floor")
        record["protocol"].update(floor_mhz=floor, candidates_mhz=candidates)
        index = highest_passing(candidates, lambda clock: qualification.trial(clock, args.seconds, "search"))
        if index is None:
            raise RuntimeError("no sustainable clock at or above the floor")
        while index < len(candidates):
            clock = candidates[index]
            if all(qualification.trial(clock, args.confirm_seconds, f"confirm-{run}") for run in (1, 2)):
                record["chosen_clock_mhz"] = clock
                record["status"] = "qualified"
                break
            index += 1
        if record["chosen_clock_mhz"] is None:
            raise RuntimeError("no candidate survived confirmation")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if qualification:
                qualification.close()
        except Exception as exc:
            record["status"] = "invalid"
            record["error"] = f"cleanup failed: {exc}"
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] == "qualified" and args.write_env:
        write_env(ROOT / ".env", record["chosen_clock_mhz"], stamp)
    print(f"\nClock qualification: {record['status']}; host={hostname}; GPUs={','.join(map(str, args.gpus))}")
    print(f"Chosen SM clock: {record['chosen_clock_mhz']} MHz (tolerance {args.tolerance} MHz)")
    if "error" in record:
        print(record["error"])
    if record["status"] == "qualified":
        for phase in record["trials"][-1]["phases"]:
            for gpu, stats in phase["gpus"].items():
                if "skipped" in stats:
                    print(f"  GPU {gpu} {phase['name']}: skipped ({stats['skipped']})")
                else:
                    print(f"  GPU {gpu} {phase['name']}: peak {stats['max_power_w']:.1f}/{stats['power_limit_w']:.0f} W; "
                          f"{stats['max_temperature_c']} C; SW cap {stats['sw_power_cap_fraction']:.1%}; "
                          f"violations={stats['violations']}")
    print(f"JSON: {output.relative_to(ROOT)}")
    print("Full run: uv run --project tools/clockqual python tools/qualify_clock.py "
          "--seconds 60 --confirm-seconds 180 --write-env")
    return 0 if record["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
