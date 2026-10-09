"""Bound profiler runs and terminate only the launched process's descendant tree."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time


def descendant_handles(root_pid):
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split(")", 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {root_pid}
    while True:
        extended = owned | {pid for pid, parent in parents.items() if parent in owned}
        if extended == owned:
            break
        owned = extended
    handles = {}
    for pid in owned:
        try:
            handles[os.pidfd_open(pid)] = pid
        except ProcessLookupError:
            pass
    return handles


def stop_tree(root_pid):
    handles = descendant_handles(root_pid)
    watcher = select.poll()
    for fd in handles:
        watcher.register(fd, select.POLLIN)
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass
    remaining = set(handles)
    deadline = time.monotonic() + 5
    while remaining and time.monotonic() < deadline:
        for fd, _ in watcher.poll(max(1, int((deadline - time.monotonic()) * 1000))):
            remaining.discard(fd)
            watcher.unregister(fd)
    forced = []
    for fd in remaining:
        try:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            forced.append(handles[fd])
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while remaining and time.monotonic() < deadline:
        for fd, _ in watcher.poll(max(1, int((deadline - time.monotonic()) * 1000))):
            remaining.discard(fd)
            watcher.unregister(fd)
    exited = not remaining
    for fd in handles:
        os.close(fd)
    if not exited:
        raise RuntimeError("owned profiler descendants did not exit after SIGKILL")
    return {"terminated_descendants": sorted(handles.values()), "forced_kill": sorted(forced),
            "all_descendants_exited": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    report = {"deadline_seconds": args.timeout, "timed_out": False}
    try:
        returncode = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        report.update(timed_out=True, **stop_tree(process.pid))
        process.wait(timeout=10)
        returncode = 124
    report.update(returncode=returncode, elapsed_seconds=time.monotonic() - started)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
