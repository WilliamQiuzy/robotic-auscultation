"""Run one owned Linux process group, aborting below 8 GiB host availability."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def mem_available_gib(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Linux MemAvailable is missing")


def _stop_group(child: subprocess.Popen) -> None:
    # A dedicated session/process group is created for this child only.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Also remove descendants if the group leader exited first.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_guard(command: list[str], *, minimum_gib: float, lock_path: Path,
              record_path: Path | None, timeout_seconds: float | None = None,
              meminfo_path: Path = Path("/proc/meminfo")) -> int:
    if minimum_gib < 8 or (timeout_seconds is not None and timeout_seconds < 1):
        raise ValueError("Require >=8 GiB reserve and timeout >=1 second")
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    monotonic_start = time.monotonic()
    report = {"command": command, "minimum_required_gib": minimum_gib,
              "start_time_unix": started, "timeout_seconds": timeout_seconds,
              "timed_out": False}
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = mem_available_gib(meminfo_path.read_text())
        report["minimum_observed_gib"] = available
        if available < minimum_gib:
            raise SystemExit(f"Host has {available:.2f} GiB available; refusing to start")
        child = subprocess.Popen(command, start_new_session=True)
        report["owned_process_group"] = child.pid
        def handle_signal(signum, _frame):
            report["interrupted_signal"] = signum
            raise KeyboardInterrupt
        previous = {s: signal.signal(s, handle_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        result = None
        try:
            while child.poll() is None:
                if timeout_seconds is not None and time.monotonic() - monotonic_start >= timeout_seconds:
                    report["timed_out"] = True
                    result = 124
                    break
                available = mem_available_gib(meminfo_path.read_text())
                report["minimum_observed_gib"] = min(report["minimum_observed_gib"], available)
                if available < minimum_gib:
                    report["aborted_low_memory"] = True
                    result = 75
                    break
                time.sleep(0.25)
            if result is None:
                result = child.wait()
        except KeyboardInterrupt:
            result = 128 + report.get("interrupted_signal", signal.SIGINT)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            _stop_group(child)
            report["elapsed_s"] = time.time() - started
            report["returncode"] = child.poll()
            report["guard_returncode"] = result
            if record_path:
                Path(record_path).parent.mkdir(parents=True, exist_ok=True)
                Path(record_path).write_text(json.dumps(report, indent=2) + "\n")
    return int(result)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minimum-gib", type=float, default=8.0)
    p.add_argument("--timeout-seconds", type=float)
    p.add_argument("--lock", type=Path, default=Path("/home/william/wam-sim/eval_curobo/heavy.lock"))
    p.add_argument("--record", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.minimum_gib < 8.0:
        p.error("minimum-gib cannot be below 8")
    if args.timeout_seconds is not None and args.timeout_seconds < 1:
        p.error("timeout-seconds must be >=1")
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        p.error("command is required")
    return run_guard(cmd, minimum_gib=args.minimum_gib, lock_path=args.lock,
                     record_path=args.record, timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
    sys.exit(main())
