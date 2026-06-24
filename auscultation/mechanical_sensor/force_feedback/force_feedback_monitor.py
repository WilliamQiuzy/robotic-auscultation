"""Monitor calibrated force feedback for the ThinkLabs passive mount.

Examples:

    python auscultation/mechanical_sensor/force_feedback/force_feedback_monitor.py \
        --calibration auscultation/mechanical_sensor/force_feedback/force_feedback_calibration.json \
        --replay-csv auscultation/mechanical_sensor/generated/isaac_contact_vast/isaac_contact_trace.csv \
        --out auscultation/mechanical_sensor/generated/force_feedback/replay_force_feedback_log.csv

Serial input accepts either JSON lines:

    {"t_ms": 1234, "disp_mm": 10.52}

or CSV lines:

    1234,10.52
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", default="auscultation/mechanical_sensor/generated/force_feedback/force_feedback_log.csv")
    parser.add_argument("--replay-csv")
    parser.add_argument("--serial-port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--max-lines", type=int, default=0, help="0 means no limit in serial mode.")
    parser.add_argument("--replay-realtime", action="store_true")
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "spring_rate_n_per_mm",
        "zero_compression_mm",
        "target_force_n",
        "max_screened_force_n",
        "travel_limit_mm",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"calibration missing keys: {', '.join(missing)}")
    return cfg


def compute_force(displacement_mm: float, cfg: dict[str, Any]) -> tuple[float, float, str]:
    compression_mm = max(0.0, displacement_mm - float(cfg["zero_compression_mm"]))
    force_n = float(cfg["spring_rate_n_per_mm"]) * compression_mm
    max_force = float(cfg["max_screened_force_n"])
    target = float(cfg["target_force_n"])
    travel = float(cfg["travel_limit_mm"])
    if force_n > max_force or compression_mm > travel:
        status = "HARD_LIMIT"
    elif abs(force_n - target) <= 0.5:
        status = "OK"
    else:
        status = "WARN"
    return compression_mm, force_n, status


def parse_sensor_line(line: str) -> tuple[float, float] | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("{"):
        payload = json.loads(text)
        return float(payload.get("t_ms", payload.get("time_ms", 0.0))) / 1000.0, float(
            payload.get("disp_mm", payload.get("compression_mm"))
        )
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 2:
        return None
    return float(parts[0]) / 1000.0, float(parts[1])


def write_rows(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "sensor_displacement_mm",
        "compression_mm",
        "force_n",
        "status",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay_csv(path: Path, cfg: dict[str, Any], realtime: bool) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    last_t = 0.0
    with path.open(newline="", encoding="utf-8") as f:
        for source in csv.DictReader(f):
            t = float(source["time_s"])
            displacement = float(source.get("spring_compression_mm", source.get("sensor_displacement_mm", "0")))
            compression, force, status = compute_force(displacement, cfg)
            rows.append(
                {
                    "time_s": t,
                    "sensor_displacement_mm": displacement,
                    "compression_mm": compression,
                    "force_n": force,
                    "status": status,
                }
            )
            print(
                f"t={t:6.3f}s compression={compression:6.3f}mm force={force:5.2f}N status={status}",
                end="\r" if realtime else "\n",
                flush=True,
            )
            if realtime:
                time.sleep(max(0.0, min(0.05, t - last_t)))
                last_t = t
    if realtime:
        print()
    return rows


def read_serial(port: str, baud: int, cfg: dict[str, Any], max_lines: int) -> list[dict[str, float | str]]:
    try:
        import serial  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("serial mode requires pyserial: python -m pip install pyserial") from exc

    rows: list[dict[str, float | str]] = []
    with serial.Serial(port, baudrate=baud, timeout=1.0) as ser:
        count = 0
        while max_lines <= 0 or count < max_lines:
            raw = ser.readline().decode("utf-8", errors="replace")
            parsed = parse_sensor_line(raw)
            if parsed is None:
                continue
            t, displacement = parsed
            compression, force, status = compute_force(displacement, cfg)
            rows.append(
                {
                    "time_s": t,
                    "sensor_displacement_mm": displacement,
                    "compression_mm": compression,
                    "force_n": force,
                    "status": status,
                }
            )
            print(f"t={t:6.3f}s compression={compression:6.3f}mm force={force:5.2f}N status={status}", flush=True)
            count += 1
    return rows


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float | int | str]:
    if not rows:
        return {"rows": 0, "status": "NO_DATA"}
    forces = [float(row["force_n"]) for row in rows]
    compressions = [float(row["compression_mm"]) for row in rows]
    return {
        "rows": len(rows),
        "peak_force_n": max(forces),
        "final_force_n": forces[-1],
        "max_compression_mm": max(compressions),
        "hard_limit_samples": sum(1 for row in rows if row["status"] == "HARD_LIMIT"),
    }


def main() -> int:
    args = parse_args()
    cfg = load_calibration(Path(args.calibration))
    if not args.replay_csv and not args.serial_port:
        raise SystemExit("pass either --replay-csv or --serial-port")

    if args.replay_csv:
        rows = replay_csv(Path(args.replay_csv), cfg, args.replay_realtime)
    else:
        rows = read_serial(args.serial_port, args.baud, cfg, args.max_lines)

    out_path = Path(args.out)
    write_rows(rows, out_path)
    summary = summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(out_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
