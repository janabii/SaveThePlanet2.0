"""
run_tracker.py — Lightweight results tracker for the mixed-fleet drone mission.

Usage in v31.py
---------------
    from run_tracker import RunTracker

    # In the finally block, after metrics are computed:
    tracker = RunTracker()
    tracker.record_run(row_dict)           # appends CSV + writes JSON snapshot
    tracker.show_overlay(cid, row_dict)    # PyBullet debug-text overlay
    tracker.print_running_averages()       # mean ± std from all rows in CSV
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
RUNS_DIR = _HERE / "runs"
RUNS_CSV = RUNS_DIR / "runs.csv"
RUNS_JSON_DIR = RUNS_DIR / "json"

CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "strategy",
    "num_drones",
    "seed",
    # FW detector metrics
    "fw_map",
    "fw_precision",
    "fw_recall",
    "fw_tp",
    "fw_fp",
    "fw_fn",
    "fw_mean_err",
    # Quad detector metrics
    "quad_map",
    "quad_precision",
    "quad_recall",
    "quad_tp",
    "quad_fp",
    "quad_fn",
    "quad_mean_err",
    # Delta + heatmap
    "delta_map",
    "heatmap_peaks",
    "heatmap_tp",
    "heatmap_fp",
    "heatmap_precision",
    "heatmap_recall",
    # Counts
    "gt_items",
    "fw_frames",
    "fw_detections",
    "photos_taken",
    # Battery per quad
    "quad0_battery_used",
    "quad1_battery_used",
    "quad2_battery_used",
    "quad3_battery_used",
    # Timing
    "sim_time_s",
    "wall_time_s",
    "avg_flight_time_s",
    # Crypto
    "crypto_total_msgs",
    "crypto_total_mj",
]


def _safe(val: Any) -> str:
    """Convert a value to a CSV-safe string; NaN/None become empty."""
    if val is None:
        return ""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


class RunTracker:
    """Append-only CSV + per-run JSON + PyBullet overlay + running averages."""

    def __init__(self, runs_dir: str | Path | None = None):
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.csv_path = self.runs_dir / "runs.csv"
        self.json_dir = self.runs_dir / "json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build row dict from mission variables
    # ------------------------------------------------------------------
    @staticmethod
    def build_row(
        *,
        strategy: str,
        num_drones: int,
        seed: int | str,
        fw_eval: dict | None,
        quad_eval: dict | None,
        heatmap_peaks: int,
        heatmap_tp: int,
        heatmap_fp: int,
        heatmap_precision: float,
        heatmap_recall: float,
        gt_items: int,
        fw_frames: int,
        fw_detections: int,
        photos_taken: int,
        battery_used_pct: list[float],
        sim_time_s: float,
        wall_time_s: float,
        flight_times_s: list[float],
        crypto_total_msgs: int,
        crypto_total_mj: float,
    ) -> dict[str, Any]:
        fw_map = fw_eval["ap_proxy"] * 100.0 if fw_eval else None
        fw_prec = fw_eval["precision"] * 100.0 if fw_eval else None
        fw_rec = fw_eval["recall"] * 100.0 if fw_eval else None
        fw_tp = fw_eval["tp"] if fw_eval else None
        fw_fp = fw_eval["fp"] if fw_eval else None
        fw_fn = fw_eval["fn"] if fw_eval else None
        fw_me = fw_eval["mean_err_m"] if fw_eval else None

        q_map = quad_eval["ap_proxy"] * 100.0 if quad_eval else None
        q_prec = quad_eval["precision"] * 100.0 if quad_eval else None
        q_rec = quad_eval["recall"] * 100.0 if quad_eval else None
        q_tp = quad_eval["tp"] if quad_eval else None
        q_fp = quad_eval["fp"] if quad_eval else None
        q_fn = quad_eval["fn"] if quad_eval else None
        q_me = quad_eval["mean_err_m"] if quad_eval else None

        delta = None
        if fw_map is not None and q_map is not None:
            delta = q_map - fw_map

        bat = [0.0] * 4
        for i, v in enumerate(battery_used_pct[:4]):
            bat[i] = v

        avg_ft = float(np.mean(flight_times_s)) if flight_times_s else 0.0

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        run_id = time.strftime("%Y%m%d_%H%M%S")

        return {
            "run_id": run_id,
            "timestamp": ts,
            "strategy": strategy,
            "num_drones": num_drones,
            "seed": seed,
            "fw_map": fw_map,
            "fw_precision": fw_prec,
            "fw_recall": fw_rec,
            "fw_tp": fw_tp,
            "fw_fp": fw_fp,
            "fw_fn": fw_fn,
            "fw_mean_err": fw_me,
            "quad_map": q_map,
            "quad_precision": q_prec,
            "quad_recall": q_rec,
            "quad_tp": q_tp,
            "quad_fp": q_fp,
            "quad_fn": q_fn,
            "quad_mean_err": q_me,
            "delta_map": delta,
            "heatmap_peaks": heatmap_peaks,
            "heatmap_tp": heatmap_tp,
            "heatmap_fp": heatmap_fp,
            "heatmap_precision": heatmap_precision,
            "heatmap_recall": heatmap_recall,
            "gt_items": gt_items,
            "fw_frames": fw_frames,
            "fw_detections": fw_detections,
            "photos_taken": photos_taken,
            "quad0_battery_used": bat[0],
            "quad1_battery_used": bat[1],
            "quad2_battery_used": bat[2],
            "quad3_battery_used": bat[3],
            "sim_time_s": sim_time_s,
            "wall_time_s": wall_time_s,
            "avg_flight_time_s": avg_ft,
            "crypto_total_msgs": crypto_total_msgs,
            "crypto_total_mj": crypto_total_mj,
        }

    # ------------------------------------------------------------------
    # 2. Append one row to the CSV (creates header if file is new)
    # ------------------------------------------------------------------
    def record_run(self, row: dict[str, Any]) -> Path:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([_safe(row.get(c)) for c in CSV_COLUMNS])

        # JSON snapshot
        json_path = self.json_dir / f"{row['run_id']}.json"
        with open(json_path, "w") as jf:
            json.dump(row, jf, indent=2, default=str)

        print(f"[RunTracker] CSV row appended -> {self.csv_path}")
        print(f"[RunTracker] JSON snapshot    -> {json_path}")
        return json_path

    # ------------------------------------------------------------------
    # 3. PyBullet debug-text overlay
    # ------------------------------------------------------------------
    @staticmethod
    def show_overlay(cid: int, row: dict[str, Any]) -> list[int]:
        """Render a summary overlay in the PyBullet GUI. Returns debug text IDs."""
        try:
            import pybullet as p
        except Exception:
            return []

        lines = [
            "=== MISSION RESULTS ===",
            f"Strategy: {row.get('strategy', '?')}   Drones: {row.get('num_drones', '?')}",
            f"FW  mAP: {_fmt_pct(row.get('fw_map'))}   P: {_fmt_pct(row.get('fw_precision'))}   R: {_fmt_pct(row.get('fw_recall'))}",
            f"Quad mAP: {_fmt_pct(row.get('quad_map'))}   P: {_fmt_pct(row.get('quad_precision'))}   R: {_fmt_pct(row.get('quad_recall'))}",
            f"Delta mAP (Q-FW): {_fmt_pct(row.get('delta_map'), signed=True)}",
            f"Heatmap peaks: {row.get('heatmap_peaks', 0)}  TP: {row.get('heatmap_tp', 0)}  FP: {row.get('heatmap_fp', 0)}",
            f"Photos: {row.get('photos_taken', 0)}   GT items: {row.get('gt_items', 0)}",
            f"Sim: {row.get('sim_time_s', 0):.1f}s   Wall: {row.get('wall_time_s', 0):.1f}s",
        ]

        ids = []
        x, y, z = 0.0, 0.0, 6.0
        for i, text in enumerate(lines):
            color = [1, 1, 0] if i == 0 else [1, 1, 1]
            size = 1.6 if i == 0 else 1.2
            tid = p.addUserDebugText(
                text,
                [x, y, z - i * 0.45],
                textColorRGB=color,
                textSize=size,
                physicsClientId=cid,
            )
            ids.append(tid)
        return ids

    # ------------------------------------------------------------------
    # 4. Running averages from the CSV
    # ------------------------------------------------------------------
    def print_running_averages(self):
        if not self.csv_path.exists():
            print("[RunTracker] No runs CSV found — skipping averages.")
            return

        rows = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        n = len(rows)
        if n == 0:
            print("[RunTracker] CSV is empty — skipping averages.")
            return

        keys = [
            ("fw_map", "FW mAP (%)"),
            ("quad_map", "Quad mAP (%)"),
            ("delta_map", "Delta mAP (%)"),
            ("fw_precision", "FW Precision (%)"),
            ("fw_recall", "FW Recall (%)"),
            ("quad_precision", "Quad Precision (%)"),
            ("quad_recall", "Quad Recall (%)"),
            ("heatmap_precision", "Heatmap Precision (%)"),
            ("heatmap_recall", "Heatmap Recall (%)"),
            ("wall_time_s", "Wall time (s)"),
        ]

        sep = "=" * 56
        print(f"\n{sep}")
        print(f"  RUNNING AVERAGES  ({n} run{'s' if n != 1 else ''})")
        print(sep)
        for col, label in keys:
            vals = []
            for r in rows:
                v = r.get(col, "")
                if v not in ("", None):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                print(f"  {label:<26s}  {mean:7.2f} +/- {std:.2f}  (n={len(vals)})")
            else:
                print(f"  {label:<26s}  n/a")
        remaining = max(0, 15 - n)
        if remaining > 0:
            print(f"\n  {remaining} more run(s) to reach 15-run target.")
        else:
            print(f"\n  15-run target reached!")
        print(sep + "\n")


def _fmt_pct(val: Any, signed: bool = False) -> str:
    if val is None:
        return "n/a"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"
