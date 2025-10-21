#!/usr/bin/env python3
"""
CARLA Log Viewer (top-down) — DataLogs aware

This GUI lets you scrub through a CARLA run with a slider, show the 2D path,
current position, and arrows for velocity & turning (yaw-rate). It matches
your naming scheme:

  DataLogs/
    └── YYYY-MM-DD/
        ├── carla_log_YYYY-MM-DD_HH-MM-SS.csv
        └── meta.json   (optional)

USAGE
-----
# Open the newest CSV for a specific date under DataLogs/<DATE>/
python visualize_datalog.py --date 2025-10-20

# Open the newest CSV anywhere under DataLogs/
python visualize_datalog.py

# Open a specific file
python visualize_datalog.py DataLogs/2025-10-20/carla_log_2025-10-20_14-35-52.csv

OPTIONS
-------
--root <path>           Root folder (default: DataLogs)
--date YYYY-MM-DD       Search only within that date folder
--fps <int>             Playback fps for the timer (default: 20)
--turn-scale <float>    Visual scale for the turn (yaw-rate) arrow (default: 5.0)
--point-size <int>      Marker size for current position (default: 8)

DEPENDENCIES
------------
pip install matplotlib pandas numpy
"""

import argparse
from pathlib import Path
import sys
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


DEFAULT_ROOT = Path("DataLogs")


def find_latest_csv(root: Path, date: str | None):
    """
    Find the newest carla_log_*.csv file.

    If 'date' is provided, search under root/<date>/ only.
    Otherwise, search recursively under root/.
    Returns (Path or None, error_message or None).
    """
    if date:
        search_dir = root / date
        if not search_dir.is_dir():
            return None, f"[viewer] No directory: {search_dir}"
        candidates = sorted(search_dir.glob("carla_log_*.csv"))
    else:
        candidates = sorted(root.glob("**/carla_log_*.csv"))

    if not candidates:
        return None, "[viewer] No CSV files found with pattern carla_log_*.csv"

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest, None


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and coerce numeric fields; ensure expected columns exist."""
    df = pd.read_csv(csv_path)

    # Coerce numerics (in case some fields were formatted as strings)
    numeric_cols = [
        "frame", "sim_time", "x", "y", "z", "yaw_deg", "speed_mps", "yawrate_rps",
        "throttle", "brake", "steer", "reverse", "gear", "hand_brake",
        "lane_id", "offset_m", "dist_left_m", "dist_right_m"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # lane_event might be NaN; convert to str with empty default
    if "lane_event" in df.columns:
        df["lane_event"] = df["lane_event"].astype(str).replace({"nan": ""})
    else:
        df["lane_event"] = ""

    # map column as string
    if "map" in df.columns:
        df["map"] = df["map"].astype(str)

    # Minimal column check
    required = [
        "frame", "sim_time", "x", "y", "z", "yaw_deg", "speed_mps", "yawrate_rps",
        "throttle", "brake", "steer", "reverse", "gear", "hand_brake",
        "lane_id", "offset_m", "dist_left_m", "dist_right_m", "lane_event", "map"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    return df


def compute_vectors(df: pd.DataFrame, turn_scale: float):
    """
    Compute display vectors:
    - Velocity vector from speed & yaw.
    - 'Turn' vector from yaw-rate, perpendicular to heading, scaled for visibility.
    """
    yaw_rad = np.deg2rad(df["yaw_deg"].values)
    hx = np.cos(yaw_rad)
    hy = np.sin(yaw_rad)

    vx = df["speed_mps"].values * hx
    vy = df["speed_mps"].values * hy

    # Perpendicular to heading (rotate by +90deg): (-sin, cos)
    tx = -hy * df["yawrate_rps"].values * turn_scale
    ty =  hx * df["yawrate_rps"].values * turn_scale

    return vx, vy, tx, ty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="Path to a CARLA log CSV (carla_log_*.csv)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Root directory for logs (default: DataLogs)")
    ap.add_argument("--date", help="Date folder under DataLogs/ to auto-pick newest CSV (YYYY-MM-DD)")
    ap.add_argument("--fps", type=int, default=20, help="Playback FPS for timer (default: 20)")
    ap.add_argument("--turn-scale", type=float, default=5.0, help="Scale factor for yaw-rate arrow (default: 5.0)")
    ap.add_argument("--point-size", type=int, default=8, help="Marker size for current position (default: 8)")
    args = ap.parse_args()

    root = Path(args.root)

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[viewer] CSV not found: {csv_path}")
            sys.exit(1)
    else:
        csv_path, err = find_latest_csv(root, args.date)
        if err:
            print(err)
            if root.exists():
                dates = sorted({p.parent.name for p in root.glob('**/carla_log_*.csv')})
                if dates:
                    print("[viewer] Available dates:", ", ".join(dates[-10:]))
            sys.exit(1)
        print(f"[viewer] Opening: {csv_path}")

    # Load and prepare data
    df = load_csv(csv_path)
    if df.empty:
        print("[viewer] CSV is empty or unreadable.")
        sys.exit(1)

    x = df["x"].values
    y = df["y"].values
    sim_t = df["sim_time"].values
    yaw = df["yaw_deg"].values
    speed = df["speed_mps"].values
    yawrate = df["yawrate_rps"].values
    lane_event = df["lane_event"].values
    offset = df["offset_m"].values
    lane_id = df["lane_id"].values

    vx, vy, tx, ty = compute_vectors(df, args.turn_scale)

    # ---- Figure and axes setup
    fig = plt.figure("CARLA Log Viewer", figsize=(10, 8))
    ax = plt.axes([0.08, 0.20, 0.88, 0.75])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    # Plot entire path
    ax.plot(x, y, linewidth=1.0)

    # Current point & vectors
    i0 = 0
    point_plot, = ax.plot([x[i0]], [y[i0]], marker="o", markersize=args.point_size)
    vel_quiv = ax.quiver(x[i0], y[i0], vx[i0], vy[i0], angles='xy', scale_units='xy', scale=1)
    turn_quiv = ax.quiver(x[i0], y[i0], tx[i0], ty[i0], angles='xy', scale_units='xy', scale=1)

    # Auto-scale with margin
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = max(1.0, 0.05 * max(1.0, xmax - xmin))
    dy = max(1.0, 0.05 * max(1.0, ymax - ymin))
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    # HUD text
    hud_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    # Slider
    idx_ax = plt.axes([0.08, 0.08, 0.70, 0.04])
    idx_slider = Slider(idx_ax, "Frame", 0, len(df) - 1, valinit=i0, valfmt="%0.0f")

    # Play/Pause button
    btn_ax = plt.axes([0.80, 0.07, 0.16, 0.06])
    btn = Button(btn_ax, "Play ▶")
    playing = {"on": False}

    def set_frame(i: int):
        """Update marker, vectors, and HUD for frame i."""
        i = int(np.clip(i, 0, len(df) - 1))

        # Update point
        point_plot.set_data([x[i]], [y[i]])

        # Update arrows (remove and redraw for simplicity)
        nonlocal vel_quiv, turn_quiv
        for q in (vel_quiv, turn_quiv):
            try:
                q.remove()
            except Exception:
                pass
        vel_quiv = ax.quiver(x[i], y[i], vx[i], vy[i], angles='xy', scale_units='xy', scale=1)
        turn_quiv = ax.quiver(x[i], y[i], tx[i], ty[i], angles='xy', scale_units='xy', scale=1)

        # HUD
        hud = (
            f"t = {sim_t[i]:.2f} s\n"
            f"pos = ({x[i]:.1f}, {y[i]:.1f}) m\n"
            f"speed = {speed[i]*3.6:.1f} km/h\n"
            f"yaw = {yaw[i]:.1f} deg  |  yawrate = {yawrate[i]:.3f} rad/s\n"
            f"lane_id = {lane_id[i]}  |  offset = {offset[i]:.2f} m\n"
            f"lane_event = {lane_event[i]}"
        )
        hud_text.set_text(hud)

        fig.canvas.draw_idle()

    def on_slider(val):
        set_frame(val)

    idx_slider.on_changed(on_slider)

    def toggle_play(_=None):
        playing["on"] = not playing["on"]
        btn.label.set_text("Pause ⏸" if playing["on"] else "Play ▶")
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_play)

    def on_key(event):
        if event.key == " ":
            toggle_play()
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Timer-based playback (portable across backends; avoids 'idle_event')
    interval_ms = max(1, int(1000 / max(1, args.fps)))
    timer = fig.canvas.new_timer(interval=interval_ms)

    def on_timer():
        if playing["on"]:
            i = int(idx_slider.val)
            if i < len(df) - 1:
                idx_slider.set_val(i + 1)
            else:
                toggle_play()  # reached end
        # draw is requested in set_frame

    timer.add_callback(on_timer)
    timer.start()

    # Initialize
    set_frame(i0)
    plt.show()


if __name__ == "__main__":
    main()
