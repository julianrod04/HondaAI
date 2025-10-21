#!/usr/bin/env python3
"""
CARLA Log Viewer (top-down)

Usage:
  python carla_log_viewer.py /path/to/carla_run_YYYY-MM-DD_HH-MM-SS.csv

Features:
- 2D path plot of (x, y)
- Slider to scrub through frames
- Play / Pause button (or press Space)
- Velocity vector arrow from speed & yaw
- "Turn" vector arrow from yaw-rate (scaled)
- HUD text: time, speed, yaw, yaw-rate, lane event, lane offset

Notes:
- Expects CSV columns from the provided logger:
  frame, sim_time, x, y, z, yaw_deg, speed_mps, yawrate_rps, throttle, brake, steer,
  reverse, gear, hand_brake, lane_id, offset_m, dist_left_m, dist_right_m, lane_event, map
"""

import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def load_csv(path):
    df = pd.read_csv(path)
    # Coerce numeric fields (CSV may be strings with formatting)
    num_cols = [
        "frame","sim_time","x","y","z","yaw_deg","speed_mps","yawrate_rps",
        "throttle","brake","steer","offset_m","dist_left_m","dist_right_m"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_vectors(df):
    # Heading unit vector from yaw (degrees→rad).
    yaw_rad = np.deg2rad(df["yaw_deg"].values)
    hx = np.cos(yaw_rad)
    hy = np.sin(yaw_rad)
    # Velocity vector (m/s) from speed and heading.
    vx = df["speed_mps"].values * hx
    vy = df["speed_mps"].values * hy
    # "Turn" vector from yaw-rate (rad/s): use perpendicular to heading, scaled for visualization.
    # Perp to heading (rotate by +90deg): (-sin, cos). Scale by yaw-rate * K.
    # Choose K so typical yawrate (0.2 rad/s) gives a short visible arrow.
    K = 5.0  # tweakable on-screen scale
    tx = -hy * df["yawrate_rps"].values * K
    ty =  hx * df["yawrate_rps"].values * K
    return vx, vy, tx, ty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="Path to CARLA log CSV")
    args = ap.parse_args()

    if not args.csv:
        print("Please provide a CSV path, e.g.:")
        print("  python carla_log_viewer.py carla_run_2025-10-20_14-35-52.csv")
        sys.exit(1)

    df = load_csv(args.csv)
    if df.empty:
        print("CSV is empty or unreadable."); sys.exit(1)

    # Basic arrays
    x = df["x"].values
    y = df["y"].values
    sim_t = df["sim_time"].values
    yaw = df["yaw_deg"].values
    speed = df["speed_mps"].values
    yawrate = df["yawrate_rps"].values
    lane_event = df["lane_event"].fillna("").astype(str).values if "lane_event" in df.columns else np.array([""]*len(df))
    offset = df["offset_m"].values if "offset_m" in df.columns else np.zeros_like(x)
    lane_id = df["lane_id"].values if "lane_id" in df.columns else np.zeros_like(x)

    vx, vy, tx, ty = compute_vectors(df)

    # Figure layout
    plt.figure("CARLA Log Viewer", figsize=(10, 8))
    ax = plt.axes([0.08, 0.20, 0.88, 0.75])   # main plot
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    # Plot full path
    path_line, = ax.plot(x, y, linewidth=1.0)

    # Current point & vectors (initialized at frame 0)
    i0 = 0
    point_plot, = ax.plot([x[i0]], [y[i0]], marker="o", markersize=8)
    vel_quiv = ax.quiver(x[i0], y[i0], vx[i0], vy[i0], angles='xy', scale_units='xy', scale=1)
    turn_quiv = ax.quiver(x[i0], y[i0], tx[i0], ty[i0], angles='xy', scale_units='xy', scale=1)

    # Auto-scale view with some margin
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = max(1.0, 0.05 * max(1.0, xmax - xmin))
    dy = max(1.0, 0.05 * max(1.0, ymax - ymin))
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    # HUD text box
    hud_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                       bbox=dict(boxstyle="round", alpha=0.2))

    # Slider (index over rows)
    idx_ax = plt.axes([0.08, 0.08, 0.70, 0.04])
    idx_slider = Slider(idx_ax, "Frame", 0, len(df)-1, valinit=i0, valfmt="%0.0f")

    # Play/Pause button
    btn_ax = plt.axes([0.80, 0.07, 0.16, 0.06])
    btn = Button(btn_ax, "Play ▶")
    playing = {"on": False}
    last_update = {"t": time.time()}

    # Update function
    def set_frame(i):
        i = int(np.clip(i, 0, len(df)-1))
        # Move point
        point_plot.set_data([x[i]], [y[i]])
        # Update quivers by removing old and drawing new (simpler than set_UVC for two quivers at once)
        nonlocal vel_quiv, turn_quiv
        for q in [vel_quiv, turn_quiv]:
            try:
                q.remove()
            except Exception:
                pass
        vel_quiv = ax.quiver(x[i], y[i], vx[i], vy[i], angles='xy', scale_units='xy', scale=1)
        turn_quiv = ax.quiver(x[i], y[i], tx[i], ty[i], angles='xy', scale_units='xy', scale=1)

        # HUD text
        hud = (f"t = {sim_t[i]:.2f} s\n"
               f"pos = ({x[i]:.1f}, {y[i]:.1f}) m\n"
               f"speed = {speed[i]*3.6:.1f} km/h\n"
               f"yaw = {yaw[i]:.1f} deg  |  yawrate = {yawrate[i]:.3f} rad/s\n"
               f"lane_id = {lane_id[i]}  |  offset = {offset[i]:.2f} m\n"
               f"lane_event = {lane_event[i]}")
        hud_text.set_text(hud)

        # Redraw
        plt.draw()

    def on_slider(val):
        set_frame(val)

    idx_slider.on_changed(on_slider)

    # Play/pause behavior
    def toggle_play(event=None):
        playing["on"] = not playing["on"]
        btn.label.set_text("Pause ⏸" if playing["on"] else "Play ▶")
        plt.draw()

    btn.on_clicked(toggle_play)

    def on_key(event):
        if event.key == " ":
            toggle_play()

    def on_idle(event):
        # Advance by ~1/20th of a second worth of frames based on CSV time spacing
        if playing["on"]:
            now = time.time()
            # advance ~real-time using sim_time increments; estimate dt between frames
            i = int(idx_slider.val)
            if i < len(df) - 1:
                # Advance by one frame per idle iteration; this keeps things simple
                idx_slider.set_val(i + 1)
            else:
                toggle_play()  # reached end

    # Connect events
    plt.gcf().canvas.mpl_connect("key_press_event", on_key)
    plt.gcf().canvas.mpl_connect("idle_event", on_idle)

    # Initialize view
    set_frame(i0)
    plt.show()

if __name__ == "__main__":
    main()
