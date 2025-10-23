#!/usr/bin/env python3
"""
Interactive CARLA datalog visualizer.

Quick start:
  python CustomPython/visualize_datalog.py --actor both

Other examples:
  python CustomPython/visualize_datalog.py                  (auto-picks newest log)
  python CustomPython/visualize_datalog.py --date 2025-10-23
  python CustomPython/visualize_datalog.py path/to/carla_log_*.csv

Keyboard shortcuts:
  Space: play / pause
  H: toggle HUD
  V: toggle velocity arrows
  T: toggle turn arrows
  P: toggle path

Expected CSV columns:
actor_tag,actor_id,frame,sim_time,x,y,z,yaw_deg,speed_mps,yawrate_rps,throttle,brake,steer,
reverse,gear,hand_brake,lane_id,offset_m,dist_left_m,dist_right_m,map
"""

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

DEFAULT_ROOT = Path("DataLogs")
ACTOR_TAGS = ("hero", "npc")
ACTOR_COLORS = {
    "hero": "#1f77b4",
    "npc": "#d62728",
}

def find_latest_csv(root: Path, date: str | None):
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
    df = pd.read_csv(csv_path)
    required = [
        "actor_tag","actor_id","frame","sim_time","x","y","z","yaw_deg","speed_mps","yawrate_rps",
        "throttle","brake","steer","reverse","gear","hand_brake","lane_id","offset_m",
        "dist_left_m","dist_right_m","map"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")
    numeric_cols = [
        "actor_id","frame","sim_time","x","y","z","yaw_deg","speed_mps","yawrate_rps",
        "throttle","brake","steer","reverse","gear","hand_brake",
        "lane_id","offset_m","dist_left_m","dist_right_m"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["actor_tag"] = df["actor_tag"].astype(str)
    df["map"] = df["map"].astype(str)
    df = df.sort_values(["frame","actor_tag"]).reset_index(drop=True)
    return df

def compute_vectors(yaw_deg, speed, yawrate, turn_scale):
    yaw_rad = np.deg2rad(yaw_deg)
    hx = np.cos(yaw_rad); hy = np.sin(yaw_rad)
    vx = speed * hx;  vy = speed * hy
    tx = -hy * yawrate * turn_scale
    ty =  hx * yawrate * turn_scale
    return vx, vy, tx, ty

def split_by_actor(df: pd.DataFrame):
    return {tag: df[df["actor_tag"] == tag].copy() for tag in df["actor_tag"].unique()}

def align_on_frames(groups: dict[str, pd.DataFrame]):
    frames_union = np.unique(np.concatenate([g["frame"].values for g in groups.values() if not g.empty]))
    frames_union.sort()
    aligned = {}
    for tag, g in groups.items():
        if g.empty:
            aligned[tag] = g
            continue
        g2 = g.set_index("frame").sort_index()
        # keep all columns then ffill
        g2 = g2.reindex(frames_union).ffill()
        g2["actor_tag"] = tag
        aligned[tag] = g2.reset_index()
    return frames_union, aligned

def build_hud_line(row, compact=True):
    if compact:
        return (f"{row['actor_tag']} (id={int(row['actor_id']) if not math.isnan(row['actor_id']) else -1}) "
                f"t={row['sim_time']:.1f}s v={row['speed_mps']*3.6:.1f}km/h "
                f"yaw={row['yaw_deg']:.1f}° off={row['offset_m']:.2f}m")
    else:
        # fuller set (still shorter than before)
        lane = int(row['lane_id']) if not math.isnan(row['lane_id']) else -1
        return (f"{row['actor_tag']} (id={int(row['actor_id']) if not math.isnan(row['actor_id']) else -1}) "
                f"t={row['sim_time']:.2f}s pos=({row['x']:.1f},{row['y']:.1f}) v={row['speed_mps']*3.6:.1f}km/h "
                f"yaw={row['yaw_deg']:.1f}° yawrate={row['yawrate_rps']:.3f} lane={lane} off={row['offset_m']:.2f}m")

def main():
    ap = argparse.ArgumentParser(
        description="Visualize CARLA CSV logs with spaced-out path and speed plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python CustomPython/visualize_datalog.py --actor both",
    )
    ap.add_argument("csv", nargs="?", help="Path to a CARLA log CSV (carla_log_*.csv)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Root directory that contains dated log folders")
    ap.add_argument("--date", help="Date folder under the root to auto-pick the newest CSV (YYYY-MM-DD)")
    ap.add_argument("--fps", type=int, default=20, help="Playback FPS")
    ap.add_argument("--tail", type=int, default=200, help="Frames to display as trail; 0 shows the full path")
    ap.add_argument("--actor", choices=["hero", "npc", "both"], default="both", help="Actors to show on load")
    ap.add_argument("--hud", choices=["off", "compact", "full"], default="compact", help="HUD verbosity")
    ap.add_argument("--no-legend", action="store_true", help="Hide legends for a cleaner look")
    ap.add_argument("--turn-scale", type=float, default=5.0, help="Scale factor for the yaw-rate arrow")
    ap.add_argument("--point-size", type=int, default=8, help="Marker size for the current position")
    args = ap.parse_args()

    root = Path(args.root)

    # Resolve CSV
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[viewer] CSV not found: {csv_path}"); sys.exit(1)
    else:
        csv_path, err = find_latest_csv(root, args.date)
        if err:
            print(err); sys.exit(1)
        print(f"[viewer] Opening: {csv_path}")

    df = load_csv(csv_path)
    groups = split_by_actor(df)
    groups = {k: v for k, v in groups.items() if k in ACTOR_TAGS}
    frames_union, aligned = align_on_frames(groups)
    if not len(frames_union):
        print("[viewer] No frames."); sys.exit(1)

    # Visibility & toggles
    visible = {"hero": args.actor in ("hero","both"), "npc": args.actor in ("npc","both")}
    show = {"path": True, "vel": True, "turn": False, "hud": args.hud != "off"}
    hud_compact = {"on": args.hud == "compact"}

    # Figure layout
    fig = plt.figure("CARLA Log Viewer", figsize=(13, 8))
    fig.suptitle(f"CARLA Log Viewer — {csv_path.name}", fontsize=14, fontweight="bold")

    # Main path axes
    # Main path axes (expanded height)
    # Main path axes (expanded height)
    ax_path = plt.axes([0.07, 0.35, 0.66, 0.60])  # slightly taller, starts lower
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title("Top-down Path", fontsize=11, pad=8)
    ax_path.grid(True, linewidth=0.5, alpha=0.3)

    # Speed timeline axes (shifted down)
    ax_speed = plt.axes([0.07, 0.15, 0.66, 0.16])



    # Speed timeline axes (km/h)
    ax_speed = plt.axes([0.07, 0.18, 0.66, 0.16])
    ax_speed.set_xlabel("Simulation Time (s)")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_title("Speed over Time", fontsize=11, pad=6)
    ax_speed.grid(True, linewidth=0.4, alpha=0.25)

    # Lines, markers, quivers
    lines = {tag: None for tag in ACTOR_TAGS}
    markers = {tag: None for tag in ACTOR_TAGS}
    vel_quiv = {tag: None for tag in ACTOR_TAGS}
    turn_quiv = {tag: None for tag in ACTOR_TAGS}
    speed_lines = {tag: None for tag in ACTOR_TAGS}
    speed_markers = {tag: None for tag in ACTOR_TAGS}

    # Create initial line objects (will update with tail)
    speed_time_arrays = []
    first_time = None
    for tag in ACTOR_TAGS:
        if tag in aligned:
            color = ACTOR_COLORS.get(tag, None)
            (line,) = ax_path.plot([], [], linewidth=1.8, label=tag.capitalize(), color=color)
            lines[tag] = line
            g = aligned[tag]
            times = g["sim_time"].values
            speeds = g["speed_mps"].values * 3.6
            if len(times):
                (speed_line,) = ax_speed.plot(times, speeds, linewidth=1.6, color=color, label=f"{tag.capitalize()} speed")
                (speed_marker,) = ax_speed.plot([times[0]], [speeds[0]], marker="o", markersize=6, color=color, alpha=0.9)
                speed_lines[tag] = speed_line
                speed_markers[tag] = speed_marker
                speed_time_arrays.append(times)
                if first_time is None:
                    first_time = times[0]

    # HUD text (smaller font, box)
    hud_text = ax_path.text(
        0.02,
        0.98,
        "",
        transform=ax_path.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    if speed_time_arrays:
        all_times = np.concatenate(speed_time_arrays)
        ax_speed.set_xlim(np.nanmin(all_times), np.nanmax(all_times))
        speed_values = []
        for tag in ACTOR_TAGS:
            if tag in aligned and not aligned[tag].empty:
                speed_values.append(aligned[tag]["speed_mps"].values * 3.6)
        if speed_values:
            all_speeds = np.concatenate(speed_values)
            try:
                smin = np.nanmin(all_speeds)
                smax = np.nanmax(all_speeds)
            except ValueError:
                smin = smax = None
            if smin is not None and smax is not None and np.isfinite(smin) and np.isfinite(smax):
                pad = max(1.0, 0.05 * max(1.0, smax - smin))
                ax_speed.set_ylim(smin - pad, smax + pad)

    speed_cursor = None
    if first_time is not None:
        speed_cursor = ax_speed.axvline(first_time, color="0.2", linestyle="--", linewidth=1.0, alpha=0.6)

    if not args.no_legend:
        ax_path.legend(loc="upper right", frameon=True)
        if any(line is not None for line in speed_lines.values()):
            ax_speed.legend(loc="upper right", frameon=True)

    # Slider
    i0 = 0
    idx_ax = plt.axes([0.07, 0.08, 0.66, 0.03])
    idx_slider = Slider(idx_ax, "Frame idx", 0, len(frames_union)-1, valinit=i0, valfmt="%0.0f")

    # Frame button
    frame_btn_ax = plt.axes([0.75, 0.08, 0.12, 0.05])
    frame_btn = Button(frame_btn_ax, f"Frame: {int(frames_union[i0])}")

    # Play/pause
    btn_ax = plt.axes([0.89, 0.08, 0.08, 0.05])
    btn = Button(btn_ax, "Play ▶")
    playing = {"on": False}

    # Radio for actor visibility
    radio_ax = plt.axes([0.77, 0.50, 0.18, 0.20])
    radio = RadioButtons(radio_ax, ("hero", "npc", "both"), active=("hero","npc","both").index(args.actor))

    # Checkboxes for Path/Vel/Turn/HUD
    check_ax = plt.axes([0.77, 0.30, 0.18, 0.16])
    checks = CheckButtons(check_ax, ("path", "vel", "turn", "hud"),
                          actives=(show["path"], show["vel"], show["turn"], show["hud"]))

    # Helpers
    def set_axes_limits():
        xs, ys = [], []
        for tag, vis in visible.items():
            if not vis or tag not in aligned:
                continue
            g = aligned[tag]
            xs.append(g["x"].values)
            ys.append(g["y"].values)
        if xs:
            x_all = np.concatenate(xs); y_all = np.concatenate(ys)
            xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)

            # make spans equal so aspect='equal' uses the full vertical space
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            span = max(xmax - xmin, ymax - ymin)
            pad  = max(1.0, 0.08 * max(1.0, span))  # generous padding

            ax_path.set_xlim(cx - 0.5*span - pad, cx + 0.5*span + pad)
            ax_path.set_ylim(cy - 0.5*span - pad, cy + 0.5*span + pad)


    def ensure_actor_drawn(tag):
        if tag not in aligned:
            return
        g = aligned[tag]
        if markers[tag] is None:
            color = ACTOR_COLORS.get(tag, "#666666")
            (pt,) = ax_path.plot(
                [g["x"].values[0]],
                [g["y"].values[0]],
                marker="o",
                markersize=args.point_size,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linestyle="None",
            )
            markers[tag] = pt
        if vel_quiv[tag] is None:
            vel_quiv[tag] = ax_path.quiver(
                g["x"].values[0],
                g["y"].values[0],
                0,
                0,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=ACTOR_COLORS.get(tag, None),
            )
        if turn_quiv[tag] is None:
            turn_quiv[tag] = ax_path.quiver(
                g["x"].values[0],
                g["y"].values[0],
                0,
                0,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=ACTOR_COLORS.get(tag, None),
            )
        if speed_lines[tag] is not None:
            speed_lines[tag].set_visible(visible[tag])
        if speed_markers[tag] is not None:
            speed_markers[tag].set_visible(visible[tag])

    for tag, vis in visible.items():
        if vis:
            ensure_actor_drawn(tag)
        if lines[tag] is not None:
            lines[tag].set_visible(visible[tag] and show["path"])
        if speed_lines[tag] is not None:
            speed_lines[tag].set_visible(visible[tag])
        if speed_markers[tag] is not None:
            speed_markers[tag].set_visible(visible[tag])
    set_axes_limits()

    def update_visibility_from_radio(label):
        if label == "hero":
            visible["hero"], visible["npc"] = True, False
        elif label == "npc":
            visible["hero"], visible["npc"] = False, True
        else:
            visible["hero"], visible["npc"] = True, True
        for tag in ACTOR_TAGS:
            if visible[tag] and markers[tag] is None:
                ensure_actor_drawn(tag)
            if lines[tag] is not None:
                lines[tag].set_visible(visible[tag] and show["path"])
            if markers[tag] is not None:
                markers[tag].set_visible(visible[tag])
            if vel_quiv[tag] is not None:
                vel_quiv[tag].set_visible(visible[tag] and show["vel"])
            if turn_quiv[tag] is not None:
                turn_quiv[tag].set_visible(visible[tag] and show["turn"])
            if speed_lines[tag] is not None:
                speed_lines[tag].set_visible(visible[tag])
            if speed_markers[tag] is not None:
                speed_markers[tag].set_visible(visible[tag])
        set_axes_limits()
        fig.canvas.draw_idle()

    radio.on_clicked(update_visibility_from_radio)

    def on_checks(label):
        show[label] = not show[label]
        # Apply globally
        for tag in ACTOR_TAGS:
            if lines[tag] is not None:
                lines[tag].set_visible(visible[tag] and show["path"])
            if vel_quiv[tag] is not None:
                vel_quiv[tag].set_visible(visible[tag] and show["vel"])
            if turn_quiv[tag] is not None:
                turn_quiv[tag].set_visible(visible[tag] and show["turn"])
        # HUD visibility
        hud_text.set_visible(show["hud"])
        fig.canvas.draw_idle()

    checks.on_clicked(on_checks)

    def set_frame(i: int):
        i = int(np.clip(i, 0, len(frames_union) - 1))
        frame_val = int(frames_union[i])
        frame_btn.label.set_text(f"Frame: {frame_val}")

        hud_lines = []
        current_time = None
        for tag, vis in visible.items():
            if not vis or tag not in aligned:
                continue
            g = aligned[tag]
            row = g.iloc[i]
            x = row["x"]
            y = row["y"]
            # Tail for path
            if lines[tag] is not None:
                if args.tail and args.tail > 0:
                    start = max(0, i - args.tail)
                    xs = g["x"].values[start : i + 1]
                    ys = g["y"].values[start : i + 1]
                else:
                    xs = g["x"].values[: i + 1]
                    ys = g["y"].values[: i + 1]
                lines[tag].set_data(xs, ys)
                lines[tag].set_visible(show["path"] and vis)
            # Marker
            if markers[tag] is not None:
                markers[tag].set_data([x], [y])
                markers[tag].set_visible(vis)
            # Vectors
            vx, vy, tx, ty = compute_vectors(
                np.array([row["yaw_deg"]]),
                np.array([row["speed_mps"]]),
                np.array([row["yawrate_rps"]]),
                args.turn_scale,
            )
            if vel_quiv[tag] is not None:
                try:
                    vel_quiv[tag].remove()
                except Exception:
                    pass
                vel_quiv[tag] = ax_path.quiver(
                    x,
                    y,
                    vx[0],
                    vy[0],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=ACTOR_COLORS.get(tag, None),
                    visible=(show["vel"] and vis),
                )
            if turn_quiv[tag] is not None:
                try:
                    turn_quiv[tag].remove()
                except Exception:
                    pass
                turn_quiv[tag] = ax_path.quiver(
                    x,
                    y,
                    tx[0],
                    ty[0],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=ACTOR_COLORS.get(tag, None),
                    visible=(show["turn"] and vis),
                )
            # Speed markers
            time_val = float(row["sim_time"]) if not math.isnan(row["sim_time"]) else None
            marker = speed_markers[tag]
            if marker is not None:
                if time_val is not None:
                    marker.set_data([time_val], [row["speed_mps"] * 3.6])
                marker.set_visible(vis and time_val is not None)
            if current_time is None and time_val is not None:
                current_time = time_val
            # HUD
            if show["hud"]:
                hud_lines.append(build_hud_line(row, compact=hud_compact["on"]))
        if speed_cursor is not None and current_time is not None:
            speed_cursor.set_xdata([current_time, current_time])
            speed_cursor.set_ydata(ax_speed.get_ylim())
        hud_text.set_text("\n".join(hud_lines))
        hud_text.set_visible(show["hud"])
        fig.canvas.draw_idle()

    def on_slider(val):
        set_frame(val)

    idx_slider.on_changed(on_slider)

    def toggle_play(_=None):
        playing["on"] = not playing["on"]
        btn.label.set_text("Pause ⏸" if playing["on"] else "Play ▶")
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_play)

    def on_key(evt):
        if evt.key == " ":
            toggle_play()
        elif evt.key in ("h", "H"):
            show["hud"] = not show["hud"]; hud_text.set_visible(show["hud"]); fig.canvas.draw_idle()
        elif evt.key in ("v", "V"):
            show["vel"] = not show["vel"]; 
            for tag in ACTOR_TAGS:
                if vel_quiv[tag] is not None: vel_quiv[tag].set_visible(show["vel"] and visible[tag])
            fig.canvas.draw_idle()
        elif evt.key in ("t", "T"):
            show["turn"] = not show["turn"]; 
            for tag in ACTOR_TAGS:
                if turn_quiv[tag] is not None: turn_quiv[tag].set_visible(show["turn"] and visible[tag])
            fig.canvas.draw_idle()
        elif evt.key in ("p", "P"):
            show["path"] = not show["path"]; 
            for tag in ACTOR_TAGS:
                if lines[tag] is not None: lines[tag].set_visible(show["path"] and visible[tag])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Timer
    interval_ms = max(1, int(1000 / max(1, args.fps)))
    timer = fig.canvas.new_timer(interval=interval_ms)

    def on_timer():
        if playing["on"]:
            i = int(idx_slider.val)
            if i < len(frames_union) - 1:
                idx_slider.set_val(i + 1)
            else:
                toggle_play()

    timer.add_callback(on_timer)
    timer.start()

    # Init
    set_frame(0)
    plt.show()

if __name__ == "__main__":
    main()
