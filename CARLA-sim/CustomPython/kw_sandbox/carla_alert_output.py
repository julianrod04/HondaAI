"""carla_alert_output.py

Runtime alert renderer for the CARLA simulator pygame window.

Reads an AlertVector (from alert_models.py) and dispatches to one of three
alert types each tick:

  0  arrow  – directional arrow pointing to the AV's position LAG seconds
               ahead. Color gradient green→red (or colorblind blue→orange).
               Optional haptic vibration trigger. Renders on windshield
               (semi-transparent overlay) or control panel.

  1  route  – full pre-recorded AV trajectory as a path overlay. Same color
               and vibration rules as arrow. No lag. Windshield or panel.

  2  sound  – plays "left"/"right" audio based on lateral offset to the AV
               LAG seconds ahead. No GUI overlay; uses cooldown to avoid spam.

Prerequisites
-------------
- pygame window already created and passed in.
- CARLA client/world already running.
- Camera sensor already attached to the hero vehicle.
- AV has completed its episode first; its trajectory is recorded in
  AVTrajectory before the human run begins.

Usage
-----
    traj = AVTrajectory()
    traj.record(av_vehicle, world)          # call each tick during AV episode

    renderer = AlertRenderer(screen, camera_actor, hero_vehicle, traj)

    # each tick of the human episode:
    renderer.tick(alert_vector, clock_seconds)
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pygame

# AlertVector is imported at runtime so this file can be used standalone
# (without the full alert_models dependency) by passing a duck-typed object.
try:
    from alert_models import AlertVector
except ImportError:
    AlertVector = None  # type: ignore

try:
    import carla  # type: ignore
except ImportError:
    carla = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes
# ─────────────────────────────────────────────────────────────────────────────

def _distance_color(
    t: float,
    colorblind: bool,
) -> Tuple[int, int, int]:
    """Return an RGB colour interpolated by normalised distance t ∈ [0, 1].

    Standard:   green (close, t=0) → red (far, t=1)
    Colorblind: blue  (close, t=0) → orange (far, t=1)
    """
    t = float(np.clip(t, 0.0, 1.0))
    if colorblind:
        r = int(0   + t * 230)
        g = int(114 - t * 114)
        b = int(178 - t * 178)
    else:
        r = int(t * 220)
        g = int((1.0 - t) * 200)
        b = 0
    return (r, g, b)


# ─────────────────────────────────────────────────────────────────────────────
# AVTrajectory — record + query AV positions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TrajectoryPoint:
    t: float          # simulation time (seconds)
    x: float
    y: float
    z: float
    yaw: float        # degrees
    speed: float = 0.0  # m/s magnitude


class AVTrajectory:
    """Records world-space positions from the AV episode for later playback.

    Record phase  (AV runs first):
        traj = AVTrajectory()
        traj.add(sim_time, av_vehicle)  # call each world tick

    Query phase  (human runs, same route):
        pos = traj.get_position_at(human_sim_time)   # interpolated x,y,z
        lat = traj.lateral_offset_at(human_sim_time, hero_vehicle)
    """

    def __init__(self) -> None:
        self._points: List[_TrajectoryPoint] = []

    # ── Recording ──────────────────────────────────────────────────────────

    def add(self, sim_time: float, av_vehicle: "carla.Vehicle") -> None:
        """Append the AV's current position and speed. Call once per world tick."""
        tf = av_vehicle.get_transform()
        vel = av_vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        self._points.append(_TrajectoryPoint(
            t=sim_time,
            x=tf.location.x,
            y=tf.location.y,
            z=tf.location.z,
            yaw=tf.rotation.yaw,
            speed=speed,
        ))

    # ── Query ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._points)

    def get_position_at(self, t: float) -> Optional[Tuple[float, float, float]]:
        """Return interpolated (x, y, z) at time t. None if out of range."""
        if len(self._points) < 2:
            return None
        if t <= self._points[0].t:
            p = self._points[0]
            return (p.x, p.y, p.z)
        if t >= self._points[-1].t:
            p = self._points[-1]
            return (p.x, p.y, p.z)
        # binary search
        lo, hi = 0, len(self._points) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._points[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self._points[lo], self._points[hi]
        alpha = (t - a.t) / max(b.t - a.t, 1e-9)
        return (
            a.x + alpha * (b.x - a.x),
            a.y + alpha * (b.y - a.y),
            a.z + alpha * (b.z - a.z),
        )

    def lateral_offset_at(
        self,
        t: float,
        hero_vehicle: "carla.Vehicle",
    ) -> Optional[float]:
        """Signed lateral offset of hero relative to AV's lane axis at time t.

        Positive = AV is to the right of the hero (hero should steer right).
        Negative = AV is to the left  (hero should steer left).
        """
        pos = self.get_position_at(t)
        if pos is None:
            return None
        # heading of the road segment at time t
        idx = self._nearest_index(t)
        fwd_yaw = self._points[min(idx + 1, len(self._points) - 1)].yaw
        fwd_rad = math.radians(fwd_yaw)
        fwd = np.array([math.cos(fwd_rad), math.sin(fwd_rad)])
        right = np.array([fwd[1], -fwd[0]])  # 90° clockwise

        hero_loc = hero_vehicle.get_location()
        av_pos = np.array([pos[0], pos[1]])
        hero_pos = np.array([hero_loc.x, hero_loc.y])
        diff = av_pos - hero_pos          # vector from hero to AV
        return float(np.dot(diff, right)) # positive = AV to the right

    def _nearest_index(self, t: float) -> int:
        if not self._points:
            return 0
        lo, hi = 0, len(self._points) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._points[mid].t <= t:
                lo = mid
            else:
                hi = mid
        return lo

    def all_positions(self) -> List[Tuple[float, float, float]]:
        """Return all recorded (x, y, z) world-space positions."""
        return [(p.x, p.y, p.z) for p in self._points]

    def get_speed_at(self, t: float) -> float:
        """Return interpolated AV speed (m/s) at time t."""
        if not self._points:
            return 0.0
        if t <= self._points[0].t:
            return self._points[0].speed
        if t >= self._points[-1].t:
            return self._points[-1].speed
        lo, hi = 0, len(self._points) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._points[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self._points[lo], self._points[hi]
        alpha = (t - a.t) / max(b.t - a.t, 1e-9)
        return a.speed + alpha * (b.speed - a.speed)


# ─────────────────────────────────────────────────────────────────────────────
# Camera projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _world_to_screen(
    world_point: Tuple[float, float, float],
    camera: "carla.Sensor",
    img_w: int,
    img_h: int,
    fov_deg: float = 90.0,
) -> Optional[Tuple[int, int]]:
    """Project a world-space (x, y, z) point into camera pixel coordinates.

    Returns (u, v) if the point is in front of the camera, else None.
    Uses the UE4 → standard camera convention: camera axes are [y, -z, x].
    """
    if carla is None:
        return None

    # Build world→camera matrix
    world_mat = np.array(camera.get_transform().get_matrix())       # 4×4
    cam_mat   = np.array(camera.get_transform().get_inverse_matrix())

    wp = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    cam_coords = cam_mat @ wp       # [x_c, y_c, z_c, 1]

    # UE4 convention: forward=X, right=Y, up=Z → camera space: forward=X
    # Standard pinhole: image right = cam_y, image up = -cam_z, depth = cam_x
    forward = cam_coords[0]
    if forward <= 0.01:             # behind camera
        return None

    f = img_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    u = int(img_w  / 2.0 + f * cam_coords[1] / forward)
    v = int(img_h  / 2.0 - f * cam_coords[2] / forward)

    if 0 <= u < img_w and 0 <= v < img_h:
        return (u, v)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sound player (non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

def _play_direction(direction: str, volume: float = 1.0) -> None:
    """Play 'left' or 'right' speech asynchronously.

    Uses `say` on macOS, `espeak` on Linux, PowerShell SpeechSynthesizer on Windows.
    """
    def _run() -> None:
        sys = platform.system()
        try:
            if sys == "Darwin":
                subprocess.run(
                    ["say", "-v", "Samantha", "-r", "220", direction],
                    check=False, timeout=3,
                )
            elif sys == "Linux":
                subprocess.run(
                    ["espeak", "-a", str(int(volume * 200)), direction],
                    check=False, timeout=3,
                )
            elif sys == "Windows":
                vol_int = int(volume * 100)
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Volume = {vol_int}; "
                    f"$s.Rate = 2; "
                    f"$s.Speak('{direction}');"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    check=False, timeout=5,
                )
        except FileNotFoundError:
            pass  # TTS not installed — silent fallback
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# AlertRenderer
# ─────────────────────────────────────────────────────────────────────────────

# Pixel dimensions used when rendering on the control panel mini-view
_PANEL_W = 320
_PANEL_H = 240

# Maximum expected car-to-car distance for colour normalisation (metres)
_MAX_DISTANCE_M = 80.0

# Distance threshold beyond which vibration is triggered (metres)
_DEFAULT_VIBRATION_DIST_M = 15.0


class AlertRenderer:
    """Dispatches alert rendering each simulation tick.

    Parameters
    ----------
    screen : pygame.Surface
        The main pygame display surface.
    camera : carla.Sensor
        The RGB camera sensor attached to the hero vehicle (used for windshield
        projection). Pass None to skip windshield projection.
    hero_vehicle : carla.Vehicle
        The human-driven vehicle.
    trajectory : AVTrajectory
        Pre-recorded AV trajectory from its earlier episode run.
    img_w, img_h : int
        Camera image resolution (must match the CARLA camera blueprint).
    fov : float
        Camera horizontal field-of-view in degrees.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        camera: Optional["carla.Sensor"],
        hero_vehicle: "carla.Vehicle",
        trajectory: AVTrajectory,
        img_w: int = 84,
        img_h: int = 84,
        fov: float = 90.0,
    ) -> None:
        self._screen = screen
        self._camera = camera
        self._hero = hero_vehicle
        self._traj = trajectory
        self._img_w = img_w
        self._img_h = img_h
        self._fov = fov

        # Sound cooldown state
        self._last_sound_t: float = -999.0
        self._sound_cooldown: float = 3.0  # default; overridden per tick

        # Vibration output flag (read externally to drive haptic hardware)
        self.vibrating: bool = False

    # ── Public API ─────────────────────────────────────────────────────────

    def tick(self, alert: "AlertVector", sim_time: float) -> None:
        """Render one frame of alerts given the current AlertVector.

        Call this once per world tick, after blitting the camera image.

        Parameters
        ----------
        alert : AlertVector
            Decoded alert vector from the alert model.
        sim_time : float
            Current simulation elapsed time in seconds (must be on the same
            clock as AVTrajectory.add() was called with).
        """
        self.vibrating = False

        if alert.gui_type == 0:
            self._render_arrow(alert, sim_time)
        elif alert.gui_type == 1:
            self._render_route(alert, sim_time)
        elif alert.gui_type == 2:
            self._tick_sound(alert, sim_time)

    # ── Arrow (gui_type = 0) ───────────────────────────────────────────────

    def _render_arrow(self, alert: "AlertVector", sim_time: float) -> None:
        """Draw a directional arrow pointing toward the AV's future position."""
        lag_t = sim_time + alert.lag
        target = self._traj.get_position_at(lag_t)
        if target is None:
            return

        scale         = float(alert.gui_params[0])  # 0→small, 1→large
        opacity       = float(alert.gui_params[1])  # 0→transparent, 1→opaque
        vibration_dist = float(alert.gui_params[2]) * _MAX_DISTANCE_M

        # Distance for colour gradient
        hero_loc = self._hero.get_location()
        dist = math.sqrt(
            (target[0] - hero_loc.x) ** 2 +
            (target[1] - hero_loc.y) ** 2
        )
        dist_t = min(dist / _MAX_DISTANCE_M, 1.0)
        color  = _distance_color(dist_t, bool(alert.color))

        self.vibrating = dist > vibration_dist and bool(alert.vibration)

        if alert.location == 1:  # windshield projection
            self._draw_windshield_arrow(target, color, scale, opacity)
        else:                    # control panel
            self._draw_panel_arrow(target, color, scale, opacity)

    def _draw_windshield_arrow(
        self,
        world_target: Tuple[float, float, float],
        color: Tuple[int, int, int],
        scale: float,
        opacity: float,
    ) -> None:
        """Project target into screen space and draw a semi-transparent arrow."""
        px = _world_to_screen(world_target, self._camera, self._img_w, self._img_h, self._fov)
        if px is None:
            return

        # Scale px to actual screen size
        sw, sh = self._screen.get_size()
        sx = int(px[0] * sw / self._img_w)
        sy = int(px[1] * sh / self._img_h)

        size = int(20 + scale * 40)  # arrow head half-size in pixels
        alpha = int(opacity * 220)

        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        # Arrow head: triangle pointing at (sx, sy) from below
        tip = (sx, sy)
        left  = (sx - size, sy + size * 2)
        right = (sx + size, sy + size * 2)
        shaft_top_l = (sx - size // 3, sy + size * 2)
        shaft_top_r = (sx + size // 3, sy + size * 2)
        shaft_bot_l = (sx - size // 3, sy + size * 3)
        shaft_bot_r = (sx + size // 3, sy + size * 3)

        arrow_color = (*color, alpha)
        pygame.draw.polygon(overlay, arrow_color, [tip, left, shaft_top_l, shaft_bot_l, shaft_bot_r, shaft_top_r, right])
        self._screen.blit(overlay, (0, 0))

    def _draw_panel_arrow(
        self,
        world_target: Tuple[float, float, float],
        color: Tuple[int, int, int],
        scale: float,
        opacity: float,
    ) -> None:
        """Draw a compass-style arrow in the bottom-right panel area."""
        sw, sh = self._screen.get_size()
        # Panel is anchored to bottom-right
        panel_x = sw - _PANEL_W - 10
        panel_y = sh - _PANEL_H - 10
        cx = panel_x + _PANEL_W // 2
        cy = panel_y + _PANEL_H // 2

        # Bearing from hero to target in screen coords
        hero_loc = self._hero.get_location()
        hero_tf  = self._hero.get_transform()
        dx = world_target[0] - hero_loc.x
        dy = world_target[1] - hero_loc.y
        # World yaw of direction vector → relative to hero heading
        world_angle = math.degrees(math.atan2(dy, dx))
        rel_angle   = math.radians(world_angle - hero_tf.rotation.yaw)

        arrow_len = int((_PANEL_H // 2 - 20) * (0.5 + scale * 0.5))
        tip_x = cx + int(arrow_len * math.sin(rel_angle))
        tip_y = cy - int(arrow_len * math.cos(rel_angle))

        alpha = int(opacity * 255)
        overlay = pygame.Surface((_PANEL_W + 20, _PANEL_H + 20), pygame.SRCALPHA)

        # Panel background
        pygame.draw.rect(overlay, (0, 0, 0, 120), (0, 0, _PANEL_W, _PANEL_H))
        # Arrow line
        local_cx = _PANEL_W // 2
        local_cy = _PANEL_H // 2
        local_tip_x = local_cx + int(arrow_len * math.sin(rel_angle))
        local_tip_y = local_cy - int(arrow_len * math.cos(rel_angle))
        pygame.draw.line(overlay, (*color, alpha), (local_cx, local_cy), (local_tip_x, local_tip_y), 4)
        # Arrow head
        head_size = max(8, int(arrow_len * 0.25))
        left_ang  = rel_angle + math.radians(150)
        right_ang = rel_angle - math.radians(150)
        lx = local_tip_x + int(head_size * math.sin(left_ang))
        ly = local_tip_y - int(head_size * math.cos(left_ang))
        rx = local_tip_x + int(head_size * math.sin(right_ang))
        ry = local_tip_y - int(head_size * math.cos(right_ang))
        pygame.draw.polygon(overlay, (*color, alpha), [(local_tip_x, local_tip_y), (lx, ly), (rx, ry)])

        self._screen.blit(overlay, (panel_x, panel_y))

    # ── Route (gui_type = 1) ───────────────────────────────────────────────

    def _render_route(self, alert: "AlertVector", sim_time: float) -> None:
        """Draw the full AV trajectory as a path overlay."""
        positions = self._traj.all_positions()
        if len(positions) < 2:
            return

        line_width   = max(1, int(1 + float(alert.gui_params[0]) * 7))
        opacity      = float(alert.gui_params[1])
        vibration_dist = float(alert.gui_params[2]) * _MAX_DISTANCE_M

        # Vibration: distance to nearest trajectory point
        hero_loc = self._hero.get_location()
        nearest_dist = min(
            math.sqrt((p[0] - hero_loc.x) ** 2 + (p[1] - hero_loc.y) ** 2)
            for p in positions
        )
        dist_t = min(nearest_dist / _MAX_DISTANCE_M, 1.0)
        color  = _distance_color(dist_t, bool(alert.color))

        self.vibrating = nearest_dist > vibration_dist and bool(alert.vibration)

        if alert.location == 1:  # windshield projection
            self._draw_windshield_route(positions, color, line_width, opacity)
        else:                    # control panel minimap
            self._draw_panel_route(positions, color, line_width, opacity)

    def _draw_windshield_route(
        self,
        positions: List[Tuple[float, float, float]],
        color: Tuple[int, int, int],
        line_width: int,
        opacity: float,
    ) -> None:
        """Project every trajectory point to screen space and draw a polyline."""
        if self._camera is None:
            return
        sw, sh = self._screen.get_size()
        alpha = int(opacity * 200)
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)

        screen_pts = []
        for pos in positions:
            px = _world_to_screen(pos, self._camera, self._img_w, self._img_h, self._fov)
            if px is not None:
                sx = int(px[0] * sw / self._img_w)
                sy = int(px[1] * sh / self._img_h)
                screen_pts.append((sx, sy))

        if len(screen_pts) >= 2:
            pygame.draw.lines(overlay, (*color, alpha), False, screen_pts, line_width)
        self._screen.blit(overlay, (0, 0))

    def _draw_panel_route(
        self,
        positions: List[Tuple[float, float, float]],
        color: Tuple[int, int, int],
        line_width: int,
        opacity: float,
    ) -> None:
        """Draw a top-down minimap of the route in the control panel area."""
        sw, sh = self._screen.get_size()
        panel_x = sw - _PANEL_W - 10
        panel_y = sh - _PANEL_H - 10

        alpha = int(opacity * 255)
        overlay = pygame.Surface((_PANEL_W, _PANEL_H), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 120), (0, 0, _PANEL_W, _PANEL_H))

        if not positions:
            self._screen.blit(overlay, (panel_x, panel_y))
            return

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        pad = 15

        def _to_panel(wx: float, wy: float) -> Tuple[int, int]:
            px = pad + int((wx - min_x) / span_x * (_PANEL_W - 2 * pad))
            py = pad + int((wy - min_y) / span_y * (_PANEL_H - 2 * pad))
            return (px, py)

        pts = [_to_panel(p[0], p[1]) for p in positions]
        if len(pts) >= 2:
            pygame.draw.lines(overlay, (*color, alpha), False, pts, line_width)

        # Hero position dot
        hero_loc = self._hero.get_location()
        hx, hy = _to_panel(hero_loc.x, hero_loc.y)
        pygame.draw.circle(overlay, (255, 255, 255, 220), (hx, hy), 5)

        self._screen.blit(overlay, (panel_x, panel_y))

    # ── Sound (gui_type = 2) ───────────────────────────────────────────────

    def _tick_sound(self, alert: "AlertVector", sim_time: float) -> None:
        """Play 'left'/'right' if AV is laterally offset beyond threshold."""
        lateral_threshold = float(alert.gui_params[0]) * 10.0  # [0,1] → [0,10] m
        cooldown          = float(alert.gui_params[1]) * 10.0  # [0,1] → [0,10] s
        volume            = float(alert.gui_params[2])

        # Enforce cooldown
        if sim_time - self._last_sound_t < cooldown:
            return

        lag_t  = sim_time + alert.lag
        offset = self._traj.lateral_offset_at(lag_t, self._hero)
        if offset is None:
            return

        if abs(offset) < lateral_threshold:
            return  # within acceptable lane — no cue needed

        direction = "right" if offset > 0 else "left"
        _play_direction(direction, volume)
        self._last_sound_t = sim_time


# ─────────────────────────────────────────────────────────────────────────────
# Minimal smoke test (no CARLA required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Build a fake trajectory (straight line, 5 m/s for 10 s)
    traj = AVTrajectory()
    for i in range(200):
        t = i * 0.05
        traj._points.append(_TrajectoryPoint(t=t, x=t * 5.0, y=0.0, z=0.0, yaw=0.0))

    print("AVTrajectory smoke test")
    pos = traj.get_position_at(5.0)
    print(f"  position at t=5s : {pos}")
    assert pos is not None and abs(pos[0] - 25.0) < 0.1, "position mismatch"

    # Test colour helper
    c = _distance_color(0.0, False)
    assert c[1] > c[0], "green should dominate at t=0"
    c = _distance_color(1.0, False)
    assert c[0] > c[1], "red should dominate at t=1"

    print("  All assertions passed.")
    print("  (Full rendering test requires a running CARLA + pygame window.)")
