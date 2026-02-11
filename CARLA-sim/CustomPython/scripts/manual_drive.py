#!/usr/bin/env python3
"""
Manual driving script - Drive the hero car from driver's seat POV.

Spawns at the same location as RL training (x=21, y=244.485397 on Town06).

Controls:
    W / Up Arrow    - Throttle
    S / Down Arrow  - Brake
    A / D           - Steer left/right
    R               - Toggle reverse
    SPACE           - Hand brake
    F5 / Backspace  - Reset (teleport back to spawn)
    ESC             - Quit

Automatic Instruction System:
    The system automatically monitors driving behavior and displays instructions:
    - Lane drifting (suggests merging back into proper lane position)
    - Speeding (suggests reducing speed when over limit)
    - Too slow (suggests increasing speed when moving too slowly)
    - Following distance (suggests slowing down or stopping when too close to NPC)

Auto-Respawn Triggers:
    - Track end: x >= 221 (goal reached)
    - Reversed too far: x < 11 (spawn_x - 10)
    - Off-road: lane offset > 7m

Also supports steering wheel/gamepad if connected.

Usage:
    python scripts/manual_drive.py
    python scripts/manual_drive.py --no-npc          # Without NPC vehicle
    python scripts/manual_drive.py --chase-cam       # Third-person chase camera
    python scripts/manual_drive.py --spectator       # Enable top-down spectator
    python scripts/manual_drive.py --spectator-height 60  # Custom bird's-eye height
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from enum import Enum
from typing import Optional

import numpy as np
import pygame

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from rl.config import DEFAULT_CONFIG, ScenarioConfig
from rl.utils import teleport_vehicle, create_spawn_transform


# =============================================================================
# INSTRUCTION SYSTEM
# =============================================================================

class AlertType(Enum):
    """Instruction types for the driving assistant."""
    NONE = 0
    # Lane drift warnings
    DRIFTING_LEFT = 1
    DRIFTING_RIGHT = 2
    # Speed instructions
    STOP = 3
    SPEED_UP = 4
    SLOW_DOWN = 5
    # Navigation instructions
    TURN_RIGHT = 6
    TURN_LEFT = 7
    STOP_AT_LIGHT = 8
    # Other
    EMERGENCY_STOP = 9
    MAINTAIN = 10


# Instruction display configuration
ALERT_CONFIG = {
    AlertType.NONE: {
        "text": "",
        "color": (255, 255, 255),
        "bg_color": None,
        "icon": None,
        "field_of_vision": False,
    },
    AlertType.DRIFTING_LEFT: {
        "text": "Drifting into left lane",
        "color": (255, 255, 255),
        "bg_color": (200, 150, 0),
        "icon": "\u2190",
        "field_of_vision": False,
    },
    AlertType.DRIFTING_RIGHT: {
        "text": "Drifting into right lane",
        "color": (255, 255, 255),
        "bg_color": (200, 150, 0),
        "icon": "\u2192",
        "field_of_vision": False,
    },
    AlertType.STOP: {
        "text": "Come to a complete stop",
        "color": (255, 255, 255),
        "bg_color": (200, 50, 50),
        "icon": "\u25a0",
        "field_of_vision": False,
    },
    AlertType.SPEED_UP: {
        "text": "Increase your speed gradually",
        "color": (255, 255, 255),
        "bg_color": (50, 150, 50),
        "icon": "\u2191",
        "field_of_vision": False,
    },
    AlertType.SLOW_DOWN: {
        "text": "Reduce your speed",
        "color": (255, 255, 255),
        "bg_color": (200, 150, 0),
        "icon": "\u2193",
        "field_of_vision": False,
    },
    AlertType.TURN_RIGHT: {
        "text": "Turn Right",
        "color": (255, 255, 255),
        "bg_color": (0, 120, 200),
        "icon": "\u2192",
        "field_of_vision": True,
    },
    AlertType.TURN_LEFT: {
        "text": "Turn Left",
        "color": (255, 255, 255),
        "bg_color": (0, 120, 200),
        "icon": "\u2190",
        "field_of_vision": True,
    },
    AlertType.STOP_AT_LIGHT: {
        "text": "Stop at the light",
        "color": (255, 255, 255),
        "bg_color": (200, 50, 50),
        "icon": "\u25cf",
        "field_of_vision": True,
    },
    AlertType.EMERGENCY_STOP: {
        "text": "Stop immediately - Emergency",
        "color": (255, 255, 255),
        "bg_color": (255, 0, 0),
        "icon": "\u26a0",
        "field_of_vision": True,
    },
    AlertType.MAINTAIN: {
        "text": "Maintain your current speed",
        "color": (200, 200, 200),
        "bg_color": (80, 80, 80),
        "icon": "\u25cf",
        "field_of_vision": False,
    },
}


class Dashboard:
    """
    Renders a car dashboard UI at the bottom of the screen.
    Displays speed, instructions, lane position, and other driving info.
    Also displays navigation instructions in the driver's field of vision.
    """

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Dashboard dimensions
        self.dashboard_height = 120
        self.dashboard_y = screen_height - self.dashboard_height

        # Dashboard alert state (bottom right - drifting, speed, etc.)
        self.current_alert: AlertType = AlertType.NONE
        self.alert_start_time: float = 0.0
        self.alert_duration: float = 3.0  # seconds

        # Navigation instruction state (field of vision - turn, stop at light, etc.)
        self.current_nav: AlertType = AlertType.NONE
        self.nav_start_time: float = 0.0
        self.nav_duration: float = 5.0  # seconds - longer for navigation

        # Fonts
        pygame.font.init()
        self.font_speed = pygame.font.SysFont("arial", 42, bold=True)
        self.font_speed_unit = pygame.font.SysFont("arial", 18)
        self.font_instruction = pygame.font.SysFont("arial", 28, bold=True)
        self.font_nav = pygame.font.SysFont("arial", 36, bold=True)  # Larger for field of vision
        self.font_label = pygame.font.SysFont("arial", 14)
        self.font_value = pygame.font.SysFont("arial", 18, bold=True)

        # Animation state
        self.fade_in_duration = 0.2
        self.fade_out_duration = 0.5

        # Instruction history for logging
        self.alert_history: list = []

        # Cooldown to prevent instruction spam
        self.last_alert_time: dict = {}
        self.alert_cooldown: float = 2.0  # seconds between same instruction type

        # Colors
        self.bg_color = (20, 20, 25)  # Dark background
        self.border_color = (60, 60, 70)
        self.text_color = (220, 220, 220)
        self.accent_color = (0, 150, 255)  # Blue accent

    def trigger_alert(self, alert_type: AlertType, duration: float = 3.0, force: bool = False) -> bool:
        """
        Trigger a new instruction (dashboard or field of vision based on config).

        Returns True if instruction was triggered, False if on cooldown.
        """
        if alert_type == AlertType.NONE:
            self.clear_alert()
            return True

        # Check cooldown (unless forced)
        if not force:
            last_time = self.last_alert_time.get(alert_type, 0)
            if time.time() - last_time < self.alert_cooldown:
                return False

        config = ALERT_CONFIG[alert_type]

        # Determine if this is a field of vision instruction or dashboard alert
        if config.get("field_of_vision", False):
            self.current_nav = alert_type
            self.nav_start_time = time.time()
            self.nav_duration = duration
        else:
            self.current_alert = alert_type
            self.alert_start_time = time.time()
            self.alert_duration = duration

        self.last_alert_time[alert_type] = time.time()

        # Log the instruction
        self.alert_history.append({
            "type": alert_type.name,
            "time": time.time(),
        })

        print(f"[INSTRUCTION] {config['text']}")
        return True

    def trigger_navigation(self, alert_type: AlertType, duration: float = 5.0) -> bool:
        """Trigger a navigation instruction in the field of vision."""
        return self.trigger_alert(alert_type, duration, force=True)

    def clear_alert(self) -> None:
        """Clear the current dashboard alert."""
        self.current_alert = AlertType.NONE

    def clear_navigation(self) -> None:
        """Clear the current navigation instruction."""
        self.current_nav = AlertType.NONE

    def get_current_alert(self) -> AlertType:
        """Get the current active dashboard alert."""
        if self.current_alert == AlertType.NONE:
            return AlertType.NONE

        # Check if instruction has expired
        elapsed = time.time() - self.alert_start_time
        if elapsed > self.alert_duration:
            self.current_alert = AlertType.NONE

        return self.current_alert

    def get_current_navigation(self) -> AlertType:
        """Get the current active navigation instruction."""
        if self.current_nav == AlertType.NONE:
            return AlertType.NONE

        # Check if instruction has expired
        elapsed = time.time() - self.nav_start_time
        if elapsed > self.nav_duration:
            self.current_nav = AlertType.NONE

        return self.current_nav

    def render(self, screen: pygame.Surface, metrics: dict, reverse_mode: bool = False) -> None:
        """Render the complete dashboard with all driving information."""

        # === FIELD OF VISION: Navigation Instructions (center-top) ===
        self._render_field_of_vision(screen)

        # Draw dashboard background
        dashboard_surface = pygame.Surface((self.screen_width, self.dashboard_height), pygame.SRCALPHA)
        dashboard_surface.fill((*self.bg_color, 230))  # Semi-transparent dark background

        screen.blit(dashboard_surface, (0, self.dashboard_y))

        # === LEFT SECTION: Speed Gauge ===
        self._render_speed_section(screen, metrics, reverse_mode)

        # === RIGHT SECTION: Lane, Distance & Instructions ===
        self._render_info_section(screen, metrics)

    def _render_field_of_vision(self, screen: pygame.Surface) -> None:
        """Render navigation instructions in the driver's field of vision (center of screen)."""
        nav = self.get_current_navigation()
        if nav == AlertType.NONE:
            return

        config = ALERT_CONFIG[nav]
        elapsed = time.time() - self.nav_start_time

        # Calculate alpha for fade effect
        if elapsed < self.fade_in_duration:
            alpha = elapsed / self.fade_in_duration
        elif elapsed > self.nav_duration - self.fade_out_duration:
            remaining = self.nav_duration - elapsed
            alpha = remaining / self.fade_out_duration
        else:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))

        # Panel dimensions and position (center of screen - in driver's field of vision)
        panel_width = 450
        panel_height = 80
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - self.dashboard_height) // 2 - panel_height // 2  # Center vertically above dashboard

        # Get colors
        bg_color = config["bg_color"]
        text_color = config["color"]

        # Create panel surface with full transparency support
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)

        # Draw background with rounded corners effect
        pygame.draw.rect(panel_surface, (*bg_color, int(220 * alpha)),
                        (0, 0, panel_width, panel_height), border_radius=15)

        # Add white border
        pygame.draw.rect(panel_surface, (255, 255, 255, int(255 * alpha)),
                       (0, 0, panel_width, panel_height), 4, border_radius=15)

        screen.blit(panel_surface, (panel_x, panel_y))

        # Draw instruction text (render fresh each time for proper display)
        display_text = config["text"]
        if config.get("icon"):
            display_text = f"{config['icon']}  {config['text']}"

        nav_text = self.font_nav.render(display_text, True, text_color)
        nav_rect = nav_text.get_rect(center=(panel_x + panel_width // 2, panel_y + panel_height // 2))

        # Create a text surface with alpha
        text_surface = pygame.Surface(nav_text.get_size(), pygame.SRCALPHA)
        text_surface.blit(nav_text, (0, 0))
        text_surface.set_alpha(int(255 * alpha))

        screen.blit(text_surface, nav_rect)

    def _render_speed_section(self, screen: pygame.Surface, metrics: dict, reverse_mode: bool) -> None:
        """Render the speed display on the left side of the dashboard."""
        speed_kmh = metrics.get("speed_kmh", 0)

        # Speed section position
        section_x = 40
        section_center_y = self.dashboard_y + self.dashboard_height // 2

        # Speed color based on value
        if speed_kmh > 80:
            speed_color = (255, 80, 80)  # Red - way too fast
        elif speed_kmh > 50:
            speed_color = (255, 200, 0)  # Yellow - over limit
        else:
            speed_color = (255, 255, 255)  # White - normal

        # Draw speed value
        speed_text = self.font_speed.render(f"{speed_kmh:.0f}", True, speed_color)
        speed_rect = speed_text.get_rect(midleft=(section_x, section_center_y - 5))
        screen.blit(speed_text, speed_rect)

        # Draw unit label
        unit_text = self.font_speed_unit.render("km/h", True, (150, 150, 150))
        unit_rect = unit_text.get_rect(midleft=(speed_rect.right + 5, section_center_y + 10))
        screen.blit(unit_text, unit_rect)

        # Draw reverse indicator if active
        if reverse_mode:
            rev_text = self.font_label.render("R", True, (255, 100, 100))
            rev_rect = rev_text.get_rect(midleft=(section_x, section_center_y + 35))
            # Draw background circle
            pygame.draw.circle(screen, (80, 30, 30), rev_rect.center, 12)
            screen.blit(rev_text, rev_rect)

    def _render_info_section(self, screen: pygame.Surface, metrics: dict) -> None:
        """Render lane position, distance info, and instructions on the right side of the dashboard."""
        section_x = self.screen_width - 200
        section_y = self.dashboard_y + 15

        # === Lane Position ===
        lane_offset = metrics.get("lane_offset", 0)
        lane_status = metrics.get("lane_status", "OK")

        # Lane label
        lane_label = self.font_label.render("LANE POSITION", True, (120, 120, 120))
        screen.blit(lane_label, (section_x, section_y))

        # Lane offset bar
        bar_width = 140
        bar_height = 12
        bar_x = section_x
        bar_y = section_y + 20

        # Background bar
        pygame.draw.rect(screen, (50, 50, 55), (bar_x, bar_y, bar_width, bar_height), border_radius=6)

        # Center marker
        pygame.draw.line(screen, (100, 100, 100),
                        (bar_x + bar_width // 2, bar_y - 2),
                        (bar_x + bar_width // 2, bar_y + bar_height + 2), 2)

        # Position indicator
        if lane_status == "CRITICAL":
            indicator_color = (255, 80, 80)
        elif lane_status == "WARNING":
            indicator_color = (255, 200, 0)
        else:
            indicator_color = (80, 200, 80)

        # Clamp indicator position
        indicator_offset = int(lane_offset * 35)
        indicator_x = bar_x + bar_width // 2 + indicator_offset
        indicator_x = max(bar_x + 8, min(bar_x + bar_width - 8, indicator_x))

        pygame.draw.circle(screen, indicator_color, (indicator_x, bar_y + bar_height // 2), 7)
        pygame.draw.circle(screen, (255, 255, 255), (indicator_x, bar_y + bar_height // 2), 7, 2)

        # Lane offset value
        offset_text = self.font_value.render(f"{lane_offset:+.1f}m", True, indicator_color)
        screen.blit(offset_text, (bar_x + bar_width + 10, bar_y - 2))

        # === Distance to Vehicle Ahead ===
        if "distance_to_npc" in metrics:
            dist = metrics["distance_to_npc"]
            follow_status = metrics.get("follow_status", "N/A")

            dist_y = section_y + 45

            # Distance label
            dist_label = self.font_label.render("VEHICLE AHEAD", True, (120, 120, 120))
            screen.blit(dist_label, (section_x, dist_y))

            # Distance color
            if follow_status == "CRITICAL":
                dist_color = (255, 80, 80)
            elif follow_status == "WARNING":
                dist_color = (255, 200, 0)
            else:
                dist_color = (150, 150, 150)

            # Distance value
            dist_text = self.font_value.render(f"{dist:.1f} m", True, dist_color)
            screen.blit(dist_text, (section_x, dist_y + 18))

            # Simple distance bar
            max_dist = 30.0
            dist_bar_width = int((min(dist, max_dist) / max_dist) * 100)
            pygame.draw.rect(screen, (50, 50, 55), (section_x + 70, dist_y + 20, 100, 8), border_radius=4)
            if dist_bar_width > 0:
                pygame.draw.rect(screen, dist_color, (section_x + 70, dist_y + 20, dist_bar_width, 8), border_radius=4)

        # === Instructions (bottom right) ===
        alert = self.get_current_alert()
        if alert != AlertType.NONE:
            config = ALERT_CONFIG[alert]
            elapsed = time.time() - self.alert_start_time

            # Calculate alpha for fade effect
            if elapsed < self.fade_in_duration:
                alpha = elapsed / self.fade_in_duration
            elif elapsed > self.alert_duration - self.fade_out_duration:
                remaining = self.alert_duration - elapsed
                alpha = remaining / self.fade_out_duration
            else:
                alpha = 1.0
            alpha = max(0.0, min(1.0, alpha))

            # Instruction panel position (bottom right, above dashboard)
            panel_width = 350
            panel_height = 50
            panel_x = self.screen_width - panel_width - 20
            panel_y = self.dashboard_y - panel_height - 10

            # Get colors
            bg_color = config["bg_color"]
            text_color = config["color"]

            # Draw instruction panel with colored background
            panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            panel_surface.fill((*bg_color, int(200 * alpha)))

            # Add border
            pygame.draw.rect(panel_surface, (255, 255, 255, int(150 * alpha)),
                           (0, 0, panel_width, panel_height), 2, border_radius=8)

            screen.blit(panel_surface, (panel_x, panel_y))

            # Draw instruction text
            instruction_text = self.font_instruction.render(config["text"], True, text_color)
            instruction_text.set_alpha(int(255 * alpha))
            instruction_rect = instruction_text.get_rect(center=(panel_x + panel_width // 2, panel_y + panel_height // 2))
            screen.blit(instruction_text, instruction_rect)

            # Pulsing border for urgent instructions
            if alert in (AlertType.STOP, AlertType.EMERGENCY_STOP):
                pulse = abs(math.sin(elapsed * 5)) * 0.5 + 0.5
                pulse_surface = pygame.Surface((panel_width + 6, panel_height + 6), pygame.SRCALPHA)
                pygame.draw.rect(pulse_surface, (255, 80, 80, int(150 * pulse * alpha)),
                               (0, 0, panel_width + 6, panel_height + 6), 3, border_radius=10)
                screen.blit(pulse_surface, (panel_x - 3, panel_y - 3))


# Keep AlertDisplay as an alias for backwards compatibility
AlertDisplay = Dashboard


class DrivingMonitor:
    """
    Monitors driving behavior and automatically provides driving instructions.

    Detects:
    - Lane drifting
    - Speeding / too slow
    - Following too close
    - Erratic steering
    """

    def __init__(self, world_map: carla.Map, config):
        self.world_map = world_map
        self.config = config

        # Thresholds
        self.lane_drift_warning = 0.8  # meters from center - warning
        self.lane_drift_critical = 1.2  # meters from center - critical
        self.max_speed_kmh = 50.0  # speed limit
        self.min_speed_kmh = 10.0  # too slow threshold
        self.follow_distance_warning = 10.0  # meters
        self.follow_distance_critical = 5.0  # meters

        # State tracking
        self.lane_offset_history: list = []
        self.history_size = 30  # frames

    def get_lane_offset(self, vehicle: carla.Vehicle) -> float:
        """Calculate lateral offset from lane center."""
        location = vehicle.get_location()
        waypoint = self.world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return 0.0

        # Calculate lateral offset
        lane_yaw = math.radians(waypoint.transform.rotation.yaw)
        normal_x = -math.sin(lane_yaw)
        normal_y = math.cos(lane_yaw)

        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y

        return dx * normal_x + dy * normal_y

    def get_distance_to_vehicle(self, ego: carla.Vehicle, other: carla.Vehicle) -> float:
        """Get distance between two vehicles."""
        ego_loc = ego.get_location()
        other_loc = other.get_location()

        dx = ego_loc.x - other_loc.x
        dy = ego_loc.y - other_loc.y

        return math.sqrt(dx * dx + dy * dy)

    def is_vehicle_ahead(self, ego: carla.Vehicle, other: carla.Vehicle) -> bool:
        """Check if other vehicle is ahead of ego."""
        ego_tf = ego.get_transform()
        other_loc = other.get_location()

        # Vector from ego to other
        dx = other_loc.x - ego_tf.location.x
        dy = other_loc.y - ego_tf.location.y

        # Ego forward vector
        yaw = math.radians(ego_tf.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)

        # Dot product - positive means ahead
        dot = dx * forward_x + dy * forward_y
        return dot > 0

    def update(
        self,
        ego: carla.Vehicle,
        npc: Optional[carla.Vehicle],
        alert_display: AlertDisplay
    ) -> dict:
        """
        Update monitoring and trigger instructions as needed.

        Returns dict with current driving metrics.
        """
        metrics = {}

        # Get vehicle state
        velocity = ego.get_velocity()
        speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_mps * 3.6
        metrics["speed_kmh"] = speed_kmh

        # Lane offset
        lane_offset = self.get_lane_offset(ego)
        metrics["lane_offset"] = lane_offset

        # Track history for smoothing
        self.lane_offset_history.append(lane_offset)
        if len(self.lane_offset_history) > self.history_size:
            self.lane_offset_history.pop(0)

        avg_offset = sum(self.lane_offset_history) / len(self.lane_offset_history)
        metrics["avg_lane_offset"] = avg_offset

        # === LANE DRIFT DETECTION ===
        abs_offset = abs(avg_offset)
        if abs_offset > self.lane_drift_critical:
            # Critical lane drift - show drifting warning
            if avg_offset > 0:
                # Drifting right
                alert_display.trigger_alert(AlertType.DRIFTING_RIGHT, duration=2.0)
            else:
                # Drifting left
                alert_display.trigger_alert(AlertType.DRIFTING_LEFT, duration=2.0)
            metrics["lane_status"] = "CRITICAL"
        elif abs_offset > self.lane_drift_warning:
            metrics["lane_status"] = "WARNING"
        else:
            metrics["lane_status"] = "OK"

        # === SPEED MONITORING ===
        if speed_kmh > self.max_speed_kmh:
            alert_display.trigger_alert(AlertType.SLOW_DOWN, duration=2.0)
            metrics["speed_status"] = "TOO_FAST"
        elif speed_kmh < self.min_speed_kmh and speed_kmh > 1.0:
            # Only alert if actually moving but too slow
            alert_display.trigger_alert(AlertType.SPEED_UP, duration=2.0)
            metrics["speed_status"] = "TOO_SLOW"
        else:
            metrics["speed_status"] = "OK"

        # === FOLLOWING DISTANCE ===
        if npc is not None:
            distance = self.get_distance_to_vehicle(ego, npc)
            metrics["distance_to_npc"] = distance

            if self.is_vehicle_ahead(ego, npc):
                if distance < self.follow_distance_critical:
                    metrics["follow_status"] = "CRITICAL"
                elif distance < self.follow_distance_warning:
                    alert_display.trigger_alert(AlertType.SLOW_DOWN, duration=2.0)
                    metrics["follow_status"] = "WARNING"
                else:
                    metrics["follow_status"] = "OK"
            else:
                metrics["follow_status"] = "N/A"

        return metrics


# =============================================================================
# TOP-DOWN SPECTATOR
# =============================================================================

class TopDownSpectator:
    """
    Background daemon thread that positions the CARLA spectator camera
    directly above the ego vehicle for a bird's-eye view.

    Finds the ego vehicle by role_name='hero' and updates the spectator
    at 20 FPS with pitch=-90 (straight down).
    """

    def __init__(self, world: carla.World, height: float = 40.0):
        self.world = world
        self.height = height
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the spectator tracking thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Spectator] Top-down view started (height={self.height}m)")

    def stop(self) -> None:
        """Stop the spectator tracking thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            print("[Spectator] Stopped")

    def _loop(self) -> None:
        """Main loop: find hero vehicle and position spectator above it."""
        spectator = self.world.get_spectator()

        while self._running:
            # Find the ego vehicle
            ego = None
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'hero':
                    ego = actor
                    break

            if ego is None:
                time.sleep(0.5)
                continue

            ego_loc = ego.get_location()

            # Position spectator directly above, looking straight down
            cam_loc = carla.Location(
                x=ego_loc.x,
                y=ego_loc.y,
                z=ego_loc.z + self.height
            )
            cam_rot = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            time.sleep(0.05)  # 20 FPS


# =============================================================================
# EPISODE RESETTER
# =============================================================================

class EpisodeResetter:
    """
    Handles resetting the ego (and optionally NPC) vehicle back to spawn.

    Used by both the manual reset keybind (F5/Backspace) and the
    automatic respawn system (track end, reversed, off-road).

    Uses teleport_vehicle() for fast resets without actor re-creation.
    """

    def __init__(
        self,
        world: carla.World,
        client: carla.Client,
        scenario_config: ScenarioConfig,
    ):
        self.world = world
        self.client = client
        self.config = scenario_config

        # Pre-build spawn transforms
        self.ego_spawn = create_spawn_transform(self.config)
        self.npc_spawn = create_spawn_transform(
            self.config,
            lane_offset=self.config.npc_lane_offset,
            x_offset=self.config.npc_x_offset,
        )

        # Boundaries for auto-respawn
        self.goal_x = self.config.goal_x            # 221
        self.min_x = self.config.spawn_x - 10.0     # 11
        self.max_lane_offset = 7.0                   # meters off-road

    def reset(
        self,
        ego: carla.Vehicle,
        npc: Optional[carla.Vehicle] = None,
    ) -> None:
        """Teleport ego (and NPC) back to spawn positions."""
        print("[Reset] Teleporting vehicles to spawn...")

        teleport_vehicle(ego, self.ego_spawn)

        if npc is not None:
            # Disable autopilot before teleport, re-enable after
            npc.set_autopilot(False)
            teleport_vehicle(npc, self.npc_spawn)

            # Re-enable autopilot via Traffic Manager
            tm = self.client.get_trafficmanager()
            npc.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(npc, DEFAULT_CONFIG.vehicle.npc_target_speed_kmh)
            tm.auto_lane_change(npc, False)

        print("[Reset] Done")

    def should_auto_respawn(self, ego: carla.Vehicle) -> bool:
        """
        Check if the ego vehicle has left the valid track area.

        Returns True if any boundary is violated:
          - Track end:      x >= goal_x (221)
          - Reversed:       x < spawn_x - 10 (11)
          - Off-road:       |lane_offset| > 7m
        """
        loc = ego.get_location()

        # Track end
        if loc.x >= self.goal_x:
            print(f"[Auto-Respawn] Track end reached (x={loc.x:.1f} >= {self.goal_x})")
            return True

        # Reversed too far
        if loc.x < self.min_x:
            print(f"[Auto-Respawn] Reversed too far (x={loc.x:.1f} < {self.min_x})")
            return True

        # Off-road check via waypoint projection
        world_map = self.world.get_map()
        waypoint = world_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is not None:
            lane_yaw = math.radians(waypoint.transform.rotation.yaw)
            normal_x = -math.sin(lane_yaw)
            normal_y = math.cos(lane_yaw)
            dx = loc.x - waypoint.transform.location.x
            dy = loc.y - waypoint.transform.location.y
            lane_offset = abs(dx * normal_x + dy * normal_y)

            if lane_offset > self.max_lane_offset:
                print(f"[Auto-Respawn] Off-road (offset={lane_offset:.1f}m > {self.max_lane_offset}m)")
                return True

        return False


# =============================================================================
# INPUT HANDLING
# =============================================================================

def get_speed(vehicle: carla.Vehicle) -> float:
    """Return vehicle speed in m/s."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def get_keyboard_control(
    keys,
    control: carla.VehicleControl,
    reverse_mode: bool
) -> carla.VehicleControl:
    """WASD keyboard control."""
    control.reverse = reverse_mode

    # Check if any driving key is pressed
    any_key = (
        keys[pygame.K_w] or keys[pygame.K_UP] or
        keys[pygame.K_s] or keys[pygame.K_DOWN] or
        keys[pygame.K_a] or keys[pygame.K_LEFT] or
        keys[pygame.K_d] or keys[pygame.K_RIGHT] or
        keys[pygame.K_SPACE]
    )

    if not any_key:
        # Coast - gradually reduce throttle
        control.throttle = max(0.0, control.throttle - 0.1)
        return control

    # Throttle
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        control.throttle = 1.0
        control.brake = 0.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        control.throttle = 0.0
        control.brake = 1.0
    else:
        control.throttle = 0.0
        control.brake = 0.0

    # Steering
    steer_amt = 0.4
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        control.steer = -steer_amt
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        control.steer = steer_amt
    else:
        control.steer = 0.0

    # Hand brake
    control.hand_brake = keys[pygame.K_SPACE]

    return control


def get_wheel_control(
    wheel: pygame.joystick.Joystick,
    control: carla.VehicleControl,
    reverse_mode: bool,
) -> carla.VehicleControl:
    """Steering wheel/gamepad control."""
    control.reverse = reverse_mode

    # Steering (Axis 0)
    steer_axis = wheel.get_axis(0)
    control.steer = max(-1.0, min(1.0, steer_axis))

    # Try common axis mappings for throttle/brake
    num_axes = wheel.get_numaxes()

    if num_axes >= 4:
        # Likely a wheel with separate pedals
        # Axis 2 or 3 for throttle, Axis 1 or 2 for brake
        accel_axis = wheel.get_axis(2)
        brake_axis = wheel.get_axis(3)
    elif num_axes >= 2:
        # Gamepad - use triggers or second stick
        accel_axis = wheel.get_axis(1)
        brake_axis = -wheel.get_axis(1)  # Same axis, opposite direction
    else:
        accel_axis = 0.0
        brake_axis = 0.0

    # Map [-1, 1] to [0, 1]
    throttle = (accel_axis + 1.0) / 2.0
    brake = (brake_axis + 1.0) / 2.0

    # Deadzone
    if throttle < 0.05:
        throttle = 0.0
    if brake < 0.05:
        brake = 0.0

    control.throttle = float(min(1.0, throttle))
    control.brake = float(min(1.0, brake))

    return control


# =============================================================================
# CORE DRIVING LOOP
# =============================================================================

def run_manual_drive(
    client: carla.Client,
    world: carla.World,
    config=None,
    no_npc: bool = False,
    chase_cam: bool = False,
    width: int = 1280,
    height: int = 720,
    enable_spectator: bool = False,
    spectator_height: float = 40.0,
):
    """
    Core manual driving loop.

    Can be called standalone (from main()) or imported by bc_sandbox/run.py.

    Args:
        client: Connected CARLA client
        world: CARLA world (already on correct map)
        config: FullConfig instance (defaults to DEFAULT_CONFIG)
        no_npc: If True, don't spawn NPC vehicle
        chase_cam: If True, use third-person chase camera
        width: Pygame window width
        height: Pygame window height
        enable_spectator: If True, start TopDownSpectator thread
        spectator_height: Height for top-down spectator camera
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Async mode for smooth driving
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # Let simulator settle
    for _ in range(10):
        world.wait_for_tick()

    # Clean up existing vehicles
    print("Cleaning up existing vehicles...")
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    for actor in world.get_actors().filter('sensor.*'):
        actor.destroy()

    blueprints = world.get_blueprint_library()

    # Spawn ego at training location
    ego_bp = blueprints.find(config.vehicle.ego_blueprint)
    ego_bp.set_attribute("role_name", "hero")

    spawn_loc = carla.Location(
        x=config.scenario.spawn_x,
        y=config.scenario.spawn_y,
        z=config.scenario.spawn_z
    )
    spawn_rot = carla.Rotation(pitch=0.0, yaw=config.scenario.spawn_yaw, roll=0.0)
    ego_tf = carla.Transform(spawn_loc, spawn_rot)

    ego = world.try_spawn_actor(ego_bp, ego_tf)
    if ego is None:
        print("Failed to spawn ego vehicle!")
        return

    print(f"Spawned hero at x={spawn_loc.x}, y={spawn_loc.y}")

    # Spawn NPC if requested
    npc = None
    if not no_npc:
        npc_bp = blueprints.find(config.vehicle.npc_blueprint)
        npc_loc = carla.Location(
            x=config.scenario.spawn_x + config.scenario.npc_x_offset,
            y=config.scenario.spawn_y + config.scenario.npc_lane_offset,
            z=config.scenario.spawn_z
        )
        npc_tf = carla.Transform(npc_loc, spawn_rot)
        npc = world.try_spawn_actor(npc_bp, npc_tf)

        if npc:
            tm = client.get_trafficmanager()
            npc.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(npc, config.vehicle.npc_target_speed_kmh)
            tm.auto_lane_change(npc, False)
            print("Spawned NPC in adjacent lane")

    # Attach camera
    W, H = width, height
    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(W))
    cam_bp.set_attribute("image_size_y", str(H))
    cam_bp.set_attribute("fov", "100")

    if chase_cam:
        # Third-person chase camera
        cam_tf = carla.Transform(
            carla.Location(x=-8.0, y=0.0, z=4.0),
            carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0),
        )
    else:
        # Driver's seat POV
        cam_tf = carla.Transform(
            carla.Location(x=0.2, y=-0.36, z=1.2),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
        )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    latest_image = None
    def on_image(img):
        nonlocal latest_image
        array = np.frombuffer(img.raw_data, dtype=np.uint8)
        array = array.reshape((img.height, img.width, 4))
        latest_image = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB

    camera.listen(on_image)

    # Pygame setup
    pygame.init()
    pygame.joystick.init()

    wheel = None
    if pygame.joystick.get_count() > 0:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print(f"Using controller: {wheel.get_name()}")
    else:
        print("No controller detected - using keyboard (WASD)")

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CARLA Manual Drive - WASD to drive, R=reverse, F5=reset, ESC=quit")

    clock = pygame.time.Clock()
    control = carla.VehicleControl()
    reverse_mode = False
    running = True

    # Create instruction display and driving monitor
    alert_display = AlertDisplay(W, H)
    world_map = world.get_map()
    driving_monitor = DrivingMonitor(world_map, config)

    # Create episode resetter
    resetter = EpisodeResetter(world, client, config.scenario)

    # Optionally start top-down spectator
    spectator = None
    if enable_spectator:
        spectator = TopDownSpectator(world, height=spectator_height)
        spectator.start()

    print("\n" + "=" * 50)
    print("MANUAL DRIVE WITH INSTRUCTION SYSTEM")
    print("=" * 50)
    print("Driving Controls:")
    print("  W/Up     - Accelerate")
    print("  S/Down   - Brake")
    print("  A/D      - Steer")
    print("  R        - Toggle reverse")
    print("  SPACE    - Hand brake")
    print("  F5/Bksp  - Reset (teleport to spawn)")
    print("  ESC      - Quit")
    print("")
    print("Navigation Instructions (test keys):")
    print("  1        - Turn Left")
    print("  2        - Turn Right")
    print("  3        - Stop at the light")
    print("")
    print("Automatic Alerts:")
    print("  - Lane drifting warnings")
    print("  - Speed recommendations")
    print("  - Following distance advice")
    print("")
    print("Auto-Respawn Triggers:")
    print(f"  - Track end:    x >= {resetter.goal_x}")
    print(f"  - Reversed:     x < {resetter.min_x}")
    print(f"  - Off-road:     offset > {resetter.max_lane_offset}m")
    if enable_spectator:
        print(f"\nTop-down spectator active (height={spectator_height}m)")
    print("=" * 50 + "\n")

    try:
        while running:
            clock.tick(60)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                        print(f"Reverse: {'ON' if reverse_mode else 'OFF'}")
                    # Reset keybinds
                    elif event.key in (pygame.K_F5, pygame.K_BACKSPACE):
                        resetter.reset(ego, npc)
                        driving_monitor.lane_offset_history.clear()
                        control = carla.VehicleControl()
                        reverse_mode = False
                    # Navigation instruction test keys
                    elif event.key == pygame.K_1:
                        alert_display.trigger_navigation(AlertType.TURN_LEFT, duration=5.0)
                    elif event.key == pygame.K_2:
                        alert_display.trigger_navigation(AlertType.TURN_RIGHT, duration=5.0)
                    elif event.key == pygame.K_3:
                        alert_display.trigger_navigation(AlertType.STOP_AT_LIGHT, duration=5.0)

                elif event.type == pygame.JOYBUTTONDOWN:
                    # Button 4 or 5 for reverse on most controllers
                    if event.button in (4, 5):
                        reverse_mode = not reverse_mode
                        print(f"Reverse: {'ON' if reverse_mode else 'OFF'}")

            # Get control input
            keys = pygame.key.get_pressed()
            if wheel:
                control = get_wheel_control(wheel, control, reverse_mode)
            control = get_keyboard_control(keys, control, reverse_mode)

            # Apply control
            ego.apply_control(control)

            # Update driving monitor (automatic instructions)
            metrics = driving_monitor.update(ego, npc, alert_display)

            # Auto-respawn check
            if resetter.should_auto_respawn(ego):
                resetter.reset(ego, npc)
                driving_monitor.lane_offset_history.clear()
                control = carla.VehicleControl()
                reverse_mode = False

            # Draw camera view
            if latest_image is not None:
                surf = pygame.surfarray.make_surface(latest_image.swapaxes(0, 1))
                screen.blit(surf, (0, 0))

            # Render the dashboard with all driving info and instructions
            alert_display.render(screen, metrics, reverse_mode)

            pygame.display.flip()

    finally:
        print("\nCleaning up...")
        if spectator is not None:
            spectator.stop()
        camera.stop()
        camera.destroy()
        if npc:
            npc.destroy()
        ego.destroy()
        pygame.quit()
        print("Done!")


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Manual driving in CARLA")
    parser.add_argument("--no-npc", action="store_true", help="Don't spawn NPC vehicle")
    parser.add_argument("--chase-cam", action="store_true", help="Use third-person chase camera")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--spectator", action="store_true", help="Enable top-down spectator view")
    parser.add_argument("--spectator-height", type=float, default=40.0, help="Spectator camera height in meters")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    # Connect to CARLA
    print("Connecting to CARLA...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    # Load Town06
    world = client.get_world()
    current_map = world.get_map().name.split('/')[-1]
    if current_map != "Town06":
        print(f"Loading Town06 (current: {current_map})...")
        world = client.load_world("Town06")

    run_manual_drive(
        client=client,
        world=world,
        config=config,
        no_npc=args.no_npc,
        chase_cam=args.chase_cam,
        width=args.width,
        height=args.height,
        enable_spectator=args.spectator,
        spectator_height=args.spectator_height,
    )


if __name__ == "__main__":
    main()
