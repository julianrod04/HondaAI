"""
Self-contained alert / notification overlay system for CARLA manual driving.

Provides:
  - AlertType enum          – all recognized alert categories
  - ALERT_CONFIG dict       – per-alert text, colors, and routing
  - AlertDisplayConfig      – global appearance tuning (inline dataclass)
  - Dashboard               – renders FOV panel + dashboard alert panel + HUD bar
  - DrivingMonitor          – detects lane drift, speeding, proximity, etc.

No dependency on the rl/ package — everything lives here.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import pygame

try:
    import carla
except ImportError:
    carla = None  # allow import for testing without carla


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlertDisplayConfig:
    """Global alert overlay appearance and layout configuration.

    Two independent panels:
      FOV panel        – large center overlay (navigation / critical alerts)
      Dashboard panel  – compact banner above the HUD bar (speed / lane alerts)
    """

    # ---- FOV panel (field-of-vision, Panel 1) --------------------------------
    width: int = 450                                    # panel width in pixels
    height: int = 80                                    # panel height in pixels
    position: str = "top-center"                        # "center", "top-center", "top-left",
                                                        # "top-right", "bottom-left", "bottom-right"
    custom_x: Optional[int] = None                      # pixel override (None = use position string)
    custom_y: Optional[int] = None
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color_override: Optional[Tuple[int, int, int]] = None  # None = use per-alert color
    alpha: int = 100                                    # 0-255 background transparency

    # ---- Dashboard alert panel (Panel 2) -------------------------------------
    dashboard_position: str = "bottom-right"            # "bottom-left", "bottom-center", "bottom-right"
    dashboard_alpha: int = 200                          # 0-255 background transparency
    dashboard_text_color: Optional[Tuple[int, int, int]] = None       # None = use per-alert color
    dashboard_bg_color_override: Optional[Tuple[int, int, int]] = None
    dashboard_padding_x: int = 24                       # horizontal padding around text (px)
    dashboard_padding_y: int = 12                       # vertical padding around text (px)

    # ---- General -------------------------------------------------------------
    show_diagnostics_bar: bool = True                   # toggle bottom HUD bar (Tab key)


# =============================================================================
# ALERT TYPES & CONFIG
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
    # Extended alerts
    SPEEDING = 11                   # speed exceeds speed limit
    SPEEDUP_FROM_STANDSTILL = 12    # accelerating from near-zero
    FRONT_PROXIMITY_WARNING = 13    # vehicle ahead 5-10m
    FRONT_PROXIMITY_CRITICAL = 14   # vehicle ahead < 5m
    REAR_PROXIMITY_WARNING = 15     # vehicle behind 5-10m
    REAR_PROXIMITY_CRITICAL = 16    # vehicle behind < 5m


# Per-alert rendering config.
#   field_of_vision = True  → routed to FOV panel  (Panel 1)
#   field_of_vision = False → routed to Dashboard panel (Panel 2)
ALERT_CONFIG = {
    AlertType.NONE: {
        "text": "", "color": (255, 255, 255),
        "bg_color": None, "icon": None, "field_of_vision": False,
    },
    AlertType.DRIFTING_LEFT: {
        "text": "Drifting into left lane", "color": (255, 255, 255),
        "bg_color": (200, 150, 0), "icon": "\u2190", "field_of_vision": False,
    },
    AlertType.DRIFTING_RIGHT: {
        "text": "Drifting into right lane", "color": (255, 255, 255),
        "bg_color": (200, 150, 0), "icon": "\u2192", "field_of_vision": False,
    },
    AlertType.STOP: {
        "text": "Come to a complete stop", "color": (255, 255, 255),
        "bg_color": (200, 50, 50), "icon": "\u25a0", "field_of_vision": False,
    },
    AlertType.SPEED_UP: {
        "text": "Increase your speed gradually", "color": (255, 255, 255),
        "bg_color": (50, 150, 50), "icon": "\u2191", "field_of_vision": False,
    },
    AlertType.SLOW_DOWN: {
        "text": "Reduce your speed", "color": (255, 255, 255),
        "bg_color": (200, 150, 0), "icon": "\u2193", "field_of_vision": False,
    },
    AlertType.TURN_RIGHT: {
        "text": "Turn Right", "color": (255, 255, 255),
        "bg_color": (0, 120, 200), "icon": "\u2192", "field_of_vision": True,
    },
    AlertType.TURN_LEFT: {
        "text": "Turn Left", "color": (255, 255, 255),
        "bg_color": (0, 120, 200), "icon": "\u2190", "field_of_vision": True,
    },
    AlertType.STOP_AT_LIGHT: {
        "text": "Stop at the light", "color": (255, 255, 255),
        "bg_color": (200, 50, 50), "icon": "\u25cf", "field_of_vision": True,
    },
    AlertType.EMERGENCY_STOP: {
        "text": "Stop immediately - Emergency", "color": (255, 255, 255),
        "bg_color": (255, 0, 0), "icon": "\u26a0", "field_of_vision": True,
    },
    AlertType.MAINTAIN: {
        "text": "Maintain your current speed", "color": (200, 200, 200),
        "bg_color": (80, 80, 80), "icon": "\u25cf", "field_of_vision": False,
    },
    AlertType.SPEEDING: {
        "text": "Speed limit exceeded", "color": (255, 255, 255),
        "bg_color": (200, 80, 0), "icon": "\u26a0", "field_of_vision": False,
    },
    AlertType.SPEEDUP_FROM_STANDSTILL: {
        "text": "Accelerating from standstill", "color": (255, 255, 255),
        "bg_color": (50, 150, 50), "icon": "\u21d1", "field_of_vision": False,
    },
    AlertType.FRONT_PROXIMITY_WARNING: {
        "text": "Vehicle ahead - slow down", "color": (255, 255, 255),
        "bg_color": (200, 150, 0), "icon": "\u26a0", "field_of_vision": False,
    },
    AlertType.FRONT_PROXIMITY_CRITICAL: {
        "text": "Too close! Brake now", "color": (255, 255, 255),
        "bg_color": (200, 30, 30), "icon": "\u26d4", "field_of_vision": True,
    },
    AlertType.REAR_PROXIMITY_WARNING: {
        "text": "Vehicle approaching from behind", "color": (255, 255, 255),
        "bg_color": (200, 150, 0), "icon": "\u2193", "field_of_vision": False,
    },
    AlertType.REAR_PROXIMITY_CRITICAL: {
        "text": "Vehicle very close behind!", "color": (255, 255, 255),
        "bg_color": (200, 30, 30), "icon": "\u26a0", "field_of_vision": False,
    },
}


# =============================================================================
# DASHBOARD (renders both FOV and dashboard alert panels + HUD bar)
# =============================================================================

class Dashboard:
    """
    Renders driving overlays on a pygame surface:
      - FOV panel: navigation / critical alerts (center-top area)
      - Dashboard alert panel: speed / lane / proximity alerts (above HUD bar)
      - HUD bar: speed gauge, lane position indicator, distance bar
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        dashboard_height: int = 120,
        alert_config: AlertDisplayConfig = None,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.alert_config = alert_config if alert_config is not None else AlertDisplayConfig()

        # Dashboard dimensions
        self.dashboard_height = dashboard_height
        self.dashboard_y = screen_height - self.dashboard_height

        # Dashboard alert state
        self.current_alert: AlertType = AlertType.NONE
        self.alert_start_time: float = 0.0
        self.alert_duration: float = 3.0

        # Navigation (FOV) alert state
        self.current_nav: AlertType = AlertType.NONE
        self.nav_start_time: float = 0.0
        self.nav_duration: float = 5.0

        # Fonts
        pygame.font.init()
        self.font_speed = pygame.font.SysFont("arial", 42, bold=True)
        self.font_speed_unit = pygame.font.SysFont("arial", 18)
        self.font_instruction = pygame.font.SysFont("arial", 28, bold=True)
        self.font_nav = pygame.font.SysFont("arial", 36, bold=True)
        self.font_label = pygame.font.SysFont("arial", 14)
        self.font_value = pygame.font.SysFont("arial", 18, bold=True)

        # Animation
        self.fade_in_duration = 0.2
        self.fade_out_duration = 0.5

        # History & cooldown
        self.alert_history: list = []
        self.last_alert_time: dict = {}
        self.alert_cooldown: float = 2.0

        # Dashboard colors
        self.bg_color = (20, 20, 25)
        self.text_color = (220, 220, 220)

    # ----- public API --------------------------------------------------------

    def trigger_alert(self, alert_type: AlertType, duration: float = 3.0, force: bool = False) -> bool:
        if alert_type == AlertType.NONE:
            self.clear_alert()
            return True

        if not force:
            last_time = self.last_alert_time.get(alert_type, 0)
            if time.time() - last_time < self.alert_cooldown:
                return False

        config = ALERT_CONFIG[alert_type]

        if config.get("field_of_vision", False):
            self.current_nav = alert_type
            self.nav_start_time = time.time()
            self.nav_duration = duration
        else:
            self.current_alert = alert_type
            self.alert_start_time = time.time()
            self.alert_duration = duration

        self.last_alert_time[alert_type] = time.time()
        self.alert_history.append({"type": alert_type.name, "time": time.time()})
        print(f"[ALERT] {config['text']}")
        return True

    def trigger_navigation(self, alert_type: AlertType, duration: float = 5.0) -> bool:
        return self.trigger_alert(alert_type, duration, force=True)

    def clear_alert(self) -> None:
        self.current_alert = AlertType.NONE

    def clear_navigation(self) -> None:
        self.current_nav = AlertType.NONE

    def get_current_alert(self) -> AlertType:
        if self.current_alert == AlertType.NONE:
            return AlertType.NONE
        if time.time() - self.alert_start_time > self.alert_duration:
            self.current_alert = AlertType.NONE
        return self.current_alert

    def get_current_navigation(self) -> AlertType:
        if self.current_nav == AlertType.NONE:
            return AlertType.NONE
        if time.time() - self.nav_start_time > self.nav_duration:
            self.current_nav = AlertType.NONE
        return self.current_nav

    # ----- rendering ---------------------------------------------------------

    def render(self, screen: pygame.Surface, metrics: dict, reverse_mode: bool = False) -> None:
        self._render_field_of_vision(screen)

        if self.alert_config.show_diagnostics_bar:
            dashboard_surface = pygame.Surface((self.screen_width, self.dashboard_height), pygame.SRCALPHA)
            dashboard_surface.fill((*self.bg_color, 230))
            screen.blit(dashboard_surface, (0, self.dashboard_y))

            self._render_speed_section(screen, metrics, reverse_mode)
            self._render_info_section(screen, metrics)

    # ----- position resolvers ------------------------------------------------

    def _resolve_panel_xy(self, w: int, h: int) -> tuple:
        if self.alert_config.custom_x is not None:
            return self.alert_config.custom_x, self.alert_config.custom_y

        sw = self.screen_width
        cam_h = self.screen_height - self.dashboard_height

        positions = {
            "center":       ((sw - w) // 2, (cam_h - h) // 2),
            "top-center":   ((sw - w) // 2, 20),
            "top-left":     (20, 20),
            "top-right":    (sw - w - 20, 20),
            "bottom-left":  (20, cam_h - h - 20),
            "bottom-right": (sw - w - 20, cam_h - h - 20),
        }
        return positions.get(self.alert_config.position, positions["center"])

    def _resolve_dashboard_alert_xy(self, w: int, h: int) -> tuple:
        sw = self.screen_width
        by = self.dashboard_y - h - 10
        positions = {
            "bottom-left":   (20, by),
            "bottom-center": ((sw - w) // 2, by),
            "bottom-right":  (sw - w - 20, by),
        }
        return positions.get(self.alert_config.dashboard_position, positions["bottom-right"])

    # ----- FOV panel ---------------------------------------------------------

    def _render_field_of_vision(self, screen: pygame.Surface) -> None:
        nav = self.get_current_navigation()
        if nav == AlertType.NONE:
            return

        alert_cfg = ALERT_CONFIG[nav]
        elapsed = time.time() - self.nav_start_time

        # Fade
        if elapsed < self.fade_in_duration:
            alpha = elapsed / self.fade_in_duration
        elif elapsed > self.nav_duration - self.fade_out_duration:
            alpha = (self.nav_duration - elapsed) / self.fade_out_duration
        else:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))

        panel_width = self.alert_config.width
        panel_height = self.alert_config.height
        panel_x, panel_y = self._resolve_panel_xy(panel_width, panel_height)

        bg_color = self.alert_config.bg_color_override or alert_cfg["bg_color"]
        text_color = self.alert_config.text_color
        base_alpha = self.alert_config.alpha

        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (*bg_color, int(base_alpha * alpha)),
                         (0, 0, panel_width, panel_height), border_radius=15)
        pygame.draw.rect(panel_surface, (255, 255, 255, int(255 * alpha)),
                         (0, 0, panel_width, panel_height), 4, border_radius=15)
        screen.blit(panel_surface, (panel_x, panel_y))

        display_text = alert_cfg["text"]
        if alert_cfg.get("icon"):
            display_text = f"{alert_cfg['icon']}  {alert_cfg['text']}"

        nav_text = self.font_nav.render(display_text, True, text_color)
        nav_rect = nav_text.get_rect(center=(panel_x + panel_width // 2, panel_y + panel_height // 2))

        text_surface = pygame.Surface(nav_text.get_size(), pygame.SRCALPHA)
        text_surface.blit(nav_text, (0, 0))
        text_surface.set_alpha(int(255 * alpha))
        screen.blit(text_surface, nav_rect)

    # ----- Speed section (HUD bar left) --------------------------------------

    def _render_speed_section(self, screen: pygame.Surface, metrics: dict, reverse_mode: bool) -> None:
        speed_kmh = metrics.get("speed_kmh", 0)
        section_x = 40
        section_center_y = self.dashboard_y + self.dashboard_height // 2

        if speed_kmh > 80:
            speed_color = (255, 80, 80)
        elif speed_kmh > 50:
            speed_color = (255, 200, 0)
        else:
            speed_color = (255, 255, 255)

        speed_text = self.font_speed.render(f"{speed_kmh:.0f}", True, speed_color)
        speed_rect = speed_text.get_rect(midleft=(section_x, section_center_y - 5))
        screen.blit(speed_text, speed_rect)

        unit_text = self.font_speed_unit.render("km/h", True, (150, 150, 150))
        unit_rect = unit_text.get_rect(midleft=(speed_rect.right + 5, section_center_y + 10))
        screen.blit(unit_text, unit_rect)

        if reverse_mode:
            rev_text = self.font_label.render("R", True, (255, 100, 100))
            rev_rect = rev_text.get_rect(midleft=(section_x, section_center_y + 35))
            pygame.draw.circle(screen, (80, 30, 30), rev_rect.center, 12)
            screen.blit(rev_text, rev_rect)

    # ----- Info section (HUD bar right) + dashboard alert panel --------------

    def _render_info_section(self, screen: pygame.Surface, metrics: dict) -> None:
        section_x = self.screen_width - 200
        section_y = self.dashboard_y + 15

        # Lane position
        lane_offset = metrics.get("lane_offset", 0)
        lane_status = metrics.get("lane_status", "OK")

        lane_label = self.font_label.render("LANE POSITION", True, (120, 120, 120))
        screen.blit(lane_label, (section_x, section_y))

        bar_width = 140
        bar_height = 12
        bar_x = section_x
        bar_y = section_y + 20

        pygame.draw.rect(screen, (50, 50, 55), (bar_x, bar_y, bar_width, bar_height), border_radius=6)
        pygame.draw.line(screen, (100, 100, 100),
                         (bar_x + bar_width // 2, bar_y - 2),
                         (bar_x + bar_width // 2, bar_y + bar_height + 2), 2)

        if lane_status == "CRITICAL":
            indicator_color = (255, 80, 80)
        elif lane_status == "WARNING":
            indicator_color = (255, 200, 0)
        else:
            indicator_color = (80, 200, 80)

        indicator_offset = int(lane_offset * 35)
        indicator_x = bar_x + bar_width // 2 + indicator_offset
        indicator_x = max(bar_x + 8, min(bar_x + bar_width - 8, indicator_x))

        pygame.draw.circle(screen, indicator_color, (indicator_x, bar_y + bar_height // 2), 7)
        pygame.draw.circle(screen, (255, 255, 255), (indicator_x, bar_y + bar_height // 2), 7, 2)

        offset_text = self.font_value.render(f"{lane_offset:+.1f}m", True, indicator_color)
        screen.blit(offset_text, (bar_x + bar_width + 10, bar_y - 2))

        # Distance to vehicle ahead
        if "distance_to_npc" in metrics:
            dist = metrics["distance_to_npc"]
            follow_status = metrics.get("follow_status", "N/A")

            dist_y = section_y + 45
            dist_label = self.font_label.render("VEHICLE AHEAD", True, (120, 120, 120))
            screen.blit(dist_label, (section_x, dist_y))

            if follow_status == "CRITICAL":
                dist_color = (255, 80, 80)
            elif follow_status == "WARNING":
                dist_color = (255, 200, 0)
            else:
                dist_color = (150, 150, 150)

            dist_text = self.font_value.render(f"{dist:.1f} m", True, dist_color)
            screen.blit(dist_text, (section_x, dist_y + 18))

            max_dist = 30.0
            dist_bar_width = int((min(dist, max_dist) / max_dist) * 100)
            pygame.draw.rect(screen, (50, 50, 55), (section_x + 70, dist_y + 20, 100, 8), border_radius=4)
            if dist_bar_width > 0:
                pygame.draw.rect(screen, dist_color, (section_x + 70, dist_y + 20, dist_bar_width, 8), border_radius=4)

        # Dashboard alert panel (auto-sized floating box)
        alert = self.get_current_alert()
        if alert != AlertType.NONE:
            alert_cfg = ALERT_CONFIG[alert]
            elapsed = time.time() - self.alert_start_time

            if elapsed < self.fade_in_duration:
                fade = elapsed / self.fade_in_duration
            elif elapsed > self.alert_duration - self.fade_out_duration:
                fade = (self.alert_duration - elapsed) / self.fade_out_duration
            else:
                fade = 1.0
            fade = max(0.0, min(1.0, fade))

            bg_color = self.alert_config.dashboard_bg_color_override or alert_cfg["bg_color"]
            text_color = self.alert_config.dashboard_text_color or alert_cfg["color"]

            display_text = alert_cfg["text"]
            if alert_cfg.get("icon"):
                display_text = f"{alert_cfg['icon']}  {alert_cfg['text']}"

            text_surf = self.font_instruction.render(display_text, True, text_color)
            tw, th = text_surf.get_size()
            px = self.alert_config.dashboard_padding_x
            py = self.alert_config.dashboard_padding_y
            panel_width = tw + px * 2
            panel_height = th + py * 2

            panel_x, panel_y = self._resolve_dashboard_alert_xy(panel_width, panel_height)
            base_alpha = self.alert_config.dashboard_alpha

            panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            panel_surface.fill((*bg_color, int(base_alpha * fade)))
            pygame.draw.rect(panel_surface, (255, 255, 255, int(150 * fade)),
                             (0, 0, panel_width, panel_height), 2, border_radius=8)
            screen.blit(panel_surface, (panel_x, panel_y))

            text_surf.set_alpha(int(255 * fade))
            text_rect = text_surf.get_rect(center=(panel_x + panel_width // 2, panel_y + panel_height // 2))
            screen.blit(text_surf, text_rect)

            # Pulsing border for urgent alerts
            if alert in (AlertType.STOP, AlertType.EMERGENCY_STOP):
                pulse = abs(math.sin(elapsed * 5)) * 0.5 + 0.5
                pulse_surface = pygame.Surface((panel_width + 6, panel_height + 6), pygame.SRCALPHA)
                pygame.draw.rect(pulse_surface, (255, 80, 80, int(150 * pulse * fade)),
                                 (0, 0, panel_width + 6, panel_height + 6), 3, border_radius=10)
                screen.blit(pulse_surface, (panel_x - 3, panel_y - 3))


# =============================================================================
# DRIVING MONITOR
# =============================================================================

class DrivingMonitor:
    """
    Monitors driving behavior and automatically triggers alerts.

    Detects: lane drift, speeding, standstill acceleration,
    front/rear proximity to other vehicles.
    """

    def __init__(self, world_map, world=None):
        self.world_map = world_map
        self.world = world

        # Thresholds
        self.lane_drift_warning = 0.8
        self.lane_drift_critical = 1.2
        self.speed_limit_kmh = 50.0
        self.min_speed_kmh = 10.0
        self.follow_distance_warning = 10.0
        self.follow_distance_critical = 5.0
        self.rear_distance_warning = 10.0
        self.rear_distance_critical = 5.0

        # Standstill detection
        self._prev_speed_kmh: float = 0.0
        self._standstill_cooldown: float = 0.0

        # Lane offset history
        self.lane_offset_history: list = []
        self.history_size = 30

    def get_lane_offset(self, vehicle) -> float:
        location = vehicle.get_location()
        waypoint = self.world_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 0.0

        lane_yaw = math.radians(waypoint.transform.rotation.yaw)
        normal_x = -math.sin(lane_yaw)
        normal_y = math.cos(lane_yaw)

        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y

        return dx * normal_x + dy * normal_y

    def get_distance_to_vehicle(self, ego, other) -> float:
        ego_loc = ego.get_location()
        other_loc = other.get_location()
        dx = ego_loc.x - other_loc.x
        dy = ego_loc.y - other_loc.y
        return math.sqrt(dx * dx + dy * dy)

    def is_vehicle_ahead(self, ego, other) -> bool:
        ego_tf = ego.get_transform()
        other_loc = other.get_location()
        dx = other_loc.x - ego_tf.location.x
        dy = other_loc.y - ego_tf.location.y
        yaw = math.radians(ego_tf.rotation.yaw)
        return (dx * math.cos(yaw) + dy * math.sin(yaw)) > 0

    def update(self, ego, npc, dashboard: Dashboard) -> dict:
        """Update monitoring and trigger alerts. Returns driving metrics dict."""
        metrics = {}

        velocity = ego.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_mps * 3.6
        metrics["speed_kmh"] = speed_kmh

        # Lane offset
        lane_offset = self.get_lane_offset(ego)
        metrics["lane_offset"] = lane_offset

        self.lane_offset_history.append(lane_offset)
        if len(self.lane_offset_history) > self.history_size:
            self.lane_offset_history.pop(0)

        avg_offset = sum(self.lane_offset_history) / len(self.lane_offset_history)
        metrics["avg_lane_offset"] = avg_offset

        # Lane drift detection
        abs_offset = abs(avg_offset)
        if abs_offset > self.lane_drift_critical:
            if avg_offset > 0:
                dashboard.trigger_alert(AlertType.DRIFTING_RIGHT, duration=2.0)
            else:
                dashboard.trigger_alert(AlertType.DRIFTING_LEFT, duration=2.0)
            metrics["lane_status"] = "CRITICAL"
        elif abs_offset > self.lane_drift_warning:
            metrics["lane_status"] = "WARNING"
        else:
            metrics["lane_status"] = "OK"

        # Speed monitoring
        if speed_kmh > self.speed_limit_kmh:
            dashboard.trigger_alert(AlertType.SPEEDING, duration=2.0)
            metrics["speed_status"] = "TOO_FAST"
        elif speed_kmh < self.min_speed_kmh and speed_kmh > 1.0:
            dashboard.trigger_alert(AlertType.SPEED_UP, duration=2.0)
            metrics["speed_status"] = "TOO_SLOW"
        else:
            metrics["speed_status"] = "OK"

        # Speedup from standstill
        now = time.time()
        if (
            self._prev_speed_kmh < 2.0
            and speed_kmh > 8.0
            and now - self._standstill_cooldown > 5.0
        ):
            dashboard.trigger_alert(AlertType.SPEEDUP_FROM_STANDSTILL, duration=3.0)
            self._standstill_cooldown = now
        self._prev_speed_kmh = speed_kmh

        # Following distance (specific NPC)
        if npc is not None:
            distance = self.get_distance_to_vehicle(ego, npc)
            metrics["distance_to_npc"] = distance

            if self.is_vehicle_ahead(ego, npc):
                if distance < self.follow_distance_critical:
                    dashboard.trigger_alert(AlertType.FRONT_PROXIMITY_CRITICAL, duration=2.0)
                    metrics["follow_status"] = "CRITICAL"
                elif distance < self.follow_distance_warning:
                    dashboard.trigger_alert(AlertType.FRONT_PROXIMITY_WARNING, duration=2.0)
                    metrics["follow_status"] = "WARNING"
                else:
                    metrics["follow_status"] = "OK"
            else:
                metrics["follow_status"] = "N/A"

        # Proximity detection (all vehicles via world)
        if self.world is not None:
            ego_loc = ego.get_location()
            ego_fwd = ego.get_transform().get_forward_vector()

            for v in self.world.get_actors().filter("vehicle.*"):
                if v.id == ego.id:
                    continue
                rel = v.get_location() - ego_loc
                dot = rel.x * ego_fwd.x + rel.y * ego_fwd.y
                dist = math.sqrt(rel.x ** 2 + rel.y ** 2 + rel.z ** 2)

                if dot > 0:
                    if dist < self.follow_distance_critical:
                        dashboard.trigger_alert(AlertType.FRONT_PROXIMITY_CRITICAL, duration=2.0)
                        metrics["follow_status"] = "CRITICAL"
                    elif dist < self.follow_distance_warning:
                        dashboard.trigger_alert(AlertType.FRONT_PROXIMITY_WARNING, duration=2.0)
                        if metrics.get("follow_status") not in ("CRITICAL",):
                            metrics["follow_status"] = "WARNING"
                else:
                    if dist < self.rear_distance_critical:
                        dashboard.trigger_alert(AlertType.REAR_PROXIMITY_CRITICAL, duration=2.0)
                    elif dist < self.rear_distance_warning:
                        dashboard.trigger_alert(AlertType.REAR_PROXIMITY_WARNING, duration=2.0)

        return metrics
