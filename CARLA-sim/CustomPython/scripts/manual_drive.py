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
    ESC             - Quit

Automatic Alert System:
    The system automatically monitors driving behavior and displays alerts:
    - Lane drifting (triggers "LANE CHANGE LEFT/RIGHT" to correct)
    - Speeding (triggers "SLOW DOWN" when over limit)
    - Too slow (triggers "SPEED UP" when moving too slowly)
    - Following distance (triggers "SLOW DOWN" or "STOP" when too close to NPC)

Also supports steering wheel/gamepad if connected.

Usage:
    python scripts/manual_drive.py
    python scripts/manual_drive.py --no-npc          # Without NPC vehicle
    python scripts/manual_drive.py --chase-cam       # Third-person chase camera
"""

import argparse
import math
import sys
import time
from pathlib import Path
from enum import Enum
from typing import Optional

import numpy as np
import pygame

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from rl.config import DEFAULT_CONFIG


# =============================================================================
# ALERT SYSTEM
# =============================================================================

class AlertType(Enum):
    """Alert types for the autonomous vehicle."""
    NONE = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    STOP = 3
    SPEED_UP = 4
    SLOW_DOWN = 5
    OVERTAKE = 6
    EMERGENCY_STOP = 7
    MAINTAIN = 8


# Alert display configuration
ALERT_CONFIG = {
    AlertType.NONE: {
        "text": "",
        "color": (255, 255, 255),
        "bg_color": None,
        "icon": None,
    },
    AlertType.LANE_CHANGE_LEFT: {
        "text": "← LANE CHANGE LEFT",
        "color": (255, 255, 255),
        "bg_color": (0, 120, 200),
        "icon": "←",
    },
    AlertType.LANE_CHANGE_RIGHT: {
        "text": "LANE CHANGE RIGHT →",
        "color": (255, 255, 255),
        "bg_color": (0, 120, 200),
        "icon": "→",
    },
    AlertType.STOP: {
        "text": "■ STOP",
        "color": (255, 255, 255),
        "bg_color": (200, 50, 50),
        "icon": "■",
    },
    AlertType.SPEED_UP: {
        "text": "↑ SPEED UP",
        "color": (255, 255, 255),
        "bg_color": (50, 150, 50),
        "icon": "↑",
    },
    AlertType.SLOW_DOWN: {
        "text": "↓ SLOW DOWN",
        "color": (255, 255, 255),
        "bg_color": (200, 150, 0),
        "icon": "↓",
    },
    AlertType.OVERTAKE: {
        "text": "⇢ OVERTAKE",
        "color": (255, 255, 255),
        "bg_color": (150, 50, 150),
        "icon": "⇢",
    },
    AlertType.EMERGENCY_STOP: {
        "text": "⚠ EMERGENCY STOP",
        "color": (255, 255, 255),
        "bg_color": (255, 0, 0),
        "icon": "⚠",
    },
    AlertType.MAINTAIN: {
        "text": "● MAINTAIN",
        "color": (200, 200, 200),
        "bg_color": (80, 80, 80),
        "icon": "●",
    },
}


class AlertDisplay:
    """
    Manages alert display with animations and timing.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.current_alert: AlertType = AlertType.NONE
        self.alert_start_time: float = 0.0
        self.alert_duration: float = 3.0  # seconds
        
        # Fonts
        pygame.font.init()
        self.font_large = pygame.font.SysFont("arial", 48, bold=True)
        self.font_medium = pygame.font.SysFont("arial", 32)
        self.font_small = pygame.font.SysFont("arial", 20)
        
        # Animation state
        self.fade_in_duration = 0.2
        self.fade_out_duration = 0.5
        
        # Alert history for logging
        self.alert_history: list = []
        
        # Cooldown to prevent alert spam
        self.last_alert_time: dict = {}
        self.alert_cooldown: float = 2.0  # seconds between same alert type
    
    def trigger_alert(self, alert_type: AlertType, duration: float = 3.0, force: bool = False) -> bool:
        """
        Trigger a new alert.
        
        Returns True if alert was triggered, False if on cooldown.
        """
        if alert_type == AlertType.NONE:
            self.clear_alert()
            return True
        
        # Check cooldown (unless forced)
        if not force:
            last_time = self.last_alert_time.get(alert_type, 0)
            if time.time() - last_time < self.alert_cooldown:
                return False
        
        self.current_alert = alert_type
        self.alert_start_time = time.time()
        self.alert_duration = duration
        self.last_alert_time[alert_type] = time.time()
        
        # Log the alert
        self.alert_history.append({
            "type": alert_type.name,
            "time": time.time(),
        })
        
        config = ALERT_CONFIG[alert_type]
        print(f"[ALERT] {config['text']}")
        return True
    
    def clear_alert(self) -> None:
        """Clear the current alert."""
        self.current_alert = AlertType.NONE
    
    def get_current_alert(self) -> AlertType:
        """Get the current active alert."""
        if self.current_alert == AlertType.NONE:
            return AlertType.NONE
        
        # Check if alert has expired
        elapsed = time.time() - self.alert_start_time
        if elapsed > self.alert_duration:
            self.current_alert = AlertType.NONE
        
        return self.current_alert
    
    def render(self, screen: pygame.Surface) -> None:
        """Render the alert on screen."""
        alert = self.get_current_alert()
        if alert == AlertType.NONE:
            return
        
        config = ALERT_CONFIG[alert]
        elapsed = time.time() - self.alert_start_time
        
        # Calculate alpha for fade effect
        if elapsed < self.fade_in_duration:
            alpha = int(255 * (elapsed / self.fade_in_duration))
        elif elapsed > self.alert_duration - self.fade_out_duration:
            remaining = self.alert_duration - elapsed
            alpha = int(255 * (remaining / self.fade_out_duration))
        else:
            alpha = 255
        
        alpha = max(0, min(255, alpha))
        
        # Create alert box
        text = config["text"]
        bg_color = config["bg_color"]
        text_color = config["color"]
        
        # Render text
        text_surface = self.font_large.render(text, True, text_color)
        text_rect = text_surface.get_rect()
        
        # Box dimensions
        padding = 30
        box_width = text_rect.width + padding * 2
        box_height = text_rect.height + padding * 2
        
        # Center position (top third of screen)
        box_x = (self.screen_width - box_width) // 2
        box_y = self.screen_height // 6
        
        # Draw semi-transparent background (more transparent)
        if bg_color:
            box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            box_surface.fill((*bg_color, int(alpha * 0.45)))  # 45% opacity for background
            
            # Add subtle border
            pygame.draw.rect(box_surface, (255, 255, 255, int(alpha * 0.6)), 
                           (0, 0, box_width, box_height), 2, border_radius=10)
            
            screen.blit(box_surface, (box_x, box_y))
        
        # Draw text with alpha (slightly transparent)
        text_surface.set_alpha(int(alpha * 0.85))
        text_x = box_x + padding
        text_y = box_y + padding
        screen.blit(text_surface, (text_x, text_y))
        
        # Draw subtle pulsing effect for urgent alerts
        if alert in (AlertType.STOP, AlertType.EMERGENCY_STOP):
            pulse = abs(math.sin(elapsed * 4)) * 0.2 + 0.3  # More subtle pulse
            border_alpha = int(alpha * pulse)
            border_surface = pygame.Surface((box_width + 10, box_height + 10), pygame.SRCALPHA)
            pygame.draw.rect(border_surface, (255, 100, 100, border_alpha),
                           (0, 0, box_width + 10, box_height + 10), 3, border_radius=12)
            screen.blit(border_surface, (box_x - 5, box_y - 5))


class DrivingMonitor:
    """
    Monitors driving behavior and automatically triggers alerts.
    
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
        Update monitoring and trigger alerts as needed.
        
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
            # Critical lane drift - suggest correction
            if avg_offset > 0:
                # Drifting right, suggest go left
                alert_display.trigger_alert(AlertType.LANE_CHANGE_LEFT, duration=2.0)
            else:
                # Drifting left, suggest go right
                alert_display.trigger_alert(AlertType.LANE_CHANGE_RIGHT, duration=2.0)
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
                    alert_display.trigger_alert(AlertType.STOP, duration=2.0)
                    metrics["follow_status"] = "CRITICAL"
                elif distance < self.follow_distance_warning:
                    alert_display.trigger_alert(AlertType.SLOW_DOWN, duration=2.0)
                    metrics["follow_status"] = "WARNING"
                else:
                    metrics["follow_status"] = "OK"
            else:
                metrics["follow_status"] = "N/A"
        
        return metrics


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


def main():
    parser = argparse.ArgumentParser(description="Manual driving in CARLA")
    parser.add_argument("--no-npc", action="store_true", help="Don't spawn NPC vehicle")
    parser.add_argument("--chase-cam", action="store_true", help="Use third-person chase camera")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
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
    if not args.no_npc:
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
            print(f"Spawned NPC in adjacent lane")
    
    # Attach camera
    W, H = args.width, args.height
    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(W))
    cam_bp.set_attribute("image_size_y", str(H))
    cam_bp.set_attribute("fov", "100")
    
    if args.chase_cam:
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
    pygame.display.set_caption("CARLA Manual Drive - WASD to drive, R=reverse, ESC=quit")
    
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 28)
    small_font = pygame.font.SysFont("consolas", 20)
    
    clock = pygame.time.Clock()
    control = carla.VehicleControl()
    reverse_mode = False
    running = True
    
    # Create alert display and driving monitor
    alert_display = AlertDisplay(W, H)
    world_map = world.get_map()
    driving_monitor = DrivingMonitor(world_map, config)
    
    print("\n" + "=" * 50)
    print("MANUAL DRIVE WITH ALERT SYSTEM")
    print("=" * 50)
    print("Driving Controls:")
    print("  W/Up     - Accelerate")
    print("  S/Down   - Brake")
    print("  A/D      - Steer")
    print("  R        - Toggle reverse")
    print("  SPACE    - Hand brake")
    print("  ESC      - Quit")
    print("")
    print("Automatic Alerts:")
    print("  - Lane drift warning")
    print("  - Speed monitoring")
    print("  - Following distance")
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
            
            # Update driving monitor (automatic alerts)
            metrics = driving_monitor.update(ego, npc, alert_display)
            
            # Get speed and position from metrics
            speed_kmh = metrics.get("speed_kmh", 0)
            lane_offset = metrics.get("lane_offset", 0)
            
            # Get position
            ego_loc = ego.get_location()
            progress = (ego_loc.x - config.scenario.spawn_x) / (config.scenario.goal_x - config.scenario.spawn_x)
            
            # Draw camera view
            if latest_image is not None:
                surf = pygame.surfarray.make_surface(latest_image.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
            
            # Draw HUD
            # Speed
            speed_color = (255, 255, 255) if speed_kmh < 50 else (255, 200, 0) if speed_kmh < 80 else (255, 100, 100)
            speed_text = font.render(f"{speed_kmh:5.1f} km/h", True, speed_color)
            screen.blit(speed_text, (20, 20))
            
            # Reverse indicator
            if reverse_mode:
                rev_text = font.render("REVERSE", True, (255, 100, 100))
                screen.blit(rev_text, (20, 55))
            
            # Lane offset indicator (right side of screen)
            lane_status = metrics.get("lane_status", "OK")
            if lane_status == "CRITICAL":
                lane_color = (255, 100, 100)
            elif lane_status == "WARNING":
                lane_color = (255, 200, 0)
            else:
                lane_color = (100, 255, 100)
            
            lane_text = small_font.render(f"Lane: {lane_offset:+.2f}m", True, lane_color)
            screen.blit(lane_text, (W - 150, 20))
            
            # Visual lane offset bar
            bar_width = 120
            bar_height = 10
            bar_x = W - 150
            bar_y = 45
            
            # Background bar
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
            
            # Center line
            pygame.draw.line(screen, (255, 255, 255), 
                           (bar_x + bar_width // 2, bar_y - 2),
                           (bar_x + bar_width // 2, bar_y + bar_height + 2), 2)
            
            # Position indicator (clamp to bar width)
            indicator_pos = bar_x + bar_width // 2 + int(lane_offset * 30)
            indicator_pos = max(bar_x + 5, min(bar_x + bar_width - 5, indicator_pos))
            pygame.draw.circle(screen, lane_color, (indicator_pos, bar_y + bar_height // 2), 6)
            
            # Distance to NPC (if available)
            if "distance_to_npc" in metrics:
                dist = metrics["distance_to_npc"]
                follow_status = metrics.get("follow_status", "N/A")
                if follow_status == "CRITICAL":
                    dist_color = (255, 100, 100)
                elif follow_status == "WARNING":
                    dist_color = (255, 200, 0)
                else:
                    dist_color = (200, 200, 200)
                dist_text = small_font.render(f"NPC: {dist:.1f}m", True, dist_color)
                screen.blit(dist_text, (W - 150, 65))
            
            # Position and progress
            pos_text = small_font.render(f"X: {ego_loc.x:.1f}  Y: {ego_loc.y:.1f}", True, (200, 200, 200))
            screen.blit(pos_text, (20, H - 60))
            
            progress_text = small_font.render(f"Progress: {progress*100:.1f}%", True, (200, 200, 200))
            screen.blit(progress_text, (20, H - 35))
            
            # Render alert system
            alert_display.render(screen)
            
            pygame.display.flip()
    
    finally:
        print("\nCleaning up...")
        camera.stop()
        camera.destroy()
        if npc:
            npc.destroy()
        ego.destroy()
        pygame.quit()
        print("Done!")


if __name__ == "__main__":
    main()

