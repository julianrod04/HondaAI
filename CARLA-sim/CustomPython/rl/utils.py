"""
CARLA connection utilities and vehicle spawning helpers.

Provides clean abstractions for connecting to CARLA servers and
managing actors (vehicles, sensors).
"""

import math
import platform
import subprocess
import time
from typing import Optional, Tuple, List

import carla

from rl.config import CarlaConfig, ScenarioConfig, VehicleConfig, CameraConfig


def connect_to_carla(
    config: CarlaConfig,
    instance_id: int = 0,
    load_map: Optional[str] = None
) -> Tuple[carla.Client, carla.World]:
    """
    Connect to a CARLA server instance.
    
    Args:
        config: CARLA connection configuration
        instance_id: Which server instance to connect to (0, 1, 2, 3...)
        load_map: Map name to load, or None to use current map
        
    Returns:
        Tuple of (client, world)
    """
    port = config.get_port(instance_id)
    client = carla.Client(config.host, port)
    client.set_timeout(config.timeout)
    
    # First, get the current world to check connection
    world = client.get_world()
    
    # Only load a different map if requested AND it's different from current
    if load_map:
        current_map = world.get_map().name.split('/')[-1]
        if current_map != load_map:
            print(f"Loading map {load_map} (current: {current_map})...")
            # Increase timeout for map loading (can take 30+ seconds)
            client.set_timeout(60.0)
            world = client.load_world(load_map)
            client.set_timeout(config.timeout)
            # Wait for world to be ready
            world.wait_for_tick()
    
    return client, world


def configure_world_settings(
    world: carla.World,
    config: CarlaConfig
) -> carla.WorldSettings:
    """
    Apply synchronous mode and fixed timestep settings to the world.
    
    Args:
        world: CARLA world object
        config: CARLA configuration
        
    Returns:
        The original settings (for restoration on cleanup)
    """
    original_settings = world.get_settings()
    
    settings = world.get_settings()
    settings.synchronous_mode = config.sync_mode
    settings.fixed_delta_seconds = config.fixed_delta if config.sync_mode else None
    world.apply_settings(settings)
    
    return original_settings


def restore_world_settings(
    world: carla.World,
    original_settings: carla.WorldSettings
) -> None:
    """Restore world settings to their original state."""
    try:
        world.apply_settings(original_settings)
    except Exception:
        pass  # Best effort restoration


def create_spawn_transform(
    config: ScenarioConfig,
    lane_offset: float = 0.0,
    x_offset: float = 0.0
) -> carla.Transform:
    """
    Create a spawn transform based on scenario configuration.
    
    Args:
        config: Scenario configuration with spawn coordinates
        lane_offset: Lateral offset in meters (positive = left, negative = right)
        x_offset: Longitudinal offset in meters (positive = ahead)
        
    Returns:
        CARLA Transform for spawning
    """
    location = carla.Location(
        x=config.spawn_x + x_offset,
        y=config.spawn_y + lane_offset,
        z=config.spawn_z
    )
    rotation = carla.Rotation(
        pitch=0.0,
        yaw=config.spawn_yaw,
        roll=0.0
    )
    return carla.Transform(location, rotation)


def spawn_ego_vehicle(
    world: carla.World,
    vehicle_config: VehicleConfig,
    scenario_config: ScenarioConfig
) -> Optional[carla.Vehicle]:
    """
    Spawn the ego (hero) vehicle at the scenario spawn point.
    
    Args:
        world: CARLA world object
        vehicle_config: Vehicle configuration
        scenario_config: Scenario configuration
        
    Returns:
        Spawned vehicle actor, or None if spawn failed
    """
    blueprints = world.get_blueprint_library()
    ego_bp = blueprints.find(vehicle_config.ego_blueprint)
    ego_bp.set_attribute("role_name", vehicle_config.ego_role_name)
    
    spawn_transform = create_spawn_transform(scenario_config)
    
    ego = world.try_spawn_actor(ego_bp, spawn_transform)
    return ego


def spawn_npc_vehicle(
    world: carla.World,
    vehicle_config: VehicleConfig,
    scenario_config: ScenarioConfig,
    client: Optional[carla.Client] = None
) -> Optional[carla.Vehicle]:
    """
    Spawn an NPC vehicle in the adjacent lane.
    
    Args:
        world: CARLA world object
        vehicle_config: Vehicle configuration
        scenario_config: Scenario configuration
        client: CARLA client (required for autopilot setup)
        
    Returns:
        Spawned vehicle actor, or None if spawn failed
    """
    blueprints = world.get_blueprint_library()
    npc_bp = blueprints.find(vehicle_config.npc_blueprint)
    npc_bp.set_attribute("role_name", vehicle_config.npc_role_name)
    
    spawn_transform = create_spawn_transform(
        scenario_config,
        lane_offset=scenario_config.npc_lane_offset,
        x_offset=scenario_config.npc_x_offset
    )
    
    npc = world.try_spawn_actor(npc_bp, spawn_transform)
    
    if npc and vehicle_config.npc_autopilot and client:
        tm = client.get_trafficmanager()
        tm_port = tm.get_port()
        npc.set_autopilot(True, tm_port)
        
        # Set target speed
        tm.set_desired_speed(npc, vehicle_config.npc_target_speed_kmh)
        
        # Disable lane changes for predictable behavior
        tm.auto_lane_change(npc, False)
    
    return npc


def spawn_camera_sensor(
    world: carla.World,
    vehicle: carla.Vehicle,
    camera_config: CameraConfig
) -> carla.Sensor:
    """
    Spawn and attach a camera sensor to a vehicle.
    
    Args:
        world: CARLA world object
        vehicle: Vehicle to attach camera to
        camera_config: Camera configuration
        
    Returns:
        Spawned camera sensor actor
    """
    blueprints = world.get_blueprint_library()
    cam_bp = blueprints.find("sensor.camera.rgb")
    
    cam_bp.set_attribute("image_size_x", str(camera_config.width))
    cam_bp.set_attribute("image_size_y", str(camera_config.height))
    cam_bp.set_attribute("fov", str(camera_config.fov))
    
    cam_transform = carla.Transform(
        carla.Location(
            x=camera_config.location_x,
            y=camera_config.location_y,
            z=camera_config.location_z
        ),
        carla.Rotation(
            pitch=camera_config.pitch,
            yaw=camera_config.yaw,
            roll=camera_config.roll
        )
    )
    
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    return camera


def spawn_collision_sensor(
    world: carla.World,
    vehicle: carla.Vehicle
) -> carla.Sensor:
    """
    Spawn and attach a collision sensor to a vehicle.
    
    Args:
        world: CARLA world object
        vehicle: Vehicle to attach sensor to
        
    Returns:
        Spawned collision sensor actor
    """
    blueprints = world.get_blueprint_library()
    collision_bp = blueprints.find("sensor.other.collision")
    
    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),  # Centered on vehicle
        attach_to=vehicle
    )
    return collision_sensor


def spawn_lane_invasion_sensor(
    world: carla.World,
    vehicle: carla.Vehicle
) -> carla.Sensor:
    """
    Spawn and attach a lane invasion sensor to a vehicle.
    
    Args:
        world: CARLA world object
        vehicle: Vehicle to attach sensor to
        
    Returns:
        Spawned lane invasion sensor actor
    """
    blueprints = world.get_blueprint_library()
    lane_bp = blueprints.find("sensor.other.lane_invasion")
    
    lane_sensor = world.spawn_actor(
        lane_bp,
        carla.Transform(),
        attach_to=vehicle
    )
    return lane_sensor


def destroy_actors(actors: List[carla.Actor]) -> int:
    """
    Safely destroy a list of actors.
    
    Args:
        actors: List of actors to destroy
        
    Returns:
        Number of actors successfully destroyed
    """
    destroyed = 0
    for actor in actors:
        if actor is not None:
            try:
                if hasattr(actor, 'stop'):
                    actor.stop()  # Stop sensors before destroying
                actor.destroy()
                destroyed += 1
            except Exception:
                pass
    return destroyed


def cleanup_world(world: carla.World, role_names: Optional[List[str]] = None) -> int:
    """
    Clean up vehicles and sensors from the world.
    
    Args:
        world: CARLA world object
        role_names: If provided, only destroy actors with these role names.
                   If None, destroys all vehicles and sensors.
                   
    Returns:
        Number of actors destroyed
    """
    destroyed = 0
    actors = world.get_actors()
    
    for actor in actors:
        should_destroy = False
        
        if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
            if role_names is None:
                should_destroy = True
            elif hasattr(actor, 'attributes'):
                actor_role = actor.attributes.get("role_name", "")
                if actor_role in role_names:
                    should_destroy = True
        
        if should_destroy:
            try:
                if hasattr(actor, 'stop'):
                    actor.stop()
                actor.destroy()
                destroyed += 1
            except Exception:
                pass
    
    return destroyed


def get_vehicle_speed(vehicle: carla.Vehicle) -> float:
    """
    Get the speed of a vehicle in meters per second.
    
    Args:
        vehicle: CARLA vehicle actor
        
    Returns:
        Speed in m/s
    """
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def get_lane_offset(
    vehicle: carla.Vehicle,
    world_map: carla.Map
) -> Tuple[float, int, float]:
    """
    Calculate the vehicle's offset from the lane center.
    
    Args:
        vehicle: CARLA vehicle actor
        world_map: CARLA map object
        
    Returns:
        Tuple of (lateral_offset, lane_id, lane_width)
        lateral_offset is positive if vehicle is left of center
    """
    transform = vehicle.get_transform()
    location = transform.location
    
    waypoint = world_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    # Calculate lateral offset from lane center
    lane_yaw = math.radians(waypoint.transform.rotation.yaw)
    
    # Normal vector pointing to the right of the lane
    normal_x = -math.sin(lane_yaw)
    normal_y = math.cos(lane_yaw)
    
    # Vector from waypoint center to vehicle
    dx = location.x - waypoint.transform.location.x
    dy = location.y - waypoint.transform.location.y
    
    # Project onto normal to get lateral offset
    lateral_offset = dx * normal_x + dy * normal_y
    
    return lateral_offset, waypoint.lane_id, waypoint.lane_width


def teleport_vehicle(
    vehicle: carla.Vehicle,
    transform: carla.Transform,
    zero_velocity: bool = True
) -> None:
    """
    Teleport a vehicle to a new location.
    
    Args:
        vehicle: Vehicle to teleport
        transform: Target transform
        zero_velocity: If True, also reset velocity to zero
    """
    vehicle.set_transform(transform)
    
    if zero_velocity:
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))


def kill_all_carla_processes() -> None:
    """
    Force-kill all CarlaUE4 server processes on this machine.

    On Linux, CarlaUE4.sh spawns a child binary (CarlaUE4-Linux-Shipping)
    that survives a SIGTERM to the shell script's process group.
    pkill -9 -f CarlaUE4 catches both the wrapper and the binary.
    """
    print("[Cleanup] Killing all CARLA server processes...")
    if platform.system() == "Windows":
        for name in ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe"):
            subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
    else:
        subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], capture_output=True)
    print("[Cleanup] Done.")


def wait_for_tick(world: carla.World, timeout: float = 10.0) -> bool:
    """
    Wait for a world tick with timeout.
    
    Args:
        world: CARLA world object
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if tick was received, False if timeout
    """
    try:
        world.wait_for_tick(timeout)
        return True
    except RuntimeError:
        return False

