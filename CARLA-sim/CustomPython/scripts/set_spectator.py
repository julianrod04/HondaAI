#!/usr/bin/env python3
"""
Set CARLA spectator camera to view the training area.

Usage:
    python scripts/set_spectator.py
    python scripts/set_spectator.py --view top      # Top-down view
    python scripts/set_spectator.py --view chase    # Chase cam behind spawn
    python scripts/set_spectator.py --follow        # Continuously follow ego vehicle
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

# Training scenario coordinates (from rl/config.py)
SPAWN_X = 21.0
SPAWN_Y = 244.485397
SPAWN_Z = 0.5
GOAL_X = 221.0


def set_spectator_view(client: carla.Client, view: str = "top"):
    """Set spectator to view the training area."""
    world = client.get_world()
    spectator = world.get_spectator()
    
    if view == "top":
        # Top-down view centered on the track
        center_x = (SPAWN_X + GOAL_X) / 2  # ~121
        location = carla.Location(x=center_x, y=SPAWN_Y, z=120.0)
        rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        print(f"Setting TOP-DOWN view at x={center_x:.1f}, y={SPAWN_Y:.1f}, z=120")
        
    elif view == "chase":
        # Behind the spawn point, looking forward
        location = carla.Location(x=SPAWN_X - 15, y=SPAWN_Y, z=8.0)
        rotation = carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)
        print(f"Setting CHASE view behind spawn at x={SPAWN_X-15:.1f}")
        
    elif view == "side":
        # Side view of the spawn area
        location = carla.Location(x=SPAWN_X + 20, y=SPAWN_Y + 30, z=10.0)
        rotation = carla.Rotation(pitch=-10.0, yaw=-90.0, roll=0.0)
        print(f"Setting SIDE view of spawn area")
        
    elif view == "spawn":
        # Right at the spawn point, ground level
        location = carla.Location(x=SPAWN_X, y=SPAWN_Y, z=2.0)
        rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        print(f"Setting SPAWN view at x={SPAWN_X:.1f}, y={SPAWN_Y:.1f}")
    
    else:
        print(f"Unknown view: {view}")
        return
    
    spectator.set_transform(carla.Transform(location, rotation))
    print("Spectator camera moved!")


def follow_ego(client: carla.Client):
    """Continuously follow the ego vehicle."""
    world = client.get_world()
    spectator = world.get_spectator()
    
    print("Following ego vehicle (Ctrl+C to stop)...")
    print("Looking for vehicle with role_name='hero'...")
    
    try:
        while True:
            # Find ego vehicle
            ego = None
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'hero':
                    ego = actor
                    break
            
            if ego is None:
                print("  No ego vehicle found, waiting...", end='\r')
                time.sleep(0.5)
                continue
            
            # Get ego transform
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            
            # Calculate chase camera position
            import math
            yaw_rad = math.radians(ego_rot.yaw)
            follow_distance = 12.0
            height = 6.0
            
            cam_x = ego_loc.x - follow_distance * math.cos(yaw_rad)
            cam_y = ego_loc.y - follow_distance * math.sin(yaw_rad)
            cam_z = ego_loc.z + height
            
            cam_loc = carla.Location(x=cam_x, y=cam_y, z=cam_z)
            cam_rot = carla.Rotation(pitch=-15.0, yaw=ego_rot.yaw, roll=0.0)
            
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            
            # Print status
            speed = math.sqrt(ego.get_velocity().x**2 + ego.get_velocity().y**2) * 3.6
            print(f"  Ego at x={ego_loc.x:.1f}, y={ego_loc.y:.1f}, speed={speed:.1f} km/h    ", end='\r')
            
            time.sleep(0.05)  # 20 FPS update
            
    except KeyboardInterrupt:
        print("\nStopped following.")


def load_town06(client: carla.Client):
    """Load Town06 map if not already loaded."""
    world = client.get_world()
    current_map = world.get_map().name.split('/')[-1]
    
    if current_map != "Town06":
        print(f"Current map is {current_map}, loading Town06...")
        client.set_timeout(60.0)
        world = client.load_world("Town06")
        client.set_timeout(10.0)
        print("Town06 loaded!")
        return client.get_world()
    else:
        print(f"Already on Town06")
        return world


def main():
    parser = argparse.ArgumentParser(description="Set CARLA spectator camera")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--view", choices=["top", "chase", "side", "spawn"], 
                        default="top", help="Camera view preset")
    parser.add_argument("--follow", action="store_true", 
                        help="Continuously follow ego vehicle")
    parser.add_argument("--load-map", action="store_true",
                        help="Load Town06 if not already loaded")
    args = parser.parse_args()
    
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    world = client.get_world()
    current_map = world.get_map().name.split('/')[-1]
    print(f"Connected! Current map: {current_map}")
    
    if args.load_map:
        world = load_town06(client)
    
    if args.follow:
        follow_ego(client)
    else:
        set_spectator_view(client, args.view)
        
        # Print helpful info
        print(f"\nTraining area coordinates:")
        print(f"  Spawn point: x={SPAWN_X}, y={SPAWN_Y}")
        print(f"  Goal point:  x={GOAL_X}, y={SPAWN_Y}")
        print(f"  Track length: {GOAL_X - SPAWN_X}m")


if __name__ == "__main__":
    main()



