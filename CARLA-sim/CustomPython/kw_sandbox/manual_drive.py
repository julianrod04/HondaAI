import carla
import random
import pygame
import math
import time
import os
# from csv_logging import CarlaCSVLogger
from parquet_logger import CarlaParquetLogger
from steering_control import get_wheel_control, get_keyboard_control
from waypoint import find_adjacent_lane_waypoint, find_forward_waypoint

def main():
    # Configure SDL2 environment variables for better XInput/Xbox controller support
    # These must be set BEFORE pygame.init()
    os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    # Force SDL to use HIDAPI backend (better for XInput devices)
    os.environ['SDL_JOYSTICK_HIDAPI'] = '1'
    # Enable XInput support explicitly
    os.environ['SDL_HINT_JOYSTICK_HIDAPI_XBOX'] = '1'
    
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("CARLA Manual Control (focus here, use WASD, R for reverse)")

    pygame.joystick.init()

    # Additional diagnostic info
    print("\n=== Joystick Debug Info ===")
    print("Pygame version:", pygame.version.ver)
    print("SDL version:", pygame.version.SDL)
    print("Joystick module initialized:", pygame.joystick.get_init())
    print("SDL_JOYSTICK_HIDAPI:", os.environ.get('SDL_JOYSTICK_HIDAPI', 'not set'))
    
    # Try multiple event pumps with longer delays to ensure detection
    # XInput devices sometimes need more time to enumerate
    print("\nScanning for joysticks...")
    for i in range(5):
        pygame.event.pump()
        time.sleep(0.2)  # Longer delay for XInput device enumeration
        current_count = pygame.joystick.get_count()
        if current_count > 0:
            print(f"  Found {current_count} device(s) after {i+1} scan(s)")
    
    joystick_count = pygame.joystick.get_count()
    print("Number of joysticks detected:", joystick_count)
    
    if joystick_count == 0:
        print(":warning:  No joysticks detected by pygame.")
        print("\nNote: If your device appears as 'Xbox peripheral' in Device Manager")
        print("      but not in joy.cpl, it's using XInput (not DirectInput).")
        print("      SDL2 should still detect it, but may need:")
        print("      - Device to be connected before running this script")
        print("      - Proper XInput drivers installed")
        print("      - Device to be powered on and not in sleep mode")
        print("\nTroubleshooting tips:")
        print("  1. Ensure joystick/gamepad is physically connected and powered on")
        print("  2. Check Windows Device Manager for the device (Xbox peripherals)")
        print("  3. Try unplugging and replugging the device")
        print("  4. Try a different USB port")
        print("  5. Restart the script with device already connected")
        print("  6. Check manufacturer's website for XInput-compatible drivers")
        print("  7. Some wheels need to be in 'PC mode' not 'Xbox mode'")
    else:
        for i in range(joystick_count):
            joy = pygame.joystick.Joystick(i)
            joy.init()
            print(f"\nJoystick {i}:")
            print("  Name:", joy.get_name())
            print("  GUID:", joy.get_guid() if hasattr(joy, "get_guid") else "N/A")
            print("  ID:", joy.get_id() if hasattr(joy, "get_id") else "N/A")
            print("  Num Axes:", joy.get_numaxes())
            print("  Num Buttons:", joy.get_numbuttons())
            print("  Num Hats:", joy.get_numhats())
    print("===========================\n")

    wheel = None

    if pygame.joystick.get_count() == 0:
        print("No joystick detected!")
    else:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print("Using wheel:", wheel.get_name())

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    ego_bp = blueprint_library.filter("model3")[0]
    npc_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn = random.choice(spawn_points)

     # Spawn ego vehicle
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle, try again")

    ego_vehicle.set_autopilot(False)
    print("Spawned ego vehicle:", ego_vehicle.id)

    # Find adjacent lane transform for NPC:
    # npc_spawn_tf = find_adjacent_lane_waypoint(world_map, ego_spawn)

    # Find forward lane transform for NPS:
    npc_spawn_tf = find_forward_waypoint(world_map, ego_spawn)
    if npc_spawn_tf is None:
        print("No adjacent lane found – spawning NPC a bit ahead in same lane instead.")
        # If no adjacent lane, just spawn ahead of ego in same lane (offset along road)
        ego_wp = world_map.get_waypoint(ego_spawn.location)
        forward_wp = ego_wp.next(10.0)[0]  # 10 meters ahead
        npc_spawn_tf = forward_wp.transform

    # Slightly offset NPC forward so they don't collide at spawn
    npc_spawn_tf.location.x += 1.0
    npc_spawn_tf.location.z += 0.5  # small lift to avoid ground collision

    npc_vehicle = world.try_spawn_actor(npc_bp, npc_spawn_tf)
    if npc_vehicle is None:
        print("Failed to spawn NPC vehicle. Only ego will be present.")
    else:
        print("Spawned NPC vehicle:", npc_vehicle.id)

    # Traffic Manager for NPC constant-ish speed
    traffic_manager = client.get_trafficmanager()
    tm_port = traffic_manager.get_port()

    if npc_vehicle is not None:
        # Register with TM
        npc_vehicle.set_autopilot(True, tm_port)

        # Make TM non-synchronous for simplicity
        traffic_manager.set_synchronous_mode(False)

        # Set desired speed: 0% difference = speed limit, positive = slower, negative = faster
        desired_slower_percent = 0.0  # change this to 20 or -20, etc.
        traffic_manager.vehicle_percentage_speed_difference(
            npc_vehicle, desired_slower_percent
        )

        # Disable lane changes if you want it to stay in its lane:
        traffic_manager.auto_lane_change(npc_vehicle, False)

    spectator = world.get_spectator()
    clock = pygame.time.Clock()
    control = carla.VehicleControl()
    reverse_mode = False  # R key toggles this

    # NPC control variables
    npc_control = carla.VehicleControl()
    npc_start_time = time.time()  # wall-clock seconds at start of loop

    # to log to csv:
    # logger = CarlaCSVLogger(world, ego_vehicle, npc_vehicle, log_dir="logs")
    # to log to parquet:
    logger = CarlaParquetLogger(world, ego_vehicle, npc_vehicle, log_dir="logs")

    try:
        while True:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    # 🔁 toggle reverse with R
                    if event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                        print("Reverse mode:", reverse_mode)
                if event.type == pygame.JOYBUTTONDOWN:
                    # Example: wheel button 4 toggles reverse
                    if event.button == 4:
                        reverse_mode = not reverse_mode
                        print("Reverse mode:", reverse_mode)
    
            if wheel is not None:
                control = get_wheel_control(wheel, control, reverse_mode)
            else:
                keys = pygame.key.get_pressed()
                control = get_keyboard_control(keys, control, reverse_mode)

            ego_vehicle.apply_control(control)
            # keys = pygame.key.get_pressed()
            # control = get_keyboard_control(keys, control, reverse_mode)
            # control = get_wheel_control(wheel, control, reverse_mode)
            # ego_vehicle.apply_control(control)

            # if npc_vehicle is not None:
            #     elapsed = time.time() - npc_start_time

            #     if elapsed < 5.0:
            #         # drive straight at constant throttle
            #         npc_control.throttle = 0.5
            #         npc_control.brake = 0.0
            #     else:
            #         # brake hard
            #         npc_control.throttle = 0.0
            #         npc_control.brake = 1.0

            #     npc_control.steer = 0.0
            #     npc_control.hand_brake = False
            #     npc_control.reverse = False

            #     npc_vehicle.apply_control(npc_control)

            # ---- follow camera ----
            transform = ego_vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation

            distance_back = 6.0
            height = 3.0
            yaw = math.radians(rotation.yaw)

            follow_x = location.x - distance_back * math.cos(yaw)
            follow_y = location.y - distance_back * math.sin(yaw)
            follow_z = location.z + height

            camera_location = carla.Location(follow_x, follow_y, follow_z)
            camera_rotation = carla.Rotation(pitch=-15.0, yaw=rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(camera_location, camera_rotation))

            logger.log_step(control)

    finally:
        logger.close()
        print("Destroying actors...")
        if npc_vehicle is not None:
            npc_vehicle.destroy()
        ego_vehicle.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
