import carla
import random
import pygame
import math
import time
import traceback
import os
import numpy as np
# from csv_logging import CarlaCSVLogger
from camera_control import follow_camera
from parquet_logger import CarlaParquetLogger
from steering_control import get_wheel_control, get_keyboard_control
from waypoint import find_adjacent_lane_waypoint, find_forward_waypoint
from alerts import AlertType, Dashboard, DrivingMonitor, AlertDisplayConfig

# Window dimensions (large enough for camera feed + overlay)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def main():
    # Configure SDL2 environment variables for better XInput/Xbox controller support
    # These must be set BEFORE pygame.init()
    os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    # Force SDL to use HIDAPI backend (better for XInput devices)
    os.environ['SDL_JOYSTICK_HIDAPI'] = '1'
    # Enable XInput support explicitly
    os.environ['SDL_HINT_JOYSTICK_HIDAPI_XBOX'] = '1'

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CARLA Manual Control (WASD to drive, R=reverse, Tab=HUD, ESC=quit)")

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
        print("No joysticks detected by pygame.")
        print("Using keyboard controls (WASD).")
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
    client.set_timeout(10.0)

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
    npc_vehicles = []
    traffic_manager = client.get_trafficmanager()
    tm_port = traffic_manager.get_port()
    for i in range(10):
        npc_spawn_tf = find_random_waypoint(world)

        # Slightly offset NPC forward so they don't collide at spawn
        # npc_spawn_tf.location.x += 1.0
        npc_spawn_tf.location.z += 0.5  # small lift to avoid ground collision

        npc_vehicle = world.try_spawn_actor(npc_bp, npc_spawn_tf)
        if npc_vehicle is None:
            print("Failed to spawn NPC vehicle. Only ego will be present.")
        else:
            print("Spawned NPC vehicle:", npc_vehicle.id)
            npc_vehicles.append(npc_vehicle)

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

    print(npc_vehicles)
    # Use the first NPC for logging compatibility (if any)
    npc_vehicle = npc_vehicles[0] if npc_vehicles else None

    event_state = EventState(traffic_manager)
    event_prob = 0.05

    # ---- Attach RGB camera to ego for pygame rendering ----
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
    cam_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
    cam_bp.set_attribute("fov", "100")

    # Driver's seat POV
    cam_tf = carla.Transform(
        carla.Location(x=0.2, y=-0.36, z=1.2),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_vehicle)

    latest_image = None
    def on_image(img):
        nonlocal latest_image
        array = np.frombuffer(img.raw_data, dtype=np.uint8)
        array = array.reshape((img.height, img.width, 4))
        latest_image = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB

    camera.listen(on_image)

    # ---- Alert system ----
    alert_config = AlertDisplayConfig()
    dashboard = Dashboard(WINDOW_WIDTH, WINDOW_HEIGHT, alert_config=alert_config)
    driving_monitor = DrivingMonitor(world_map, world=world)

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
    logger = CarlaParquetLogger(world, ego_vehicle, npc_vehicles, log_dir="logs")

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
    print("Navigation Alerts (test keys):")
    print("  1        - Turn Left")
    print("  2        - Turn Right")
    print("  3        - Stop at the light")
    print("  TAB      - Toggle diagnostics bar")
    print("")
    print("Automatic Alerts:")
    print("  - Lane drifting warnings")
    print("  - Speed limit exceeded")
    print("  - Acceleration from standstill")
    print("  - Front/rear proximity warnings")
    print("=" * 50 + "\n")

    try:
        while True:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Received QUIT event")
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                        print("Reverse mode:", reverse_mode)
                    # Navigation alert test keys
                    elif event.key == pygame.K_1:
                        dashboard.trigger_navigation(AlertType.TURN_LEFT, duration=5.0)
                    elif event.key == pygame.K_2:
                        dashboard.trigger_navigation(AlertType.TURN_RIGHT, duration=5.0)
                    elif event.key == pygame.K_3:
                        dashboard.trigger_navigation(AlertType.STOP_AT_LIGHT, duration=5.0)
                    # Toggle diagnostics bar
                    elif event.key == pygame.K_TAB:
                        dashboard.alert_config.show_diagnostics_bar = \
                            not dashboard.alert_config.show_diagnostics_bar
                        state = "ON" if dashboard.alert_config.show_diagnostics_bar else "OFF"
                        print(f"[HUD] Diagnostics bar: {state}")
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
            # follow_camera(ego_vehicle)
            # transform = ego_vehicle.get_transform()
            # location = transform.location
            # rotation = transform.rotation

            # distance_back = 6.0
            # height = 3.0
            # yaw = math.radians(rotation.yaw)

            # follow_x = location.x - distance_back * math.cos(yaw)
            # follow_y = location.y - distance_back * math.sin(yaw)
            # follow_z = location.z + height

            # camera_location = carla.Location(follow_x, follow_y, follow_z)
            # camera_rotation = carla.Rotation(pitch=-15.0, yaw=rotation.yaw, roll=0.0)
            camera_transform = follow_camera(ego_vehicle)
            spectator.set_transform(camera_transform)

            # throw random npc behavior
            if not event_state.in_progress and random.random() < event_prob:
                # first find nearest npc
                nearest_npc = find_nearest_npc_vehicle(world, ego_vehicle)
                lane_relative_to_ego, ahead_or_behind_ego, distance_to_ego = get_relative_lane_and_longitudinal(world_map, ego_vehicle, nearest_npc)
                print(lane_relative_to_ego, ahead_or_behind_ego)
                event_state.select_behavior(nearest_npc, lane_relative_to_ego, ahead_or_behind_ego)
                print(event_state.in_progress)
            elif event_state.in_progress:
                event_state.tick_override_event()

            # ---- Render camera feed + overlays to pygame ----
            if latest_image is not None:
                surf = pygame.surfarray.make_surface(latest_image.swapaxes(0, 1))
                screen.blit(surf, (0, 0))

            dashboard.render(screen, metrics, reverse_mode)
            pygame.display.flip()

            logger.log_step(control)

    except KeyboardInterrupt:
        print("KeyboardInterrupt (Ctrl+C)")

    except Exception as e:
        print("Exception occurred:", repr(e))
        traceback.print_exc()
    
    finally:
        logger.close()
        print("Destroying actors...")
        for npc in npc_vehicles:
            npc.destroy()
        camera.stop()
        camera.destroy()
        if npc_vehicle is not None:
            npc_vehicle.destroy()
        ego_vehicle.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
