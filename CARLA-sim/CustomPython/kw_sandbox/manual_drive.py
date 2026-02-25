import carla
import random
import pygame
import math
import time
import traceback
# from csv_logging import CarlaCSVLogger
from camera_control import follow_camera
from parquet_logger import CarlaParquetLogger
from steering_control import get_wheel_control, get_keyboard_control
from waypoint import find_adjacent_lane_waypoint, find_forward_waypoint, find_nearest_npc_vehicle, find_random_waypoint, get_relative_lane_and_longitudinal
from globals import EventState

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("CARLA Manual Control (focus here, use WASD, R for reverse)")

    pygame.joystick.init()

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

    try:
        while True:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Received QUIT event")
                    raise SystemExit
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
        ego_vehicle.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
