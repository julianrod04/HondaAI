import carla
import random
import pygame
import math
import time
# from csv_logging import CarlaCSVLogger
from parquet_logger import CarlaParquetLogger

def get_wheel_control(wheel, control, reverse_mode: bool) -> carla.VehicleControl:
    """
    Logitech wheel + pedals input handler for mapping where:
      - Axis 0: steering  (-1 left, +1 right, ~0 centered)
      - Axis 2: accelerator ( -1 at rest -> +1 fully pressed )
      - Axis 3: brake       ( -1 at rest -> +1 fully pressed )

    If your gas/brake are on different axes, just change the axis indices.
    """
    # Reset
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # Reverse gear flag (you toggle reverse_mode elsewhere)
    control.reverse = reverse_mode

    # ---- Steering (Axis 0) ----
    steer_axis = wheel.get_axis(0)  # -1 left, +1 right
    print(steer_axis)
    # deadzone = 0.05
    # if abs(steer_axis) < deadzone:
        # steer_axis = 0.0
    control.steer = float(max(-1.0, min(1.0, steer_axis)))

    # ---- Accelerator (Axis 2) ----
    accel_axis = wheel.get_axis(1)  # -1 rest, +1 pressed
    print(accel_axis)
    # Map [-1, 1] -> [0, 1]
    throttle = (accel_axis + 1.0) / 2.0
    control.throttle = float(max(0.15, min(1.0, throttle)))

    # ---- Brake (Axis 3) ----
    brake_axis = wheel.get_axis(3)  # -1 rest, +1 pressed
    brake_val = (brake_axis + 1.0) / 2.0
    brake_val = math.exp(10*(brake_val**(2.5))) - 1
    print(brake_val)
    control.brake = float(max(0.0, min(1.0, brake_val)))

    # ---- Hand brake (pick any button you like, example: button 5) ----
    handbrake_button = wheel.get_button(5)
    control.hand_brake = bool(handbrake_button)

    return control


def get_keyboard_control(keys, control, reverse_mode: bool) -> carla.VehicleControl:
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # set reverse flag based on current gear mode
    control.reverse = reverse_mode

    # Throttle (forward or reverse depending on control.reverse)
    if keys[pygame.K_w]:
        control.throttle = 1.0

    # Brake
    if keys[pygame.K_s]:
        control.brake = 1.0

    # Steering
    steer_amt = 0.35
    if keys[pygame.K_a]:
        control.steer = -steer_amt
    if keys[pygame.K_d]:
        control.steer = steer_amt

    # Hand brake
    control.hand_brake = keys[pygame.K_SPACE]

    return control

def find_adjacent_lane_waypoint(world_map, ego_spawn_transform):
    """
    Given the ego spawn transform, find a driving lane directly to the right if possible.
    If no right lane exists, try the left lane.
    Returns a carla.Transform for the adjacent lane, or None if none exists.
    """
    ego_wp = world_map.get_waypoint(
        ego_spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    print(ego_wp)

    # Try right lane first (typical opposite-direction in many maps)
    right_wp = ego_wp.get_right_lane()
    print(right_wp.transform)
    if right_wp is not None and (right_wp.lane_type & carla.LaneType.Driving):
        return right_wp.transform

    # Fallback: try left lane
    left_wp = ego_wp.get_left_lane()
    print(left_wp.transform)
    if left_wp is not None and (left_wp.lane_type & carla.LaneType.Driving):
        return left_wp.transform

    # No adjacent lane found
    return None

def find_forward_waypoint(world_map, ego_spawn_transform):
    ego_wp = world_map.get_waypoint(
        ego_spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    ahead_wps = ego_wp.next(8.0)  # 8 meters in front
    if ahead_wps:
        return ahead_wps[0].transform
    
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("CARLA Manual Control (focus here, use WASD, R for reverse)")

    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected!")
    else:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print("Using wheel:", wheel.get_name())

    for i in range(wheel.get_numaxes()):
        print(f"Axis {i}: {wheel.get_axis(i)}")

    for i in range(wheel.get_numbuttons()):
        print(f"Button {i}: {wheel.get_button(i)}")


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
            control = get_wheel_control(wheel, control, reverse_mode)
            ego_vehicle.apply_control(control)

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
