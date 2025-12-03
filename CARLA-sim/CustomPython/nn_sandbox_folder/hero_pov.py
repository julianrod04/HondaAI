import os
import sys
import math
import time

import numpy as np
import pygame

# -------------------------------------------------------------------
# Add CustomPython and kw_sandbox to sys.path
# -------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../CustomPython/nn_sandbox_folder
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../CustomPython
KW_SANDBOX = os.path.join(PROJECT_ROOT, "kw_sandbox")          # .../CustomPython/kw_sandbox

for path in (PROJECT_ROOT, KW_SANDBOX):
    if path not in sys.path:
        sys.path.append(path)

# -------------------------------------------------------------------
# Make sure CARLA PythonAPI is on the path (adjust if your path differs)
# -------------------------------------------------------------------
sys.path.append(r"C:\Users\natha\Downloads\CARLA_0.9.16\PythonAPI")
sys.path.append(r"C:\Users\natha\Downloads\CARLA_0.9.16\PythonAPI\carla\dist")

import carla  # noqa: E402

from parquet_logger import CarlaParquetLogger


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_speed(vehicle: carla.Vehicle) -> float:
    """Return vehicle speed in m/s."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def connect_and_load_world() -> tuple[carla.Client, carla.World]:
    """Connect to CARLA and load Town06 with a generous timeout and async mode."""
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    print("Connecting to CARLA and loading Town06...")
    world = client.load_world("Town06")
    print("World loaded, map name:", world.get_map().name)

    # Force async mode so sim steps automatically without world.tick()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # Let the simulator tick a few times to settle
    for _ in range(10):
        world.wait_for_tick()

    return client, world


# -------------------------------------------------------------------
# Input helpers
# -------------------------------------------------------------------
def get_wheel_control(
    wheel: pygame.joystick.Joystick,
    control: carla.VehicleControl,
    reverse_mode: bool,
) -> carla.VehicleControl:
    """
    Logitech wheel + pedals input handler.

    In your current mapping:
      - Axis 0: steering
      - Axis 3: throttle (gas)
      - Axis 1: brake
    """
    # Reset main axes
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # Reverse flag
    control.reverse = reverse_mode

    # ---- Steering (Axis 0) ----
    steer_axis = wheel.get_axis(0)  # -1 left, +1 right typically
    steer_axis = max(-1.0, min(1.0, steer_axis))
    control.steer = float(steer_axis)

    # ---- Accelerator (Axis 3) ----
    accel_axis = wheel.get_axis(3)          # -1 at rest, +1 fully pressed
    accel_axis = max(-1.0, min(1.0, accel_axis))
    throttle = (accel_axis + 1.0) / 2.0     # [-1, 1] -> [0, 1]
    control.throttle = float(max(0.0, min(1.0, throttle)))

    # ---- Brake (Axis 1) ----
    brake_axis = wheel.get_axis(1)          # -1 at rest, +1 fully pressed
    brake_axis = max(-1.0, min(1.0, brake_axis))
    brake_val = (brake_axis + 1.0) / 2.0    # [-1, 1] -> [0, 1]

    # Small deadzone so resting pedal doesn't drag
    if brake_val < 0.02:
        brake_val = 0.0

    control.brake = float(max(0.0, min(1.0, brake_val)))

    # ---- Hand brake (example: button 5) ----
    handbrake_button = wheel.get_button(5)
    control.hand_brake = bool(handbrake_button)

    return control


def get_keyboard_control(
    keys, control: carla.VehicleControl, reverse_mode: bool
) -> carla.VehicleControl:
    """
    WASD control that can:
      - be the only control (no wheel), or
      - override wheel when keys are pressed.

    If no relevant key is pressed, we keep the existing control values.
    """
    # Always keep reverse flag updated
    control.reverse = reverse_mode

    any_key = (
        keys[pygame.K_w]
        or keys[pygame.K_s]
        or keys[pygame.K_a]
        or keys[pygame.K_d]
        or keys[pygame.K_SPACE]
    )

    if not any_key:
        # Don't clobber wheel values if nothing is pressed
        return control

    # If we are using keyboard for this frame, reset axes
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # Throttle
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


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
def main():
    client, world = connect_and_load_world()
    blueprints = world.get_blueprint_library()
    spectator = world.get_spectator()

    # ----------------------------------------------------------------
    # Clean up leftover vehicles/sensors from previous runs
    # ----------------------------------------------------------------
    print("Cleaning up previous vehicles/sensors...")
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
            try:
                actor.destroy()
            except Exception:
                pass
    print("Cleanup complete.")

    # (Initial spectator view doesn't matter much; we'll override it each frame.)

    # ----------------------------------------------------------------
    # Spawn ego (hero) and NPC police in adjacent lane
    # ----------------------------------------------------------------
    ego_bp = blueprints.find("vehicle.dodge.charger_2020")
    ego_bp.set_attribute("role_name", "hero")

    ego_spawn_loc = carla.Location(x=21, y=244.485397, z=0.5)
    ego_spawn_rot = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    ego_tf = carla.Transform(ego_spawn_loc, ego_spawn_rot)

    npc_bp = blueprints.find("vehicle.dodge.charger_police_2020")
    npc_spawn_loc = carla.Location(x=15, y=244.485397 - 3.5, z=0.5)  # left lane, slightly behind
    npc_spawn_rot = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    npc_tf = carla.Transform(npc_spawn_loc, npc_spawn_rot)

    ego = world.try_spawn_actor(ego_bp, ego_tf)
    npc = world.try_spawn_actor(npc_bp, npc_tf)

    print("Ego:", ego)
    print("NPC:", npc)

    if ego is None:
        print("Failed to spawn ego vehicle. Exiting.")
        return
    if npc is None:
        print("Failed to spawn NPC vehicle. Exiting.")
        ego.destroy()
        return

    # ----------------------------------------------------------------
    # Traffic Manager and autopilot for NPC
    # ----------------------------------------------------------------
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    npc.set_autopilot(True, tm_port)   # NPC uses Traffic Manager
    ego.set_autopilot(False)           # Ego is manual

    print("Traffic Manager port:", tm_port)

    # Disable TM collision avoidance between NPC and ego
    tm.collision_detection(npc, ego, False)

    # ----------------------------------------------------------------
    # Attach camera to ego (driver POV)
    # ----------------------------------------------------------------
    W, H = 1920, 1080  # Pygame display + sensor resolution

    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(W))
    cam_bp.set_attribute("image_size_y", str(H))
    cam_bp.set_attribute("fov", "90")

    cam_tf = carla.Transform(
        carla.Location(x=0.2, y=-0.36, z=1.2),
        carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    latest_image = None

    def on_image(img: carla.Image):
        nonlocal latest_image
        array = np.frombuffer(img.raw_data, dtype=np.uint8)
        array = array.reshape((img.height, img.width, 4))
        latest_image = array[:, :, :3]  # RGB

    camera.listen(on_image)

    # ----------------------------------------------------------------
    # Pygame setup
    # ----------------------------------------------------------------
    pygame.init()
    print("pygame.get_init():", pygame.get_init())
    print("display.get_init():", pygame.display.get_init())

    pygame.joystick.init()
    wheel = None
    if pygame.joystick.get_count() == 0:
        print("No joystick detected – starting with keyboard control.")
    else:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print("Using wheel:", wheel.get_name())

    try:
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Hero POV (Wheel/Keyboard + Logging)")

        pygame.font.init()
        hud_font = pygame.font.SysFont("consolas", 24)
    except Exception as e:
        print("Error creating display:", e)
        camera.stop()
        ego.destroy()
        npc.destroy()
        pygame.quit()
        return

    print("display surface right after set_mode:", pygame.display.get_surface())
    print("video driver:", pygame.display.get_driver())

    clock = pygame.time.Clock()

    print("Camera attached and Pygame window should now be visible.")
    print("Click on the Hero POV window and use wheel or WASD to drive, ESC to quit.")
    MAX_SPEED = 30.0  # km/h
    npc_locked_speed = False
    running = True

    control = carla.VehicleControl()
    reverse_mode = False  # toggled by R key or wheel button

    logger = CarlaParquetLogger(world, ego, npc, log_dir="logs")
    print("Parquet logging enabled.")

    # ---- Initial spectator chase-cam state (for smoothing) ----
    spec_tf = spectator.get_transform()
    spec_loc = spec_tf.location
    spec_yaw = spec_tf.rotation.yaw

    try:
        while running:
            if not pygame.get_init() or not pygame.display.get_surface():
                print("Pygame window closed — stopping loop.")
                break

            # ----- Event handling -----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUIT event received.")
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("ESC pressed, exiting loop.")
                        running = False
                    if event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                        print("Reverse mode:", reverse_mode)

                if event.type == pygame.JOYBUTTONDOWN and wheel is not None:
                    # Example: toggle reverse on button 4
                    if event.button == 4:
                        reverse_mode = not reverse_mode
                        print("Reverse mode:", reverse_mode)

            keys = pygame.key.get_pressed()

            # ----- Base control from wheel or keyboard -----
            if wheel is not None:
                control = get_wheel_control(wheel, control, reverse_mode)
            else:
                control = get_keyboard_control(keys, control, reverse_mode)

            # ----- Allow keyboard to override wheel when keys are pressed -----
            control = get_keyboard_control(keys, control, reverse_mode)

            # Hero speed
            hero_speed = get_speed(ego)          # m/s
            hero_speed_kmh = hero_speed * 3.6    # km/h

            # ---- Light launch assist: only when basically stopped and gas > 0 ----
            if hero_speed < 0.5 and control.brake < 0.01 and control.throttle > 0.0:
                # Slight minimum throttle so you don't need to floor it to get rolling
                control.throttle = max(control.throttle, 0.18)

            # Apply control to hero
            ego.apply_control(control)

            # Log this frame
            logger.log_step(control)

            # ------------------------------------------------------------
            # NPC speed logic: follow hero until hero hits 30 km/h once
            # ------------------------------------------------------------
            if not npc_locked_speed:
                if hero_speed_kmh < 1.0:
                    # Hero basically stopped: don't move the NPC at all
                    tm.set_desired_speed(npc, 0.0)
                else:
                    # Hero moving: match hero speed
                    tm.set_desired_speed(npc, hero_speed_kmh)

                if hero_speed_kmh >= MAX_SPEED:
                    npc_locked_speed = True
                    tm.set_desired_speed(npc, MAX_SPEED)
                    print(f"NPC max speed locked at {MAX_SPEED} km/h")
            else:
                tm.set_desired_speed(npc, MAX_SPEED)

            # ---- Smooth chase-camera spectator (behind ego) ----
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation

            follow_distance = 12.0  # meters behind the car
            height = 6.0            # meters above the car

            yaw_rad = math.radians(ego_rot.yaw)
            # Position behind ego based on yaw
            target_loc = carla.Location(
                x=ego_loc.x - follow_distance * math.cos(yaw_rad),
                y=ego_loc.y - follow_distance * math.sin(yaw_rad),
                z=ego_loc.z + height,
            )
            target_yaw = ego_rot.yaw

            # Simple exponential smoothing
            alpha = 0.18  # smaller = smoother, larger = more responsive

            spec_loc.x = spec_loc.x + alpha * (target_loc.x - spec_loc.x)
            spec_loc.y = spec_loc.y + alpha * (target_loc.y - spec_loc.y)
            spec_loc.z = spec_loc.z + alpha * (target_loc.z - spec_loc.z)

            # Yaw wrap handling (avoid spinning the long way around)
            yaw_diff = (target_yaw - spec_yaw + 180.0) % 360.0 - 180.0
            spec_yaw = spec_yaw + alpha * yaw_diff

            spectator.set_transform(
                carla.Transform(
                    spec_loc,
                    carla.Rotation(pitch=-15.0, yaw=spec_yaw, roll=0.0),
                )
            )

            # ----- Draw latest camera image if available -----
            if latest_image is not None:
                # Avoid mirror/rotate weirdness: swap axes like CARLA examples
                surf = pygame.surfarray.make_surface(latest_image.swapaxes(0, 1))
                screen.blit(surf, (0, 0))

                speed_text = hud_font.render(f"{hero_speed_kmh:5.1f} km/h", True, (255, 255, 255))
                screen.blit(speed_text, (10, 10))

            pygame.display.flip()
            clock.tick(30)

    finally:
        print("Exiting control loop, cleaning up...")

        try:
            logger.close()
        except Exception:
            pass

        try:
            camera.stop()
        except Exception:
            pass

        try:
            ego.destroy()
        except Exception:
            pass

        try:
            npc.destroy()
        except Exception:
            pass

        actors = world.get_actors()
        for actor in actors:
            if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
                try:
                    actor.destroy()
                except Exception:
                    pass

        pygame.quit()
        print("Cleaned up vehicles, sensors, logger, and pygame.")


if __name__ == "__main__":
    main()
