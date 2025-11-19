import sys
import math
import time

import numpy as np
import pygame

# -------------------------------------------------------------------
# Make sure CARLA PythonAPI is on the path (adjust if your path differs)
# -------------------------------------------------------------------
sys.path.append(r"C:\Users\natha\Downloads\CARLA_0.9.16\PythonAPI")
sys.path.append(r"C:\Users\natha\Downloads\CARLA_0.9.16\PythonAPI\carla\dist")

import carla  # noqa: E402


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_speed(vehicle: carla.Vehicle) -> float:
    """Return vehicle speed in m/s."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def connect_and_load_world() -> tuple[carla.Client, carla.World]:
    """Connect to CARLA and load Town06 with a generous timeout."""
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)  # seconds

    print("Connecting to CARLA and loading Town06...")
    world = client.load_world("Town06")
    print("World loaded, map name:", world.get_map().name)

    # Let the simulator tick a few times to settle
    for _ in range(10):
        world.wait_for_tick()

    return client, world


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
            actor.destroy()
    print("Cleanup complete.")

    # ----------------------------------------------------------------
    # Set spectator top-down
    # ----------------------------------------------------------------
    top_loc = carla.Location(x=116, y=244.485397, z=100.0)
    top_rot = carla.Rotation(pitch=270.0, yaw=270.0, roll=0.0)
    spectator.set_transform(carla.Transform(top_loc, top_rot))

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
    # Traffic Manager and autopilot
    # ----------------------------------------------------------------
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    npc.set_autopilot(True, tm_port)   # NPC uses Traffic Manager
    ego.set_autopilot(False)           # Ego is manual

    print("Traffic Manager port:", tm_port)

    # ----------------------------------------------------------------
    # Attach camera to ego (driver POV)
    # ----------------------------------------------------------------
    W, H = 800, 450

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

    try:
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Hero POV")
        
        # Font for speed display (HUD)
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
    print("Click on the Hero POV window and use WASD to drive, ESC to quit.")
    MAX_SPEED = 30.0  # km/h
    npc_locked_speed = False
    running = True

    try:
        # ------------------------------------------------------------
        # Main control loop
        # ------------------------------------------------------------
        while running:
            # Safety: if pygame/display is gone, exit cleanly
            if not pygame.get_init() or not pygame.display.get_surface():
                print("Pygame window closed — stopping loop.")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUIT event received.")
                    running = False

            keys = pygame.key.get_pressed()

            throttle = 0.0
            brake = 0.0
            steer = 0.0

            if keys[pygame.K_w]:
                throttle = 0.6
            if keys[pygame.K_s]:
                brake = 1.0
            if keys[pygame.K_a]:
                steer = -0.4
            if keys[pygame.K_d]:
                steer = 0.4
            if keys[pygame.K_ESCAPE]:
                print("ESC pressed, exiting loop.")
                running = False

            # Apply control to hero
            ego.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                )
            )

            # ------------------------------------------------------------
            # NPC speed logic: follow hero until hero hits 30 km/h once
            # ------------------------------------------------------------
            hero_speed = get_speed(ego)          # m/s
            hero_speed_kmh = hero_speed * 3.6    # km/h

            if not npc_locked_speed:
                # Until hero hits 30 km/h, NPC matches hero speed
                tm.set_desired_speed(npc, hero_speed_kmh)

                # Once hero reaches or exceeds 30 km/h, lock NPC at 30 forever
                if hero_speed_kmh >= MAX_SPEED:
                    npc_locked_speed = True
                    tm.set_desired_speed(npc, MAX_SPEED)
                    print(f"NPC max speed locked at {MAX_SPEED} km/h")
            else:
                # After lock: always keep NPC at 30 km/h
                tm.set_desired_speed(npc, MAX_SPEED)


            # Draw latest camera image if available
            if latest_image is not None:
                surf = pygame.surfarray.make_surface(np.rot90(latest_image))
                screen.blit(surf, (0, 0))

            pygame.display.flip()
            clock.tick(30)  # ~30 FPS

    finally:
        # ------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------
        print("Exiting control loop, cleaning up...")

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

        # Optional: also destroy any leftover vehicles/sensors
        actors = world.get_actors()
        for actor in actors:
            if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
                try:
                    actor.destroy()
                except Exception:
                    pass

        pygame.quit()
        print("Cleaned up vehicles, sensors, and pygame.")


if __name__ == "__main__":
    main()