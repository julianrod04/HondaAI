# carla_logger.py

import csv
import datetime
import math
from pathlib import Path


class CarlaCSVLogger:
    """
    Owns:
      - the CSV file handle
      - the csv.writer
      - all per-frame logging logic

    You just call logger.log_step(control) from your main loop.
    """

    def __init__(self, world, ego_vehicle, npc_vehicle=None, log_dir="logs"):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.npc_vehicle = npc_vehicle

        # create log directory if needed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.log_dir / f"carla_run_{timestamp_str}.csv"

        # open file ONCE and keep it open
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.writer(self._file)

        # write header once
        self._writer.writerow([
            "frame",
            "sim_time",

            "ego_x", "ego_y", "ego_z",
            "ego_yaw",
            "ego_vx", "ego_vy", "ego_vz",
            "ego_speed",

            "ego_throttle", "ego_steer",
            "ego_brake", "ego_reverse", "ego_hand_brake",

            "npc_present",
            "npc_x", "npc_y", "npc_z",
            "npc_yaw",
            "npc_vx", "npc_vy", "npc_vz",
            "npc_speed",

            "distance_ego_npc"
        ])

        print(f"[CarlaCSVLogger] Logging to: {self.file_path}")

    def log_step(self, control):
        """
        Call this once per simulation step from your main loop.
        `control` is the ego vehicle's VehicleControl.
        """

        world = self.world
        ego_vehicle = self.ego_vehicle
        npc_vehicle = self.npc_vehicle

        snapshot = world.get_snapshot()
        sim_time = snapshot.timestamp.elapsed_seconds
        frame_id = snapshot.frame

        # ego state
        ego_tf = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_rot = ego_tf.rotation
        ego_vel = ego_vehicle.get_velocity()

        ego_speed = math.sqrt(
            ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2
        )

        ego_throttle = control.throttle
        ego_steer = control.steer
        ego_brake = control.brake
        ego_reverse_flag = control.reverse
        ego_hand_brake_flag = control.hand_brake

        # npc state (if exists)
        npc_present = npc_vehicle is not None
        npc_x = npc_y = npc_z = 0.0
        npc_yaw = 0.0
        npc_vx = npc_vy = npc_vz = 0.0
        npc_speed = 0.0
        distance_ego_npc = -1.0  # -1 => no NPC

        if npc_vehicle is not None:
            npc_tf = npc_vehicle.get_transform()
            npc_loc = npc_tf.location
            npc_rot = npc_tf.rotation
            npc_vel = npc_vehicle.get_velocity()

            npc_x, npc_y, npc_z = npc_loc.x, npc_loc.y, npc_loc.z
            npc_yaw = npc_rot.yaw
            npc_vx, npc_vy, npc_vz = npc_vel.x, npc_vel.y, npc_vel.z
            npc_speed = math.sqrt(
                npc_vx ** 2 + npc_vy ** 2 + npc_vz ** 2
            )

            # distance on ground plane
            dx = ego_loc.x - npc_loc.x
            dy = ego_loc.y - npc_loc.y
            distance_ego_npc = math.sqrt(dx ** 2 + dy ** 2)

        self._writer.writerow([
            frame_id,
            sim_time,

            ego_loc.x, ego_loc.y, ego_loc.z,
            ego_rot.yaw,
            ego_vel.x, ego_vel.y, ego_vel.z,
            ego_speed,

            ego_throttle, ego_steer,
            ego_brake, ego_reverse_flag, ego_hand_brake_flag,

            int(npc_present),
            npc_x, npc_y, npc_z,
            npc_yaw,
            npc_vx, npc_vy, npc_vz,
            npc_speed,

            distance_ego_npc
        ])

    def close(self):
        """Explicit cleanup."""
        if not self._file.closed:
            self._file.close()
            print(f"[CarlaCSVLogger] Closed log file: {self.file_path}")

    # Optional: let you use `with CarlaCSVLogger(...) as logger:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
