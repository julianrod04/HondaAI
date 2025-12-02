# carla_parquet_logger.py
#
# Parquet logger for CARLA simulations.
#
# Usage from your main script:
#
#   from carla_parquet_logger import CarlaParquetLogger
#
#   logger = CarlaParquetLogger(world, ego_vehicle, npc_vehicle, log_dir="logs")
#   ...
#   while running:
#       control = get_keyboard_control(...)
#       ego_vehicle.apply_control(control)
#       logger.log_step(control)
#   ...
#   logger.close()
#
# Requires:
#   pip install pyarrow

import math
import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class CarlaParquetLogger:
    """
    Owns:
      - in-memory buffers for each logged column
      - Parquet writing at the end of the run

    You:
      - create one instance per run
      - call log_step(control) once per frame
      - call close() at the end
    """

    def __init__(self, world, ego_vehicle, npc_vehicle=None, log_dir="logs", run_id=None):
        """
        Parameters
        ----------
        world : carla.World
        ego_vehicle : carla.Vehicle
        npc_vehicle : carla.Vehicle or None
        log_dir : str
            Directory in which Parquet file will be written.
        run_id : str or None
            Optional identifier to include in the filename.
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.npc_vehicle = npc_vehicle

        # Prepare output path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id is None:
            filename = f"carla_run_{timestamp_str}.parquet"
        else:
            filename = f"carla_run_{run_id}_{timestamp_str}.parquet"

        self.file_path = self.log_dir / filename

        # In-memory column buffers; keys are column names, values are Python lists
        self._cols = {
            "frame": [],
            "sim_time": [],

            "ego_x": [],
            "ego_y": [],
            "ego_z": [],
            "ego_yaw": [],
            "ego_vx": [],
            "ego_vy": [],
            "ego_vz": [],
            "ego_speed": [],

            "ego_throttle": [],
            "ego_steer": [],
            "ego_brake": [],
            "ego_reverse": [],
            "ego_hand_brake": [],

            "npc_present": [],
            "npc_x": [],
            "npc_y": [],
            "npc_z": [],
            "npc_yaw": [],
            "npc_vx": [],
            "npc_vy": [],
            "npc_vz": [],
            "npc_speed": [],

            "distance_ego_npc": [],
        }

        self._closed = False
        print(f"[CarlaParquetLogger] Logging to: {self.file_path}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def log_step(self, control):
        """
        Log one simulation frame.

        Parameters
        ----------
        control : carla.VehicleControl
            The control applied to the ego vehicle this frame.
        """
        if self._closed:
            # Silently ignore or raise; here we raise to catch misuse
            raise RuntimeError("CarlaParquetLogger.log_step() called after close().")

        world = self.world
        ego_vehicle = self.ego_vehicle
        npc_vehicle = self.npc_vehicle

        # Snapshot / timing
        snapshot = world.get_snapshot()
        sim_time = snapshot.timestamp.elapsed_seconds
        frame_id = snapshot.frame

        # Ego state
        ego_tf = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_rot = ego_tf.rotation
        ego_vel = ego_vehicle.get_velocity()

        ego_speed = math.sqrt(
            ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2
        )

        # Controls
        ego_throttle = float(control.throttle)
        ego_steer = float(control.steer)
        ego_brake = float(control.brake)
        ego_reverse_flag = bool(control.reverse)
        ego_hand_brake_flag = bool(control.hand_brake)

        # NPC state
        npc_present = npc_vehicle is not None
        if npc_present:
            npc_tf = npc_vehicle.get_transform()
            npc_loc = npc_tf.location
            npc_rot = npc_tf.rotation
            npc_vel = npc_vehicle.get_velocity()

            npc_x = float(npc_loc.x)
            npc_y = float(npc_loc.y)
            npc_z = float(npc_loc.z)
            npc_yaw = float(npc_rot.yaw)
            npc_vx = float(npc_vel.x)
            npc_vy = float(npc_vel.y)
            npc_vz = float(npc_vel.z)
            npc_speed = math.sqrt(npc_vx ** 2 + npc_vy ** 2 + npc_vz ** 2)

            dx = ego_loc.x - npc_loc.x
            dy = ego_loc.y - npc_loc.y
            distance_ego_npc = math.sqrt(dx ** 2 + dy ** 2)
        else:
            npc_x = npc_y = npc_z = 0.0
            npc_yaw = 0.0
            npc_vx = npc_vy = npc_vz = 0.0
            npc_speed = 0.0
            distance_ego_npc = -1.0  # sentinel meaning "no npc"

        # Append to buffers
        self._cols["frame"].append(int(frame_id))
        self._cols["sim_time"].append(float(sim_time))

        self._cols["ego_x"].append(float(ego_loc.x))
        self._cols["ego_y"].append(float(ego_loc.y))
        self._cols["ego_z"].append(float(ego_loc.z))
        self._cols["ego_yaw"].append(float(ego_rot.yaw))
        self._cols["ego_vx"].append(float(ego_vel.x))
        self._cols["ego_vy"].append(float(ego_vel.y))
        self._cols["ego_vz"].append(float(ego_vel.z))
        self._cols["ego_speed"].append(float(ego_speed))

        self._cols["ego_throttle"].append(ego_throttle)
        self._cols["ego_steer"].append(ego_steer)
        self._cols["ego_brake"].append(ego_brake)
        self._cols["ego_reverse"].append(ego_reverse_flag)
        self._cols["ego_hand_brake"].append(ego_hand_brake_flag)

        self._cols["npc_present"].append(bool(npc_present))
        self._cols["npc_x"].append(float(npc_x))
        self._cols["npc_y"].append(float(npc_y))
        self._cols["npc_z"].append(float(npc_z))
        self._cols["npc_yaw"].append(float(npc_yaw))
        self._cols["npc_vx"].append(float(npc_vx))
        self._cols["npc_vy"].append(float(npc_vy))
        self._cols["npc_vz"].append(float(npc_vz))
        self._cols["npc_speed"].append(float(npc_speed))

        self._cols["distance_ego_npc"].append(float(distance_ego_npc))

    def close(self):
        """
        Convert the buffers to a PyArrow Table and write to a Parquet file.
        Safe to call multiple times, but only writes once.
        """
        if self._closed:
            return

        # Build Arrow arrays from column buffers
        arrays = {}
        for name, data in self._cols.items():
            arrays[name] = pa.array(data)

        table = pa.table(arrays)

        # Optionally, you could define an explicit schema for full control.
        pq.write_table(table, self.file_path)
        self._closed = True
        print(f"[CarlaParquetLogger] Wrote Parquet with {table.num_rows} rows to: {self.file_path}")

    # Optional context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
