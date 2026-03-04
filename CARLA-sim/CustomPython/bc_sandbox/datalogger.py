#!/usr/bin/env python3
import csv, math, time, signal, datetime, os, json
import carla

def mag(v): return (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5

def main():
    # ----- graceful stop wiring -----
    stop = {"now": False}
    def _stop_handler(sig, frame):
        print("\n[datalogger] received signal, stopping gracefully...")
        stop["now"] = True
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    # ----- timestamp & folder setup -----
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    folder = f"DataLogs/{date_str}"
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, f"carla_log_{date_str}_{time_str}.csv")

    # ----- connect to CARLA -----
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()

    # save and modify settings
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # find ego (vehicle spawned by manual_control.py)
    ego = None
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name", "") == "hero":
            ego = a
            break
    if ego is None:
        raise RuntimeError("No ego vehicle with role_name=hero found. Start manual_control.py first.")

    # lane invasion sensor
    lane_evt = {"markings": ""}
    bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
    lane_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    def on_lane(evt):
        types = sorted(set(str(x.type).split(".")[-1] for x in evt.crossed_lane_markings))
        lane_evt["markings"] = "|".join(types)
    lane_sensor.listen(on_lane)

    # ----- metadata -----
    meta = {
        "date": date_str,
        "time_start": time_str,
        "map": carla_map.name.split("/")[-1],
        "vehicle": ego.type_id,
        "notes": "Logged via datalogger.py"
    }
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # CSV setup
    fields = [
        "frame","sim_time",
        "x","y","z","yaw_deg","speed_mps","yawrate_rps",
        "throttle","brake","steer","reverse","gear","hand_brake",
        "lane_id","offset_m","dist_left_m","dist_right_m",
        "lane_event","map"
    ]

    try:
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()

            print(f"[datalogger] started {date_str} {time_str}")
            print(f"[datalogger] logging to {fname}")

            while not stop["now"]:
                try:
                    world.tick()
                except RuntimeError:
                    print("[datalogger] world.tick() failed (server likely closed). Exiting.")
                    break

                tf = ego.get_transform()
                vel = ego.get_velocity()
                ang = ego.get_angular_velocity()
                ctrl = ego.get_control()

                wp = carla_map.get_waypoint(tf.location, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                lane_yaw = math.radians(wp.transform.rotation.yaw)
                n_hat = carla.Vector3D(-math.sin(lane_yaw), math.cos(lane_yaw), 0.0)

                dx = tf.location.x - wp.transform.location.x
                dy = tf.location.y - wp.transform.location.y
                offset = dx * n_hat.x + dy * n_hat.y
                half_w = 0.5 * wp.lane_width
                dist_left  = half_w + offset
                dist_right = half_w - offset

                ts = world.get_snapshot().timestamp
                w.writerow(dict(
                    frame=ts.frame, sim_time=f"{ts.elapsed_seconds:.3f}",
                    x=f"{tf.location.x:.3f}", y=f"{tf.location.y:.3f}", z=f"{tf.location.z:.3f}",
                    yaw_deg=f"{tf.rotation.yaw:.3f}",
                    speed_mps=f"{mag(vel):.3f}",
                    yawrate_rps=f"{ang.z:.4f}",
                    throttle=f"{ctrl.throttle:.3f}",
                    brake=f"{ctrl.brake:.3f}",
                    steer=f"{ctrl.steer:.3f}",
                    reverse=int(ctrl.reverse),
                    gear=ctrl.gear,
                    hand_brake=int(ctrl.hand_brake),
                    lane_id=wp.lane_id,
                    offset_m=f"{offset:.3f}",
                    dist_left_m=f"{dist_left:.3f}",
                    dist_right_m=f"{dist_right:.3f}",
                    lane_event=lane_evt["markings"],
                    map=carla_map.name.split("/")[-1]
                ))
                lane_evt["markings"] = ""
                f.flush()

    finally:
        try:
            if lane_sensor and lane_sensor.is_alive:
                lane_sensor.stop()
                lane_sensor.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original)
        except Exception:
            pass
        print(f"[datalogger] closed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[datalogger] CSV saved to: {fname}")

if __name__ == "__main__":
    main()

