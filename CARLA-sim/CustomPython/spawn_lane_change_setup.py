#!/usr/bin/env python3
# spawn_lane_change_setup.py
#
# Robust spawner for lane-change scenarios:
# - Ego at given coords (or default) with optional lane yaw alignment
# - NPC in adjacent lane (left/right/auto) at same longitudinal position (or +/- npc_gap)
# - Blueprint fallback: if requested vehicle ID is missing, pick a sensible available one
#
# Usage examples:
#   python3 spawn_lane_change_setup.py --x 20 --y 248 --z 0.5 --yaw 90 --ego vehicle.tesla.model3
#   python3 spawn_lane_change_setup.py --npc-side left --npc_gap 10
#   python3 spawn_lane_change_setup.py --town Town06 --align-to-lane --ego vehicle.audi.tt --npc vehicle.mustang.mustang

import argparse
import math
import time
from typing import Optional

import carla


# ---------------------------- helpers ----------------------------

def set_sync(world, delta=0.05):
    """Enable synchronous mode for deterministic steps."""
    settings = world.get_settings()
    orig = carla.WorldSettings(
        no_rendering_mode=settings.no_rendering_mode,
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds
    )
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    return orig


def magnitude(v): 
    return (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5


def project_to_lane(world, loc, align_yaw_to_lane=True):
    """
    Snap a location to the nearest driving-lane waypoint.
    If align_yaw_to_lane: use lane's rotation; else keep yaw as-is later.
    """
    m = world.get_map()
    wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    tf = carla.Transform(wp.transform.location, wp.transform.rotation if align_yaw_to_lane else carla.Rotation())
    return wp, tf


def choose_vehicle_blueprint(world, preferred: Optional[str] = None, role_name: Optional[str] = None):
    """
    Return a vehicle blueprint:
      - If 'preferred' exists, use it.
      - Else choose a reasonable fallback (4-wheeled, drivable).
    """
    lib = world.get_blueprint_library()
    bp = None
    if preferred:
        try:
            bp = lib.find(preferred)
        except IndexError:
            bp = None

    if bp is None:
        # Filter to common 4-wheeled vehicles (avoid bikes/2-wheelers if present)
        candidates = [b for b in lib.filter("vehicle.*") if b.id.count('.') >= 2]
        # Prefer sedans/compacts if available
        priority_substrings = ["tesla.model3", "audi.tt", "mercedes", "lincoln", "nissan", "mini", "volkswagen", "seat"]
        def score(b):
            name = b.id.lower()
            for i, key in enumerate(priority_substrings):
                if key in name:
                    return i
            return 999
        candidates.sort(key=score)
        if not candidates:
            raise RuntimeError("No vehicle blueprints found in this CARLA build.")
        bp = candidates[0]

    if role_name:
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role_name)
        else:
            # Role name is not a standard attribute on vehicle bps, but we can attach it as a 'fake' attribute via tags.
            try:
                bp.set_attribute("role_name", role_name)
            except Exception:
                pass

    # Give a random color if supported (helps distinguish cars)
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", colors[0])
    return bp


def try_spawn(world, bp, transform, attempts=6, z_lift=0.5):
    """
    Try to spawn with small vertical nudges if the spot is blocked.
    """
    tf = carla.Transform(transform.location, transform.rotation)
    tf.location.z = max(tf.location.z, z_lift)
    veh = None
    for i in range(attempts):
        veh = world.try_spawn_actor(bp, tf)
        if veh is not None:
            return veh
        # nudge Z a bit if blocked
        tf.location.z += 0.1
    raise RuntimeError(f"Failed to spawn {bp.id} after {attempts} attempts at {transform}")


# ---------------------------- main logic ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town06", help="Map to load (default: Town06)")

    # Ego pose (optional)
    ap.add_argument("--x", type=float, help="Desired X")
    ap.add_argument("--y", type=float, help="Desired Y")
    ap.add_argument("--z", type=float, default=0.5, help="Desired Z (m)")
    ap.add_argument("--yaw", type=float, help="Desired yaw (deg, 0=+X/East)")

    ap.add_argument("--align-to-lane", action="store_true", help="Align ego yaw to lane direction after snapping")
    ap.add_argument("--ego", default=None, help="Ego blueprint id (e.g., vehicle.tesla.model3); fallback used if missing")
    ap.add_argument("--npc", default=None, help="NPC blueprint id; fallback used if missing")

    # Adjacent-lane placement
    ap.add_argument("--npc-side", choices=["left", "right", "auto"], default="auto",
                    help="Adjacent lane for NPC (auto tries right then left)")
    ap.add_argument("--npc_gap", type=float, default=0.0,
                    help="Longitudinal gap (m) for NPC relative to ego (+ahead, -behind)")
    args = ap.parse_args()

    # Connect client
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Load map if needed
    try:
        current = world.get_map().name.split("/")[-1]
    except Exception:
        current = None
    if current != args.town:
        print(f"[map] loading {args.town} (was {current})")
        world = client.load_world(args.town)
        time.sleep(0.5)

    # Sync mode
    orig_settings = set_sync(world, delta=0.05)

    # Ego transform
    if args.x is not None and args.y is not None and args.yaw is not None:
        raw_tf = carla.Transform(carla.Location(args.x, args.y, args.z),
                                 carla.Rotation(yaw=args.yaw))
        ego_wp, ego_tf = project_to_lane(world, raw_tf.location, align_yaw_to_lane=args.align_to_lane)
        if not args.align_to_lane:
            # keep user's yaw if not aligning
            ego_tf.rotation = raw_tf.rotation
    else:
        sp = world.get_map().get_spawn_points()
        if not sp:
            raise RuntimeError("Map has no spawn points.")
        # pick a spawn point and snap/align to lane
        ego_wp, ego_tf = project_to_lane(world, sp[0].location, align_yaw_to_lane=True)

    # Ego blueprint (robust)
    ego_bp = choose_vehicle_blueprint(world, preferred=args.ego, role_name="hero")
    ego = try_spawn(world, ego_bp, ego_tf)
    print(f"[ego] {ego.type_id} @ {ego.get_transform()}")

    # Choose adjacent lane for NPC
    side_order = {"left": ["left", "right"],
                  "right": ["right", "left"],
                  "auto": ["right", "left"]}[args.npc_side]

    npc_wp = None
    npc_side_used = None
    for side in side_order:
        cand = ego_wp.get_left_lane() if side == "left" else ego_wp.get_right_lane()
        if cand and cand.lane_type == carla.LaneType.Driving:
            npc_wp = cand
            npc_side_used = side
            break

    if npc_wp is None:
        # Fallback: create a lateral offset within same lane (not ideal, but better than failing)
        print("[npc] No adjacent lane. Using same-lane lateral offset of -1.5 m.")
        yaw_lane = math.radians(ego_wp.transform.rotation.yaw)
        n_hat = carla.Vector3D(-math.sin(yaw_lane), math.cos(yaw_lane), 0.0)
        npc_loc = ego_tf.location + n_hat * (-1.5)
        npc_tf = carla.Transform(npc_loc, ego_tf.rotation)
    else:
        # Apply longitudinal gap along NPC lane if requested
        if abs(args.npc_gap) > 0.01:
            if args.npc_gap > 0:
                wps = npc_wp.next(args.npc_gap)
                if wps: npc_wp = wps[0]
            else:
                wps = npc_wp.previous(abs(args.npc_gap))
                if wps: npc_wp = wps[0]
        npc_tf = npc_wp.transform
        print(f"[npc] Using {npc_side_used} adjacent lane.")

    # NPC blueprint (robust)
    npc_bp = choose_vehicle_blueprint(world, preferred=args.npc, role_name="npc")
    npc = try_spawn(world, npc_bp, npc_tf)
    print(f"[npc] {npc.type_id} @ {npc.get_transform()}")

    print("\nAll set:")
    print(f"  Ego (role=hero): {ego.type_id}")
    print(f"  NPC (role=npc) : {npc.type_id}")
    print("Tip: run `manual_control.py --sync` and take over the ego.")

    # keep alive a moment so you can see them before other scripts connect
    try:
        for _ in range(50):
            world.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # Uncomment to auto-clean on exit:
        # for a in (npc, ego):
        #     if a and a.is_alive:
        #         a.destroy()
        # world.apply_settings(orig_settings)
        pass


if __name__ == "__main__":
    main()

