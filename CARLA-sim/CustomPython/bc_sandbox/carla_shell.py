#!/usr/bin/env python3
"""
CARLA Interactive Shell (persistent)
------------------------------------
A small, persistent REPL that stays connected to a CARLA server and accepts
friendly commands until Ctrl+C or 'exit'.

What you can do:
- hero xyzyaw X Y Z YAW         : spawn or move the hero to absolute pose
- hero spawn_idx N               : spawn or move hero to map spawn point N
- npc spawn left|right [--gap m] [--lat m] [--bp vehicle.audi.tt]
- npc speed S                    : set straight constant speed (m/s) via set_target_velocity
- npc stop                       : zero speed (and disable autopilot)
- npc tm on  [--kmh K] [--ignore]: enable Traffic Manager lane-following at K km/h
- npc tm off                     : disable TM autopilot
- log start hero|both [--dir DataLogs] : start CSV logging
- log stop                       : stop logging
- list spawns                    : list map spawn points
- cleanup [--role-name NAME]     : destroy vehicles with role_name (default npc_neighbor)
- kill ID [ID ...]               : destroy specific actor ids
- sync on|off [--dt 0.05]        : toggle synchronous mode
- status                         : print current actors and speeds
- help                           : show this help
- exit / Ctrl+D                  : quit

Notes
- Works alongside manual_control.py (does not force sync unless you turn it on).
- Default hero role_name='hero', NPC role_name='npc_neighbor'.
- Straight-speed mode keeps the NPC's yaw fixed and sets a world velocity.
"""

from __future__ import annotations
import argparse
import cmd
import csv
import math
import os
import shlex
import signal
import sys
import threading
import time
import datetime
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import carla
    from carla import command as cmdc
except Exception as e:
    print("[carla_shell] Failed to import CARLA API. Ensure PYTHONPATH is set to CARLA/PythonAPI.")
    raise

ROLE_HERO = "hero"
ROLE_NPC  = "npc_neighbor"

# --------------------------- helpers ----------------------------

def right_vector_from_yaw_deg(yaw_deg: float) -> Tuple[float, float]:
    # UE/CARLA axes: X forward, Y right; yaw clockwise from +X.
    yaw = math.radians(yaw_deg)
    return -math.sin(yaw), math.cos(yaw)


def transform_from_xyzyaw(x: float, y: float, z: float, yaw_deg: float) -> carla.Transform:
    return carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z)),
                           carla.Rotation(pitch=0.0, yaw=float(yaw_deg), roll=0.0))


def lateral_offset_transform(base_tf: carla.Transform, meters_right: float) -> carla.Transform:
    rx, ry = right_vector_from_yaw_deg(base_tf.rotation.yaw)
    loc = base_tf.location + carla.Location(x=rx*meters_right, y=ry*meters_right, z=0.0)
    return carla.Transform(loc, base_tf.rotation)


def adjacent_lane_transform(carla_map: carla.Map, ego_tf: carla.Transform, side: str, gap_ahead: float) -> Optional[carla.Transform]:
    ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    side_wp = ego_wp.get_right_lane() if side == 'right' else ego_wp.get_left_lane()
    if side_wp is None or side_wp.lane_type != carla.LaneType.Driving:
        return None
    s_remaining = float(gap_ahead)
    step = 1.0 if s_remaining >= 0.0 else -1.0
    wp = side_wp
    while abs(s_remaining) > 1e-6:
        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]
        s_remaining -= step
    return wp.transform


def speed_of(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


# --------------------------- logger ----------------------------
class CsvLogger:
    def __init__(self, world: carla.World, carla_map: carla.Map, root: Path, mode: str, hero: Optional[carla.Actor], npc: Optional[carla.Actor]):
        self.world = world
        self.map = carla_map
        self.root = root
        self.mode = mode  # 'hero' or 'both'
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        tstr = datetime.datetime.now().strftime('%H-%M-%S')
        folder = root / date
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"carla_log_{date}_{tstr}.csv"
        self.f = open(self.path, 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=[
            'actor_tag','actor_id','frame','sim_time','x','y','z','yaw_deg','speed_mps','yawrate_rps',
            'throttle','brake','steer','reverse','gear','hand_brake','lane_id','offset_m','dist_left_m','dist_right_m','map'
        ])
        self.w.writeheader()
        self.hero = hero
        self.npc = npc
        self.lock = threading.Lock()
        self.active = True
        self.cb = self.world.on_tick(self._on_tick)
        print(f"[log] writing to {self.path}")

    def _on_tick(self, snapshot: carla.WorldSnapshot):
        if not self.active: return
        with self.lock:
            try:
                if self.hero:
                    self._log_one(self.hero, 'hero', snapshot)
                if self.mode == 'both' and self.npc:
                    self._log_one(self.npc, 'npc', snapshot)
                self.f.flush()
            except Exception:
                # Don't crash the tick thread; best-effort logging
                pass

    def _log_one(self, actor: carla.Actor, tag: str, snap: carla.WorldSnapshot):
        tf = actor.get_transform()
        vel = actor.get_velocity()
        ang = actor.get_angular_velocity()
        ctrl = actor.get_control() if hasattr(actor, 'get_control') else None
        wp = self.map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_yaw = math.radians(wp.transform.rotation.yaw)
        n_hat = carla.Vector3D(-math.sin(lane_yaw), math.cos(lane_yaw), 0.0)
        dx = tf.location.x - wp.transform.location.x
        dy = tf.location.y - wp.transform.location.y
        offset = dx*n_hat.x + dy*n_hat.y
        half_w = 0.5 * wp.lane_width
        dist_left, dist_right = half_w + offset, half_w - offset
        self.w.writerow(dict(
            actor_tag=tag, actor_id=actor.id, frame=snap.frame, sim_time=f"{snap.timestamp.elapsed_seconds:.3f}",
            x=f"{tf.location.x:.3f}", y=f"{tf.location.y:.3f}", z=f"{tf.location.z:.3f}", yaw_deg=f"{tf.rotation.yaw:.3f}",
            speed_mps=f"{math.sqrt(vel.x*vel.x+vel.y*vel.y+vel.z*vel.z):.3f}", yawrate_rps=f"{ang.z:.4f}",
            throttle=f"{getattr(ctrl,'throttle',0.0):.3f}", brake=f"{getattr(ctrl,'brake',0.0):.3f}", steer=f"{getattr(ctrl,'steer',0.0):.3f}",
            reverse=int(getattr(ctrl,'reverse',False)), gear=getattr(ctrl,'gear',0), hand_brake=int(getattr(ctrl,'hand_brake',False)),
            lane_id=wp.lane_id, offset_m=f"{offset:.3f}", dist_left_m=f"{dist_left:.3f}", dist_right_m=f"{dist_right:.3f}",
            map=self.map.name.split('/')[-1]
        ))

    def stop(self):
        if not self.active: return
        self.active = False
        try:
            if self.cb: self.world.remove_on_tick(self.cb)
        except Exception:
            pass
        try:
            self.f.close()
        except Exception:
            pass
        print(f"[log] closed {self.path}")


# --------------------------- REPL ----------------------------
class CarlaShell(cmd.Cmd):
    intro = "CARLA shell — type 'help' for commands, Ctrl+C to quit."
    prompt = "carla> "

    def __init__(self, host='localhost', port=2000):
        super().__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.hero: Optional[carla.Actor] = self._find_hero()
        self.npc: Optional[carla.Actor] = None
        self.logger: Optional[CsvLogger] = None
        self.sync_prev = self.world.get_settings()
        print(f"Connected to {host}:{port} — map {self.map.name.split('/')[-1]}")
        if self.hero:
            loc = self.hero.get_transform().location
            print(f"Found hero id={self.hero.id} at ({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")

    # ------------- utilities -------------
    def _find_hero(self) -> Optional[carla.Actor]:
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.attributes.get('role_name') == ROLE_HERO:
                return a
        return None

    def _spawn_vehicle(self, bp_name: str, tf: carla.Transform, role: str) -> Optional[carla.Actor]:
        bp = self.world.get_blueprint_library().find(bp_name)
        bp.set_attribute('role_name', role)
        actor = self.world.try_spawn_actor(bp, tf)
        return actor

    def _destroy_ids(self, ids: List[int]) -> int:
        try:
            res = self.client.apply_batch_sync([cmdc.DestroyActor(i) for i in ids], False)
        except Exception:
            return 0
        return sum(1 for r in (res or []) if not getattr(r, 'error', None))

    # ------------- hero commands -------------
    def do_hero(self, arg: str):
        """hero xyzyaw X Y Z YAW | hero spawn_idx N |
        hero speed S | hero stop |
        hero tm on [--kmh K] [--ignore] | hero tm off
        
        Spawn or reposition the hero actor (role_name=hero). If no hero exists, one will be
        spawned with blueprint vehicle.tesla.model3. You can also set a straight constant speed,
        stop the hero, or enable/disable Traffic Manager autopilot for the hero.
        """
        try:
            parts = shlex.split(arg)
            if not parts:
                print("usage: hero xyzyaw X Y Z YAW | hero spawn_idx N | hero speed S | hero stop | hero tm on [--kmh K] [--ignore] | hero tm off")
                return
            sub = parts[0]
            # --- pose controls ---
            if sub == 'xyzyaw' and len(parts) == 5:
                x, y, z, yaw = map(float, parts[1:])
                tf = transform_from_xyzyaw(x, y, z, yaw)
                if self.hero is None:
                    self.hero = self._spawn_vehicle('vehicle.tesla.model3', tf, ROLE_HERO)
                    if self.hero is None:
                        print("[hero] failed to spawn (spot blocked)")
                    else:
                        print(f"[hero] spawned id={self.hero.id}")
                else:
                    self.hero.set_transform(tf)
                    print(f"[hero] moved id={self.hero.id}")
                return
            elif sub == 'spawn_idx' and len(parts) == 2:
                idx = int(parts[1])
                sps = self.map.get_spawn_points()
                if idx < 0 or idx >= len(sps):
                    print(f"index out of range 0..{len(sps)-1}")
                    return
                tf = sps[idx]
                if self.hero is None:
                    self.hero = self._spawn_vehicle('vehicle.tesla.model3', tf, ROLE_HERO)
                    if self.hero is None:
                        print("[hero] failed to spawn (spot blocked)")
                    else:
                        print(f"[hero] spawned id={self.hero.id}")
                else:
                    self.hero.set_transform(tf)
                    print(f"[hero] moved id={self.hero.id}")
                return
            
            # --- motion controls ---
            if sub == 'speed' and len(parts) == 2:
                if self.hero is None:
                    print("[hero] no hero. Use 'hero xyzyaw ...' or 'hero spawn_idx ...'")
                    return
                S = float(parts[1])
                try:
                    self.hero.set_autopilot(False)
                except Exception:
                    pass
                yaw = math.radians(self.hero.get_transform().rotation.yaw)
                vx = S * math.cos(yaw)
                vy = S * math.sin(yaw)
                self.hero.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
                self.hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print(f"[hero] constant straight speed = {S:.2f} m/s")
                return
            elif sub == 'stop':
                if self.hero is None:
                    print("[hero] no hero.")
                    return
                try:
                    self.hero.set_autopilot(False)
                except Exception:
                    pass
                self.hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print("[hero] stopped")
                return
            elif sub == 'tm':
                if len(parts) < 2:
                    print("usage: hero tm on [--kmh K] [--ignore] | hero tm off")
                    return
                if self.hero is None:
                    print("[hero] no hero.")
                    return
                mode = parts[1]
                if mode == 'on':
                    kmh = 20.0
                    ignore = False
                    i = 2
                    while i < len(parts):
                        if parts[i] == '--kmh':
                            kmh = float(parts[i+1]); i += 2
                        elif parts[i] == '--ignore':
                            ignore = True; i += 1
                        else:
                            print(f"unknown opt {parts[i]}"); return
                    tm = self.client.get_trafficmanager()
                    self.hero.set_autopilot(True, tm.get_port())
                    try:
                        tm.set_desired_speed(self.hero, float(kmh))
                    except Exception:
                        pass
                    if ignore:
                        tm.distance_to_leading_vehicle(self.hero, 0.0)
                        tm.ignore_lights_percentage(self.hero, 100)
                        tm.ignore_signs_percentage(self.hero, 100)
                    print(f"[hero] TM on at {kmh:.1f} km/h, ignore={ignore}")
                    return
                elif mode == 'off':
                    self.hero.set_autopilot(False)
                    print("[hero] TM off")
                    return
                else:
                    print("usage: hero tm on [--kmh K] [--ignore] | hero tm off")
                    return
            else:
                print("usage: hero xyzyaw X Y Z YAW | hero spawn_idx N | hero speed S | hero stop | hero tm on [--kmh K] [--ignore] | hero tm off")
        except Exception as e:
            print(f"[hero] error: {e}")

    # ------------- NPC commands -------------
    def do_npc(self, arg: str):
        """npc spawn left|right [--gap m] [--lat m] [--bp vehicle.audi.tt]
        npc speed S | npc stop | npc tm on [--kmh K] [--ignore] | npc tm off"""
        parts = shlex.split(arg)
        if not parts:
            print(self.do_npc.__doc__)
            return
        try:
            sub = parts[0]
            if sub == 'spawn':
                if self.hero is None:
                    print("[npc] need a hero first (use 'hero ...')")
                    return
                if len(parts) < 2 or parts[1] not in ('left','right'):
                    print("usage: npc spawn left|right [--gap m] [--lat m] [--bp name]")
                    return
                side = parts[1]
                gap = 0.0
                lat = 3.5
                bp  = 'vehicle.audi.tt'
                # parse flags
                i=2
                while i < len(parts):
                    if parts[i] == '--gap':
                        gap = float(parts[i+1]); i+=2
                    elif parts[i] == '--lat':
                        lat = float(parts[i+1]); i+=2
                    elif parts[i] == '--bp':
                        bp = parts[i+1]; i+=2
                    else:
                        print(f"unknown option {parts[i]}"); return
                base_tf = self.hero.get_transform()
                tf = adjacent_lane_transform(self.map, base_tf, side, gap)
                if tf is None:
                    meters_right = lat if side=='right' else -lat
                    tf = lateral_offset_transform(base_tf, meters_right)
                tf.location.z += 0.5
                self.npc = self._spawn_vehicle(bp, tf, ROLE_NPC)
                if self.npc is None:
                    print("[npc] failed to spawn (spot blocked)")
                else:
                    print(f"[npc] spawned id={self.npc.id}")
            elif sub == 'speed':
                if self.npc is None:
                    print("[npc] no NPC. Use 'npc spawn ...'")
                    return
                S = float(parts[1])
                # disable TM if active
                try: self.npc.set_autopilot(False)
                except Exception: pass
                yaw = math.radians(self.npc.get_transform().rotation.yaw)
                vx = S*math.cos(yaw); vy = S*math.sin(yaw)
                self.npc.set_target_velocity(carla.Vector3D(vx,vy,0.0))
                self.npc.set_target_angular_velocity(carla.Vector3D(0,0,0))
                print(f"[npc] constant straight speed = {S:.2f} m/s")
            elif sub == 'stop':
                if self.npc is None:
                    print("[npc] no NPC.")
                    return
                try: self.npc.set_autopilot(False)
                except Exception: pass
                self.npc.set_target_velocity(carla.Vector3D(0,0,0))
                print("[npc] stopped")
            elif sub == 'tm':
                if len(parts) < 2:
                    print("usage: npc tm on [--kmh K] [--ignore] | npc tm off")
                    return
                if self.npc is None:
                    print("[npc] no NPC.")
                    return
                mode = parts[1]
                if mode == 'on':
                    kmh = 20.0
                    ignore = False
                    i=2
                    while i < len(parts):
                        if parts[i] == '--kmh': kmh=float(parts[i+1]); i+=2
                        elif parts[i] == '--ignore': ignore=True; i+=1
                        else: print(f"unknown opt {parts[i]}"); return
                    tm = self.client.get_trafficmanager()
                    self.npc.set_autopilot(True, tm.get_port())
                    try:
                        tm.set_desired_speed(self.npc, float(kmh))
                    except Exception:
                        pass
                    if ignore:
                        tm.distance_to_leading_vehicle(self.npc, 0.0)
                        tm.ignore_lights_percentage(self.npc, 100)
                        tm.ignore_signs_percentage(self.npc, 100)
                    print(f"[npc] TM on at {kmh:.1f} km/h, ignore={ignore}")
                elif mode == 'off':
                    self.npc.set_autopilot(False)
                    print("[npc] TM off")
                else:
                    print("usage: npc tm on [--kmh K] [--ignore] | npc tm off")
            else:
                print(self.do_npc.__doc__)
        except Exception as e:
            print(f"[npc] error: {e}")

    # ------------- logging -------------
    def do_log(self, arg: str):
        """log start hero|both [--dir PATH] | log stop"""
        parts = shlex.split(arg)
        if not parts:
            print(self.do_log.__doc__); return
        try:
            if parts[0] == 'start':
                if len(parts) < 2 or parts[1] not in ('hero','both'):
                    print("usage: log start hero|both [--dir PATH]"); return
                mode = parts[1]
                outdir = Path('DataLogs')
                i=2
                while i < len(parts):
                    if parts[i] == '--dir': outdir = Path(parts[i+1]); i+=2
                    else: print(f"unknown opt {parts[i]}"); return
                if self.logger: self.logger.stop(); self.logger=None
                self.logger = CsvLogger(self.world, self.map, outdir, mode, self.hero, self.npc)
            elif parts[0] == 'stop':
                if self.logger: self.logger.stop(); self.logger=None
                else: print('[log] not running')
            else:
                print(self.do_log.__doc__)
        except Exception as e:
            print(f"[log] error: {e}")

    # ------------- utilities -------------
    def do_list(self, arg: str):
        """list spawns"""
        parts = shlex.split(arg)
        if parts == ['spawns']:
            sps = self.map.get_spawn_points()
            print(f"{len(sps)} spawn points:")
            for i, tf in enumerate(sps):
                l, r = tf.location, tf.rotation
                print(f"  #{i:02d} (x={l.x:.1f}, y={l.y:.1f}, z={l.z:.1f}) yaw={r.yaw:.1f}")
        else:
            print(self.do_list.__doc__)

    def do_cleanup(self, arg: str):
        """cleanup [--role-name NAME] : destroy vehicles with role_name (default npc_neighbor)"""
        role = ROLE_NPC
        parts = shlex.split(arg)
        i=0
        while i < len(parts):
            if parts[i] == '--role-name': role = parts[i+1]; i+=2
            else: print(f"unknown opt {parts[i]}"); return
        actors = [a for a in self.world.get_actors().filter('vehicle.*') if a.attributes.get('role_name') == role]
        n = self._destroy_ids([a.id for a in actors])
        print(f"[cleanup] destroyed {n} vehicles with role_name={role}")

    def do_kill(self, arg: str):
        """kill ID [ID ...] : destroy specific actor ids"""
        parts = shlex.split(arg)
        if not parts:
            print(self.do_kill.__doc__); return
        ids = [int(p) for p in parts]
        n = self._destroy_ids(ids)
        print(f"[kill] destroyed {n}/{len(ids)}")

    def do_sync(self, arg: str):
        """sync on|off [--dt 0.05] : toggle synchronous mode"""
        parts = shlex.split(arg)
        if not parts or parts[0] not in ('on','off'):
            print(self.do_sync.__doc__); return
        mode = parts[0]
        dt = 0.05
        i=1
        while i < len(parts):
            if parts[i] == '--dt': dt=float(parts[i+1]); i+=2
            else: print(f"unknown opt {parts[i]}"); return
        st = self.world.get_settings()
        if mode == 'on':
            st.synchronous_mode = True
            st.fixed_delta_seconds = dt
            self.world.apply_settings(st)
            print(f"[sync] on, dt={dt}")
        else:
            # restore async
            st.synchronous_mode = False
            st.fixed_delta_seconds = None
            self.world.apply_settings(st)
            print("[sync] off")

    def do_status(self, arg: str):
        """status : print info for hero and npc"""
        if self.hero:
            tf = self.hero.get_transform(); sp = speed_of(self.hero)
            print(f"hero id={self.hero.id} at ({tf.location.x:.1f},{tf.location.y:.1f}) yaw={tf.rotation.yaw:.1f} speed={sp:.2f} m/s")
        else:
            print("hero: none")
        if self.npc:
            tf = self.npc.get_transform(); sp = speed_of(self.npc)
            print(f"npc  id={self.npc.id} at ({tf.location.x:.1f},{tf.location.y:.1f}) yaw={tf.rotation.yaw:.1f} speed={sp:.2f} m/s")
        else:
            print("npc : none")

    def do_exit(self, arg: str):
        """exit : quit the shell"""
        raise KeyboardInterrupt

    def do_EOF(self, arg: str):
        print()
        return True

    # graceful stop
    def postloop(self):
        try:
            if self.logger: self.logger.stop()
            # leave world settings as-is
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='CARLA interactive shell')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    shell = CarlaShell(host=args.host, port=args.port)

    def handle_sigint(sig, frame):
        print("\nExiting...")
        shell.postloop()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        handle_sigint(None, None)


if __name__ == '__main__':
    main()
