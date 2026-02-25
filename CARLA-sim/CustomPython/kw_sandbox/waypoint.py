import carla
import random

def find_adjacent_lane_waypoint(world_map, ego_spawn_transform):
    """
    Find a driving lane directly to the right of the ego vehicle if possible.
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

def find_random_waypoint(world, max_tries=50):
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available.")

    for _ in range(max_tries):
        spawn_point = random.choice(spawn_points)
        actor = world.try_spawn_actor(
            world.get_blueprint_library().filter("vehicle.*")[0],
            spawn_point
        )
        if actor is not None:
            actor.destroy()
            return spawn_point

    raise RuntimeError("Could not find a free spawn point.")

def find_nearest_npc_vehicle(world, ego_vehicle, max_distance=None):
    """
    Return the nearest non-ego vehicle actor to ego_vehicle.
    If max_distance is set, returns None when no vehicle is within that radius.
    """
    ego_loc = ego_vehicle.get_location()
    vehicles = world.get_actors().filter("vehicle.*")

    nearest = None
    nearest_dist = float("inf")

    for v in vehicles:
        if v.id == ego_vehicle.id:
            continue
        d = ego_loc.distance(v.get_location())
        if d < nearest_dist:
            nearest = v
            nearest_dist = d

    if max_distance is not None and nearest_dist > max_distance:
        return None
    return nearest

def get_relative_lane_and_longitudinal(world_map, reference_vehicle, other_vehicle):
    """
    Determine whether other_vehicle is in the left/right/same lane as reference_vehicle,
    and whether it is ahead/behind along the reference lane direction.
    Returns (lane_relation, longitudinal_relation, distance_m).
    """
    ref_loc = reference_vehicle.get_location()
    other_loc = other_vehicle.get_location()

    ref_wp = world_map.get_waypoint(
        ref_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    other_wp = world_map.get_waypoint(
        other_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    # Lane relation: same road/section and lane_id adjacency
    if (ref_wp.road_id == other_wp.road_id and
        ref_wp.section_id == other_wp.section_id):
        if other_wp.lane_id == ref_wp.lane_id:
            lane_relation = "same"
        elif other_wp.lane_id > ref_wp.lane_id:
            lane_relation = "left"
        elif other_wp.lane_id < ref_wp.lane_id:
            lane_relation = "right"
        else:
            lane_relation = "other"
    else:
        lane_relation = "other"

    # Longitudinal relation: project vector onto reference lane forward direction
    ref_forward = ref_wp.transform.get_forward_vector()
    dx = other_loc.x - ref_loc.x
    dy = other_loc.y - ref_loc.y
    dz = other_loc.z - ref_loc.z
    dot = dx * ref_forward.x + dy * ref_forward.y + dz * ref_forward.z
    if dot > 0:
        longitudinal_relation = "ahead"
    elif dot < 0:
        longitudinal_relation = "behind"
    else:
        longitudinal_relation = "side_by_side"

    distance_m = ref_loc.distance(other_loc)
    return lane_relation, longitudinal_relation, distance_m