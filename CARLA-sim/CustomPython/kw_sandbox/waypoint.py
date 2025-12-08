import carla

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