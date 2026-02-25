import carla
import math

def follow_camera(ego_vehicle):
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
    
    return carla.Transform(camera_location, camera_rotation)