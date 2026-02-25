import time
import carla

def npc_brake(npc_car, event_state, duration=3):
    control = npc_car.get_control()
    control.brake = 1
    event_state.start_npc_override(npc_car, control, duration)

def npc_swerve_left(npc_car, event_state, duration=3):
    control = npc_car.get_control()
    control.steer = -1  #do we want to make this value a parameter?
    event_state.start_npc_override(npc_car, control, duration)

def npc_swerve_right(npc_car, event_state, duration=3):
    control = npc_car.get_control()
    control.steer = 1   #parametrize
    event_state.start_npc_override(npc_car, control, duration)

def npc_accelerate(npc_car, event_state, duration=3):
    control = npc_car.get_control()
    control.throttle = 1    #definitely want to parametrize this
    event_state.start_npc_override(npc_car, control, duration)

behaviors = {
    "brake": npc_brake,
    "swerve_left": npc_swerve_left,
    "swerve_right": npc_swerve_right,
    "accelerate": npc_accelerate,
}

def select_behavior(npc_vehicle, lane_relative_to_ego, ahead_or_behind_ego, event_state):
    if lane_relative_to_ego == "same" and ahead_or_behind_ego == "ahead":
        event_state.behaviors["brake"](npc_vehicle, duration=3)
    elif lane_relative_to_ego ==  "left" and ahead_or_behind_ego == "ahead":
        event_state.behaviors["swerve_right"](npc_vehicle, duration=3)
    elif lane_relative_to_ego == "right" and ahead_or_behind_ego == "ahead":
        event_state.behaviors["swerve_left"](npc_vehicle, duration=3)
    elif lane_relative_to_ego == ("left" or "right") and ahead_or_behind_ego == "behind":
        event_state.behaviors["accelerate"](npc_vehicle, duration=3)