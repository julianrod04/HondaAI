import time

class EventState():
    def __init__(self, traffic_manager):
        self.in_progress = False
        self.end_time = None
        self.tm = traffic_manager
        self.npc_car = None
        self.restore_behavior = None

    def npc_brake(self, npc_car, duration):
        self.tm.auto_lane_change(npc_car, False)
        self.tm.vehicle_percentage_speed_difference(npc_car, 100)
        self.restore_behavior = self.restore_npc_speed
        self.in_progress = True
        self.end_time = time.time() + duration
        self.npc_car = npc_car

    def npc_swerve_left(self, npc_car, duration):
        self.tm.auto_lane_change(npc_car, False)
        self.tm.force_lane_change(npc_car, False)
        self.restore_behavior = self.restore_lane_change
        self.in_progress = True
        self.end_time = time.time() + duration
        self.npc_car = npc_car

    def npc_swerve_right(self, npc_car, duration):
        self.tm.auto_lane_change(npc_car, False)
        self.tm.force_lane_change(npc_car, True)
        self.restore_behavior = self.restore_lane_change
        self.in_progress = True
        self.end_time = time.time() + duration
        self.npc_car = npc_car

    def npc_accelerate(self, npc_car, duration):
        self.tm.auto_lane_change(npc_car, False)
        self.tm.vehicle_percentage_speed_difference(npc_car, -60)
        self.restore_behavior = self.restore_npc_speed
        self.in_progress = True
        self.end_time = time.time() + duration
        self.npc_car = npc_car

    def restore_lane_change(self):
        self.tm.auto_lane_change(self.npc_car, True)
        self.in_progress = False
        self.override_control = None
        self.npc_car = None
    
    def restore_npc_speed(self):
        self.tm.auto_lane_change(self.npc_car, True)
        self.tm.vehicle_percentage_speed_difference(self.npc_car, 0)
        self.in_progress = False
        self.override_control = None
        self.npc_car = None

    def tick_override_event(self):
        if not self.in_progress:
            return
        if time.time() >= self.end_time:
            print("resetting npc car")
            self.restore_behavior()

    def select_behavior(self, npc_vehicle, lane_relative_to_ego, ahead_or_behind_ego):
        if lane_relative_to_ego == "same" and ahead_or_behind_ego == "ahead":
            self.npc_brake(npc_vehicle, duration=3)
        elif lane_relative_to_ego ==  "left" and ahead_or_behind_ego == "ahead":
            self.npc_swerve_right(npc_vehicle, duration=3)
        elif lane_relative_to_ego == "right" and ahead_or_behind_ego == "ahead":
            self.npc_swerve_left(npc_vehicle, duration=3)
        elif lane_relative_to_ego in ("left", "right") and ahead_or_behind_ego == "behind":
            self.npc_accelerate(npc_vehicle, duration=3)