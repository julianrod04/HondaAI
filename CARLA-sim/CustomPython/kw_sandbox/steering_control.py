import carla
import math
import pygame

def get_wheel_control(wheel, control, reverse_mode: bool) -> carla.VehicleControl:
    """
    Logitech wheel + pedals input handler for mapping where:
      - Axis 0: steering  (-1 left, +1 right, ~0 centered)
      - Axis 2: accelerator ( -1 at rest -> +1 fully pressed )
      - Axis 3: brake       ( -1 at rest -> +1 fully pressed )

    If your gas/brake are on different axes, just change the axis indices.
    """
    # Reset
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # Reverse gear flag (you toggle reverse_mode elsewhere)
    control.reverse = reverse_mode

    # ---- Steering (Axis 0) ----
    steer_axis = wheel.get_axis(0)  # -1 left, +1 right
    print(steer_axis)
    # deadzone = 0.05
    # if abs(steer_axis) < deadzone:
        # steer_axis = 0.0
    control.steer = float(max(-1.0, min(1.0, steer_axis)))

    # ---- Accelerator (Axis 2) ----
    accel_axis = wheel.get_axis(1)  # -1 rest, +1 pressed
    print(accel_axis)
    # Map [-1, 1] -> [0, 1]
    throttle = (accel_axis + 1.0) / 2.0
    control.throttle = float(max(0.15, min(1.0, throttle)))

    # ---- Brake (Axis 3) ----
    brake_axis = wheel.get_axis(3)  # -1 rest, +1 pressed
    brake_val = (brake_axis + 1.0) / 2.0
    brake_val = math.exp(10*(brake_val**(2.5))) - 1
    print(brake_val)
    control.brake = float(max(0.0, min(1.0, brake_val)))

    # ---- Hand brake (pick any button you like, example: button 5) ----
    handbrake_button = wheel.get_button(5)
    control.hand_brake = bool(handbrake_button)

    return control

def get_keyboard_control(keys, control, reverse_mode: bool) -> carla.VehicleControl:
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0

    # set reverse flag based on current gear mode
    control.reverse = reverse_mode

    # Throttle (forward or reverse depending on control.reverse)
    if keys[pygame.K_w]:
        control.throttle = 1.0

    # Brake
    if keys[pygame.K_s]:
        control.brake = 1.0

    # Steering
    steer_amt = 0.35
    if keys[pygame.K_a]:
        control.steer = -steer_amt
    if keys[pygame.K_d]:
        control.steer = steer_amt

    # Hand brake
    control.hand_brake = keys[pygame.K_SPACE]

    return control