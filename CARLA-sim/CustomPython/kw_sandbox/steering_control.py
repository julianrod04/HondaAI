import carla
import math
import pygame

def get_wheel_control(
    wheel,
    control,
    reverse_mode: bool,
    debug: bool = False,
) -> carla.VehicleControl:
    """
    Logitech G920 wheel + pedals input handler.

    Axis mapping (G920):
      - Axis 0: steering      (-1 left, +1 right, ~0 centered)
      - Axis 1: brake         (-1 at rest -> +1 fully pressed)
      - Axis 2: clutch        (ignored)
      - Axis 3: accelerator   (-1 at rest -> +1 fully pressed)

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
    deadzone = 0.05
    if abs(steer_axis) < deadzone:
        steer_axis = 0.0
    # Power curve: softens centre, full lock still reaches ±1.0
    # Increase exponent for softer centre (2.0 = moderate, 3.0 = very soft)
    steer_sensitivity = 2.5
    sign = 1.0 if steer_axis >= 0 else -1.0
    steer_axis = sign * (abs(steer_axis) ** steer_sensitivity)
    control.steer = float(max(-1.0, min(1.0, steer_axis)))
    if debug:
        print(f"[wheel] steer={control.steer:.3f}")

    # ---- Accelerator (Axis 3) ----
    accel_axis = wheel.get_axis(3)  # -1 rest, +1 pressed
    # Map [-1, 1] -> [0, 1]
    throttle = (accel_axis + 1.0) / 2.0
    # Deadzone so pedal at rest reads exactly 0
    if throttle < 0.05:
        throttle = 0.0
    control.throttle = float(max(0.0, min(1.0, throttle)))
    if debug:
        print(f"[wheel] throttle={control.throttle:.3f}")

    # ---- Brake (Axis 1) ----
    brake_axis = wheel.get_axis(1)  # -1 rest, +1 pressed
    brake_val = (brake_axis + 1.0) / 2.0
    if brake_val < 0.05:
        brake_val = 0.0
    # Gentle power curve: light tap = light brake, full press = full brake
    brake_val = brake_val ** 1.5
    control.brake = float(max(0.0, min(1.0, brake_val)))
    if debug:
        print(f"[wheel] brake={control.brake:.3f}")

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


# =============================================================================
# AXIS DIAGNOSTIC TOOL
# =============================================================================

def run_axis_diagnostic():
    """
    Visual pygame window showing all joystick axes, buttons, and hats in
    real-time.  Run with:  python steering_control.py

    Move each pedal / wheel one at a time to identify which axis index
    corresponds to which physical control.
    """
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick/wheel detected. Plug in your G920 and try again.")
        pygame.quit()
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    name = js.get_name()
    num_axes = js.get_numaxes()
    num_buttons = js.get_numbuttons()
    num_hats = js.get_numhats()

    width, height = 600, 80 + num_axes * 40 + num_buttons * 22 + num_hats * 22 + 60
    screen = pygame.display.set_mode((width, max(height, 300)))
    pygame.display.set_caption(f"Axis Diagnostic — {name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill((30, 30, 30))
        y = 10

        # Title
        title = font.render(f"{name}  (axes={num_axes}  btns={num_buttons}  hats={num_hats})", True, (200, 200, 200))
        screen.blit(title, (10, y))
        y += 35

        # Axes
        bar_w = 300
        for i in range(num_axes):
            val = js.get_axis(i)
            label = font.render(f"Axis {i}: {val:+.4f}", True, (180, 220, 255))
            screen.blit(label, (10, y))

            # Bar background
            bx = 250
            pygame.draw.rect(screen, (60, 60, 60), (bx, y + 2, bar_w, 20))
            # Center line
            pygame.draw.line(screen, (120, 120, 120), (bx + bar_w // 2, y + 2), (bx + bar_w // 2, y + 22), 1)
            # Value bar (green if positive, red if negative)
            fill = int(abs(val) * (bar_w // 2))
            if val >= 0:
                pygame.draw.rect(screen, (80, 200, 80), (bx + bar_w // 2, y + 2, fill, 20))
            else:
                pygame.draw.rect(screen, (200, 80, 80), (bx + bar_w // 2 - fill, y + 2, fill, 20))

            y += 32

        y += 10

        # Buttons
        btn_label = small.render("Buttons:", True, (180, 180, 180))
        screen.blit(btn_label, (10, y))
        y += 20
        cols = 10
        for i in range(num_buttons):
            pressed = js.get_button(i)
            col = i % cols
            row = i // cols
            bx = 10 + col * 55
            by = y + row * 22
            color = (80, 200, 80) if pressed else (80, 80, 80)
            pygame.draw.rect(screen, color, (bx, by, 48, 18), border_radius=3)
            txt = small.render(str(i), True, (255, 255, 255) if pressed else (140, 140, 140))
            screen.blit(txt, (bx + 18, by + 1))

        y += ((num_buttons // cols) + 1) * 22 + 10

        # Hats
        for i in range(num_hats):
            hx, hy = js.get_hat(i)
            hat_label = small.render(f"Hat {i}: ({hx:+d}, {hy:+d})", True, (220, 200, 140))
            screen.blit(hat_label, (10, y))
            y += 22

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run_axis_diagnostic()
