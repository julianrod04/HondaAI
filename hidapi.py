from inputs import get_gamepad
import sys

print("Listening for G920 data... (Press Ctrl+C to stop)")

try:
    while True:
        events = get_gamepad()
        for event in events:
            # distinct event types allows you to filter out noise
            if event.ev_type != "Sync":
                print(f"Event: {event.code}, State: {event.state}")
except KeyboardInterrupt:
    sys.exit()
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to close...")