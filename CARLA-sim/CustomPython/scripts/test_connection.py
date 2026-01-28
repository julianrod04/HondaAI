#!/usr/bin/env python3
"""
Test CARLA connection before training.

Usage:
    python scripts/test_connection.py
    python scripts/test_connection.py --port 2000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla


def test_connection(host: str = "localhost", port: int = 2000, timeout: float = 10.0):
    """Test connection to CARLA server."""
    print(f"Testing connection to CARLA at {host}:{port}...")
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        
        # Try to get the world
        world = client.get_world()
        map_name = world.get_map().name.split('/')[-1]
        
        print(f"✓ Connected successfully!")
        print(f"  Map: {map_name}")
        print(f"  Server version: {client.get_server_version()}")
        print(f"  Client version: {client.get_client_version()}")
        
        # Count actors
        actors = world.get_actors()
        vehicles = [a for a in actors if a.type_id.startswith('vehicle')]
        print(f"  Vehicles in world: {len(vehicles)}")
        
        # Get weather
        weather = world.get_weather()
        print(f"  Weather: cloudiness={weather.cloudiness:.0f}%, rain={weather.precipitation:.0f}%")
        
        return True
        
    except RuntimeError as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure CARLA server is running:")
        print("  1. Open a new terminal")
        print("  2. cd C:\\Users\\bc35638\\Desktop\\HondaAI\\CARLA-sim\\CARLA_0.9.16")
        print("  3. .\\CarlaUE4.exe")
        print("  4. Wait for the window to fully load (30-60 seconds)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test CARLA connection")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout")
    args = parser.parse_args()
    
    success = test_connection(args.host, args.port, args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



