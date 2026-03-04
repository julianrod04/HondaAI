#!/usr/bin/env python3
"""
Unified CARLA Workflow CLI - Single command to start server, load map, and drive.

Handles everything needed for a manual driving session:
1. Finds CARLA installation
2. Starts the CARLA server (skippable with --no-server)
3. Waits for server readiness
4. Loads Town06 if not already loaded
5. Launches manual driving with HUD, auto-respawn, and top-down spectator

Usage:
    # Full workflow (start server + drive)
    python bc_sandbox/run.py

    # Server already running
    python bc_sandbox/run.py --no-server

    # Kill server on exit, no NPC
    python bc_sandbox/run.py --kill-on-exit --no-npc

    # Custom options
    python bc_sandbox/run.py --no-server --chase-cam --no-spectator
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import scripts/ and rl/
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from scripts.launch_servers import find_carla_root, get_carla_executable, launch_server
from scripts.manual_drive import run_manual_drive
from rl.config import DEFAULT_CONFIG

log = logging.getLogger("run")


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

def wait_for_server(host: str, port: int, timeout: float = 120.0, interval: float = 3.0) -> carla.Client:
    """
    Poll until the CARLA server is ready to accept connections.

    Args:
        host: Server hostname
        port: Server port
        timeout: Maximum seconds to wait
        interval: Seconds between connection attempts

    Returns:
        Connected carla.Client

    Raises:
        TimeoutError: If server doesn't respond within timeout
    """
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            log.debug(f"Connection attempt {attempt} to {host}:{port}...")
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            # This call will fail if the server isn't ready
            client.get_server_version()
            log.info(f"Connected to CARLA server on port {port}")
            client.set_timeout(20.0)
            return client
        except RuntimeError:
            remaining = deadline - time.time()
            if remaining > 0:
                print(f"  Waiting for server... ({remaining:.0f}s remaining)", end="\r")
                time.sleep(interval)
            continue

    raise TimeoutError(
        f"CARLA server did not respond on {host}:{port} within {timeout}s"
    )


def ensure_map(client: carla.Client, map_name: str = "Town06") -> carla.World:
    """
    Load the target map if not already loaded.

    Args:
        client: Connected CARLA client
        map_name: Map to load

    Returns:
        CARLA world object
    """
    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]

    if current_map != map_name:
        print(f"Loading {map_name} (current: {current_map})...")
        client.set_timeout(60.0)
        world = client.load_world(map_name)
        client.set_timeout(20.0)
        print(f"{map_name} loaded!")
    else:
        print(f"Already on {map_name}")

    return world


# =============================================================================
# CLEANUP
# =============================================================================

def kill_server(process) -> None:
    """Terminate a CARLA server process."""
    if process is None:
        return
    if process.poll() is not None:
        log.debug("Server process already exited")
        return

    print("Stopping CARLA server...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=10)
        print("Server stopped")
    except subprocess.TimeoutExpired:
        process.kill()
        print("Server killed (force)")
    except Exception as e:
        log.warning(f"Error stopping server: {e}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CARLA workflow: start server, load map, and drive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server options
    server = parser.add_argument_group("Server")
    server.add_argument(
        "--no-server", action="store_true",
        help="Skip server launch (connect to already-running server)",
    )
    server.add_argument(
        "--carla-path", type=str, default=None,
        help="Override CARLA installation path (or set CARLA_ROOT env var)",
    )
    server.add_argument(
        "--port", type=int, default=2000,
        help="CARLA server port",
    )
    server.add_argument(
        "--quality", choices=["Low", "Medium", "High", "Epic"], default="Low",
        help="Rendering quality for CARLA server",
    )
    server.add_argument(
        "--keep-server", action="store_true",
        help="Keep CARLA server running after the driving session ends",
    )
    server.add_argument(
        "--headless", action="store_true",
        help="Hide the CARLA server viewport window (offscreen rendering)",
    )

    # Driving options
    driving = parser.add_argument_group("Driving")
    driving.add_argument(
        "--no-npc", action="store_true",
        help="Don't spawn NPC vehicle",
    )
    driving.add_argument(
        "--chase-cam", action="store_true",
        help="Use third-person chase camera instead of driver POV",
    )

    # Spectator options
    spec = parser.add_argument_group("Spectator")
    spec.add_argument(
        "--no-spectator", action="store_true",
        help="Disable top-down spectator camera",
    )
    spec.add_argument(
        "--spectator-height", type=float, default=40.0,
        help="Bird's-eye spectator height in meters",
    )

    # Misc
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    server_process = None

    try:
        # ── Step 1: Start server (unless --no-server) ────────────────────
        if not args.no_server:
            # Find CARLA
            if args.carla_path:
                carla_root = Path(args.carla_path)
            else:
                carla_root = find_carla_root()

            if carla_root is None or not carla_root.exists():
                print("ERROR: Could not find CARLA installation.")
                print("  Set CARLA_ROOT env var, use --carla-path, or use --no-server")
                sys.exit(1)

            carla_exe = get_carla_executable(carla_root)
            print(f"CARLA root: {carla_root}")
            print(f"Executable: {carla_exe}")

            print(f"Starting CARLA server on port {args.port} (quality={args.quality})...")
            server_process = launch_server(
                carla_executable=carla_exe,
                port=args.port,
                quality=args.quality,
                offscreen=args.headless,
            )
            print(f"Server started (PID={server_process.pid})")

        # ── Step 2: Wait for server readiness ─────────────────────────────
        print(f"Connecting to CARLA on localhost:{args.port}...")
        client = wait_for_server("localhost", args.port)
        version = client.get_server_version()
        print(f"Connected! Server version: {version}")

        # ── Step 3: Load map ──────────────────────────────────────────────
        world = ensure_map(client, "Town06")

        # ── Step 4: Launch manual driving ─────────────────────────────────
        print("\nLaunching manual driving session...\n")
        run_manual_drive(
            client=client,
            world=world,
            config=DEFAULT_CONFIG,
            no_npc=args.no_npc,
            chase_cam=args.chase_cam,
            enable_spectator=not args.no_spectator,
            spectator_height=args.spectator_height,
        )

    except TimeoutError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        if server_process is not None:
            if args.keep_server:
                print(f"\nCARLA server still running (PID={server_process.pid})")
            else:
                kill_server(server_process)


if __name__ == "__main__":
    main()
