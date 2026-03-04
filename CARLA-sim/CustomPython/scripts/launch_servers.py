#!/usr/bin/env python3
"""
Launch multiple CARLA server instances for parallel training.

Each instance runs on a different port (2000, 2002, 2004, 2006).
CARLA uses two consecutive ports: the base port for the simulator
and base_port+1 for the streaming port.

Usage:
    python scripts/launch_servers.py --num-instances 4
    python scripts/launch_servers.py --num-instances 4 --carla-path /path/to/CARLA
    python scripts/launch_servers.py --stop  # Stop all running instances

Requirements:
    - CARLA simulator must be installed
    - Set CARLA_ROOT environment variable or use --carla-path
"""

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# Default CARLA installation paths
DEFAULT_CARLA_PATHS = [
    # Windows
    r"C:\CARLA_0.9.16",
    r"C:\Users\{user}\CARLA_0.9.16",
    # Linux
    "/opt/carla-simulator",
    os.path.expanduser("~/CARLA_0.9.16"),
    # Relative to this repo
    str(Path(__file__).parent.parent.parent / "CARLA_0.9.16"),
]


def find_carla_root() -> Optional[Path]:
    """
    Find the CARLA installation directory.
    
    Returns:
        Path to CARLA root, or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get("CARLA_ROOT")
    if env_path:
        carla_path = Path(env_path)
        if carla_path.exists():
            return carla_path
    
    # Check default locations
    username = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    for path_template in DEFAULT_CARLA_PATHS:
        path_str = path_template.format(user=username)
        carla_path = Path(path_str)
        if carla_path.exists():
            return carla_path
    
    return None


def get_carla_executable(carla_root: Path) -> Path:
    """
    Get the path to the CARLA executable.
    
    Args:
        carla_root: CARLA installation directory
        
    Returns:
        Path to the executable
    """
    system = platform.system()
    
    if system == "Windows":
        # Try different Windows executable names
        candidates = [
            carla_root / "CarlaUE4.exe",
            carla_root / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe",
            carla_root / "WindowsNoEditor" / "CarlaUE4.exe",
        ]
    else:
        # Linux
        candidates = [
            carla_root / "CarlaUE4.sh",
            carla_root / "CarlaUE4" / "Binaries" / "Linux" / "CarlaUE4-Linux-Shipping",
        ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Fall back to expected name
    if system == "Windows":
        return carla_root / "CarlaUE4.exe"
    else:
        return carla_root / "CarlaUE4.sh"


def launch_server(
    carla_executable: Path,
    port: int,
    gpu_id: int = 0,
    quality: str = "Low",
    offscreen: bool = True,
    extra_args: Optional[List[str]] = None
) -> subprocess.Popen:
    """
    Launch a single CARLA server instance.
    
    Args:
        carla_executable: Path to CARLA executable
        port: Port number for this instance
        gpu_id: GPU device ID to use
        quality: Rendering quality ("Low", "Medium", "High", "Epic")
        offscreen: Run without display (for training)
        extra_args: Additional command line arguments
        
    Returns:
        Popen process object
    """
    cmd = [str(carla_executable)]
    
    # Port configuration
    cmd.extend(["-carla-port", str(port)])
    
    # Quality setting
    cmd.extend(["-quality-level", quality])
    
    # GPU selection (CUDA device)
    cmd.extend(["-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=" + str(gpu_id)])
    
    # Offscreen rendering (no window)
    if offscreen:
        cmd.append("-RenderOffScreen")
    
    # Disable sound
    cmd.append("-nosound")
    
    # Disable log file
    cmd.append("-carla-no-log")
    
    # Additional arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Set environment for GPU selection
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Launch process
    system = platform.system()
    
    if system == "Windows":
        # Windows: use CREATE_NEW_PROCESS_GROUP for better signal handling
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        # Linux: set process group for signal handling
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
    
    return process


def launch_servers(
    num_instances: int,
    carla_root: Path,
    base_port: int = 2000,
    quality: str = "Low",
    offscreen: bool = True,
    startup_delay: float = 5.0,
) -> List[subprocess.Popen]:
    """
    Launch multiple CARLA server instances.
    
    Args:
        num_instances: Number of instances to launch
        carla_root: CARLA installation directory
        base_port: Starting port number (instances use +2 increments)
        quality: Rendering quality
        offscreen: Run without display
        startup_delay: Seconds to wait between launching instances
        
    Returns:
        List of Popen process objects
    """
    carla_executable = get_carla_executable(carla_root)
    
    if not carla_executable.exists():
        raise FileNotFoundError(
            f"CARLA executable not found at {carla_executable}"
        )
    
    print(f"Launching {num_instances} CARLA server instances...")
    print(f"CARLA executable: {carla_executable}")
    print(f"Quality: {quality}, Offscreen: {offscreen}")
    print()
    
    processes = []
    
    for i in range(num_instances):
        port = base_port + (i * 2)
        gpu_id = i % max(1, get_gpu_count())  # Distribute across GPUs
        
        print(f"  Starting instance {i}: port={port}, GPU={gpu_id}...", end=" ")
        
        try:
            process = launch_server(
                carla_executable,
                port=port,
                gpu_id=gpu_id,
                quality=quality,
                offscreen=offscreen,
            )
            processes.append(process)
            print(f"PID={process.pid}")
        except Exception as e:
            print(f"FAILED: {e}")
            # Clean up already started processes
            stop_servers(processes)
            raise
        
        # Wait between launches to avoid resource contention
        if i < num_instances - 1:
            time.sleep(startup_delay)
    
    print()
    print(f"All {num_instances} instances started!")
    print("Ports:", [base_port + (i * 2) for i in range(num_instances)])
    
    return processes


def _kill_all_carla_processes() -> None:
    """Force-kill any remaining CarlaUE4 processes (wrapper + shipping binary)."""
    print("  Sweeping for remaining CARLA processes...")
    if platform.system() == "Windows":
        for name in ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe"):
            subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
    else:
        # pkill -9 catches both CarlaUE4.sh and CarlaUE4-Linux-Shipping
        subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], capture_output=True)


def stop_servers(processes: List[subprocess.Popen]) -> None:
    """
    Stop all running CARLA server instances.

    Args:
        processes: List of Popen process objects
    """
    print(f"Stopping {len(processes)} CARLA instances...")

    system = platform.system()

    for i, process in enumerate(processes):
        try:
            if process.poll() is None:  # Still running
                print(f"  Stopping instance {i} (PID={process.pid})...", end=" ")

                if system == "Windows":
                    # Windows: send CTRL_BREAK_EVENT
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    # Linux: send SIGTERM to process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print("stopped")
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    process.kill()
                    print("killed")
            else:
                print(f"  Instance {i} already stopped")
        except Exception as e:
            print(f"  Error stopping instance {i}: {e}")

    # Final sweep — catches CarlaUE4-Linux-Shipping children that survive SIGTERM
    _kill_all_carla_processes()
    print("All instances stopped.")


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        pass
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return 1  # Assume at least one GPU


def write_pid_file(processes: List[subprocess.Popen], pid_file: Path) -> None:
    """Write process IDs to a file for later cleanup."""
    with open(pid_file, "w") as f:
        for process in processes:
            f.write(f"{process.pid}\n")
    print(f"PID file written: {pid_file}")


def read_pid_file(pid_file: Path) -> List[int]:
    """Read process IDs from a file."""
    if not pid_file.exists():
        return []
    
    with open(pid_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def stop_servers_from_pid_file(pid_file: Path) -> None:
    """Stop servers using PIDs from file."""
    pids = read_pid_file(pid_file)
    
    if not pids:
        print(f"No PIDs found in {pid_file}")
        return
    
    print(f"Stopping {len(pids)} processes from {pid_file}...")
    
    system = platform.system()
    
    for pid in pids:
        try:
            print(f"  Stopping PID {pid}...", end=" ")
            
            if system == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            
            print("stopped")
        except (ProcessLookupError, OSError) as e:
            print(f"not running ({e})")
    
    # Final sweep for any remaining CARLA processes
    _kill_all_carla_processes()

    # Clean up PID file
    try:
        pid_file.unlink()
        print(f"Removed PID file: {pid_file}")
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple CARLA server instances for parallel training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-instances", "-n",
        type=int,
        default=4,
        help="Number of CARLA instances to launch"
    )
    
    parser.add_argument(
        "--carla-path",
        type=str,
        default=None,
        help="Path to CARLA installation (or set CARLA_ROOT env var)"
    )
    
    parser.add_argument(
        "--base-port",
        type=int,
        default=2000,
        help="Starting port number"
    )
    
    parser.add_argument(
        "--quality",
        choices=["Low", "Medium", "High", "Epic"],
        default="Low",
        help="Rendering quality (Low recommended for training)"
    )
    
    parser.add_argument(
        "--no-offscreen",
        action="store_true",
        help="Run with display windows (not recommended for training)"
    )
    
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between launching instances"
    )
    
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop all running instances (using PID file)"
    )
    
    parser.add_argument(
        "--pid-file",
        type=str,
        default=".carla_pids",
        help="File to store/read process IDs"
    )
    
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for servers to exit (keeps script running)"
    )
    
    args = parser.parse_args()
    
    pid_file = Path(args.pid_file)
    
    # Handle stop command
    if args.stop:
        stop_servers_from_pid_file(pid_file)
        return
    
    # Find CARLA installation
    if args.carla_path:
        carla_root = Path(args.carla_path)
    else:
        carla_root = find_carla_root()
    
    if carla_root is None or not carla_root.exists():
        print("ERROR: Could not find CARLA installation.")
        print("Please either:")
        print("  1. Set the CARLA_ROOT environment variable")
        print("  2. Use --carla-path /path/to/CARLA")
        sys.exit(1)
    
    print(f"CARLA root: {carla_root}")
    
    # Launch servers
    try:
        processes = launch_servers(
            num_instances=args.num_instances,
            carla_root=carla_root,
            base_port=args.base_port,
            quality=args.quality,
            offscreen=not args.no_offscreen,
            startup_delay=args.startup_delay,
        )
        
        # Write PID file
        write_pid_file(processes, pid_file)
        
        if args.wait:
            print()
            print("Press Ctrl+C to stop all instances...")
            
            try:
                while True:
                    # Check if any process has died
                    for i, p in enumerate(processes):
                        if p.poll() is not None:
                            print(f"Instance {i} (PID={p.pid}) exited with code {p.returncode}")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nReceived interrupt signal...")
                stop_servers(processes)
        else:
            print()
            print("Servers launched in background.")
            print(f"To stop them later: python {__file__} --stop")
    
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

