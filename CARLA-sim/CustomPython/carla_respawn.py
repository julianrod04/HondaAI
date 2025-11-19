"""
Auto-respawn monitor for CARLA vehicles.
Monitors vehicle positions and time, automatically respawning when conditions are met.
"""
import threading
import time
import carla
from typing import Optional, Callable, List


class AutoRespawnMonitor:
    """Monitors vehicles and respawns them when conditions are met."""
    
    def __init__(self, world: carla.World, ego_spawn_point: carla.Transform, 
                 npc_spawn_point: carla.Transform, ego_blueprint: carla.ActorBlueprint, 
                 npc_blueprint: carla.ActorBlueprint, camera_blueprint: carla.ActorBlueprint, 
                 camera_transform: carla.Transform, display=None,
                 camera_callback: Optional[Callable[[carla.Image], None]] = None,
                 spawn_x_threshold: float = 200, time_threshold: float = 30):
        """
        Initialize the auto-respawn monitor.
        
        Args:
            world: CARLA world object
            ego_spawn_point: Transform for ego vehicle spawn location
            npc_spawn_point: Transform for NPC vehicle spawn location
            ego_blueprint: Blueprint for ego vehicle
            npc_blueprint: Blueprint for NPC vehicle
            camera_blueprint: Blueprint for camera sensor
            camera_transform: Transform for camera relative to vehicle
            display: Optional display handler (CarlaDisplay) for camera images
            camera_callback: Optional callback invoked with each camera image
            spawn_x_threshold: Distance in x direction to trigger respawn (default: 200)
            time_threshold: Time in seconds to trigger respawn (default: 30)
        """
        self.world = world
        self.ego_spawn_point = ego_spawn_point
        self.npc_spawn_point = npc_spawn_point
        self.ego_blueprint = ego_blueprint
        self.npc_blueprint = npc_blueprint
        self.camera_blueprint = camera_blueprint
        self.camera_transform = camera_transform
        self.display = display
        self.camera_callbacks: List[Callable[[carla.Image], None]] = []
        if display is not None:
            self.camera_callbacks.append(display.on_image)
        if camera_callback is not None:
            self.camera_callbacks.append(camera_callback)
        
        self.spawn_x = ego_spawn_point.location.x
        self.x_threshold = self.spawn_x + spawn_x_threshold
        self.time_threshold = time_threshold
        
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.npc_vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        
        # Track autopilot state to preserve it during respawn
        self.ego_autopilot: bool = False
        self.npc_autopilot: bool = False
        
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def add_camera_callback(self, callback: Callable[[carla.Image], None]):
        """Register additional camera callbacks (deduplicated)."""
        if callback not in self.camera_callbacks:
            self.camera_callbacks.append(callback)
    
    def _relay_camera_image(self, image: carla.Image):
        """Fan-out camera images to registered callbacks."""
        for callback in list(self.camera_callbacks):
            try:
                callback(image)
            except Exception as exc:
                print(f"Camera callback error: {exc}")
    
    @staticmethod
    def _discard_image(image: carla.Image):
        """Fallback no-op camera sink."""
        return
    
    def respawn_vehicles(self):
        """Destroy and respawn both vehicles and camera."""
        print("Respawn triggered!")
        
        # Destroy existing actors
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        if self.npc_vehicle:
            self.npc_vehicle.destroy()
        
        # Spawn new vehicles
        self.ego_vehicle = self.world.spawn_actor(self.ego_blueprint, self.ego_spawn_point)
        self.npc_vehicle = self.world.spawn_actor(self.npc_blueprint, self.npc_spawn_point)
        
        # Restore autopilot state
        self.ego_vehicle.set_autopilot(self.ego_autopilot)
        self.npc_vehicle.set_autopilot(self.npc_autopilot)
        
        # Spawn and attach camera
        self.camera = self.world.spawn_actor(
            self.camera_blueprint, 
            self.camera_transform, 
            attach_to=self.ego_vehicle
        )
        callback = self._relay_camera_image if self.camera_callbacks else self._discard_image
        self.camera.listen(callback)
        
        print(f"Vehicles respawned at x={self.spawn_x} (ego_autopilot={self.ego_autopilot}, npc_autopilot={self.npc_autopilot})")
        return self.ego_vehicle, self.npc_vehicle, self.camera
    
    def force_respawn(self, ego_autopilot: Optional[bool] = None, 
                      npc_autopilot: Optional[bool] = None):
        """
        Synchronously respawn vehicles and camera without starting the monitor thread.
        Useful for RL environments that own the simulation loop.
        """
        if ego_autopilot is not None:
            self.ego_autopilot = ego_autopilot
        if npc_autopilot is not None:
            self.npc_autopilot = npc_autopilot
        return self.respawn_vehicles()
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        start_time = time.time()
        
        while self.running:
            if self.ego_vehicle is None or self.npc_vehicle is None:
                time.sleep(0.1)
                continue
            
            # Check time condition
            elapsed_time = time.time() - start_time
            
            # Check position condition
            try:
                ego_location = self.ego_vehicle.get_location()
                npc_location = self.npc_vehicle.get_location()
                ego_x = ego_location.x
                npc_x = npc_location.x
                
                # Check if either vehicle has traveled x += threshold from spawn
                position_triggered = (ego_x >= self.x_threshold) or (npc_x >= self.x_threshold)
                
                # Check if time threshold has been reached
                time_triggered = elapsed_time >= self.time_threshold
                
                if position_triggered or time_triggered:
                    trigger_reason = []
                    if position_triggered:
                        trigger_reason.append(
                            f"Position (ego_x={ego_x:.2f}, npc_x={npc_x:.2f} >= {self.x_threshold})"
                        )
                    if time_triggered:
                        trigger_reason.append(
                            f"Time ({elapsed_time:.2f}s >= {self.time_threshold}s)"
                        )
                    print(f"Respawn condition met: {', '.join(trigger_reason)}")
                    
                    self.respawn_vehicles()
                    start_time = time.time()  # Reset timer after respawn
                    
            except Exception as e:
                # Vehicle might have been destroyed externally
                print(f"Error monitoring vehicles: {e}")
                time.sleep(0.1)
                continue
            
            time.sleep(0.1)  # Check every 100ms
    
    def start(self, ego_vehicle: carla.Vehicle, npc_vehicle: carla.Vehicle, 
              camera: carla.Sensor, ego_autopilot: Optional[bool] = None, 
              npc_autopilot: Optional[bool] = None):
        """
        Start monitoring with initial vehicles.
        
        Args:
            ego_vehicle: Initial ego vehicle
            npc_vehicle: Initial NPC vehicle
            camera: Initial camera sensor
            ego_autopilot: Initial autopilot state for ego vehicle (None to auto-detect)
            npc_autopilot: Initial autopilot state for NPC vehicle (None to auto-detect)
        """
        self.ego_vehicle = ego_vehicle
        self.npc_vehicle = npc_vehicle
        self.camera = camera
        
        # Set autopilot state if provided, otherwise try to detect from vehicle attributes
        # Note: CARLA doesn't expose autopilot state directly, so we default to False
        # and allow manual setting via set_autopilot methods
        if ego_autopilot is not None:
            self.ego_autopilot = ego_autopilot
        if npc_autopilot is not None:
            self.npc_autopilot = npc_autopilot
        
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Auto-respawn monitor started")
    
    def set_autopilot(self, ego: Optional[bool] = None, npc: Optional[bool] = None):
        """
        Update autopilot state for vehicles.
        This state will be preserved when vehicles respawn.
        
        Args:
            ego: Autopilot state for ego vehicle (None to leave unchanged)
            npc: Autopilot state for NPC vehicle (None to leave unchanged)
        """
        if ego is not None:
            self.ego_autopilot = ego
            if self.ego_vehicle:
                self.ego_vehicle.set_autopilot(ego)
        if npc is not None:
            self.npc_autopilot = npc
            if self.npc_vehicle:
                self.npc_vehicle.set_autopilot(npc)
    
    def stop(self):
        """Stop monitoring."""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            print("Auto-respawn monitor stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

