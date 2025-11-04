"""
Pygame display handler for CARLA camera feed.
Runs in a separate thread to allow notebook execution to continue.
"""
import pygame
import numpy as np
import threading
import queue
import carla
from typing import Optional


class CarlaDisplay:
    """Manages pygame display in a background thread."""
    
    def __init__(self, width: int = 758, height: int = 396):
        self.width = width
        self.height = height
        self.image_queue = queue.Queue(maxsize=1)
        self.running = False
        self.display_thread = None
        self.display = None
        
    def to_surface(self, image):
        """Convert CARLA image to pygame surface."""
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def _display_loop(self):
        """Main display loop running in background thread."""
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height), 
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        clock = pygame.time.Clock()
        
        latest_surface = None
        
        while self.running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
            
            # Get latest image from queue (non-blocking)
            try:
                image = self.image_queue.get_nowait()
                latest_surface = self.to_surface(image)
            except queue.Empty:
                pass
            
            # Display the latest surface
            if latest_surface is not None:
                self.display.blit(latest_surface, (0, 0))
                pygame.display.flip()
            
            clock.tick_busy_loop(60)
        
        pygame.quit()
    
    def on_image(self, image):
        """Callback for CARLA camera - adds image to queue."""
        if self.running:
            # Put image in queue (discard old one if queue is full)
            try:
                self.image_queue.put_nowait(image)
            except queue.Full:
                try:
                    self.image_queue.get_nowait()  # Remove old image
                    self.image_queue.put_nowait(image)  # Add new one
                except queue.Empty:
                    pass
    
    def start(self):
        """Start the display thread."""
        if not self.running:
            self.running = True
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
            print("Pygame display started in background thread")
    
    def stop(self):
        """Stop the display thread."""
        if self.running:
            self.running = False
            if self.display_thread:
                self.display_thread.join(timeout=2.0)
            print("Pygame display stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

