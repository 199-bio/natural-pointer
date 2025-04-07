"""
Mouse movement tracker for collecting training data.
Captures mouse movements, clicks, and timing information.
"""

import json
import time
from pynput import mouse
import pyautogui
from collections import deque

class MouseTracker:
    """Records natural mouse movements and events for training a neural network."""
    
    def __init__(self, save_path="mouse_data.json"):
        """
        Initialize the mouse tracker.
        
        Args:
            save_path: Path to save the recorded data.
        """
        self.movements = []
        self.save_path = save_path
        self.recording = False
        self.last_time = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_positions = deque(maxlen=5)
        self.listener = None
        
    def start_recording(self):
        """Start recording mouse movements and events."""
        self.recording = True
        self.movements = []
        self.last_positions.clear()
        self.last_time = time.time()
        
        # Start mouse listener
        self.listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll)
        self.listener.start()
        
        return self
        
    def stop_recording(self):
        """Stop recording mouse movements and return the collected data."""
        if self.listener:
            self.listener.stop()
        self.recording = False
        return self.movements
        
    def _on_move(self, x, y):
        """Callback for mouse movement events."""
        if not self.recording:
            return
            
        current_time = time.time()
        # Reduce the minimum interval to capture smoother movements
        # 5ms minimum interval - captures more detailed movements while still
        # preventing excessive data points
        if self.last_time and current_time - self.last_time < 0.005:  # 5ms minimum
            return
            
        # Normalize coordinates to 0-1 range for screen independence
        nx = x / self.screen_width
        ny = y / self.screen_height
        
        # Calculate time delta from last event in milliseconds
        time_delta = 0 if self.last_time is None else (current_time - self.last_time) * 1000
        self.last_time = current_time
        
        # Calculate velocity if we have previous positions (in normalized units per second)
        vx, vy = 0, 0
        if len(self.last_positions) > 0:
            prev_x, prev_y = self.last_positions[-1][:2]
            vx = (nx - prev_x) / (time_delta/1000) if time_delta > 0 else 0
            vy = (ny - prev_y) / (time_delta/1000) if time_delta > 0 else 0
        
        # Add current position to history
        self.last_positions.append((nx, ny, time_delta, vx, vy, current_time))
        
        # Record movement with features
        self.movements.append({
            'type': 'move',
            'x': nx,
            'y': ny,
            'time': time_delta,
            'vx': vx,
            'vy': vy,
            'timestamp': current_time
        })
        
    def _on_click(self, x, y, button, pressed):
        """Callback for mouse click events."""
        if not self.recording:
            return
            
        current_time = time.time()
        
        # Normalize coordinates
        nx = x / self.screen_width
        ny = y / self.screen_height
        
        # Calculate time delta from last event
        time_delta = 0 if self.last_time is None else (current_time - self.last_time) * 1000
        self.last_time = current_time
        
        # Record click event
        self.movements.append({
            'type': 'click',
            'button': str(button),
            'pressed': pressed,
            'x': nx,
            'y': ny,
            'time': time_delta,
            'vx': 0,
            'vy': 0,
            'timestamp': current_time
        })
    
    def _on_scroll(self, x, y, dx, dy):
        """Callback for mouse scroll events."""
        if not self.recording:
            return
            
        current_time = time.time()
        
        # Normalize coordinates
        nx = x / self.screen_width
        ny = y / self.screen_height
        
        # Calculate time delta from last event
        time_delta = 0 if self.last_time is None else (current_time - self.last_time) * 1000
        self.last_time = current_time
        
        # Record scroll event
        self.movements.append({
            'type': 'scroll',
            'x': nx,
            'y': ny,
            'dx': dx,
            'dy': dy,
            'time': time_delta,
            'vx': 0,
            'vy': 0,
            'timestamp': current_time
        })
        
    def save_data(self):
        """Save the recorded data to a JSON file."""
        if not self.movements:
            print("No movements recorded yet.")
            return False
            
        # Save the recorded movements directly
        with open(self.save_path, 'w') as f:
            # Save the list as is, preserving original fields including 'time' delta
            json.dump(self.movements, f, indent=2) 
        
        print(f"Data saved to {self.save_path}")
        return True
