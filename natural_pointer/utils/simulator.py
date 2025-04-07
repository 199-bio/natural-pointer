"""
Mouse movement simulation using trained neural network models.
"""

import torch
import numpy as np
import pyautogui
import time
import random
import math
# from collections import deque # Unused
from typing import List, Dict, Tuple, Optional, Union
import os
# import json # Unused

from ..models.neural_network import MouseMovementNN

class MouseSimulator:
    """
    Simulates natural mouse movements using a trained neural network.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the mouse simulator.
        
        Args:
            model_path: Path to the trained model file. If None, will look for 'mouse_model.pth'.
        """
        self.screen_width, self.screen_height = pyautogui.size()
        self.model = MouseMovementNN()
        
        # Try to load model
        self.model_path = model_path or os.path.join(os.getcwd(), 'mouse_model.pth')
        self.model_loaded = self._load_model() # Store the loading status
    
    def _load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                return True
            else:
                print(f"Model not found at {self.model_path}. Using untrained model.")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_path(self, start_x: float, start_y: float, 
                      end_x: float, end_y: float, 
                      steps: int = None) -> List[Dict[str, float]]:
        """
        Generate a natural path from start to end coordinates.
        
        Args:
            start_x: Starting x-coordinate.
            start_y: Starting y-coordinate.
            end_x: Target x-coordinate.
            end_y: Target y-coordinate.
            steps: Number of steps in the path. If None, calculated based on distance.
            
        Returns:
            List of path points with x, y coordinates and time intervals.
        """
        # Convert to normalized coordinates (0-1)
        start_x_norm = start_x / self.screen_width
        start_y_norm = start_y / self.screen_height
        end_x_norm = end_x / self.screen_width
        end_y_norm = end_y / self.screen_height
        
        # Calculate distance
        dist = math.sqrt((end_x_norm - start_x_norm)**2 + (end_y_norm - start_y_norm)**2)
        
        # Calculate appropriate number of steps based on distance
        if steps is None:
            steps = max(10, min(50, int(dist * 100)))
        
        # Create initial path points with linear interpolation + small randomness
        path = []
        
        # Initialize with 5 points to have enough history for the model
        for i in range(5):
            progress = i / 4
            # Move 20% of the way to target in these initial points
            x = start_x_norm + progress * (end_x_norm - start_x_norm) * 0.2
            y = start_y_norm + progress * (end_y_norm - start_y_norm) * 0.2
            
            # Add some randomness
            x += random.uniform(-0.01, 0.01) * (1 - progress)
            y += random.uniform(-0.01, 0.01) * (1 - progress)
            
            # Ensure coordinates are within screen bounds
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            # Calculate timing and velocity
            if i == 0:
                time_ms = 0
                vx, vy = 0, 0
            else:
                time_ms = random.uniform(10, 30)  # Initial random timing
                prev_x, prev_y = path[-1]['x'], path[-1]['y']
                vx = (x - prev_x) / (time_ms/1000) if time_ms > 0 else 0
                vy = (y - prev_y) / (time_ms/1000) if time_ms > 0 else 0
            
            path.append({
                'x': x,
                'y': y,
                'time': time_ms,
                'vx': vx,
                'vy': vy,
                'type': 'move'
            })
        
        # Generate the rest of the path using the neural network
        if not hasattr(self.model, 'model'):
            # Fall back to bezier curve if model isn't properly loaded
            return self._generate_bezier_path(start_x, start_y, end_x, end_y, steps)
        
        # Use the model to generate the rest of the path
        try:
            with torch.no_grad():
                for i in range(steps - 5):
                    # Prepare input features from the last 5 points
                    features = []
                    for j in range(5):
                        point = path[-5 + j]
                        is_click = 1.0 if point.get('type', '') == 'click' else 0.0
                        features.extend([
                            point['x'],
                            point['y'],
                            point['time']/1000,  # Convert to seconds
                            point['vx'],
                            point['vy'],
                            is_click
                        ])
                    
                    # Convert to tensor
                    input_tensor = torch.tensor([features], dtype=torch.float32)
                    
                    # Get prediction (x, y, time)
                    prediction = self.model(input_tensor).numpy()[0]
                    pred_x, pred_y, pred_time = prediction
                    
                    # Blend with target as we progress
                    progress = (i + 5) / steps
                    # Increase pull toward target as we get closer
                    blend_factor = min(0.8, progress * 2)
                    
                    # Blend model prediction with target
                    x = pred_x * (1 - blend_factor) + end_x_norm * blend_factor
                    y = pred_y * (1 - blend_factor) + end_y_norm * blend_factor
                    
                    # Add small controlled randomness (reduces as we approach target)
                    noise_factor = max(0, 0.4 - progress)
                    x += random.uniform(-0.005, 0.005) * noise_factor
                    y += random.uniform(-0.005, 0.005) * noise_factor
                    
                    # Ensure we're in bounds
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    # Calculate velocity
                    prev_x, prev_y = path[-1]['x'], path[-1]['y']
                    vx = (x - prev_x) / (pred_time if pred_time > 0 else 0.01)
                    vy = (y - prev_y) / (pred_time if pred_time > 0 else 0.01)
                    
                    # Occasionally add a small pause/hover (human behavior)
                    pause_prob = 0.05  # 5% chance of pause
                    if random.random() < pause_prob and i > 5 and i < steps - 10:
                        pause_time = random.uniform(0.1, 0.3)  # 100-300ms pause
                        path.append({
                            'x': prev_x,
                            'y': prev_y,
                            'time': pause_time * 1000,  # Convert to milliseconds
                            'vx': 0,
                            'vy': 0,
                            'type': 'move'
                        })
                    
                    # Add predicted point to path
                    path.append({
                        'x': x,
                        'y': y,
                        'time': pred_time * 1000,  # Convert back to milliseconds
                        'vx': vx,
                        'vy': vy,
                        'type': 'move'
                    })
            
            # Ensure the final point is exactly at the target
            path[-1]['x'] = end_x_norm
            path[-1]['y'] = end_y_norm
            
            # Convert normalized coordinates back to screen coordinates
            screen_path = []
            for point in path:
                screen_path.append({
                    'x': int(point['x'] * self.screen_width),
                    'y': int(point['y'] * self.screen_height),
                    'time': point['time'],
                    'type': point['type']
                })
            
            return screen_path
            
        except Exception as e:
            print(f"Error generating path with neural network: {str(e)}")
            # Fall back to bezier curve
            return self._generate_bezier_path(start_x, start_y, end_x, end_y, steps)
    
    def _generate_bezier_path(self, start_x: float, start_y: float, 
                             end_x: float, end_y: float, 
                             steps: int = 20) -> List[Dict[str, float]]:
        """
        Fallback method to generate path using bezier curves when model fails.
        
        Args:
            start_x, start_y: Starting coordinates.
            end_x, end_y: Target coordinates.
            steps: Number of steps in the path.
            
        Returns:
            List of path points.
        """
        # Number of control points based on distance
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        num_control_points = min(5, max(2, int(distance / 300)))
        
        # Generate control points
        control_points = [(start_x, start_y)]
        
        for i in range(num_control_points):
            # Position along the path
            progress = (i + 1) / (num_control_points + 1)
            
            # Base point on the direct line
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            
            # Add randomness, reducing as we get closer to target
            variance = 0.2 * (1 - progress)
            rand_x = x + (end_x - start_x) * random.uniform(-variance, variance)
            rand_y = y + (end_y - start_y) * random.uniform(-variance, variance)
            
            control_points.append((rand_x, rand_y))
            
        control_points.append((end_x, end_y))
        
        # Generate bezier curve points
        path = []
        prev_x, prev_y = start_x, start_y
        total_time = 0.5 + (distance / 500)  # Longer for greater distances
        
        for t in np.linspace(0, 1, steps):
            # Get point on bezier curve
            x, y = self._bezier_point(control_points, t)
            
            # Calculate timing - slower at start and end (easing)
            progress = t
            time_factor = 1 - 0.5 * math.sin(progress * math.pi)
            point_time = total_time / steps * time_factor * 1000  # Convert to ms
            
            # Skip first point
            if t > 0:
                path.append({
                    'x': int(x),
                    'y': int(y),
                    'time': point_time,
                    'type': 'move'
                })
            
            prev_x, prev_y = x, y
            
        return path
    
    def _bezier_point(self, control_points: List[Tuple[float, float]], 
                     t: float) -> Tuple[float, float]:
        """
        Calculate a point on a bezier curve.
        
        Args:
            control_points: List of control points as (x, y) tuples.
            t: Parameter value between 0 and 1.
            
        Returns:
            (x, y) coordinates of the point.
        """
        n = len(control_points) - 1
        x, y = 0, 0
        
        for i, point in enumerate(control_points):
            binomial = math.comb(n, i)
            factor = binomial * (t ** i) * ((1 - t) ** (n - i))
            x += point[0] * factor
            y += point[1] * factor
            
        return x, y
    
    def move_to(self, end_x: float, end_y: float, 
               start_x: Optional[float] = None, 
               start_y: Optional[float] = None, 
               click: bool = False, right_click: bool = False,
               double_click: bool = False) -> None:
        """
        Move the mouse to target coordinates with natural motion.
        
        Args:
            end_x, end_y: Target coordinates.
            start_x, start_y: Starting coordinates. If None, current position is used.
            click: Whether to perform a left click after moving.
            right_click: Whether to perform a right click after moving.
            double_click: Whether to perform a double click after moving.
        """
        # If start not provided, use current position
        if start_x is None or start_y is None:
            start_x, start_y = pyautogui.position()
            
        # Generate path
        path = self.generate_path(start_x, start_y, end_x, end_y)
        
        # Execute the movement using interpolation for smoother motion
        if len(path) < 2:
            return
            
        # For smoother movement, use a continuous approach instead of discrete jumps
        # Create a list of points and cumulative times for interpolation
        points = []
        times = []
        cumulative_time = 0
        
        for point in path:
            if point["time"] > 0:
                cumulative_time += point["time"] / 1000
                points.append((point["x"], point["y"]))
                times.append(cumulative_time)
        
        if not points:
            # Fallback to direct movement if no valid points
            pyautogui.moveTo(end_x, end_y)
            return
        
        start_time = time.time()
        pyautogui.moveTo(start_x, start_y, duration=0)
        
        # Use more granular interpolation for smoother movement
        total_duration = cumulative_time if cumulative_time > 0 else 0.5
        steps = min(int(total_duration * 100), 250)  # 100 steps per second, max 250
        
        for step in range(1, steps + 1):
            # Calculate the current time position
            progress = step / steps
            current_time = total_duration * progress
            
            # Find the appropriate segment in the path
            idx = 0
            while idx < len(times) - 1 and current_time > times[idx]:
                idx += 1
                
            # Interpolate between points
            if idx == 0:
                # Before first point, interpolate from start to first point
                t = current_time / times[0] if times[0] > 0 else 1
                x = start_x + (points[0][0] - start_x) * t
                y = start_y + (points[0][1] - start_y) * t
            else:
                # Between two points in the path
                prev_time = times[idx-1]
                next_time = times[idx]
                t = (current_time - prev_time) / (next_time - prev_time) if next_time > prev_time else 1
                
                x = points[idx-1][0] + (points[idx][0] - points[idx-1][0]) * t
                y = points[idx-1][1] + (points[idx][1] - points[idx-1][1]) * t
            
            # Move to the interpolated position
            pyautogui.moveTo(x, y, _pause=False)
            
            # Calculate time until next step should occur
            elapsed = time.time() - start_time
            next_time = total_duration * ((step + 1) / steps)
            sleep_time = max(0, next_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Ensure we end at exactly the target position
        pyautogui.moveTo(end_x, end_y, _pause=False)
        
        # Perform click actions if requested
        if double_click:
            pyautogui.doubleClick()
        elif click:
            pyautogui.click()
        elif right_click:
            pyautogui.rightClick()
