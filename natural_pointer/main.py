"""
Main module for natural pointer - provides easy access to the library functionality.
This module focuses only on using an already trained model to simulate natural
mouse movements.
"""

import os
import time
import pyautogui
from typing import Optional, Dict, Any

from .models.neural_network import MouseMovementNN
from .utils.simulator import MouseSimulator

class NaturalPointer:
    """
    Main class for the natural pointer library - provides a simple interface
    for natural mouse movements in automated tasks.
    """
    
    def __init__(self, model_path: str = "mouse_model.pth"):
        """
        Initialize the natural pointer with a trained model.
        
        Args:
            model_path: Path to the trained model file. If the model doesn't exist,
                       it will fall back to using bezier curves for movement.
        """
        self.model_path = model_path
        
        # Create simulator which handles model loading internally
        self.simulator = MouseSimulator(model_path=model_path)
    
    def move_to(self, x: int, y: int, click: bool = False, 
                right_click: bool = False, double_click: bool = False) -> None:
        """
        Move the mouse to the specified coordinates with natural motion.
        
        Args:
            x, y: Target coordinates.
            click: Whether to perform a left click after moving.
            right_click: Whether to perform a right click after moving.
            double_click: Whether to perform a double click after moving.
        """
        self.simulator.move_to(
            end_x=x, 
            end_y=y, 
            click=click,
            right_click=right_click,
            double_click=double_click
        )
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """
        Perform a click at the specified coordinates.
        If no coordinates are provided, clicks at the current position.
        
        Args:
            x, y: Coordinates to click at. If None, clicks at current position.
        """
        if x is not None and y is not None:
            self.move_to(x, y, click=True)
        else:
            pyautogui.click()
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """
        Perform a right click at the specified coordinates.
        If no coordinates are provided, right-clicks at the current position.
        
        Args:
            x, y: Coordinates to right-click at. If None, right-clicks at current position.
        """
        if x is not None and y is not None:
            self.move_to(x, y, right_click=True)
        else:
            pyautogui.rightClick()
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """
        Perform a double click at the specified coordinates.
        If no coordinates are provided, double-clicks at the current position.
        
        Args:
            x, y: Coordinates to double-click at. If None, double-clicks at current position.
        """
        if x is not None and y is not None:
            self.move_to(x, y, double_click=True)
        else:
            pyautogui.doubleClick()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the natural pointer.
        
        Returns:
            Dictionary with information about the natural pointer.
        """
        screen_width, screen_height = pyautogui.size()
        
        return {
            "model_path": self.model_path,
            "model_loaded": self.simulator.model_loaded, # Get status from simulator
            "screen_width": screen_width,
            "screen_height": screen_height,
            "current_position": pyautogui.position()
        }
