"""
Basic example of using the natural-pointer library.
"""

import time
from natural_pointer import NaturalPointer

def main():
    """
    Demonstrates the basic usage of the Natural Pointer library.
    """
    # Create a natural pointer instance
    pointer = NaturalPointer()
    
    # Option 1: Record mouse movements for training
    print("Recording mouse movements for 20 seconds. Move your mouse naturally...")
    pointer.record(duration=20)
    
    # Option 2: Train the model
    print("\nTraining model on recorded data...")
    pointer.train(epochs=50)
    
    # Option 3: Use the trained model to move the mouse
    print("\nDemonstrating natural mouse movements:")
    
    # Move to center of screen
    screen_width, screen_height = pointer.simulator.screen_width, pointer.simulator.screen_height
    center_x, center_y = screen_width // 2, screen_height // 2
    
    print(f"Moving to center: ({center_x}, {center_y})")
    pointer.move_to(center_x, center_y)
    time.sleep(1)
    
    # Move to top-left quadrant
    dest_x, dest_y = screen_width // 4, screen_height // 4
    print(f"Moving to top-left: ({dest_x}, {dest_y})")
    pointer.move_to(dest_x, dest_y)
    time.sleep(1)
    
    # Move to bottom-right quadrant with click
    dest_x, dest_y = screen_width * 3 // 4, screen_height * 3 // 4
    print(f"Moving to bottom-right with click: ({dest_x}, {dest_y})")
    pointer.move_to(dest_x, dest_y, click=True)
    time.sleep(1)
    
    # Return to center with right-click
    print(f"Moving back to center with right-click: ({center_x}, {center_y})")
    pointer.move_to(center_x, center_y, right_click=True)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()