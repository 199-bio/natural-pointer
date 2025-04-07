"""
Simple example of using the Natural Pointer library.
"""

from natural_pointer import NaturalPointer
import pyautogui
import time

def main():
    """
    Demonstrates basic usage of the Natural Pointer library.
    """
    print("Natural Pointer Demo")
    print("-------------------")
    
    # Initialize the natural pointer with a trained model
    # (uses fallback bezier curves if model doesn't exist)
    pointer = NaturalPointer(model_path="mouse_model.pth")
    
    # Get information about current setup
    info = pointer.get_info()
    print(f"Screen size: {info['screen_width']}x{info['screen_height']}")
    print(f"Model loaded: {info['model_loaded']}")
    print(f"Current mouse position: {info['current_position']}")
    
    # Get screen center 
    center_x = info["screen_width"] // 2
    center_y = info["screen_height"] // 2
    
    print(f"\nDemonstrating natural mouse movements")
    print("Moving mouse to various screen positions with natural motion")
    print("Press Ctrl+C at any time to stop the demo\n")
    
    try:
        # Move to center
        print(f"Moving to center: ({center_x}, {center_y})")
        pointer.move_to(center_x, center_y)
        time.sleep(1)
        
        # Move to top-left with click
        dest_x, dest_y = info["screen_width"] // 4, info["screen_height"] // 4
        print(f"Moving to top-left with click: ({dest_x}, {dest_y})")
        pointer.click(dest_x, dest_y)
        time.sleep(1)
        
        # Move to top-right with right click
        dest_x, dest_y = info["screen_width"] * 3 // 4, info["screen_height"] // 4
        print(f"Moving to top-right with right-click: ({dest_x}, {dest_y})")
        pointer.right_click(dest_x, dest_y)
        time.sleep(1)
        
        # Move to bottom-left with double click
        dest_x, dest_y = info["screen_width"] // 4, info["screen_height"] * 3 // 4
        print(f"Moving to bottom-left with double-click: ({dest_x}, {dest_y})")
        pointer.double_click(dest_x, dest_y)
        time.sleep(1)
        
        # Move to bottom-right
        dest_x, dest_y = info["screen_width"] * 3 // 4, info["screen_height"] * 3 // 4
        print(f"Moving to bottom-right: ({dest_x}, {dest_y})")
        pointer.move_to(dest_x, dest_y)
        time.sleep(1)
        
        # Return to center
        print(f"Returning to center: ({center_x}, {center_y})")
        pointer.move_to(center_x, center_y)
        
        print("\nDemo completed!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

if __name__ == "__main__":
    main()