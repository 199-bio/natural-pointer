#!/usr/bin/env python3
"""
Natural Pointer Automation Test

This script demonstrates how to use the Natural Pointer library in a real-world 
automation scenario. It uses a pre-trained model to simulate natural mouse 
movements while performing automated tasks.

Model: boris.pth (pre-trained model)
"""

import os
import sys
import time
import random
import pyautogui

# Ensure the required packages are available
try:
    from natural_pointer import NaturalPointer
except ImportError:
    print("Error: Natural Pointer package not found.")
    print("Make sure you've installed it with 'pip install -e .'")
    sys.exit(1)

# Set up constants
MODEL_PATH = "boris.pth"  # Path to your pre-trained model
PAUSE_BETWEEN_ACTIONS = 1.5  # Pause duration between actions (seconds)

class AutomationDemo:
    """Demonstrates Natural Pointer in automation scenarios."""
    
    def __init__(self):
        """Initialize the automation demo."""
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file {MODEL_PATH} not found.")
            print(f"The automation will use bezier curves instead of the neural network.")
            print(f"Expected model at: {os.path.abspath(MODEL_PATH)}")
        
        # Initialize Natural Pointer with the trained model
        self.pointer = NaturalPointer(model_path=MODEL_PATH)
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize successful action counter
        self.successful_actions = 0
        self.total_actions = 0
    
    def pause(self, duration=None):
        """Add a natural pause between actions."""
        if duration is None:
            # Add slight randomness to pauses for more natural behavior
            duration = PAUSE_BETWEEN_ACTIONS * random.uniform(0.8, 1.2)
        
        time.sleep(duration)
    
    def simulate_form_filling(self):
        """Simulate filling out a form."""
        print("\n📋 Form Filling Simulation")
        print("========================")
        
        # Define form field positions (center and quadrants of screen)
        # In a real scenario, these would be actual UI element coordinates
        form_fields = {
            "name": (self.screen_width // 4, self.screen_height // 3),
            "email": (self.screen_width // 4, self.screen_height // 2),
            "submit": (self.screen_width // 4, self.screen_height * 2 // 3)
        }
        
        try:
            # Click on name field
            print("🖱️ Moving to name field...")
            self.total_actions += 1
            self.pointer.click(form_fields["name"][0], form_fields["name"][1])
            self.successful_actions += 1
            self.pause()
            
            # Type name (using PyAutoGUI directly)
            print("⌨️ Typing name: John Doe")
            pyautogui.typewrite("John Doe", interval=0.08)  # Slight delay between keystrokes
            self.pause()
            
            # Click on email field
            print("🖱️ Moving to email field...")
            self.total_actions += 1
            self.pointer.click(form_fields["email"][0], form_fields["email"][1])
            self.successful_actions += 1
            self.pause()
            
            # Type email
            print("⌨️ Typing email: john.doe@example.com")
            pyautogui.typewrite("john.doe@example.com", interval=0.07)
            self.pause()
            
            # Click submit button
            print("🖱️ Moving to submit button...")
            self.total_actions += 1
            self.pointer.click(form_fields["submit"][0], form_fields["submit"][1])
            self.successful_actions += 1
            self.pause()
            
            print("✅ Form submission simulated successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error in form filling simulation: {str(e)}")
            return False
    
    def simulate_drag_and_drop(self):
        """Simulate drag and drop operations."""
        print("\n🖱️ Drag and Drop Simulation")
        print("=========================")
        
        # Define source and target positions
        sources = [
            (self.screen_width // 4, self.screen_height // 4),
            (self.screen_width // 4, self.screen_height * 3 // 4)
        ]
        
        targets = [
            (self.screen_width * 3 // 4, self.screen_height // 4),
            (self.screen_width * 3 // 4, self.screen_height * 3 // 4)
        ]
        
        try:
            for i, (source, target) in enumerate(zip(sources, targets)):
                print(f"🖱️ Drag and drop item {i+1}:")
                print(f"   From: ({source[0]}, {source[1]})")
                print(f"   To: ({target[0]}, {target[1]})")
                
                # Move to source
                self.total_actions += 1
                self.pointer.move_to(source[0], source[1])
                self.successful_actions += 1
                self.pause(0.5)
                
                # Perform drag and drop (using PyAutoGUI)
                pyautogui.mouseDown()
                self.pause(0.3)
                
                # Move to target with natural motion
                self.total_actions += 1
                self.pointer.move_to(target[0], target[1])
                self.successful_actions += 1
                self.pause(0.3)
                
                # Release
                pyautogui.mouseUp()
                self.pause()
            
            print("✅ Drag and drop operations completed successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error in drag and drop simulation: {str(e)}")
            return False
    
    def simulate_menu_navigation(self):
        """Simulate navigating through a menu hierarchy."""
        print("\n📋 Menu Navigation Simulation")
        print("===========================")
        
        # Define menu positions
        # In a real automation, these would be actual menu coordinates
        menu_positions = {
            "main_menu": (self.screen_width // 2, self.screen_height // 6),
            "submenu_1": (self.screen_width // 2 + 100, self.screen_height // 6 + 50),
            "submenu_2": (self.screen_width // 2 + 200, self.screen_height // 6 + 100),
            "menu_item": (self.screen_width // 2 + 200, self.screen_height // 6 + 150)
        }
        
        try:
            # Click main menu
            print("🖱️ Opening main menu...")
            self.total_actions += 1
            self.pointer.click(menu_positions["main_menu"][0], menu_positions["main_menu"][1])
            self.successful_actions += 1
            self.pause()
            
            # Hover over first submenu
            print("🖱️ Hovering over first submenu...")
            self.total_actions += 1
            self.pointer.move_to(menu_positions["submenu_1"][0], menu_positions["submenu_1"][1])
            self.successful_actions += 1
            self.pause()
            
            # Hover over second submenu
            print("🖱️ Hovering over second submenu...")
            self.total_actions += 1
            self.pointer.move_to(menu_positions["submenu_2"][0], menu_positions["submenu_2"][1])
            self.successful_actions += 1
            self.pause()
            
            # Click menu item
            print("🖱️ Clicking menu item...")
            self.total_actions += 1
            self.pointer.click(menu_positions["menu_item"][0], menu_positions["menu_item"][1])
            self.successful_actions += 1
            self.pause()
            
            print("✅ Menu navigation completed successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error in menu navigation simulation: {str(e)}")
            return False
    
    def simulate_drawing(self):
        """Simulate drawing a simple shape."""
        print("\n🎨 Drawing Simulation")
        print("===================")
        
        # Define the center of our drawing area
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        radius = min(self.screen_width, self.screen_height) // 6
        
        try:
            # Move to starting position
            print("🖱️ Moving to starting position...")
            self.total_actions += 1
            start_x = center_x + radius
            start_y = center_y
            self.pointer.move_to(start_x, start_y)
            self.successful_actions += 1
            self.pause(0.5)
            
            # Start drawing (press mouse down)
            print("🖱️ Drawing a circle...")
            pyautogui.mouseDown()
            self.pause(0.2)
            
            # Draw a circle using 8 points
            points = 12
            for i in range(1, points + 1):
                angle = 2 * 3.14159 * i / points
                x = center_x + int(radius * math.cos(angle))
                y = center_y + int(radius * math.sin(angle))
                
                self.total_actions += 1
                self.pointer.move_to(x, y)
                self.successful_actions += 1
                self.pause(0.2)
            
            # Finish drawing (release mouse)
            pyautogui.mouseUp()
            self.pause()
            
            print("✅ Drawing completed successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error in drawing simulation: {str(e)}")
            # Make sure to release the mouse button if an error occurs
            pyautogui.mouseUp()
            return False
    
    def run_all_demos(self):
        """Run all automation demos."""
        print("\n🚀 Starting Natural Pointer Automation Demos")
        print("==========================================")
        print(f"Using model: {MODEL_PATH}")
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        print("\nNOTE: You can press Ctrl+C at any time to stop the demo.")
        
        try:
            # Pause before starting to give user time to prepare
            print("\nStarting demos in 3 seconds...")
            time.sleep(3)
            
            # Run form filling demo
            self.simulate_form_filling()
            
            # Run drag and drop demo
            self.simulate_drag_and_drop()
            
            # Run menu navigation demo
            self.simulate_menu_navigation()
            
            # Try to import math for the drawing demo
            try:
                import math
                # Run drawing demo
                self.simulate_drawing()
            except ImportError:
                print("Skipping drawing demo (math module not available)")
            
            # Return to center
            print("\n🖱️ Returning to screen center...")
            self.pointer.move_to(self.screen_width // 2, self.screen_height // 2)
            
            # Show completion message with success rate
            success_rate = (self.successful_actions / self.total_actions * 100) if self.total_actions > 0 else 0
            print("\n✨ All automation demos completed!")
            print(f"Success rate: {success_rate:.1f}% ({self.successful_actions}/{self.total_actions} actions)")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Automation demos interrupted by user.")
            print("Returning mouse to center of screen...")
            self.pointer.move_to(self.screen_width // 2, self.screen_height // 2)
        except Exception as e:
            print(f"\n\n❌ Error in automation demos: {str(e)}")
            print("Returning mouse to center of screen...")
            try:
                self.pointer.move_to(self.screen_width // 2, self.screen_height // 2)
            except:
                pass

if __name__ == "__main__":
    # Create and run the automation demo
    demo = AutomationDemo()
    demo.run_all_demos()