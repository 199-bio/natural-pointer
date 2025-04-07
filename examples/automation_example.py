"""
Example of using natural-pointer for automation tasks.
"""

import time
import pyautogui
from natural_pointer import NaturalPointer

def automate_form_filling():
    """
    Demonstrates using natural pointer for automating form filling.
    
    This example assumes a simple form is open in the browser.
    Adjust coordinates as needed for your specific use case.
    """
    # Initialize pointer with pre-trained model (if exists)
    pointer = NaturalPointer(model_path="form_filling_model.pth")
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Example coordinates for form elements (adjust for your screen)
    form_elements = {
        "name_field": (screen_width // 2, screen_height // 3),
        "email_field": (screen_width // 2, screen_height // 3 + 50),
        "submit_button": (screen_width // 2, screen_height // 3 + 150)
    }
    
    print("Starting automated form filling with natural mouse movements...")
    
    # Click on name field
    print("Moving to name field...")
    pointer.move_to(
        form_elements["name_field"][0], 
        form_elements["name_field"][1],
        click=True
    )
    time.sleep(0.5)
    
    # Type name
    print("Typing name...")
    pyautogui.typewrite("John Doe", interval=0.1)
    time.sleep(1)
    
    # Click on email field
    print("Moving to email field...")
    pointer.move_to(
        form_elements["email_field"][0],
        form_elements["email_field"][1],
        click=True
    )
    time.sleep(0.5)
    
    # Type email
    print("Typing email...")
    pyautogui.typewrite("john.doe@example.com", interval=0.1)
    time.sleep(1)
    
    # Click submit button
    print("Moving to and clicking submit button...")
    pointer.move_to(
        form_elements["submit_button"][0],
        form_elements["submit_button"][1],
        click=True
    )
    
    print("Form submitted!")

def automation_demo():
    """Run the complete automation demo."""
    try:
        print("This demo will simulate filling out a form with natural mouse movements.")
        print("Please note: no actual form will appear - this is just a simulation.")
        print("In a real application, you would integrate this with actual UI elements.")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        automate_form_filling()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

if __name__ == "__main__":
    automation_demo()