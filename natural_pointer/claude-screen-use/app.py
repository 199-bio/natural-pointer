import os
import time
import json
import base64
import pyautogui
import anthropic
from natural_pointer import NaturalPointer

# Get API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it before running the script.")

# Initialize Claude client with the API key
client = anthropic.Anthropic(api_key=api_key)

# Initialize Natural Pointer
pointer = NaturalPointer()

def take_screenshot():
    """Take a screenshot and save it to a file"""
    screenshot = pyautogui.screenshot()
    screenshot_path = 'screenshot.png'
    screenshot.save(screenshot_path)
    return screenshot_path

def encode_image_to_base64(image_path):
    """Convert image to base64 encoding for Claude API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_element_with_claude(screenshot_path, element_description):
    """
    Use Claude Vision to find an element in the screenshot
    Returns coordinates if found, None otherwise
    """
    base64_image = encode_image_to_base64(screenshot_path)
    
    # Prepare the message for Claude with the screenshot and task
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system="You are a computer vision assistant that helps identify UI elements in screenshots. Return ONLY a JSON object with the coordinates of the element described.",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Find the {element_description} in this screenshot. Return a JSON object with the format: {{\"x\": center_x_coordinate, \"y\": center_y_coordinate}}. If you can't find it, return {{\"found\": false}}."
                    }
                ]
            }
        ]
    )
    
    # Extract the response text
    response_text = message.content[0].text
    
    # Try to parse the JSON response
    try:
        # Extract just the JSON part if there's other text
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
            
        result = json.loads(json_str)
        
        if "found" in result and result["found"] is False:
            print(f"Element '{element_description}' not found")
            return None
        
        if "x" in result and "y" in result:
            print(f"Found {element_description} at coordinates: ({result['x']}, {result['y']})")
            return result["x"], result["y"]
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Claude's response: {e}")
        print(f"Raw response: {response_text}")
    
    return None

def click_element(element_description):
    """
    Take a screenshot, find the element using Claude vision, and click it using Natural Pointer
    """
    screenshot_path = take_screenshot()
    coordinates = find_element_with_claude(screenshot_path, element_description)
    
    if coordinates:
        x, y = coordinates
        print(f"Moving to and clicking at coordinates: ({x}, {y})")
        pointer.click(x, y)
        return True
    else:
        print(f"Could not find and click '{element_description}'")
        return False

if __name__ == "__main__":
    # Example usage
    element_to_find = input("Enter the UI element to find and click (e.g., 'close button', 'login button'): ")
    click_element(element_to_find)
