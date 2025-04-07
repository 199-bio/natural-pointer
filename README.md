# Natural Pointer

A lightweight Python library for AI-driven mouse movement simulation that replicates human-like mouse behavior.

## Overview

Natural Pointer uses machine learning to analyze and mimic your unique mouse movement patterns, bringing natural motion to automated tasks. The library can fall back to sophisticated bezier curves when no trained model is available.

## Installation

```bash
# Install from PyPI
pip install natural-pointer


## Simple Usage

```python
from natural_pointer import NaturalPointer

# Create an instance (works even without a trained model)
pointer = NaturalPointer()

# Basic movement
pointer.move_to(500, 300)

# Click operations
pointer.click(800, 400)       # Move and left-click
pointer.right_click(700, 200) # Move and right-click
pointer.double_click(300, 500) # Move and double-click
```

## Features

- Fluid, natural mouse movement that mimics human behavior
- Automatic adaptation to any screen resolution
- Natural acceleration and deceleration curves
- Fine-tuned micro-adjustments and pauses
- Integrated click operations (left, right, double)
- Fallback to bezier curves if no model is available

## Training a Custom Model

For optimal results, train a model based on your own mouse movements:

```bash
# Record your mouse movements
python -m natural_pointer.recorder record

# Train a model on your recorded data
python -m natural_pointer.recorder train

# Use your custom model
from natural_pointer import NaturalPointer
pointer = NaturalPointer(model_path="mouse_model.pth")
```

## Automation Example

```python
from natural_pointer import NaturalPointer
import pyautogui
import time

pointer = NaturalPointer()

# Simple form fill automation
def automate_login():
    # Navigate to username field and enter text
    pointer.click(500, 200)
    pyautogui.typewrite("username")
    time.sleep(0.5)
    
    # Navigate to password field and enter text
    pointer.click(500, 250)
    pyautogui.typewrite("password")
    time.sleep(0.5)
    
    # Click login button
    pointer.click(500, 300)

automate_login()
```

## API Reference

### Main Class

- `NaturalPointer(model_path="mouse_model.pth")` - Initialize with optional path to trained model

### Movement Methods

- `move_to(x, y)` - Move to coordinates with natural motion
- `click(x, y)` - Move and perform left click
- `right_click(x, y)` - Move and perform right click
- `double_click(x, y)` - Move and perform double click

### Information Method

- `get_info()` - Returns dictionary with cursor position, screen size, and model information

## Requirements

- Python 3.7+
- PyTorch
- PyAutoGUI
- pynput
- numpy
- matplotlib (optional, for visualization)

## Developed By

This tool was developed as part of the initiatives at [199 Longevity](https://199longevity.com), a group focused on extending the frontiers of human health and longevity.

Learn more about our work in biotechnology at 199.bio.

Project contributor: Boris Djordjevic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
