# Claude Screen Use

This is a simple MVP application that combines Claude Vision capabilities with Natural-pointer to:
1. Take a screenshot of your screen
2. Use Claude Vision to identify UI elements
3. Move the mouse naturally and click on the identified element

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

When prompted, enter a description of the UI element you want to find and click (e.g., "close button", "Settings icon", "File menu").

## How it works

1. The application takes a screenshot of your current screen
2. The screenshot is sent to Claude's Vision API with your description
3. Claude returns the coordinates of the identified element
4. Natural-pointer moves the mouse in a human-like pattern to those coordinates and clicks

## Requirements

- Python 3.7+
- Anthropic API key
- Internet connection for Claude API calls