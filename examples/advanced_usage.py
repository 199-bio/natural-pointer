"""
Advanced usage examples for the natural-pointer library.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict

from natural_pointer import NaturalPointer
from natural_pointer.models.neural_network import MouseMovementNN
from natural_pointer.data.tracker import MouseTracker
from natural_pointer.utils.simulator import MouseSimulator

def visualize_movements(path: List[Dict]):
    """
    Visualize a generated mouse movement path.
    
    Args:
        path: List of points with x, y coordinates
    """
    # Extract x, y coordinates
    x_values = [point['x'] for point in path]
    y_values = [point['y'] for point in path]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot the path as a line
    plt.plot(x_values, y_values, 'b-', alpha=0.5, label='Path')
    
    # Plot start and end points
    plt.plot(x_values[0], y_values[0], 'go', markersize=10, label='Start')
    plt.plot(x_values[-1], y_values[-1], 'ro', markersize=10, label='End')
    
    # Plot all points
    plt.scatter(x_values, y_values, c=range(len(x_values)), 
                cmap='viridis', s=30, alpha=0.8, label='Points')
    
    # Invert y-axis to match screen coordinates (origin at top-left)
    plt.gca().invert_yaxis()
    
    # Add labels and legend
    plt.title('Mouse Movement Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.colorbar(label='Point Sequence')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def custom_model_training():
    """
    Demonstrates creating and training a custom model.
    """
    # Create a custom model with different architecture
    custom_model = MouseMovementNN(
        input_size=30,  # 5 points with 6 features each
        hidden_size=48, # Larger hidden layer
        output_size=3   # x, y, time
    )
    
    # Add an extra layer for more complexity
    custom_model.model = torch.nn.Sequential(
        torch.nn.Linear(30, 48),
        torch.nn.ReLU(),
        torch.nn.Linear(48, 48),
        torch.nn.ReLU(),
        torch.nn.Linear(48, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 3)
    )
    
    # Initialize the trainer with this model
    pointer = NaturalPointer(
        data_path="custom_data.json",
        model_path="custom_model.pth"
    )
    pointer.model = custom_model
    pointer.trainer.model = custom_model
    
    # Record some data - commented out to avoid actual recording during example
    # Set a specific duration to avoid needing to interrupt manually
    # pointer.record(duration=30)
    
    # Train with custom parameters
    # pointer.trainer.train(epochs=200, batch_size=16, validation_split=0.2)
    # pointer.trainer.save_model()
    
    # The model can now be used as usual
    # pointer.move_to(500, 500, click=True)
    print("Custom model example - see code for details")

def compare_bezier_vs_neural():
    """
    Compare bezier curve movement with neural network movement.
    """
    # Initialize simulator
    simulator = MouseSimulator()
    
    # Coordinates for testing
    start_x, start_y = 100, 100
    end_x, end_y = 700, 500
    
    # Generate bezier path (using internal method)
    print("Generating bezier curve path...")
    bezier_path = simulator._generate_bezier_path(
        start_x, start_y, end_x, end_y, steps=30
    )
    
    # Simulate the bezier movement
    print("Moving along bezier path...")
    pyautogui.moveTo(start_x, start_y, duration=0)
    for point in bezier_path:
        pyautogui.moveTo(point["x"], point["y"], _pause=False)
        time.sleep(point["time"] / 1000)
    
    time.sleep(1)
    
    # Try to load neural model - will fall back to bezier if not available
    print("Testing neural network path (if model exists)...")
    neural_path = simulator.generate_path(start_x, start_y, end_x, end_y)
    
    # Visualize the paths (uncomment to see plots)
    # visualize_movements(bezier_path)
    # visualize_movements(neural_path)
    
    print("Paths generated. Uncomment visualization code to see plots.")

def main():
    """Run the advanced examples."""
    print("Advanced Natural Pointer Examples")
    print("---------------------------------")
    print("1. Custom model training (code example)")
    print("2. Compare bezier vs neural network movement")
    print("3. Run both examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-3): ")
    
    if choice == "1":
        custom_model_training()
    elif choice == "2":
        compare_bezier_vs_neural()
    elif choice == "3":
        custom_model_training()
        print("\n")
        compare_bezier_vs_neural()
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()