#!/usr/bin/env python3
"""
Mouse Movement Recorder and Trainer

This standalone application records mouse movements and trains a neural network
model that can later be used by the natural_pointer library to simulate natural
mouse movements in automated tasks.

Usage:
    python recorder.py record  # Record mouse movements (Ctrl+C to stop)
    python recorder.py train   # Train model on recorded data
    python recorder.py visualize  # Visualize recorded movements
    python recorder.py analyze  # Analyze existing data and suggest training parameters
"""

import os
import sys
import time
import json
import argparse
import datetime

# Check for required packages before importing them
def check_dependencies():
    """Check if all required packages are installed and provide installation instructions if not."""
    missing_packages = []
    
    try:
        import pyautogui
    except ImportError:
        missing_packages.append("pyautogui")
    
    try:
        import pynput
    except ImportError:
        missing_packages.append("pynput")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import matplotlib.pyplot
    except ImportError:
        missing_packages.append("matplotlib")
    
    if missing_packages:
        print("Error: Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install the missing packages using one of these methods:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

# Check dependencies before continuing
check_dependencies()

# Now it's safe to import the packages
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project modules
from natural_pointer.data.tracker import MouseTracker
from natural_pointer.models.neural_network import MouseModelTrainer, MouseMovementNN

def load_existing_data(file_path):
    """Load existing mouse movement data from a file, if it exists."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                return existing_data
        except Exception as e:
            print(f"Warning: Error loading existing data from {file_path}: {str(e)}")
            print("Starting with empty dataset.")
    return []

def save_merged_data(file_path, data):
    """Save merged mouse movement data to a file."""
    try:
        # Create backup of original file if it exists
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup.{int(time.time())}"
            try:
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                print(f"Backup created at {backup_path}")
            except Exception as e:
                print(f"Warning: Failed to create backup: {str(e)}")
        
        # Save the merged data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def record_movements(args):
    """Record mouse movements and add to existing data if available."""
    # Load existing data if file exists and append mode is enabled
    existing_data = []
    if args.append and os.path.exists(args.output):
        existing_data = load_existing_data(args.output)
        data_points = len(existing_data)
        print(f"Found existing data with {data_points} events. New movements will be appended.")
    
    print(f"Recording mouse movements to {args.output}")
    print("Move your mouse naturally. Press Ctrl+C to stop recording.")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Recording started at: {timestamp}")
    
    tracker = MouseTracker(save_path=args.output)
    tracker.start_recording()
    
    try:
        if args.duration:
            print(f"Recording will automatically stop after {args.duration} seconds.")
            time.sleep(args.duration)
            movements = tracker.stop_recording()
            print(f"Recording completed after {args.duration} seconds.")
        else:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        movements = tracker.stop_recording()
        print("\nRecording stopped by user.")
    
    # Merge with existing data if in append mode
    if args.append and existing_data:
        # Add a small timestamp field to separate recording sessions
        session_timestamp = time.time()
        for m in movements:
            m['session'] = session_timestamp
        
        merged_data = existing_data + movements
        print(f"Adding {len(movements)} new events to existing {len(existing_data)} events.")
        
        if save_merged_data(args.output, merged_data):
            print(f"Merged data saved to {args.output}")
            print(f"Total events: {len(merged_data)}")
            return True
    else:
        # Save new data directly through tracker
        if tracker.save_data():
            print(f"Recorded {len(movements)} mouse events.")
            print(f"Data saved to {args.output}")
            return True
        else:
            print("No data was recorded or there was an error saving the data.")
            return False

def train_model(args):
    """Train a model on recorded mouse movements."""
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found!")
        return
    
    # Analyze data first to provide feedback on training parameters
    data_stats = analyze_data(args, print_results=False)
    
    if data_stats['total_events'] < 100:
        print(f"Warning: Dataset contains only {data_stats['total_events']} events.")
        print("For better results, consider recording more data (recommended: 500+ events).")
    
    # Suggest epochs based on data size
    suggested_epochs = min(200, max(50, data_stats['total_events'] // 10))
    if args.epochs is None:
        args.epochs = suggested_epochs
        print(f"Using {args.epochs} epochs based on data size.")
    elif args.epochs < suggested_epochs * 0.5:
        print(f"Warning: Specified epochs ({args.epochs}) may be too low for your data size.")
        print(f"Suggested epochs: {suggested_epochs}")
    elif args.epochs > suggested_epochs * 2:
        print(f"Warning: Specified epochs ({args.epochs}) may be unnecessarily high.")
        print(f"Suggested epochs: {suggested_epochs}")
    
    # Suggest batch size based on data size
    suggested_batch = min(64, max(8, data_stats['total_events'] // 20))
    if args.batch_size is None:
        args.batch_size = suggested_batch
        print(f"Using batch size {args.batch_size} based on data size.")
    
    print(f"Training model on data from {args.data}")
    print(f"This may take a few minutes...")
    print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
    
    # Create model and trainer
    model = MouseMovementNN()
    trainer = MouseModelTrainer(
        model=model,
        data_path=args.data,
        model_save_path=args.model
    )
    
    try:
        # Train the model
        history = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        
        # Save model
        if trainer.save_model():
            print(f"Model trained and saved to {args.model}")
            
            # Print final loss values
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            print(f"Final training loss: {final_train_loss:.6f}")
            print(f"Final validation loss: {final_val_loss:.6f}")
            
            # Evaluate if the model might be overfitting or underfitting
            if final_train_loss < 0.001:
                print("\nNote: Training loss is very low, which might indicate overfitting.")
                print("      Consider using more data or fewer epochs.")
            
            if final_val_loss > 0.1:
                print("\nNote: Validation loss is high, which might indicate underfitting.")
                print("      Consider using more data or more epochs.")
            
            ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 1
            if ratio > 1.5:
                print("\nNote: Validation loss is significantly higher than training loss,")
                print("      which might indicate overfitting.")
                print("      Consider using more varied mouse movement data.")
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.splitext(args.model)[0] + '_training.png'
            plt.savefig(plot_path)
            print(f"Training plot saved to {plot_path}")
            
            if args.show_plot:
                plt.show()
                
            return True
        else:
            print("Error: Failed to save the model.")
            return False
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\nThis might be due to invalid data format or insufficient data.")
        print("Try recording more mouse movements or check the data file format.")
        return False

def visualize_data(args):
    """Visualize recorded mouse movement data."""
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found!")
        return
    
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("Error: No data found in the file.")
            return
        
        print(f"Visualizing {len(data)} mouse events from {args.data}")
        
        # Check if data has multiple sessions
        sessions = set()
        for point in data:
            if 'session' in point:
                sessions.add(point['session'])
        
        if len(sessions) > 1 and not args.combine_sessions:
            print(f"Found {len(sessions)} recording sessions in the data.")
            print("Visualizing the most recent session. Use --combine-sessions to view all.")
            latest_session = max(sessions)
            data = [point for point in data if point.get('session', latest_session) == latest_session]
            print(f"Showing {len(data)} events from the latest session.")
        
        # Extract movement data
        move_events = [point for point in data if point.get('type', '') == 'move']
        click_events = [point for point in data if point.get('type', '') == 'click' and point.get('pressed', False)]
        scroll_events = [point for point in data if point.get('type', '') == 'scroll']
        
        if not move_events:
            print("Error: No movement data found.")
            return
        
        # Extract coordinates
        x_values = [point['x'] for point in move_events]
        y_values = [point['y'] for point in move_events]
        
        # Create main plot
        plt.figure(figsize=(12, 8))
        
        # Plot the movement path
        plt.plot(x_values, y_values, 'b-', alpha=0.5, linewidth=1, label='Mouse Path')
        
        # Plot points with color gradient to show sequence
        points = plt.scatter(x_values, y_values, c=range(len(x_values)), 
                  cmap='viridis', s=15, alpha=0.7)
        plt.colorbar(points, label='Time Sequence')
        
        # Plot click events
        if click_events:
            click_x = [point['x'] for point in click_events]
            click_y = [point['y'] for point in click_events]
            plt.scatter(click_x, click_y, c='red', s=100, marker='x', label='Clicks')
        
        # Plot scroll events
        if scroll_events:
            scroll_x = [point['x'] for point in scroll_events]
            scroll_y = [point['y'] for point in scroll_events]
            plt.scatter(scroll_x, scroll_y, c='green', s=80, marker='^', label='Scrolls')
        
        # Mark start and end
        plt.scatter([x_values[0]], [y_values[0]], c='green', s=100, marker='o', label='Start')
        plt.scatter([x_values[-1]], [y_values[-1]], c='red', s=100, marker='o', label='End')
        
        # Invert y-axis to match screen coordinates
        plt.gca().invert_yaxis()
        
        # Add labels and title
        title = 'Recorded Mouse Movements'
        if len(sessions) > 1:
            if args.combine_sessions:
                title += f' (All {len(sessions)} Sessions)'
            else:
                title += ' (Latest Session)'
        plt.title(title)
        plt.xlabel('X (normalized)')
        plt.ylabel('Y (normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add information about the data
        info_text = (
            f"Total events: {len(data)}\n"
            f"Movement points: {len(move_events)}\n"
            f"Click events: {len(click_events)}\n"
            f"Scroll events: {len(scroll_events)}"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10)
        
        plt.tight_layout()
        
        # Save the visualization
        if args.output:
            plt.savefig(args.output)
            print(f"Visualization saved to {args.output}")
        
        if args.show_plot:
            plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error visualizing data: {str(e)}")
        return False

def analyze_data(args, print_results=True):
    """Analyze mouse movement data and suggest training parameters."""
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found!")
        return {}
    
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("Error: No data found in the file.")
            return {}
        
        # Basic statistics
        total_events = len(data)
        move_events = [point for point in data if point.get('type', '') == 'move']
        click_events = [point for point in data if point.get('type', '') == 'click']
        scroll_events = [point for point in data if point.get('type', '') == 'scroll']
        
        # Time statistics (if available)
        time_deltas = [point.get('time', 0) for point in move_events if 'time' in point]
        avg_time_delta = sum(time_deltas) / len(time_deltas) if time_deltas else 0
        
        # Session information
        sessions = set()
        for point in data:
            if 'session' in point:
                sessions.add(point['session'])
        
        # Calculate data quality metrics
        movement_coverage = 0
        if move_events:
            x_values = [point['x'] for point in move_events]
            y_values = [point['y'] for point in move_events]
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            movement_coverage = (x_range * y_range)  # Normalized area coverage
        
        # Calculate training recommendations
        recommended_epochs = min(200, max(50, total_events // 10))
        recommended_batch = min(64, max(8, total_events // 20))
        
        # Determine data adequacy
        data_adequacy = "insufficient"
        if total_events >= 1000:
            data_adequacy = "excellent"
        elif total_events >= 500:
            data_adequacy = "good"
        elif total_events >= 200:
            data_adequacy = "adequate"
        elif total_events >= 100:
            data_adequacy = "minimal"
        
        # Create results object
        results = {
            'total_events': total_events,
            'move_events': len(move_events),
            'click_events': len(click_events),
            'scroll_events': len(scroll_events),
            'avg_time_delta': avg_time_delta,
            'sessions': len(sessions),
            'movement_coverage': movement_coverage,
            'recommended_epochs': recommended_epochs,
            'recommended_batch': recommended_batch,
            'data_adequacy': data_adequacy
        }
        
        if print_results:
            print("\n📊 Mouse Movement Data Analysis")
            print("==============================")
            print(f"Data file: {args.data}")
            print(f"Total events: {total_events}")
            print(f"Movement events: {len(move_events)}")
            print(f"Click events: {len(click_events)}")
            print(f"Scroll events: {len(scroll_events)}")
            
            if len(sessions) > 1:
                print(f"Recording sessions: {len(sessions)}")
            
            print(f"\nAverage time between movements: {avg_time_delta:.2f}ms")
            
            print(f"\nData adequacy: {data_adequacy.upper()}")
            if data_adequacy == "insufficient":
                print("⚠️ Your dataset is very small. Consider recording more data.")
                print("   Aim for at least 200 events for adequate training.")
            elif data_adequacy == "minimal":
                print("⚠️ Your dataset meets minimal requirements but more data would improve results.")
                print("   Consider recording additional sessions.")
            
            print("\n📈 Training Recommendations:")
            print(f"Recommended epochs: {recommended_epochs}")
            print(f"Recommended batch size: {recommended_batch}")
            
            if movement_coverage < 0.3:
                print("\n⚠️ Limited mouse movement coverage detected.")
                print("   For better results, record movements across more of the screen.")
            
            if len(click_events) < 10:
                print("\n⚠️ Few click events detected.")
                print("   For better results, include more clicks in your recordings.")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {}

def main():
    """Main function for the recorder CLI."""
    parser = argparse.ArgumentParser(
        description="Mouse Movement Recorder and Trainer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record mouse movements")
    record_parser.add_argument("--output", "-o", type=str, default="mouse_data.json",
                              help="Output file for recorded data (default: mouse_data.json)")
    record_parser.add_argument("--duration", "-d", type=float, default=None,
                              help="Recording duration in seconds (default: until Ctrl+C)")
    record_parser.add_argument("--append", "-a", action="store_true",
                              help="Append to existing data file instead of overwriting")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train neural network model")
    train_parser.add_argument("--data", "-d", type=str, default="mouse_data.json",
                             help="Input data file (default: mouse_data.json)")
    train_parser.add_argument("--model", "-m", type=str, default="mouse_model.pth",
                             help="Output model file (default: mouse_model.pth)")
    train_parser.add_argument("--epochs", "-e", type=int, default=None,
                             help="Number of training epochs (default: auto-determined)")
    train_parser.add_argument("--batch-size", "-b", type=int, default=None,
                             help="Training batch size (default: auto-determined)")
    train_parser.add_argument("--show-plot", "-p", action="store_true",
                             help="Show training plot after training")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize recorded data")
    visualize_parser.add_argument("--data", "-d", type=str, default="mouse_data.json",
                                help="Input data file (default: mouse_data.json)")
    visualize_parser.add_argument("--output", "-o", type=str, default="mouse_visualization.png",
                                help="Output image file (default: mouse_visualization.png)")
    visualize_parser.add_argument("--show-plot", "-p", action="store_true",
                                help="Show plot window")
    visualize_parser.add_argument("--combine-sessions", "-c", action="store_true",
                                help="Visualize all recording sessions (not just the latest)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data and recommend training parameters")
    analyze_parser.add_argument("--data", "-d", type=str, default="mouse_data.json",
                              help="Input data file (default: mouse_data.json)")
    
    args = parser.parse_args()
    
    if args.command == "record":
        record_movements(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "visualize":
        visualize_data(args)
    elif args.command == "analyze":
        analyze_data(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()