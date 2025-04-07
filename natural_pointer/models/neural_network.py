"""
Neural network models for predicting natural mouse movements.
"""

import torch
import torch.nn as nn
# import numpy as np # Unused
import os
import json
from typing import List, Tuple, Dict, Optional, Union

class MouseMovementNN(nn.Module):
    """Neural network for modeling mouse movement patterns."""
    
    def __init__(self, input_size: int = 30, hidden_size: int = 24, output_size: int = 3):
        """
        Initialize the mouse movement neural network.
        
        Args:
            input_size: Size of the input feature vector (default: 30 for 5 points with 6 features each)
            hidden_size: Size of the hidden layers
            output_size: Size of the output vector (default: 3 for x, y, time)
        """
        super(MouseMovementNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        return self.model(x)


class MouseModelTrainer:
    """Trainer for the mouse movement neural network."""
    
    def __init__(self, model: Optional[MouseMovementNN] = None, 
                 data_path: str = "mouse_data.json", 
                 model_save_path: str = "mouse_model.pth"):
        """
        Initialize the mouse model trainer.
        
        Args:
            model: Optional pre-initialized model. If None, a new one will be created.
            data_path: Path to the recorded mouse data.
            model_save_path: Path to save the trained model.
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = model if model else MouseMovementNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def _prepare_sequences(self, data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence data for training.
        
        Args:
            data: List of mouse movement data points.
            
        Returns:
            Tuple of (input_sequences, target_values) as PyTorch tensors.
        """
        X, y = [], []
        seq_length = 5  # Use 5 consecutive points to predict next point
        
        for i in range(len(data) - seq_length):
            # Extract features from sequence
            features = []
            for j in range(seq_length):
                point = data[i + j]
                # Create feature vector: [x, y, time, vx, vy, is_click]
                # is_move = 1.0 if point.get('type', 'move') == 'move' else 0.0 # Unused
                is_click = 1.0 if point.get('type', '') == 'click' else 0.0
                # is_scroll = 1.0 if point.get('type', '') == 'scroll' else 0.0 # Unused
                
                # Create a 6-dimensional feature vector for each point
                features.append([
                    point['x'], 
                    point['y'], 
                    point['time']/1000,  # Normalize time to seconds
                    point['vx'], 
                    point['vy'],
                    is_click,  # Binary indicator for clicks
                ])
            
            # Flatten the sequence
            features_flat = [val for sublist in features for val in sublist]
            X.append(features_flat)
            
            # Target is next point's position and timing
            target_point = data[i + seq_length]
            y.append([
                target_point['x'],
                target_point['y'],
                target_point['time']/1000  # Normalize time to seconds
            ])
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return X_tensor, y_tensor
        
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and prepare training data.
        
        Returns:
            Tuple of (input_sequences, target_values) as PyTorch tensors.
        """
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            return self._prepare_sequences(data)
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def train(self, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Train the neural network on mouse movement data.
        
        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
            
        Returns:
            Dictionary with training and validation loss history.
        """
        try:
            # Load and prepare data
            X, y = self.load_data()
            
            # Split into training and validation sets
            val_size = int(len(X) * validation_split)
            indices = torch.randperm(len(X))
            
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            # Training setup
            history = {'train_loss': [], 'val_loss': []}
            
            # Train the model
            for epoch in range(epochs):
                # Training mode
                self.model.train()
                
                # Process in batches
                train_loss = 0
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.loss_fn(outputs, batch_y)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item() * len(batch_X)
                
                avg_train_loss = train_loss / len(X_train)
                history['train_loss'].append(avg_train_loss)
                
                # Validation mode
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.loss_fn(val_outputs, y_val).item()
                    history['val_loss'].append(val_loss)
                
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {val_loss:.6f}')
            
            print("Training complete!")
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Model saved to {self.model_save_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if os.path.exists(self.model_save_path):
                self.model.load_state_dict(torch.load(self.model_save_path))
                self.model.eval()
                print(f"Model loaded from {self.model_save_path}")
                return True
            else:
                print(f"Model file not found: {self.model_save_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
