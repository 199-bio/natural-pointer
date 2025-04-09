"""
Neural network models for predicting natural mouse movements.
"""

import torch
import torch.nn as nn
import torch.utils.data as data # Add this for Dataset, DataLoader
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


# NEW CLASS
class MouseDataDataset(data.Dataset):
    """
    PyTorch Dataset for loading mouse movement data efficiently.
    Loads data points as needed, avoiding loading the entire dataset into memory at once
    if implemented with a streaming JSON parser (currently loads all in __init__ for simplicity).
    """
    def __init__(self, data_path: str, seq_length: int = 5):
        """
        Initializes the dataset.

        Args:
            data_path: Path to the JSON file containing mouse data.
            seq_length: The number of consecutive data points to use as input features.
        """
        self.data_path = data_path
        self.seq_length = seq_length

        try:
            with open(self.data_path, 'r') as f:
                # Load data once to get length and for easy access in __getitem__
                # Note: For extremely large files (> RAM), a streaming parser (e.g., ijson)
                # and pre-calculating length would be necessary for true memory efficiency.
                # This implementation is a trade-off for simplicity.
                self.data = json.load(f)
            if not isinstance(self.data, list) or len(self.data) <= self.seq_length:
                 raise ValueError(f"Data in {data_path} is not a list or is too short for seq_length={self.seq_length}.")
            # Calculate the total number of sequences possible
            self.num_sequences = len(self.data) - self.seq_length
            if self.num_sequences <= 0:
                 raise ValueError(f"Data length ({len(self.data)}) is not greater than seq_length ({self.seq_length}). No sequences can be formed.")
            print(f"Dataset initialized with {self.num_sequences} sequences.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.data_path}")
            raise
        except ValueError as ve:
             print(f"Error initializing dataset: {ve}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred loading data for dataset: {str(e)}")
            raise

    def __len__(self) -> int:
        """Returns the total number of sequences in the dataset."""
        return self.num_sequences

    def _prepare_single_sequence(self, sequence_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a single input sequence tensor and its corresponding target tensor.

        Args:
            sequence_data: A list of (seq_length + 1) consecutive data points.

        Returns:
            A tuple containing (input_tensor, target_tensor).
        """
        features = []
        # Input sequence uses the first 'seq_length' points
        for point in sequence_data[:self.seq_length]:
            # Create feature vector: [x, y, time, vx, vy, is_click]
            is_click = 1.0 if point.get('type', '') == 'click' else 0.0
            # Ensure all expected keys exist, provide defaults if necessary
            features.append([
                point.get('x', 0.0),
                point.get('y', 0.0),
                point.get('time', 0.0) / 1000.0,  # Normalize time to seconds
                point.get('vx', 0.0),
                point.get('vy', 0.0),
                is_click,
            ])

        # Flatten the sequence features into a single vector
        features_flat = [val for sublist in features for val in sublist]
        input_tensor = torch.tensor(features_flat, dtype=torch.float32)

        # Target is the next point's position and timing (the last point in sequence_data)
        target_point = sequence_data[-1]
        target_tensor = torch.tensor([
            target_point.get('x', 0.0),
            target_point.get('y', 0.0),
            target_point.get('time', 0.0) / 1000.0 # Normalize time to seconds
        ], dtype=torch.float32)

        return input_tensor, target_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input sequence and target tensor for a given index.

        Args:
            idx: The index of the sequence to retrieve.

        Returns:
            A tuple containing (input_tensor, target_tensor).

        Raises:
            IndexError: If the index is out of bounds.
        """
        if not 0 <= idx < self.num_sequences:
            raise IndexError(f"Dataset index {idx} out of range for dataset size {self.num_sequences}")

        # Get the slice of raw data points needed: seq_length for input, 1 for target
        raw_sequence = self.data[idx : idx + self.seq_length + 1]

        # Prepare the tensors for this specific sequence
        try:
            input_tensor, target_tensor = self._prepare_single_sequence(raw_sequence)
            return input_tensor, target_tensor
        except KeyError as ke:
            print(f"Error processing sequence at index {idx}: Missing key {ke} in data point.")
            # Optionally return dummy data or raise a more specific error
            # For now, re-raise to halt execution
            raise RuntimeError(f"Data format error at index {idx}") from ke
        except Exception as e:
            print(f"Error preparing sequence at index {idx}: {str(e)}")
            raise RuntimeError(f"Error preparing sequence at index {idx}") from e


class MouseModelTrainer:
    """Trainer for the mouse movement neural network."""

    def __init__(self, model: Optional[MouseMovementNN] = None,
                 data_path: str = "mouse_data.json",
                 model_save_path: str = "mouse_model.pth",
                 seq_length: int = 5): # Added seq_length
        """
        Initialize the mouse model trainer.

        Args:
            model: Optional pre-initialized model. If None, a new one will be created.
            data_path: Path to the recorded mouse data.
            model_save_path: Path to save the trained model.
            seq_length: Length of the input sequence for prediction (default: 5).
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.seq_length = seq_length # Store seq_length
        # Determine input size based on seq_length and features per point (6)
        # Ensure model input size matches the flattened features from the dataset
        input_size = seq_length * 6
        self.model = model if model else MouseMovementNN(input_size=input_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    # REMOVED _prepare_sequences method (logic moved to MouseDataDataset)
    # REMOVED load_data method (replaced by Dataset/DataLoader)

    def train(self, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Train the neural network on mouse movement data using DataLoader.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.

        Returns:
            Dictionary with training and validation loss history.
        """
        try:
            # 1. Create the Dataset
            # This might raise errors if file not found or format is wrong
            full_dataset = MouseDataDataset(self.data_path, self.seq_length)

            # 2. Split Dataset indices for training and validation
            dataset_size = len(full_dataset)
            if dataset_size == 0:
                print("Error: Dataset is empty. Cannot train.")
                return {'train_loss': [], 'val_loss': []} # Return empty history

            indices = list(range(dataset_size))
            split = int(dataset_size * validation_split)

            # Shuffle indices before splitting using torch.randperm for reproducibility if seed is set
            indices = torch.randperm(dataset_size).tolist()

            train_indices, val_indices = indices[split:], indices[:split]

            if not train_indices:
                print("Warning: No training samples after split. Check validation_split.")
                return {'train_loss': [], 'val_loss': []}
            if not val_indices:
                print("Warning: No validation samples after split. Training will proceed without validation.")
                # Optionally, proceed without validation or raise an error

            # 3. Create Samplers
            train_sampler = data.SubsetRandomSampler(train_indices)
            # Use SubsetRandomSampler for validation as well for consistency
            # Handle case where val_indices might be empty
            val_sampler = data.SubsetRandomSampler(val_indices) if val_indices else None

            # 4. Create DataLoaders
            # num_workers > 0 can speed up loading but might cause issues on some systems/platforms
            # pin_memory=True can speed up CPU->GPU transfer if using CUDA
            train_loader = data.DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=False)
            # Only create val_loader if there are validation samples
            val_loader = None
            if val_sampler:
                val_loader = data.DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=False)


            # Training setup
            history = {'train_loss': [], 'val_loss': []}

            print(f"Starting training: {len(train_indices)} train samples, {len(val_indices)} validation samples.")

            # Train the model
            for epoch in range(epochs):
                # Training mode
                self.model.train()

                epoch_train_loss = 0.0
                num_train_batches = 0
                # Iterate over batches from DataLoader
                for batch_X, batch_y in train_loader:
                    # Ensure data is on the correct device if using GPU
                    # batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.loss_fn(outputs, batch_y)

                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate loss correctly
                    epoch_train_loss += loss.item() # loss.item() is the average loss for the batch
                    num_train_batches += 1

                # Calculate average loss for the epoch
                avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
                history['train_loss'].append(avg_train_loss)

                # Validation mode (only if val_loader exists)
                avg_val_loss = 0.0
                num_val_batches = 0
                if val_loader:
                    self.model.eval()
                    epoch_val_loss = 0.0
                    with torch.no_grad():
                        # Iterate over validation DataLoader
                        for batch_X_val, batch_y_val in val_loader:
                            # Ensure data is on the correct device if using GPU
                            # batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)

                            val_outputs = self.model(batch_X_val)
                            loss = self.loss_fn(val_outputs, batch_y_val)
                            epoch_val_loss += loss.item()
                            num_val_batches += 1

                    # Calculate average validation loss for the epoch
                    avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0.0
                    history['val_loss'].append(avg_val_loss)
                elif epoch == 0: # Print warning once if no validation loader
                    print("No validation samples; skipping validation loss calculation.")

                if (epoch + 1) % 10 == 0:
                    val_loss_str = f'{avg_val_loss:.6f}' if num_val_batches > 0 else 'N/A'
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {val_loss_str}')

            print("Training complete!")
            return history

        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Consider more specific error handling or logging (e.g., logging traceback)
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
                # Need to know the input size to reconstruct the model before loading state_dict
                # Assuming default seq_length=5 if not otherwise known, leading to input_size=30
                # A better approach might involve saving model architecture or input size with the state_dict
                input_size = self.seq_length * 6 # Use stored seq_length
                # Re-initialize model structure before loading state dict
                self.model = MouseMovementNN(input_size=input_size)
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
