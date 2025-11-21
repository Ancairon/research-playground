"""
LSTM with Attention Mechanism for Time Series Forecasting

Features:
- Attention mechanism to focus on relevant historical patterns
- Multiple scaling options: StandardScaler, None (manual normalization)
- Bias correction to fix systematic over/under-prediction
- Differencing mode to learn changes instead of absolute values (eliminates drift)
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseTimeSeriesModel


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for time series."""
    
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return max(0, len(self.data) - self.lookback - self.horizon + 1)
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]
        return torch.FloatTensor(X), torch.FloatTensor(y)


class Attention(nn.Module):
    """Scaled dot-product attention for LSTM hidden states."""
    
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) - weighted combination of hidden states
            attention_weights: (batch, seq_len) - attention scores
        """
        energy = torch.tanh(self.attn(hidden_states))
        attention = self.v(energy).squeeze(-1)
        attention_weights = torch.softmax(attention, dim=1)
        context = (attention_weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return context, attention_weights


class LSTMAttentionNetwork(nn.Module):
    """LSTM with attention mechanism for multi-step forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float = 0.2):
        super(LSTMAttentionNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size)
        
        # Multi-layer decoder for horizon predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            predictions: (batch, horizon)
        """
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        predictions = self.decoder(context)
        return predictions


class LSTMAttentionModel(BaseTimeSeriesModel):
    """
    LSTM with Attention for time series forecasting.
    
    Args:
        horizon: Number of steps to predict
        lookback: Number of historical steps to use
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Optimizer learning rate
        epochs: Training epochs
        batch_size: Training batch size
        scaler_type: 'standard' (StandardScaler) or 'none' (manual normalization)
        bias_correction: Enable bias correction to fix systematic prediction errors
        use_differencing: Learn changes instead of absolute values (eliminates drift)
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        scaler_type: str = 'none',
        bias_correction: bool = False,
        use_differencing: bool = False,
        **kwargs
    ):
        super().__init__(horizon, random_state)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.bias_correction = bias_correction
        self.use_differencing = use_differencing
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'standard' or 'none'")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_data = None
        self.last_data_raw = None
        self.last_value = None
        self.bias_offset = 0.0
        self.data_mean = None
        self.data_std = None
        
        # Warn if using CPU for large models
        if not torch.cuda.is_available() and (hidden_size >= 128 or lookback >= 180):
            warnings.warn("Training on CPU - large model may be slow. Consider using GPU.")
        
        # Set random seeds for full reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        
        # Make PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train LSTM with Attention."""
        start_time = time.time()
        max_train_loss = kwargs.get('max_train_loss', None)
        
        # Validate data size
        if len(data) < self.lookback + self.horizon:
            raise ValueError(
                f"Insufficient data for training. Need at least {self.lookback + self.horizon} points, "
                f"got {len(data)}"
            )
        
        # Apply differencing if enabled
        if self.use_differencing:
            # Store the last value for undifferencing predictions
            self.last_value = data.values[-1]
            # Convert to first differences
            data_values = np.diff(data.values)
            # Pad with first value to maintain length
            data_values = np.concatenate([[0], data_values])
        else:
            data_values = data.values
        
        # Scale data based on scaler type
        if self.scaler is None:
            # No scaler - use raw data without normalization
            data_scaled = data_values
            self.last_data_raw = data.values[-self.lookback:]
            # Store mean/std as None to indicate no scaling
            self.data_mean = None
            self.data_std = None
        else:
            # Use StandardScaler
            data_scaled = self.scaler.fit_transform(data_values.reshape(-1, 1)).flatten()
        
        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.lookback, self.horizon)
        
        if len(dataset) == 0:
            raise ValueError("No training samples created")
        
        # Use single worker for stability with large lookback/horizon
        # Multiple workers can cause memory corruption with large tensors
        # Note: We use num_workers=0 which fixed the corruption issue, so no need to reduce batch_size
        num_workers = 0  # Always use main process for data loading
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,  # Use full batch_size - no reduction needed
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,  # Disable on CPU to save memory
            persistent_workers=False  # Disable persistent workers
        )
        
        # Create model
        self.model = LSTMAttentionNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler - reduce LR when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Early stopping setup
        best_loss = float('inf')
        patience = 7  # Increased patience for pattern learning
        patience_counter = 0
        
        # Training loop
        self.model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                
                # Update learning rate based on loss
                scheduler.step(avg_loss)
                
                # Print loss every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
                
                # Allow early-abort if caller signalled a max_train_loss
                if max_train_loss is not None:
                    try:
                        if float(max_train_loss) is not None and avg_loss > float(max_train_loss):
                            print(f"  Aborting training: avg_loss={avg_loss:.6f} > max_train_loss={max_train_loss}")
                            best_loss = min(best_loss, avg_loss)
                            break
                    except Exception:
                        pass

                # Early stopping check
                if avg_loss < best_loss - 1e-6:  # Improvement threshold
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                        break
        
        self.is_trained = True
        self.last_data = data_scaled[-self.lookback:]
        
        # Calculate bias correction if enabled
        if self.bias_correction:
            self._calculate_bias_correction(data_scaled)
        
        training_time = time.time() - start_time
        # Expose best loss for external monitoring/guards
        try:
            self.last_train_loss = float(best_loss)
        except Exception:
            self.last_train_loss = float('nan')
        return training_time
    
    def _calculate_bias_correction(self, data_scaled: np.ndarray):
        """
        Calculate bias offset by comparing predictions to actuals on training data.
        This helps correct systematic under/over-estimation.
        """
        try:
            self.model.eval()
            prediction_errors = []
            
            # Sample up to 50 windows from training data for bias estimation
            num_samples = min(50, len(data_scaled) - self.lookback - self.horizon)
            if num_samples <= 0:
                self.bias_offset = 0.0
                return
                
            step = max(1, (len(data_scaled) - self.lookback - self.horizon) // num_samples)
            
            with torch.no_grad():
                for i in range(0, len(data_scaled) - self.lookback - self.horizon, step):
                    X = torch.FloatTensor(data_scaled[i:i+self.lookback]).unsqueeze(0).unsqueeze(-1).to(self.device)
                    y_true = data_scaled[i+self.lookback:i+self.lookback+self.horizon]
                    y_pred = self.model(X).cpu().numpy().flatten()
                    
                    # Calculate error in scaled space
                    error = y_true - y_pred
                    prediction_errors.extend(error)
            
            # Calculate mean bias (positive = under-prediction, negative = over-prediction)
            self.bias_offset = np.mean(prediction_errors)
            
            if abs(self.bias_offset) > 0.01:  # Only report significant bias
                print(f"  Bias correction: {self.bias_offset:.4f} (scaled space)")
                
        except Exception as e:
            print(f"  Warning: Bias correction failed: {e}")
            self.bias_offset = 0.0
    
    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        if self.last_data is None or len(self.last_data) < self.lookback:
            raise RuntimeError(f"Insufficient data for prediction")
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(self.last_data[-self.lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)
            predictions_scaled = self.model(X).cpu().numpy().flatten()
        
        # Apply bias correction if enabled
        if self.bias_correction and abs(self.bias_offset) > 0:
            predictions_scaled = predictions_scaled + self.bias_offset
        
        # Inverse transform based on scaler type
        if self.scaler is None:
            # No scaler - use predictions as-is (no denormalization needed)
            predictions = predictions_scaled
        else:
            # StandardScaler - use inverse_transform
            predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        # Reverse differencing if enabled
        if self.use_differencing:
            # Predictions are changes, convert back to absolute values
            # Start from last known value and accumulate changes
            absolute_predictions = np.zeros(len(predictions))
            current_value = self.last_value
            for i in range(len(predictions)):
                current_value = current_value + predictions[i]
                absolute_predictions[i] = current_value
            predictions = absolute_predictions
        
        return predictions.tolist()
    
    def update(self, data: pd.Series, **kwargs):
        """Update last_data window for prediction."""
        if not self.is_trained:
            warnings.warn("Model not trained yet")
            return
        
        try:
            # Update last_value for differencing
            if self.use_differencing:
                self.last_value = data.values[-1]
                # Convert to differences
                data_values = np.diff(data.values)
                if len(data_values) == 0:
                    return
            else:
                data_values = data.values
            
            if self.scaler is None:
                # No scaler - use raw data
                data_scaled = data_values
                self.last_data_raw = np.concatenate([self.last_data_raw, data.values])[-self.lookback:]
            else:
                # StandardScaler
                data_scaled = self.scaler.transform(data_values.reshape(-1, 1)).flatten()
            
            if self.last_data is None:
                self.last_data = data_scaled[-self.lookback:]
            else:
                self.last_data = np.concatenate([self.last_data, data_scaled])[-self.lookback:]
        except Exception as e:
            warnings.warn(f"Update failed: {e}")
    
    def get_model_name(self) -> str:
        return "LSTM+Attention"
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "scaler_type": self.scaler_type,
            "bias_correction": self.bias_correction,
            "use_differencing": self.use_differencing,
            "device": str(self.device),
            "online_learning": False,
            "attention": True
        }
    
    def supports_online_learning(self) -> bool:
        return False
