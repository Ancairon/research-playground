"""
LSTM with Attention mechanism for improved long-horizon forecasting.
Attention allows the model to focus on relevant parts of the input sequence.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from models.base_model import BaseTimeSeriesModel


class TimeSeriesDataset(Dataset):
    """Dataset for sliding window time series."""
    
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]
        return torch.FloatTensor(X), torch.FloatTensor(y)


class Attention(nn.Module):
    """Attention mechanism for sequence models."""
    
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        # Calculate attention scores
        energy = torch.tanh(self.attn(hidden_states))  # (batch, seq_len, hidden_size)
        attention = self.v(energy).squeeze(-1)  # (batch, seq_len)
        attention_weights = torch.softmax(attention, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        # (batch, seq_len, 1) * (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size)
        context = attention_weights.unsqueeze(-1) * hidden_states
        context = context.sum(dim=1)  # (batch, hidden_size)
        
        return context, attention_weights


class LSTMAttentionNetwork(nn.Module):
    """LSTM with Attention for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float = 0.2):
        super(LSTMAttentionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_size)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, horizon)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        context, attn_weights = self.attention(lstm_out)  # context: (batch, hidden_size)
        
        # Generate predictions
        predictions = self.fc(context)  # (batch, horizon)
        
        return predictions


class LSTMAttentionModel(BaseTimeSeriesModel):
    """
    LSTM with Attention mechanism for time series forecasting.
    
    Attention allows the model to focus on the most relevant historical points,
    improving performance on long-horizon forecasts.
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
        **kwargs
    ):
        """Initialize LSTM with Attention model."""
        super().__init__(horizon, random_state)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_data = None
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train LSTM with Attention."""
        start_time = time.time()
        
        # Validate data size
        if len(data) < self.lookback + self.horizon:
            raise ValueError(
                f"Insufficient data for training. Need at least {self.lookback + self.horizon} points, "
                f"got {len(data)}"
            )
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        
        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.lookback, self.horizon)
        
        if len(dataset) == 0:
            raise ValueError("No training samples created")
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
                
                # Print loss every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        self.last_data = data_scaled[-self.lookback:]
        training_time = time.time() - start_time
        return training_time
    
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
        
        # Debug: Check if predictions are all zeros in scaled space
        if np.all(predictions_scaled == 0):
            print(f"WARNING: Model output is all zeros in scaled space!")
            print(f"  Input data range (scaled): [{self.last_data.min():.4f}, {self.last_data.max():.4f}]")
            print(f"  Scaler range: [{self.scaler.data_min_[0]:.4f}, {self.scaler.data_max_[0]:.4f}]")
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions.tolist()
    
    def update(self, data: pd.Series, **kwargs):
        """Update last_data window for prediction."""
        if not self.is_trained or self.scaler is None:
            warnings.warn("Model not trained yet")
            return
        
        try:
            data_scaled = self.scaler.transform(data.values.reshape(-1, 1)).flatten()
            
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
            "device": str(self.device),
            "online_learning": False,
            "attention": True
        }
    
    def supports_online_learning(self) -> bool:
        return False
