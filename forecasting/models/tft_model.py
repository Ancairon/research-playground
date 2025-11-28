"""
Temporal Fusion Transformer (TFT) - Simplified implementation.
Multi-head attention with gating for interpretable long-horizon forecasting.

Inspired by: https://arxiv.org/abs/1912.09363
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

from .base_model import BaseTimeSeriesModel


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


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature processing."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection
        if input_size != hidden_size:
            self.skip = nn.Linear(input_size, hidden_size)
        else:
            self.skip = None
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size) or (batch, input_size)
        residual = x if self.skip is None else self.skip(x)
        
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        
        # Gating
        gate = self.sigmoid(self.gate(h))
        h = h * gate
        
        # Residual connection and normalization
        out = self.layer_norm(h + residual)
        return out


class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer."""
    
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, num_layers: int, 
                 horizon: int, dropout: float = 0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        
        # Input embedding
        self.input_embed = nn.Linear(1, hidden_size)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, input_size, hidden_size))
        
        # Variable selection network (simplified)
        self.variable_selection = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated residual networks for post-attention processing
        self.grn1 = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        self.grn2 = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, horizon)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Embed input
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_embed(x)  # (batch, seq_len, hidden_size)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Variable selection
        x = self.variable_selection(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Multi-head self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        
        # Residual connection
        x = lstm_out + attn_out
        
        # Post-attention processing
        x = self.grn1(x)
        
        # Aggregate over sequence (use last timestep)
        x = x[:, -1, :]  # (batch, hidden_size)
        
        # Final transformation
        x = self.grn2(x)
        
        # Generate forecast
        forecast = self.output_layer(x)  # (batch, horizon)
        
        return forecast


class TFTModel(BaseTimeSeriesModel):
    """
    Temporal Fusion Transformer (TFT) - Simplified implementation.
    
    Combines LSTM, multi-head attention, and gated residual networks
    for interpretable multi-horizon forecasting. Designed for long-range
    predictions with variable selection and temporal processing.
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        lookback: int = 120,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize TFT model."""
        super().__init__(horizon, random_state)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_data = None
        
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
        """Train TFT."""
        start_time = time.time()
        max_train_loss = kwargs.get('max_train_loss', None)
        
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
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        self.model = TemporalFusionTransformer(
            input_size=self.lookback,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Early stopping setup
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop
        self.model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(self.epochs):
                epoch_loss = 0
                batch_count = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                
                avg_loss = epoch_loss / batch_count

                # Early abort if caller signalled a maximum acceptable loss
                if max_train_loss is not None:
                    try:
                        if float(max_train_loss) is not None and avg_loss > float(max_train_loss):
                            print(f"  Aborting training: avg_loss={avg_loss:.6f} > max_train_loss={max_train_loss}")
                            best_loss = min(best_loss, avg_loss)
                            break
                    except Exception:
                        pass

                # Early stopping check
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        self.is_trained = True
        self.last_data = data_scaled[-self.lookback:]
        training_time = time.time() - start_time
        # Expose best loss for external monitoring
        try:
            self.last_train_loss = float(best_loss)
        except Exception:
            self.last_train_loss = float('nan')
        return training_time
    
    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        if self.last_data is None or len(self.last_data) < self.lookback:
            raise RuntimeError(f"Insufficient data for prediction")
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(self.last_data[-self.lookback:]).unsqueeze(0).to(self.device)
            predictions_scaled = self.model(X).cpu().numpy().flatten()
        
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
        return "TFT"
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "online_learning": False,
            "architecture": "temporal_fusion_transformer"
        }
    
    def supports_online_learning(self) -> bool:
        return False
