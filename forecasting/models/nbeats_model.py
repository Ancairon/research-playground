"""
N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting).
Pure forecasting architecture designed specifically for long-horizon predictions.

Paper: https://arxiv.org/abs/1905.10437
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


class NBeatsBlock(nn.Module):
    """Single N-BEATS block."""
    
    def __init__(self, input_size: int, theta_size: int, horizon: int, hidden_size: int = 256, num_layers: int = 4):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        
        # Fully connected layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)
        
        # Basis expansion (simple polynomial for generic block)
        self.backcast_basis = self._polynomial_basis(input_size, theta_size)
        self.forecast_basis = self._polynomial_basis(horizon, theta_size)
        
    def _polynomial_basis(self, size: int, degree: int):
        """Create polynomial basis functions."""
        # T will be [0, 1, ..., size-1]
        # Normalize to [0, 1]
        T = torch.arange(size, dtype=torch.float32) / max(1, size - 1)
        basis = torch.stack([T ** i for i in range(degree)], dim=1)  # (size, degree)
        return basis
    
    def forward(self, x):
        # x: (batch, input_size)
        h = self.fc_layers(x)  # (batch, hidden_size)
        
        theta_b = self.theta_b(h)  # (batch, theta_size)
        theta_f = self.theta_f(h)  # (batch, theta_size)
        
        # Basis expansion
        backcast_basis = self.backcast_basis.to(x.device)  # (input_size, theta_size)
        forecast_basis = self.forecast_basis.to(x.device)  # (horizon, theta_size)
        
        # Backcast: (batch, theta_size) @ (theta_size, input_size) -> (batch, input_size)
        backcast = torch.matmul(theta_b, backcast_basis.T)
        
        # Forecast: (batch, theta_size) @ (theta_size, horizon) -> (batch, horizon)
        forecast = torch.matmul(theta_f, forecast_basis.T)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    """Stack of N-BEATS blocks."""
    
    def __init__(self, num_blocks: int, input_size: int, theta_size: int, horizon: int, hidden_size: int = 256):
        super(NBeatsStack, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, horizon, hidden_size)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # x: (batch, input_size)
        forecast = 0
        residual = x
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        
        return forecast


class NBEATSNetwork(nn.Module):
    """N-BEATS model with multiple stacks."""
    
    def __init__(self, input_size: int, horizon: int, num_stacks: int = 2, num_blocks: int = 3, 
                 theta_size: int = 8, hidden_size: int = 256):
        super(NBEATSNetwork, self).__init__()
        self.stacks = nn.ModuleList([
            NBeatsStack(num_blocks, input_size, theta_size, horizon, hidden_size)
            for _ in range(num_stacks)
        ])
        
    def forward(self, x):
        # x: (batch, input_size)
        forecast = 0
        for stack in self.stacks:
            stack_forecast = stack(x)
            forecast = forecast + stack_forecast
        
        return forecast


class NBEATSModel(BaseTimeSeriesModel):
    """
    N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting).
    
    A pure forecasting architecture designed specifically for time series.
    Uses basis expansion and doubly residual stacking for interpretable predictions.
    Excellent for long-horizon forecasting.
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        lookback: int = 120,
        num_stacks: int = 2,
        num_blocks: int = 3,
        theta_size: int = 8,
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize N-BEATS model."""
        super().__init__(horizon, random_state)
        self.lookback = lookback
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.theta_size = theta_size
        self.hidden_size = hidden_size
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
        """Train N-BEATS."""
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
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        self.model = NBEATSNetwork(
            input_size=self.lookback,
            horizon=self.horizon,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            theta_size=self.theta_size,
            hidden_size=self.hidden_size
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(self.epochs):
                epoch_loss = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
        
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
        return "N-BEATS"
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "lookback": self.lookback,
            "num_stacks": self.num_stacks,
            "num_blocks": self.num_blocks,
            "theta_size": self.theta_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "online_learning": False,
            "architecture": "doubly_residual_stacking"
        }
    
    def supports_online_learning(self) -> bool:
        return False
