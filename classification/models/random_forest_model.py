"""
Random Forest classifier for time series classification.
"""

import time
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseClassificationModel


class RandomForestModel(BaseClassificationModel):
    """
    Random Forest classifier for time series classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features for splits (sqrt, log2, or int)
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state=random_state, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the sklearn model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            
        Returns:
            Training time in seconds
        """
        start_time = time.time()
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        train_time = time.time() - start_time
        return train_time
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for the input features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate probability predictions for all classes.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_classes) with probability for each class
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_model_name(self) -> str:
        """Return the model's display name."""
        return "Random Forest Classifier"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Return current model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return {f'feature_{i}': imp for i, imp in enumerate(importances)}
