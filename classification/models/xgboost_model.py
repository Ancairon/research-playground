"""
XGBoost classifier for time series classification.
"""

import time
import numpy as np
from typing import Dict, Any

from .base_model import BaseClassificationModel


class XGBoostModel(BaseClassificationModel):
    """
    XGBoost classifier for time series classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state=random_state, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the XGBoost model."""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1,
            )
        except ImportError as exc:
            raise ImportError("XGBoost is required. Install with: pip install xgboost") from exc
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Train the XGBoost model.
        
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
        return "XGBoost Classifier"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Return current model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return {f'feature_{i}': imp for i, imp in enumerate(importances)}
