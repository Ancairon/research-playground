"""
Base interface for time series classification models.
All classifier models must implement this interface.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseClassificationModel(ABC):
    """
    Abstract base class for time series classification models.
    
    Any model (Random Forest, XGBoost, etc.) must implement these methods
    to be compatible with the classification framework.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):  # noqa: ARG002
        """
        Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility
            **kwargs: Model-specific parameters (used by subclasses)
        """
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.classes_: Optional[np.ndarray] = None
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            **kwargs: Model-specific training parameters
            
        Returns:
            Training time in seconds
        """
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for the input features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            **kwargs: Model-specific prediction parameters
            
        Returns:
            Array of predicted class labels
        """
        ...
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate probability predictions for all classes.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            **kwargs: Model-specific prediction parameters
            
        Returns:
            Array of shape (n_samples, n_classes) with probability for each class
        """
        ...
    
    def get_max_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the maximum probability and corresponding class for each sample.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (max_proba, predicted_classes) arrays
        """
        probas = self.predict_proba(X)
        max_proba = np.max(probas, axis=1)
        predicted_classes = self.classes_[np.argmax(probas, axis=1)]
        return max_proba, predicted_classes
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model's display name."""
        ...
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Return current model parameters for logging/debugging."""
        ...
    
    def save(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        import joblib
        joblib.dump({
            'model': self.model,
            'classes_': self.classes_,
            'is_trained': self.is_trained,
        }, path)
    
    def load(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.classes_ = data['classes_']
        self.is_trained = data['is_trained']
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance scores if available.
        
        Returns:
            Dict of feature names -> importance scores, or None if not supported
        """
        return None
