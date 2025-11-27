"""
Time Series Classifier

Handles feature extraction from time series data, classification into known classes,
detection of new patterns, and automatic retraining when new classes are discovered.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import yaml
import json

from .models import create_model
from .models.base_model import BaseClassificationModel
from .feature_extractor import TimeSeriesFeatureExtractor


@dataclass
class ClassificationResult:
    """Result of a classification operation."""
    predicted_class: Optional[str]
    confidence: float
    is_new_class: bool
    all_probabilities: Dict[str, float]
    features: Dict[str, float]
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'is_new_class': self.is_new_class,
            'all_probabilities': self.all_probabilities,
            'features': self.features,
            'timestamp': self.timestamp,
        }


class TimeSeriesClassifier:
    """
    Main classifier for time series data.
    
    Handles:
    - Loading time series data from CSV
    - Feature extraction
    - Classification into known classes
    - Detection of new patterns (low confidence)
    - Automatic retraining with new classes
    """
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.60  # 60% confidence threshold
    
    def __init__(
        self,
        model_name: str = 'randomforest',
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        data_dir: Optional[str] = None,
        random_state: int = 42,
        **model_kwargs
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the classification model to use
            confidence_threshold: Minimum confidence to accept a classification (0-1)
            data_dir: Directory for storing classifier data (models, training data)
            random_state: Random seed for reproducibility
            **model_kwargs: Additional arguments passed to the model
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        
        # Set up data directory
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            data_dir = os.path.join(repo_root, 'classification', 'data')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize paths
        self.model_path = self.data_dir / 'classifier_model.joblib'
        self.classes_path = self.data_dir / 'classes.json'
        self.training_data_path = self.data_dir / 'training_data.npz'
        
        # Initialize components
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.model: Optional[BaseClassificationModel] = None
        self.classes_: List[str] = []
        self.training_features_: Optional[np.ndarray] = None
        self.training_labels_: Optional[np.ndarray] = None
        
        # Load existing model if available
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing model and training data if available."""
        if self.model_path.exists() and self.classes_path.exists():
            try:
                # Load classes
                with open(self.classes_path, 'r', encoding='utf-8') as f:
                    self.classes_ = json.load(f)
                
                # Load model
                self.model = create_model(
                    self.model_name,
                    random_state=self.random_state,
                    **self.model_kwargs
                )
                self.model.load(str(self.model_path))
                
                # Load training data if exists
                if self.training_data_path.exists():
                    data = np.load(self.training_data_path)
                    self.training_features_ = data['features']
                    self.training_labels_ = data['labels']
                
                print(f"[Classifier] Loaded existing model with classes: {self.classes_}")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                print(f"[Classifier] Failed to load existing model: {e}")
                self.model = None
                self.classes_ = []
    
    def _save(self):
        """Save model, classes, and training data."""
        if self.model is not None and self.model.is_trained:
            # Save model
            self.model.save(str(self.model_path))
            
            # Save classes
            with open(self.classes_path, 'w', encoding='utf-8') as f:
                json.dump(self.classes_, f)
            
            # Save training data
            if self.training_features_ is not None and self.training_labels_ is not None:
                np.savez(
                    self.training_data_path,
                    features=self.training_features_,
                    labels=self.training_labels_
                )
            
            print(f"[Classifier] Saved model and data to {self.data_dir}")
    
    def load_csv(self, csv_path: str, value_column: str = 'value') -> pd.Series:
        """
        Load time series data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            value_column: Name of the column containing values
            
        Returns:
            pd.Series with the time series data
        """
        df = pd.read_csv(csv_path)
        
        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
        
        if value_column not in df.columns:
            # Try to find the value column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
            else:
                raise ValueError(f"Could not find value column. Available: {list(df.columns)}")
        
        return df[value_column]
    
    def load_from_config(self, config_path: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Load time series data using a config YAML file.
        
        Args:
            config_path: Path to the YAML config file
            
        Returns:
            Tuple of (time series data, config dict)
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Find CSV file path
        csv_file = config.get('csv-file') or config.get('csv_file')
        
        if csv_file is None:
            # Infer from config filename
            config_basename = os.path.splitext(os.path.basename(config_path))[0]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            csv_file = os.path.join(repo_root, 'csv', f"{config_basename}.csv")
        elif not os.path.isabs(csv_file):
            # Relative path - resolve from repo root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            csv_file = os.path.join(repo_root, 'csv', csv_file)
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        data = self.load_csv(csv_file)
        return data, config
    
    def extract_features(self, data: pd.Series) -> Dict[str, float]:
        """
        Extract features from time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of feature name -> value
        """
        return self.feature_extractor.extract(data)
    
    def train(
        self,
        training_data: Dict[str, pd.Series],
        save: bool = True
    ) -> float:
        """
        Train the classifier on labeled time series data.
        
        Args:
            training_data: Dict mapping class labels to time series data
            save: Whether to save the trained model
            
        Returns:
            Training time in seconds
        """
        print(f"[Classifier] Training on {len(training_data)} classes...")
        
        # Extract features for all training samples
        features_list = []
        labels_list = []
        
        for label, data in training_data.items():
            features = self.extract_features(data)
            features_array = np.array(list(features.values()))
            features_list.append(features_array)
            labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Store for later retraining
        self.training_features_ = X
        self.training_labels_ = y
        self.classes_ = list(set(y))
        
        # Create and train model
        self.model = create_model(
            self.model_name,
            random_state=self.random_state,
            **self.model_kwargs
        )
        
        train_time = self.model.train(X, y)
        
        print(f"[Classifier] Training complete in {train_time:.2f}s")
        print(f"[Classifier] Classes: {self.classes_}")
        
        if save:
            self._save()
        
        return train_time
    
    def train_from_config(
        self,
        training_config_path: Optional[str] = None,
        train_dir: Optional[str] = None,
        save: bool = True
    ) -> float:
        """
        Train the classifier from a training configuration file.
        
        The config file maps class names to CSV files in the train directory.
        
        Args:
            training_config_path: Path to training_config.yaml (default: classification/train/training_config.yaml)
            train_dir: Directory containing training CSV files (default: classification/train/)
            save: Whether to save the trained model
            
        Returns:
            Training time in seconds
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        
        # Set default paths
        if train_dir is None:
            train_dir = os.path.join(repo_root, 'csv')
        
        if training_config_path is None:
            training_config_path = os.path.join(script_dir, 'train', 'training_config.yaml')
        
        # Load training config
        if not os.path.exists(training_config_path):
            raise FileNotFoundError(f"Training config not found: {training_config_path}")
        
        with open(training_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        classes_config = config.get('classes', {})
        
        if not classes_config:
            raise ValueError("No classes defined in training config. Add class mappings under 'classes:' key.")
        
        print(f"[Classifier] Loading training data from: {train_dir}")
        print(f"[Classifier] Config: {training_config_path}")
        print(f"[Classifier] Classes defined: {list(classes_config.keys())}")
        
        # Load all training data
        training_data: Dict[str, List[pd.Series]] = {}
        
        for class_name, csv_files in classes_config.items():
            if not csv_files:
                print(f"[Classifier] Warning: No CSV files for class '{class_name}', skipping")
                continue
            
            training_data[class_name] = []
            
            for csv_file in csv_files:
                # Resolve CSV path
                if os.path.isabs(csv_file):
                    csv_path = csv_file
                else:
                    csv_path = os.path.join(train_dir, csv_file)
                
                if not os.path.exists(csv_path):
                    print(f"[Classifier] Warning: CSV not found: {csv_path}, skipping")
                    continue
                
                # Load the CSV
                data = self.load_csv(csv_path)
                training_data[class_name].append(data)
                print(f"[Classifier]   Loaded: {csv_file} -> {class_name} ({len(data)} samples)")
        
        if not training_data:
            raise ValueError("No training data loaded. Check your config and CSV files.")
        
        # Train with multiple samples per class
        return self.train_multi(training_data, save=save)
    
    def train_multi(
        self,
        training_data: Dict[str, List[pd.Series]],
        save: bool = True
    ) -> float:
        """
        Train the classifier with multiple samples per class.
        
        Args:
            training_data: Dict mapping class labels to list of time series samples
            save: Whether to save the trained model
            
        Returns:
            Training time in seconds
        """
        total_samples = sum(len(samples) for samples in training_data.values())
        print(f"[Classifier] Training on {len(training_data)} classes, {total_samples} total samples...")
        
        # Extract features for all training samples
        features_list = []
        labels_list = []
        
        for label, data_samples in training_data.items():
            for data in data_samples:
                features = self.extract_features(data)
                features_array = np.array(list(features.values()))
                features_list.append(features_array)
                labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Store for later retraining
        self.training_features_ = X
        self.training_labels_ = y
        self.classes_ = sorted(list(set(y)))
        
        # Create and train model
        self.model = create_model(
            self.model_name,
            random_state=self.random_state,
            **self.model_kwargs
        )
        
        train_time = self.model.train(X, y)
        
        print(f"[Classifier] Training complete in {train_time:.2f}s")
        print(f"[Classifier] Classes: {self.classes_}")
        print(f"[Classifier] Samples per class: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        if save:
            self._save()
        
        return train_time
    
    @classmethod
    def load(cls, data_dir: Optional[str] = None, **kwargs) -> 'TimeSeriesClassifier':
        """
        Load a trained classifier from disk.
        
        Args:
            data_dir: Directory containing saved classifier data
            **kwargs: Additional arguments for classifier initialization
            
        Returns:
            Loaded TimeSeriesClassifier instance
        """
        classifier = cls(data_dir=data_dir, **kwargs)
        
        if not classifier.model_path.exists():
            raise FileNotFoundError(f"No saved model found at: {classifier.model_path}")
        
        # Model is loaded in __init__ via _load_if_exists()
        if classifier.model is None or not classifier.model.is_trained:
            raise RuntimeError("Failed to load trained model")
        
        print(f"[Classifier] Loaded model with {len(classifier.classes_)} classes: {classifier.classes_}")
        return classifier
    
    def classify(self, data: pd.Series) -> ClassificationResult:
        """
        Classify a time series.
        
        Args:
            data: Time series data to classify
            
        Returns:
            ClassificationResult with prediction, confidence, and metadata
        """
        if self.model is None or not self.model.is_trained:
            raise RuntimeError("Model must be trained before classification. Call train() first.")
        
        # Extract features
        features = self.extract_features(data)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Get predictions and probabilities
        max_proba, predicted = self.model.get_max_confidence(features_array)
        confidence = float(max_proba[0])
        predicted_class = str(predicted[0])
        
        # Get all class probabilities
        probas = self.model.predict_proba(features_array)[0]
        all_probabilities = {
            cls: float(prob) 
            for cls, prob in zip(self.model.classes_, probas)
        }
        
        # Check if this is a new class (low confidence)
        is_new_class = confidence < self.confidence_threshold
        
        if is_new_class:
            predicted_class = None  # Don't assign a class if confidence is too low
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            is_new_class=is_new_class,
            all_probabilities=all_probabilities,
            features=features,
        )
    
    def add_new_class(
        self,
        class_name: str,
        data: pd.Series,
        retrain: bool = True
    ) -> Optional[float]:
        """
        Add a new class and optionally retrain the model.
        
        Args:
            class_name: Name for the new class
            data: Time series data representing this class
            retrain: Whether to retrain the model
            
        Returns:
            Training time if retrain=True, else None
        """
        print(f"[Classifier] Adding new class: {class_name}")
        
        # Extract features
        features = self.extract_features(data)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Add to training data
        if self.training_features_ is None:
            self.training_features_ = features_array
            self.training_labels_ = np.array([class_name])
        else:
            self.training_features_ = np.vstack([self.training_features_, features_array])
            self.training_labels_ = np.append(self.training_labels_, class_name)
        
        # Update classes list
        if class_name not in self.classes_:
            self.classes_.append(class_name)
        
        if retrain:
            # Create new model and train on all data
            self.model = create_model(
                self.model_name,
                random_state=self.random_state,
                **self.model_kwargs
            )
            
            train_time = self.model.train(self.training_features_, self.training_labels_)
            self._save()
            
            print(f"[Classifier] Retrained with {len(self.classes_)} classes in {train_time:.2f}s")
            return train_time
        
        return None
    
    def classify_and_update_csv(
        self,
        config_path: str,
        class_key: str = 'class',
        new_class_name: Optional[str] = None,
        auto_retrain: bool = True,
        skip_new_class: bool = False
    ) -> ClassificationResult:
        """
        Classify a time series from config and update the config with the result.
        
        Args:
            config_path: Path to YAML config file
            class_key: Key to use in config for storing the class
            new_class_name: Name for new class (if detected). If None, uses config basename.
            auto_retrain: Whether to automatically retrain when new class is detected
            skip_new_class: If True, don't add new class - just report the best match
            
        Returns:
            ClassificationResult
        """
        # Load data
        data, config = self.load_from_config(config_path)
        
        # Handle first-time use (no model yet)
        if self.model is None or not self.model.is_trained:
            # Use config basename as default class name
            if new_class_name is None:
                new_class_name = os.path.splitext(os.path.basename(config_path))[0]
            
            print(f"[Classifier] No existing model - creating new class: {new_class_name}")
            
            # Train with this as the first class
            self.train({new_class_name: data})
            
            # Create result
            result = ClassificationResult(
                predicted_class=new_class_name,
                confidence=1.0,
                is_new_class=True,
                all_probabilities={new_class_name: 1.0},
                features=self.extract_features(data),
            )
        else:
            # Classify
            result = self.classify(data)
            
            # Handle new class detection
            if result.is_new_class:
                if skip_new_class:
                    print(f"[Classifier] New class detected (confidence: {result.confidence:.2%}) - skipping")
                    # Return result with best match, but still mark as new class
                else:
                    if new_class_name is None:
                        new_class_name = os.path.splitext(os.path.basename(config_path))[0]
                    
                    print(f"[Classifier] New class detected (confidence: {result.confidence:.2%})")
                    print(f"[Classifier] Adding new class: {new_class_name}")
                    
                    if auto_retrain:
                        self.add_new_class(new_class_name, data, retrain=True)
                        result = ClassificationResult(
                            predicted_class=new_class_name,
                            confidence=result.confidence,
                            is_new_class=True,
                            all_probabilities=result.all_probabilities,
                            features=result.features,
                        )
        
        # Update config with class (unless skipping new class)
        if not (result.is_new_class and skip_new_class):
            config[class_key] = result.predicted_class
            
            # Write back to config
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"[Classifier] Updated {config_path} with class: {result.predicted_class}")
        else:
            print(f"[Classifier] Skipped updating config (new class detection skipped)")
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get classifier information."""
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'classes': self.classes_,
            'n_training_samples': len(self.training_labels_) if self.training_labels_ is not None else 0,
            'is_trained': self.model is not None and self.model.is_trained,
            'data_dir': str(self.data_dir),
            'model_params': self.model.get_model_params() if self.model else {},
        }
