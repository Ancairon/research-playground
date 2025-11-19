import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_ROOT = Path(__file__).parent / "cache"


@dataclass
class ConfigFingerprint:
    """Canonical representation of a run's configuration for caching.

    Only include options that affect model behaviour or data.
    """

    model: str
    csv_file: Optional[str]
    ip: Optional[str]
    context: Optional[str]
    dimension: Optional[str]
    window: int
    train_window: int
    horizon: int
    lookback: Optional[int]
    hidden_size: Optional[int]
    num_layers: Optional[int]
    dropout: Optional[float]
    learning_rate: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    prediction_smoothing: int
    aggregation_method: str
    aggregation_weight_tau: float
    scaler_type: Optional[str] = 'standard'
    bias_correction: Optional[bool] = True
    use_differencing: Optional[bool] = False

    @classmethod
    def from_args(cls, args) -> "ConfigFingerprint":
        train_window = args.train_window if getattr(args, "train_window", None) is not None else args.window
        
        # Normalize csv_file to just basename for consistent hashing
        csv_file = getattr(args, "csv_file", None)
        if csv_file:
            csv_file = os.path.basename(csv_file)
        
        return cls(
            model=args.model,
            csv_file=csv_file,
            ip=getattr(args, "ip", None),
            context=getattr(args, "context", None),
            dimension=getattr(args, "dimension", None),
            window=args.window,
            train_window=train_window,
            horizon=args.horizon,
            lookback=getattr(args, "lookback", None),
            hidden_size=getattr(args, "hidden_size", None),
            num_layers=getattr(args, "num_layers", None),
            dropout=getattr(args, "dropout", None),
            learning_rate=getattr(args, "learning_rate", None),
            epochs=getattr(args, "epochs", None),
            batch_size=getattr(args, "batch_size", None),
            prediction_smoothing=args.prediction_smoothing,
            aggregation_method=args.aggregation_method,
            aggregation_weight_tau=args.aggregation_weight_tau,
            scaler_type=getattr(args, "scaler_type", "standard"),
            bias_correction=getattr(args, "bias_correction", True),
            use_differencing=getattr(args, "use_differencing", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure deterministic key order
        return {k: data[k] for k in sorted(data.keys())}

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def get_readable_name(self) -> str:
        """Generate a human-readable cache directory name."""
        parts = []
        
        # Data source
        if self.csv_file:
            # Remove extension and use as base name
            base = os.path.splitext(self.csv_file)[0]
            parts.append(base)
        elif self.context and self.dimension:
            parts.append(f"{self.context}_{self.dimension}")
        
        # Model
        parts.append(self.model)
        
        # Key parameters
        if self.lookback:
            parts.append(f"lb{self.lookback}")
        if self.hidden_size:
            parts.append(f"h{self.hidden_size}")
        if self.num_layers:
            parts.append(f"l{self.num_layers}")
        
        # Horizon
        parts.append(f"hor{self.horizon}")
        
        return "_".join(parts)


def get_cache_dir(config_hash: str, readable_name: str = None) -> Path:
    """Get cache directory, optionally with readable name."""
    if readable_name:
        # Use readable name with short hash suffix for uniqueness
        dir_name = f"{readable_name}_{config_hash[:8]}"
    else:
        dir_name = config_hash
    return CACHE_ROOT / dir_name


def load_cached_results(config_hash: str, readable_name: str = None) -> Optional[Dict[str, Any]]:
    """Load cached results if they exist."""
    # Try with readable name first
    if readable_name:
        cache_dir = get_cache_dir(config_hash, readable_name)
        if cache_dir.exists():
            results_path = cache_dir / "results.json"
            if results_path.exists():
                with results_path.open("r") as f:
                    return json.load(f)
    
    # Fallback to hash-only (for backwards compatibility)
    cache_dir = get_cache_dir(config_hash)
    results_path = cache_dir / "results.json"
    if not results_path.exists():
        return None
    with results_path.open("r") as f:
        return json.load(f)


def save_cache(config_hash: str, fingerprint: ConfigFingerprint, results: Dict[str, Any]) -> None:
    """Save results to cache with readable directory name."""
    readable_name = fingerprint.get_readable_name()
    cache_dir = get_cache_dir(config_hash, readable_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"
    results_path = cache_dir / "results.json"

    meta = {
        "hash": config_hash,
        "readable_name": readable_name,
        "fingerprint": fingerprint.to_dict(),
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
