import hashlib
import json
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

    @classmethod
    def from_args(cls, args) -> "ConfigFingerprint":
        train_window = args.train_window if getattr(args, "train_window", None) is not None else args.window
        return cls(
            model=args.model,
            csv_file=getattr(args, "csv_file", None),
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
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure deterministic key order
        return {k: data[k] for k in sorted(data.keys())}

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_dir(config_hash: str) -> Path:
    return CACHE_ROOT / config_hash


def load_cached_results(config_hash: str) -> Optional[Dict[str, Any]]:
    cache_dir = get_cache_dir(config_hash)
    results_path = cache_dir / "results.json"
    if not results_path.exists():
        return None
    with results_path.open("r") as f:
        return json.load(f)


def save_cache(config_hash: str, fingerprint: ConfigFingerprint, results: Dict[str, Any]) -> None:
    cache_dir = get_cache_dir(config_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"
    results_path = cache_dir / "results.json"

    meta = {
        "hash": config_hash,
        "fingerprint": fingerprint.to_dict(),
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
