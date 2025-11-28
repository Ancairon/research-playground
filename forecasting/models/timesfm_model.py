"""
TimesFM wrapper model

This model wraps the Hugging Face TimesFM pre-trained checkpoints (eg. google/timesfm-2.5-200m-pytorch)
and exposes a lightweight adapter so it can be used by the existing UniversalForecaster.

Notes:
- The TimesFM model is pre-trained and this wrapper only supports prediction (no training of weights).
- The code will attempt to import Transformers / PyTorch and load the HF checkpoint lazily. If HF
  libraries or the checkpoint aren't available at runtime this wrapper falls back to a simple
  naive predictor (repeat the last value or linear extrapolation) so it remains safe to use
  in environments that don't have the HF artifacts downloaded.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import time

from .base_model import BaseTimeSeriesModel


class TimesFMModel(BaseTimeSeriesModel):
    """Wrapper for pre-trained TimesFM checkpoints from Hugging Face.

    Usage
    -----
    - model_name: HF model id, default 'google/timesfm-2.5-200m-pytorch'
    - horizon: number of steps to forecast
    - lookback: length of history to provide to the model. If None, the wrapper will
      use the whole history provided to `train`/`update`.

    Behaviour
    ---------
    - train(...) will not train model weights. Instead it prepares (lazy loads)
      the HF model if the libraries are present and stores the last seen history
      window for predictions.
    - predict() returns a list[float] with length == horizon. If a HF model cannot
      be used, a simple fallback forecast is produced.
    """

    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        model_name: str = 'google/timesfm-2.5-200m-pytorch',
        lookback: Optional[int] = None,
        device: Optional[str] = None,
        strict: bool = True,
        # Optional kwargs forwarded to transformers.from_pretrained (eg. cache_dir, local_files_only)
        hf_load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(horizon=horizon, random_state=random_state)
        self.model_name = model_name
        self.lookback = lookback
        self.device = device
        # If strict==True: require HF artifacts to be present and usable.
        # If strict==False: fallback to simple predictor on load/forward failures.
        self.strict = bool(strict)
        self.hf_load_kwargs = hf_load_kwargs or {}

        # Hugging Face objects will be lazily loaded
        self._hf_model = None
        self._hf_processor = None
        self._hf_available = False

        # last observed values (1D numpy array)
        self.last_data: Optional[np.ndarray] = None

        # A small flag used by the framework; mark as True after train() is called
        self.is_trained = False

    # ---------------------- Internal HF loader ----------------------
    def _try_load_hf(self) -> None:
        """Attempt to import and load the HF model. If anything fails we set
        _hf_available to False and continue using the fallback path.
        """
        if self._hf_model is not None or self._hf_available:
            return

        try:
            # Import locally so environments without the libs are not impacted
            import torch
            from transformers import AutoModel, AutoConfig

            # Try to load a generic AutoModel first; some timesfm checkpoints ship
            # custom classes and this will still work for simple forward-based models
            # if the repo provides the expected weights.
            cfg = AutoConfig.from_pretrained(self.model_name)

            # Prefer a forecasting-specific class if present, otherwise fall back
            # to generic AutoModel
            # Prefer plain AutoModel but enable trust_remote_code so custom
            # implementation classes included in the HF repo are used (TimesFmModel).
            load_kwargs = dict(self.hf_load_kwargs)
            # always enable trust_remote_code for timesfm to pick up the right class
            load_kwargs.setdefault('trust_remote_code', True)

            model = AutoModel.from_pretrained(self.model_name, **load_kwargs)

            # Move to device if available
            if self.device is None:
                self.device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

            self._hf_model = model
            try:
                self._hf_model.to(self.device)
            except Exception:
                # Some HF model objects may not support .to() directly; ignore
                pass

            # No complex processor required here; many time-series checkpoints
            # rely on raw numpy arrays. We'll mark available and rely on the
            # forward pass in predict() or fallback.
            self._hf_available = True
        except Exception as e:
            # Any import/load failure -> no HF usage
            self._hf_available = False
            # In strict mode we want to surface the underlying error so callers
            # can fail early instead of silently falling back.
            if getattr(self, 'strict', False):
                raise RuntimeError(f"TimesFMModel strict mode: failed to import/load HF model '{self.model_name}': {e}")

    # ---------------------- Interface methods ------------------------
    def train(self, data: pd.Series, **kwargs) -> float:
        """No-op training path for pre-trained TimesFM.

        The method sets up internal state (last_data) and tries to lazily
        load a HF model if possible. Returns 0.0 as training time.
        """
        start = time.time()

        if not isinstance(data, pd.Series):
            # try to turn it into a series if possible
            try:
                data = pd.Series(data)
            except Exception:
                raise ValueError("TimesFMModel.train requires a pandas.Series or array-like data")

        arr = np.asarray(data.values, dtype=float)

        if self.lookback is not None and len(arr) >= int(self.lookback):
            self.last_data = arr[-int(self.lookback):]
        else:
            self.last_data = arr.copy()

        # Try to load HF model lazily. In strict mode we raise on failures.
        self._try_load_hf()

        self.is_trained = True
        elapsed = time.time() - start
        # training time is effectively zero for a pre-trained wrapper
        return elapsed

    def predict(self, **kwargs) -> List[float]:
        """Produce a forecast for the next `horizon` steps.

        If the HF model is available and appears to be plausible we attempt to
        run it. If not, a simple fallback predictor is used:
         - If there are at least 2 points in last_data: linear extrapolation
         - Otherwise: repeat the last observed value
        """
        if not self.is_trained:
            raise RuntimeError("TimesFMModel: train() must be called before predict()")

        horizon = int(self.horizon)
        if self.last_data is None or len(self.last_data) == 0:
            # No data -> return zeros
            return [0.0] * horizon

        # Try HF path first
        try:
            # Ensure HF model is available. _try_load_hf will raise in strict mode
            self._try_load_hf()
            if self._hf_available and getattr(self, '_hf_model', None) is not None:
                # A best-effort inference using PyTorch tensors and a standard
                # forward() call. Many HF time-series checkpoints expect a
                # (batch, seq_len, features) float tensor and return a dict with
                # 'predictions' or a plain tensor. Use try/except with safe
                # fallbacks.
                import torch

                seq = np.asarray(self.last_data, dtype=float)

                # If the user provided a lookback, respect it
                if self.lookback is not None and int(self.lookback) > 0:
                    if seq.ndim == 1 and seq.size > int(self.lookback):
                        seq = seq[-int(self.lookback):]

                # If HF model config exposes a context_length or patch_length,
                # use it to determine/pad the sequence length so the model's
                # internal reshape operations succeed.
                cfg = getattr(self._hf_model, 'config', None)
                context_len = None
                patch_len = None
                try:
                    if cfg is not None:
                        context_len = getattr(cfg, 'context_length', None)
                        patch_len = getattr(cfg, 'patch_length', None)
                except Exception:
                    context_len = None
                    patch_len = None

                # Decide target seq length
                seq_len = seq.shape[0] if seq.ndim == 1 else seq.shape[1]
                target_len = seq_len if context_len is None else int(context_len)

                # If seq longer than target, truncate the left side (keep most recent)
                if seq_len > target_len:
                    seq = seq[-target_len:]
                    seq_len = target_len

                # Ensure divisibility by patch_len by padding to next multiple if needed
                if patch_len is not None and patch_len > 0:
                    if seq_len % int(patch_len) != 0:
                        new_len = ((seq_len + int(patch_len) - 1) // int(patch_len)) * int(patch_len)
                        # If we had a context_len and new_len > context_len, extend target to new_len
                        target_len = max(target_len, new_len)
                else:
                    target_len = seq_len

                # If sequence shorter than target_len, pad at the end with zeros
                if seq_len < target_len:
                    if seq.ndim == 1:
                        seq = np.pad(seq, (target_len - seq_len, 0), mode='constant') if False else np.pad(seq, (0, target_len - seq_len), mode='constant')
                        # above: pad at the end
                    elif seq.ndim == 2:
                        seq = np.pad(seq, ((0, 0), (0, target_len - seq_len)), mode='constant')
                    elif seq.ndim == 3:
                        seq = np.pad(seq, ((0, 0), (0, target_len - seq_len), (0, 0)), mode='constant')
                    seq_len = target_len

                # Build candidate tensor shapes. Many TimesFM implementations accept
                # past_values as 2D (batch, seq_len) or 3D (batch, seq_len, features).
                if seq.ndim == 1:
                    seq_2d = seq.reshape(1, seq.shape[0])
                    seq_3d = seq.reshape(1, seq.shape[0], 1)
                elif seq.ndim == 2:
                    seq_2d = seq
                    seq_3d = seq.reshape(seq.shape[0], seq.shape[1], 1)
                elif seq.ndim == 3:
                    seq_3d = seq
                    seq_2d = seq.reshape(seq.shape[0], seq.shape[1])
                else:
                    # fallback coercions
                    seq_2d = seq.reshape(1, -1)
                    seq_3d = seq.reshape(1, -1, 1)

                # If the underlying HF TimesFmModel requires sequences grouped by
                # a patch length (eg. patch_length=32) the forward pass may try to
                # reshape the inputs into patches and fail if the sequence length
                # isn't divisible by the patch size. Try to pad to the next
                # multiple of patch_length when possible.
                patch_len = None
                try:
                    cfg = getattr(self._hf_model, 'config', None)
                    patch_len = getattr(cfg, 'patch_length', None)
                except Exception:
                    patch_len = None

                def pad_to_patch(arr, patch_len):
                    if patch_len is None:
                        return arr
                    seq_len = arr.shape[1]
                    if seq_len % patch_len == 0:
                        return arr
                    target = ((seq_len + patch_len - 1) // patch_len) * patch_len
                    pad_amt = target - seq_len
                    # pad at the end with zeros
                    if arr.ndim == 2:
                        return np.pad(arr, ((0, 0), (0, pad_amt)), mode='constant', constant_values=0.0)
                    elif arr.ndim == 3:
                        return np.pad(arr, ((0, 0), (0, pad_amt), (0, 0)), mode='constant', constant_values=0.0)
                    else:
                        return arr

                seq_2d = pad_to_patch(seq_2d, patch_len)
                seq_3d = pad_to_patch(seq_3d, patch_len)

                # Build a 3D tensor for past_values (batch, seq_len, 1)
                t = torch.as_tensor(seq_3d, dtype=torch.float32)
                if self.device is not None:
                    try:
                        t = t.to(self.device)
                    except Exception:
                        pass

                # Build padding and freq tensors aligned with t
                try:
                    batch, seq_len, feat = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
                except Exception:
                    batch, seq_len, feat = 1, int(t.shape[1]) if t.ndim > 1 else 1, 1

                try:
                    pv_pad = torch.zeros(batch, seq_len, dtype=torch.long, device=t.device)
                except Exception:
                    pv_pad = torch.zeros(batch, seq_len, dtype=torch.long)

                try:
                    freq = torch.zeros(batch, dtype=torch.long, device=t.device)
                except Exception:
                    freq = torch.zeros(batch, dtype=torch.long)

                out = self._hf_model(past_values=t, past_values_padding=pv_pad, freq=freq)

                # out could be a tuple, a tensor, or a dict-like object
                preds = None
                if isinstance(out, tuple) and len(out) > 0:
                    maybe = out[0]
                    if isinstance(maybe, torch.Tensor):
                        preds = maybe
                elif hasattr(out, 'predictions'):
                    preds = out.predictions
                elif isinstance(out, torch.Tensor):
                    preds = out

                # If we got a predictions-like tensor, try to convert
                if preds is not None:
                    # Move to cpu and convert
                    try:
                        preds = preds.detach().cpu().numpy()
                    except Exception:
                        preds = np.asarray(preds)

                    # preds may have shape (batch, horizon, feat) or (batch, horizon)
                    if preds.ndim == 3:
                        preds = preds[0, :, 0]
                    elif preds.ndim == 2 and preds.shape[0] == 1:
                        preds = preds[0, :]

                    preds = np.asarray(preds, dtype=float)
                    # If length doesn't match horizon: try to trim/pad
                    if preds.size >= horizon:
                        preds = preds[:horizon]
                    else:
                        preds = np.pad(preds, (0, horizon - preds.size), mode='edge')

                    return [float(x) for x in preds.tolist()]

                # NOTE: forward exceptions are handled above (we attempted multiple shapes)
                # Any exception here will be handled by the outer except below.

        except Exception as e:
            # In strict mode propagate errors. Otherwise, swallow and use fallback.
            if getattr(self, 'strict', False):
                raise
            # else continue to fallback
            pass

        # ------------------- Fallback predictor ------------------
        # If at least two points are available, use simple linear extrapolation
        arr = np.asarray(self.last_data, dtype=float)
        if arr.size >= 2:
            # compute slope over last two points (could be improved — this keeps it tiny)
            slope = float(arr[-1] - arr[-2])
            base = float(arr[-1])
            preds = [base + slope * (i + 1) for i in range(horizon)]
        else:
            val = float(arr[-1])
            preds = [val for _ in range(horizon)]

        return [float(x) for x in preds]

    def update(self, data: pd.Series, **kwargs):
        """Update stored history window used for predictions (no online learning).

        If lookback is set we maintain a rolling buffer; otherwise we append.
        """
        if data is None:
            return

        try:
            arr = np.asarray(data.values, dtype=float)
        except Exception:
            arr = np.asarray(data, dtype=float)

        if self.last_data is None:
            if self.lookback is not None and len(arr) >= int(self.lookback):
                self.last_data = arr[-int(self.lookback):]
            else:
                self.last_data = arr.copy()
            return

        # Append and trim
        combined = np.concatenate([self.last_data, arr])
        if self.lookback is not None and len(combined) > int(self.lookback):
            combined = combined[-int(self.lookback):]

        self.last_data = combined

    def get_model_name(self) -> str:
        return f"TimesFM ({self.model_name})"

    def get_model_params(self) -> Dict[str, Any]:
        return {
            'horizon': self.horizon,
            'model_name': self.model_name,
            'lookback': self.lookback,
            'hf_available': bool(self._hf_available),
            'strict': bool(self.strict)
        }

    def supports_online_learning(self) -> bool:
        return False
