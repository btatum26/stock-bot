import numpy as np

class MLInference:
    """
    Vectorized inference helper for converting model outputs into signals.

    The engine's canonical path uses SignalValidator in backtester.py. This
    class remains as a small ML-side helper for callers that need numpy arrays.
    """
    
    def apply_activation(self, predictions: np.ndarray) -> np.ndarray:
        """
        Compresses raw model outputs into the range [-1.0, 1.0].
        """
        values = np.asarray(predictions, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(np.tanh(values), -1.0, 1.0)

    def generate_signal_array(self, model_output: np.ndarray) -> np.ndarray:
        """
        Converts predictions into the standardized [-1.0, 1.0] signal array.

        One-dimensional outputs are treated as raw scores and passed through
        tanh. Two-column binary probability outputs are converted to
        ``p_long - p_short``. Wider softmax-style class matrices are rejected
        because the engine expects native fractional conviction, not class IDs.
        """
        values = np.asarray(model_output, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim == 2:
            if values.shape[1] == 1:
                values = values[:, 0]
            elif values.shape[1] == 2:
                values = values[:, 1] - values[:, 0]
                values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                return np.clip(values, -1.0, 1.0)
            else:
                raise ValueError("Multi-class softmax outputs are not supported; return a 1D fractional signal.")
        elif values.ndim > 2:
            raise ValueError("model_output must be a 1D score array or 2D binary probability matrix.")

        return self.apply_activation(values)
