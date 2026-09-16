import numpy as np
import pandas as pd
from typing import Dict, Any


class PeerRelativeFeatureExtractor:
    """Computes peer-relative deviation and z-scores against approved reference population."""

    @staticmethod
    def extract(target_val: float, peers_df: pd.DataFrame, param_key: str) -> Dict[str, Any]:
        if peers_df.empty or param_key not in peers_df.columns:
            return {
                f"{param_key}_peer_mean": np.nan,
                f"{param_key}_peer_std": np.nan,
                f"{param_key}_peer_diff": 0.0,
                f"{param_key}_peer_zscore": 0.0,
                "peer_status": "unavailable_no_peers"
            }

        peer_vals = peers_df[param_key].dropna().values.astype(float)
        if len(peer_vals) == 0:
            return {
                f"{param_key}_peer_mean": np.nan,
                f"{param_key}_peer_std": np.nan,
                f"{param_key}_peer_diff": 0.0,
                f"{param_key}_peer_zscore": 0.0,
                "peer_status": "unavailable_no_peer_measurements"
            }

        peer_mean = float(np.mean(peer_vals))
        peer_std = float(np.std(peer_vals))
        diff = float(target_val - peer_mean)
        zscore = float(diff / (peer_std + 1e-8))

        return {
            f"{param_key}_peer_mean": peer_mean,
            f"{param_key}_peer_std": peer_std,
            f"{param_key}_peer_diff": diff,
            f"{param_key}_peer_zscore": zscore,
            "peer_status": "computed"
        }
