import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class BaselineFeatureExtractor:
    """Extracts component level and delta-from-baseline features."""

    @staticmethod
    def extract(trajectory: pd.DataFrame, param_key: str) -> Dict[str, Any]:
        """
        Calculates current level, baseline value, and relative change.
        trajectory must be chronologically sorted.
        """
        if trajectory.empty or param_key not in trajectory.columns:
            return {
                f"{param_key}_current": np.nan,
                f"{param_key}_baseline": np.nan,
                f"{param_key}_delta_from_baseline": np.nan,
                f"{param_key}_ratio_from_baseline": np.nan,
                "baseline_status": "unavailable_no_records"
            }

        values = trajectory[param_key].values
        current_val = float(values[-1])
        baseline_val = float(values[0])
        delta = current_val - baseline_val

        # Avoid division by zero
        ratio = (delta / baseline_val) if abs(baseline_val) > 1e-9 else 0.0

        return {
            f"{param_key}_current": current_val,
            f"{param_key}_baseline": baseline_val,
            f"{param_key}_delta_from_baseline": delta,
            f"{param_key}_ratio_from_baseline": ratio,
            "baseline_status": "computed"
        }
