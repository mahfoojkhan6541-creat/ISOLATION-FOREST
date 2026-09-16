import numpy as np
import pandas as pd
from typing import Dict, Any


class DriftFeatureExtractor:
    """Computes trajectory drift, slope, curvature, and volatility."""

    @staticmethod
    def extract(trajectory: pd.DataFrame, param_key: str) -> Dict[str, Any]:
        n_points = len(trajectory)

        if n_points < 2:
            return {
                f"{param_key}_drift_slope": 0.0,
                f"{param_key}_trajectory_std": 0.0,
                f"{param_key}_curvature": np.nan,
                "drift_status": "unavailable_insufficient_points"
            }

        values = trajectory[param_key].values.astype(float)

        # Elapsed time or checkpoint as x
        if "elapsed_time" in trajectory.columns and trajectory["elapsed_time"].nunique() > 1:
            t = trajectory["elapsed_time"].values.astype(float)
        elif "checkpoint" in trajectory.columns:
            t = trajectory["checkpoint"].values.astype(float)
        else:
            t = np.arange(n_points, dtype=float)

        # 1. Slope / Drift rate (linear fit)
        try:
            # simple polyfit degree 1
            denom = np.sum((t - np.mean(t)) ** 2)
            if denom > 1e-12:
                slope = float(np.sum((t - np.mean(t)) * (values - np.mean(values))) / denom)
            else:
                slope = float(values[-1] - values[0])
        except Exception:
            slope = float(values[-1] - values[0])

        # 2. Volatility / STD
        volatility = float(np.std(values))

        # 3. Curvature / Acceleration (requires >= 3 points)
        if n_points >= 3:
            # second difference
            d2 = (values[-1] - values[-2]) - (values[-2] - values[-3])
            curvature = float(d2)
            curvature_status = "computed"
        else:
            curvature = np.nan
            curvature_status = "unavailable_insufficient_points"

        return {
            f"{param_key}_drift_slope": slope,
            f"{param_key}_trajectory_std": volatility,
            f"{param_key}_curvature": curvature,
            "drift_status": "computed",
            "curvature_status": curvature_status
        }
