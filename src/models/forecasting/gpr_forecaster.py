import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from typing import Dict, Any, Optional, Tuple, List


class GPRTrajectoryForecaster:
    """
    Gaussian Process Regression (GPR) Forecast Branch adhering to Section 25 of the guide.
    Estimates future trajectory drift while quantifying predictive uncertainty.
    """

    def __init__(self, model_version: str = "gpr_v1", random_state: int = 42):
        self.model_version = model_version
        self.random_state = random_state

    def fit_and_predict(
        self,
        trajectory: pd.DataFrame,
        param_key: str,
        future_checkpoints: List[float],
        min_history_points: int = 3
    ) -> Dict[str, Any]:
        """
        Fits GPR strictly on past trusted trajectory points and projects future distribution.
        """
        clean_traj = trajectory.dropna(subset=[param_key])
        n_points = len(clean_traj)

        # Enforce Section 25.4 rule: If history is insufficient, return explicit status
        if n_points < min_history_points:
            return {
                "forecast_status": "unavailable_insufficient_history",
                "param_key": param_key,
                "history_points": n_points,
                "forecast_mean": None,
                "forecast_std": None,
                "interval": None,
                "predictions_by_checkpoint": {},
                "model_version": self.model_version
            }

        # Time variable
        if "elapsed_time" in clean_traj.columns and clean_traj["elapsed_time"].nunique() > 1:
            X_train = clean_traj["elapsed_time"].values.reshape(-1, 1).astype(float)
        elif "checkpoint" in clean_traj.columns:
            X_train = clean_traj["checkpoint"].values.reshape(-1, 1).astype(float)
        else:
            X_train = np.arange(n_points).reshape(-1, 1).astype(float)

        y_train = clean_traj[param_key].values.astype(float)

        # GPR Kernel: ConstantKernel * RBF + WhiteKernel (sensor noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e1))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.random_state
        )

        try:
            gpr.fit(X_train, y_train)

            X_pred = np.array(future_checkpoints).reshape(-1, 1).astype(float)
            y_mean, y_std = gpr.predict(X_pred, return_std=True)

            predictions_dict = {}
            for cp, mean_val, std_val in zip(future_checkpoints, y_mean, y_std):
                predictions_dict[float(cp)] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "lower_2sigma": float(mean_val - 2.0 * std_val),
                    "upper_2sigma": float(mean_val + 2.0 * std_val)
                }

            latest_pred = predictions_dict[float(future_checkpoints[-1])]

            return {
                "forecast_status": "available",
                "param_key": param_key,
                "history_points": n_points,
                "forecast_horizon": float(future_checkpoints[-1]),
                "forecast_mean": latest_pred["mean"],
                "forecast_std": latest_pred["std"],
                "interval": {
                    "lower": latest_pred["lower_2sigma"],
                    "upper": latest_pred["upper_2sigma"]
                },
                "predictions_by_checkpoint": predictions_dict,
                "model_version": self.model_version
            }

        except Exception as e:
            return {
                "forecast_status": "fit_error",
                "error_details": str(e),
                "param_key": param_key,
                "history_points": n_points,
                "model_version": self.model_version
            }
