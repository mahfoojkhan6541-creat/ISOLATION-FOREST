import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support


class ModelEvaluator:
    """Evaluates anomaly detection and forecast performance without label leakage."""

    @staticmethod
    def evaluate_anomaly_branch(
        y_true: pd.Series,
        anomaly_scores: pd.Series,
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Calculates ranking and classification metrics for anomaly detection.
        Excludes unlabelled records.
        """
        valid_mask = y_true.notnull()
        y_t = y_true[valid_mask].astype(int).values
        scores = anomaly_scores[valid_mask].values

        if len(np.unique(y_t)) < 2:
            return {
                "status": "insufficient_classes",
                "total_evaluated": len(y_t),
                "positive_count": int(np.sum(y_t))
            }

        ap = float(average_precision_score(y_t, scores))
        auc = float(roc_auc_score(y_t, scores))

        y_pred = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_t, y_pred, average="binary", zero_division=0)
        alert_rate = float(np.mean(y_pred))

        return {
            "average_precision": ap,
            "roc_auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "alert_rate": alert_rate,
            "threshold_used": threshold,
            "total_evaluated": len(y_t),
            "positive_count": int(np.sum(y_t))
        }

    @staticmethod
    def evaluate_forecast_branch(
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculates regression and interval calibration metrics for forecasting."""
        errors = y_true - y_pred_mean
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        coverage = None
        if lower_bound is not None and upper_bound is not None:
            in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            coverage = float(np.mean(in_interval))

        return {
            "mae": mae,
            "rmse": rmse,
            "interval_coverage": coverage,
            "sample_count": len(y_true)
        }
