import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional, List, Tuple, Union


class IsolationForestAnomalyDetector:
    """
    Production-grade Isolation Forest Anomaly Branch adhering strictly to
    Section 24, Section 26, and Section 33 of the Generalized Data Pipeline Implementation Guide.

    Features:
      - Frozen model configuration support (e.g. V2 enhanced without acceleration)
      - MaterialID-level / Component-level score aggregation (mean, max, p90)
      - Calibrated continuous anomaly scoring in [0.0, 1.0]
      - Section 24.4 compliant output schema
      - Section 33 Plain-English anomaly feature attribution & baseline comparison
      - Serialization safety with native Python primitives (no uncoerced numpy scalars)
    """

    def __init__(
        self,
        model_version: str = "if_v2_material",
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
        decision_level: str = "MaterialID",
        score_aggregation: str = "mean_score",
        feature_selection_rule: Optional[str] = "exclude_acceleration",
        n_jobs: int = -1
    ):
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.decision_level = decision_level
        self.score_aggregation = score_aggregation
        self.feature_selection_rule = feature_selection_rule
        self.n_jobs = n_jobs

        self.model: Optional[IsolationForest] = None
        self.feature_names: List[str] = []
        self.baseline_mean: Optional[pd.Series] = None
        self.baseline_std: Optional[pd.Series] = None
        self.training_center: Optional[np.ndarray] = None

        # Calibration parameters
        self.calib_min: float = 0.35
        self.calib_max: float = 0.55
        self.frozen_threshold: Optional[float] = None
        self.is_frozen: bool = False
        self.config_metadata: Dict[str, Any] = {}

    @classmethod
    def load_frozen(
        cls,
        config_path: str = "models/D2_v2_no_acceleration_config.pkl",
        model_path: str = "models/D2_v2_no_acceleration_frozen_model.pkl"
    ) -> "IsolationForestAnomalyDetector":
        """Factory method to load validated frozen model artifacts."""
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Frozen model or config not found at: {model_path} / {config_path}")

        config = joblib.load(config_path)
        frozen_bundle = joblib.load(model_path)

        # Unpack model from dictionary bundle if needed
        if isinstance(frozen_bundle, dict):
            frozen_model = frozen_bundle.get("model", frozen_bundle)
            preprocessing = frozen_bundle.get("preprocessing", {})
        else:
            frozen_model = frozen_bundle
            preprocessing = {}

        detector = cls(
            model_version=f"{config.get('dataset', 'D2')}_frozen_v2",
            n_estimators=int(config.get("n_estimators", 200)),
            contamination=float(config.get("candidate_contamination", 0.01)),
            random_state=int(config.get("random_state", 42)),
            decision_level=str(config.get("decision_level", "MaterialID")),
            score_aggregation=str(config.get("score_aggregation", "mean_score")),
            feature_selection_rule="exclude_acceleration"
        )

        detector.model = frozen_model
        detector.feature_names = list(config.get("model_features", []))
        detector.frozen_threshold = float(config.get("validation_threshold", 0.393578))
        detector.is_frozen = True
        detector.config_metadata = config

        # Populate baseline stats if scaler is present in preprocessing
        if isinstance(preprocessing, dict) and "scaler" in preprocessing:
            scaler = preprocessing["scaler"]
            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                detector.baseline_mean = pd.Series(scaler.mean_, index=detector.feature_names)
                detector.baseline_std = pd.Series(scaler.scale_, index=detector.feature_names)
                detector.training_center = scaler.mean_

        # Set calibrated bounds based on frozen threshold
        # Maps frozen threshold ~0.393578 to calibrated review threshold ~0.65
        detector.calib_min = float(detector.frozen_threshold - 0.05)
        detector.calib_max = float(detector.frozen_threshold + 0.10)

        return detector

    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """Fits unsupervised Isolation Forest on validated behaviour features."""
        # 1. Feature filtering
        if feature_names is not None:
            selected_cols = [c for c in feature_names if c in X.columns]
        else:
            selected_cols = list(X.select_dtypes(include=[np.number]).columns)
            if self.feature_selection_rule == "exclude_acceleration":
                selected_cols = [c for c in selected_cols if "_acceleration" not in c]

        self.feature_names = selected_cols
        X_fit = X[self.feature_names].fillna(0.0)

        # 2. Store baseline reference population metrics for explainability
        self.baseline_mean = X_fit.mean(axis=0)
        self.baseline_std = X_fit.std(axis=0).replace(0.0, 1e-6)
        self.training_center = X_fit.values.mean(axis=0)

        # 3. Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model.fit(X_fit.values)

        # 4. Calibrate score bounds
        raw_train_scores = -self.model.score_samples(X_fit.values)
        self.calib_min = float(np.percentile(raw_train_scores, 5))
        self.calib_max = float(np.percentile(raw_train_scores, 95))
        if self.calib_max <= self.calib_min:
            self.calib_max = self.calib_min + 0.1

        return self

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures incoming data has all model features aligned in exact order without fragmentation."""
        if not self.feature_names:
            num_cols = list(X.select_dtypes(include=[np.number]).columns)
            if self.feature_selection_rule == "exclude_acceleration":
                num_cols = [c for c in num_cols if "_acceleration" not in c]
            return X[num_cols].fillna(0.0)

        X_aligned = X.reindex(columns=self.feature_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X_aligned

    def score(
        self,
        X: pd.DataFrame,
        group_col: Optional[str] = None,
        watch_threshold: float = 0.55,
        review_threshold: float = 0.65,
        reject_threshold: float = 0.85
    ) -> pd.DataFrame:
        """
        Generates calibrated continuous anomaly scores in [0.0, 1.0].
        Adheres to Section 24.4 of Implementation Guide.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # 1. Feature Alignment & Raw Scoring
        X_aligned = self._align_features(X)
        raw_scores = -self.model.score_samples(X_aligned.values)

        # 2. Continuous Calibration
        rng = max(self.calib_max - self.calib_min, 1e-6)
        calibrated_scores = np.clip((raw_scores - self.calib_min) / rng, 0.0, 1.0)

        temp_df = pd.DataFrame({
            "raw_anomaly_score": raw_scores,
            "anomaly_score": calibrated_scores
        }, index=X.index)

        # 3. Decision Level & Score Aggregation
        active_group_col = group_col
        if not active_group_col and self.decision_level in X.columns:
            active_group_col = self.decision_level

        if active_group_col and active_group_col in X.columns:
            temp_df[active_group_col] = X[active_group_col]
            if "mean" in self.score_aggregation.lower():
                agg_df = temp_df.groupby(active_group_col)[["raw_anomaly_score", "anomaly_score"]].transform("mean")
            elif "max" in self.score_aggregation.lower():
                agg_df = temp_df.groupby(active_group_col)[["raw_anomaly_score", "anomaly_score"]].transform("max")
            else:
                agg_df = temp_df.groupby(active_group_col)[["raw_anomaly_score", "anomaly_score"]].transform(lambda s: s.quantile(0.90))

            raw_eval = agg_df["raw_anomaly_score"].values
            calib_eval = agg_df["anomaly_score"].values
        else:
            raw_eval = raw_scores
            calib_eval = calibrated_scores

        # 4. Status assignment per Section 24.4
        statuses = []
        for raw_s, calib_s in zip(raw_eval, calib_eval):
            # If frozen threshold is configured, check against frozen validation threshold
            if self.is_frozen and self.frozen_threshold is not None:
                if raw_s >= self.frozen_threshold:
                    statuses.append("high")
                elif raw_s >= (self.frozen_threshold - 0.03):
                    statuses.append("watch")
                else:
                    statuses.append("normal")
            else:
                if calib_s >= review_threshold:
                    statuses.append("high")
                elif calib_s >= watch_threshold:
                    statuses.append("watch")
                else:
                    statuses.append("normal")

        results_df = pd.DataFrame({
            "anomaly_score": [round(float(s), 5) for s in calib_eval],
            "raw_anomaly_score": [round(float(s), 5) for s in raw_eval],
            "anomaly_status": statuses,
            "model_version": self.model_version,
            "decision_level": self.decision_level if active_group_col else "record_level",
            "feature_version": f"v_{len(self.feature_names)}_features",
            "decision_threshold_version": "frozen_val" if self.is_frozen else "calibrated_std"
        }, index=X.index)

        return results_df

    def explain_component(
        self,
        X_row: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Computes feature attribution for anomaly explainability (Section 33).
        Ranks top k features deviating most significantly from baseline reference population.
        """
        if self.baseline_mean is None or self.baseline_std is None or not self.feature_names:
            return []

        if isinstance(X_row, pd.DataFrame):
            row_series = X_row[self.feature_names].iloc[0]
        elif isinstance(X_row, pd.Series):
            row_series = X_row.reindex(self.feature_names).fillna(0.0)
        else:
            row_series = pd.Series(X_row, index=self.feature_names[:len(X_row)])

        z_scores = ((row_series - self.baseline_mean) / self.baseline_std).abs()
        top_features = z_scores.sort_values(ascending=False).head(top_k)

        attributions = []
        for feat, z_val in top_features.items():
            actual_val = float(row_series[feat])
            mean_val = float(self.baseline_mean[feat])
            direction = "elevated" if actual_val >= mean_val else "depressed"
            attributions.append({
                "feature": str(feat),
                "z_score": round(float(z_val), 3),
                "actual_value": round(actual_val, 4),
                "baseline_mean": round(mean_val, 4),
                "direction": direction,
                "impact": "High outlier deviation" if z_val >= 3.0 else "Moderate drift deviation"
            })

        return attributions

    def compare_against_baseline(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Isolation Forest against Centroid Distance Baseline per Section 24 & Section 27.
        """
        if self.training_center is None:
            X_align = self._align_features(X)
            self.training_center = X_align.values.mean(axis=0)

        X_align = self._align_features(X)
        if_scores = -self.model.score_samples(X_align.values)
        baseline_scores = np.linalg.norm(X_align.values - self.training_center, axis=1)

        result = {
            "sample_count": len(X),
            "if_score_mean": round(float(np.mean(if_scores)), 5),
            "if_score_std": round(float(np.std(if_scores)), 5),
            "baseline_distance_mean": round(float(np.mean(baseline_scores)), 5),
            "baseline_distance_std": round(float(np.std(baseline_scores)), 5)
        }

        if y_true is not None and len(np.unique(y_true.dropna())) >= 2:
            from sklearn.metrics import average_precision_score, roc_auc_score
            y_t = y_true.values.astype(int)
            result["if_roc_auc"] = round(float(roc_auc_score(y_t, if_scores)), 4)
            result["if_average_precision"] = round(float(average_precision_score(y_t, if_scores)), 4)
            result["baseline_roc_auc"] = round(float(roc_auc_score(y_t, baseline_scores)), 4)
            result["baseline_average_precision"] = round(float(average_precision_score(y_t, baseline_scores)), 4)

        return result

    def save(self, filepath: str):
        """Persists model with metadata, feature names, baseline stats, and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_version": self.model_version,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "decision_level": self.decision_level,
            "score_aggregation": self.score_aggregation,
            "feature_names": self.feature_names,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "training_center": self.training_center,
            "calib_min": self.calib_min,
            "calib_max": self.calib_max,
            "frozen_threshold": self.frozen_threshold,
            "is_frozen": self.is_frozen,
            "config_metadata": self.config_metadata
        }, filepath)

    def load(self, filepath: str):
        """Restores model and metadata from serialized bundle."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.model_version = data["model_version"]
        self.n_estimators = data["n_estimators"]
        self.contamination = data["contamination"]
        self.random_state = data["random_state"]
        self.decision_level = data.get("decision_level", "MaterialID")
        self.score_aggregation = data.get("score_aggregation", "mean_score")
        self.feature_names = data.get("feature_names", [])
        self.baseline_mean = data.get("baseline_mean")
        self.baseline_std = data.get("baseline_std")
        self.training_center = data.get("training_center")
        self.calib_min = float(data.get("calib_min", 0.35))
        self.calib_max = float(data.get("calib_max", 0.55))
        self.frozen_threshold = data.get("frozen_threshold")
        self.is_frozen = bool(data.get("is_frozen", False))
        self.config_metadata = data.get("config_metadata", {})
        return self
