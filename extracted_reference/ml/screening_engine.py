"""Deterministic, explainable burn-in screening engine for SIH26170.

The engine intentionally keeps the core models transparent. It uses robust
peer statistics for Module A and compares a linear ridge forecaster with a
polynomial ridge forecaster for Module B. Early decisions never read 96h or
168h values: those values are only used for training and retrospective tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORTED_TIMES = (0, 24, 96, 168)
EARLY_TIMES = (0, 24)
REQUIRED_COLUMNS = ("component_id", "parameter", "unit")
OPTIONAL_DEFAULTS: dict[str, Any] = {
    "lot_id": "UNSPECIFIED_LOT",
    "device_type": "UNSPECIFIED_DEVICE",
    "temperature_c": np.nan,
    "voltage_v": np.nan,
    "spec_min": np.nan,
    "spec_max": np.nan,
    "label": "unknown",
}

ALIASES = {
    "component": "component_id",
    "componentid": "component_id",
    "part_id": "component_id",
    "partnumber": "device_type",
    "part_number": "device_type",
    "device": "device_type",
    "lot": "lot_id",
    "batch": "lot_id",
    "measurement": "parameter",
    "param": "parameter",
    "measurement_unit": "unit",
    "time": "time_hours",
    "hours": "time_hours",
    "test_time_hours": "time_hours",
    "measurement_value": "value",
    "reading": "value",
    "temp_c": "temperature_c",
    "temperature": "temperature_c",
    "voltage": "voltage_v",
    "lower_spec_limit": "spec_min",
    "lsl": "spec_min",
    "upper_spec_limit": "spec_max",
    "usl": "spec_max",
    "status": "label",
}


@dataclass(frozen=True)
class ValidationResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    summary: dict[str, Any]


def _snake(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _stable_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_index(axis=1).sort_values(list(frame.columns)).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_input(path: str | Path) -> pd.DataFrame:
    """Load CSV, XLSX, or JSON into a DataFrame without altering source values."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "records" in raw:
            raw = raw["records"]
        if not isinstance(raw, list):
            raise ValueError("JSON must be an array of records or contain a 'records' array.")
        return pd.DataFrame(raw)
    raise ValueError("Only CSV, XLSX, XLS, and JSON input files are supported.")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    renamed = {_column: ALIASES.get(_snake(_column), _snake(_column)) for _column in result.columns}
    result = result.rename(columns=renamed)
    duplicate_columns = result.columns[result.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Input contains duplicate canonical columns: {sorted(set(duplicate_columns))}")
    return result


def normalise_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Support wide `value_0h`/`value_24h` and canonical long input formats."""
    source = _normalise_columns(frame)
    wide_columns: dict[str, int] = {}
    for column in source.columns:
        compact = column.replace("_", "")
        for hour in SUPPORTED_TIMES:
            if compact in {f"value{hour}h", f"value{hour}", f"measurement{hour}h"}:
                wide_columns[column] = hour
    if wide_columns:
        id_columns = [column for column in source.columns if column not in wide_columns]
        rows: list[pd.DataFrame] = []
        for column, hour in wide_columns.items():
            part = source[id_columns].copy()
            part["time_hours"] = hour
            part["value"] = source[column]
            rows.append(part)
        source = pd.concat(rows, ignore_index=True)
    if "time_hours" not in source or "value" not in source:
        raise ValueError(
            "Use either long data with time_hours and value, or wide data with value_0h, value_24h, value_96h, and value_168h."
        )
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in source:
            source[column] = default
    for column in REQUIRED_COLUMNS:
        if column not in source:
            source[column] = np.nan
    ordered = [
        "component_id", "lot_id", "device_type", "parameter", "time_hours", "value", "unit",
        "temperature_c", "voltage_v", "spec_min", "spec_max", "label",
    ]
    return source[ordered].copy()


def validate_frame(frame: pd.DataFrame) -> ValidationResult:
    """Validate canonical records and quarantine invalid individual rows.

    Missing 96h/168h points are not invalid for an early screening batch. They
    are reported as quality indicators and are disallowed only by train/evaluate
    commands which require a 168h target.
    """
    result = normalise_to_long(frame)
    result["_row"] = np.arange(len(result))
    for column in ("component_id", "lot_id", "device_type", "parameter", "unit", "label"):
        result[column] = result[column].astype("string").str.strip()
    for column in ("time_hours", "value", "temperature_c", "voltage_v", "spec_min", "spec_max"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    reasons: list[list[str]] = [[] for _ in range(len(result))]
    for index, row in result.iterrows():
        if any(pd.isna(row[column]) or not str(row[column]).strip() for column in REQUIRED_COLUMNS):
            reasons[index].append("MISSING_REQUIRED_IDENTIFIER")
        if not np.isfinite(row["time_hours"]) or row["time_hours"] not in SUPPORTED_TIMES:
            reasons[index].append("UNSUPPORTED_TIME_POINT")
        if not np.isfinite(row["value"]):
            reasons[index].append("NON_NUMERIC_MEASUREMENT")
        if np.isfinite(row["spec_min"]) and np.isfinite(row["spec_max"]) and row["spec_min"] > row["spec_max"]:
            reasons[index].append("INVALID_SPECIFICATION_RANGE")

    result["validation_reasons"] = [";".join(reason) for reason in reasons]
    duplicate_mask = result.duplicated(["component_id", "parameter", "time_hours"], keep=False)
    result.loc[duplicate_mask, "validation_reasons"] = result.loc[duplicate_mask, "validation_reasons"].apply(
        lambda current: ";".join(filter(None, [current, "DUPLICATE_MEASUREMENT"]))
    )

    valid = result[result["validation_reasons"] == ""].copy()
    for _, group in valid.groupby(["component_id", "parameter"], dropna=False):
        if group["unit"].nunique(dropna=True) > 1:
            valid.loc[group.index, "validation_reasons"] = "UNIT_MISMATCH_WITHIN_COMPONENT"
    clean = valid[valid["validation_reasons"] == ""].drop(columns=["_row"]).reset_index(drop=True)
    quarantine = pd.concat(
        [result[result["validation_reasons"] != ""], valid[valid["validation_reasons"] != ""]],
        ignore_index=True,
    ).drop(columns=["_row"], errors="ignore").drop_duplicates().reset_index(drop=True)

    observed = set(clean["time_hours"].dropna().astype(int).tolist())
    summary = {
        "input_rows": int(len(result)),
        "accepted_rows": int(len(clean)),
        "quarantined_rows": int(len(quarantine)),
        "component_parameter_series": int(clean.groupby(["component_id", "parameter"]).ngroups),
        "observed_time_points": sorted(observed),
        "has_early_points": set(EARLY_TIMES).issubset(observed),
        "has_168h_target": 168 in observed,
        "data_hash": _stable_hash(clean) if len(clean) else None,
    }
    return ValidationResult(clean=clean, quarantine=quarantine, summary=summary)


def _safe_mad(values: pd.Series | np.ndarray) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) < 2:
        return 1.0
    median = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - median))) * 1.4826
    return mad if mad > 1e-9 else max(float(np.std(numeric)), 1e-6)


def _percentile(value: float, peer_values: Iterable[float]) -> float:
    peers = np.asarray(list(peer_values), dtype=float)
    peers = peers[np.isfinite(peers)]
    if not len(peers) or not np.isfinite(value):
        return 0.5
    return float(np.mean(peers <= value))


def build_feature_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Build strictly interpretable, early-safe features for each component/parameter series."""
    keys = ["component_id", "lot_id", "device_type", "parameter", "unit", "temperature_c", "voltage_v", "spec_min", "spec_max", "label"]
    pivot = clean.pivot_table(index=keys, columns="time_hours", values="value", aggfunc="first").reset_index()
    pivot.columns = [f"value_{int(c)}h" if isinstance(c, (int, float, np.integer, np.floating)) else c for c in pivot.columns]
    for hour in SUPPORTED_TIMES:
        column = f"value_{hour}h"
        if column not in pivot:
            pivot[column] = np.nan
    pivot["delta_0_24"] = pivot["value_24h"] - pivot["value_0h"]
    pivot["relative_delta_0_24"] = pivot["delta_0_24"] / pivot["value_0h"].abs().clip(lower=1e-9)
    pivot["slope_0_24"] = pivot["delta_0_24"] / 24.0
    pivot["slope_24_96"] = (pivot["value_96h"] - pivot["value_24h"]) / 72.0
    pivot["slope_96_168"] = (pivot["value_168h"] - pivot["value_96h"]) / 72.0
    pivot["curvature_early"] = pivot["slope_24_96"] - pivot["slope_0_24"]
    pivot["measurement_quality_missing_count"] = pivot[[f"value_{hour}h" for hour in SUPPORTED_TIMES]].isna().sum(axis=1)
    pivot["has_early_data"] = pivot[["value_0h", "value_24h"]].notna().all(axis=1)
    pivot["has_target_168h"] = pivot["value_168h"].notna()

    group_columns = ["lot_id", "device_type", "parameter", "unit"]
    peer_median = pivot.groupby(group_columns, dropna=False)["value_24h"].transform("median")
    peer_mad = pivot.groupby(group_columns, dropna=False)["value_24h"].transform(_safe_mad)
    slope_median = pivot.groupby(group_columns, dropna=False)["slope_0_24"].transform("median")
    slope_mad = pivot.groupby(group_columns, dropna=False)["slope_0_24"].transform(_safe_mad)
    pivot["peer_median_24h"] = peer_median
    pivot["peer_mad_24h"] = peer_mad.fillna(1.0)
    pivot["peer_robust_z_24h"] = (pivot["value_24h"] - peer_median) / pivot["peer_mad_24h"].clip(lower=1e-9)
    pivot["peer_slope_median"] = slope_median
    pivot["peer_slope_mad"] = slope_mad.fillna(1.0)
    pivot["slope_robust_z"] = (pivot["slope_0_24"] - slope_median) / pivot["peer_slope_mad"].clip(lower=1e-9)
    pivot["peer_percentile_24h"] = pivot.groupby(group_columns, dropna=False)["value_24h"].transform(
        lambda values: values.rank(pct=True, method="average")
    ).fillna(0.5)
    return pivot.reset_index(drop=True)


def _feature_columns() -> list[str]:
    return [
        "value_0h", "value_24h", "delta_0_24", "relative_delta_0_24", "slope_0_24",
        "peer_robust_z_24h", "slope_robust_z", "peer_percentile_24h", "temperature_c", "voltage_v",
    ]


def _split_groups(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = sorted(features["lot_id"].fillna("UNSPECIFIED_LOT").astype(str).unique())
    if len(groups) < 2:
        indices = np.arange(len(features))
        test_mask = indices % 5 == 0
    else:
        test_groups = set(groups[::5] or [groups[-1]])
        test_mask = features["lot_id"].astype(str).isin(test_groups).to_numpy()
    if test_mask.sum() == 0 or (~test_mask).sum() == 0:
        test_mask = np.arange(len(features)) % 5 == 0
    return features.loc[~test_mask].copy(), features.loc[test_mask].copy()


def _prepare_matrix(frame: pd.DataFrame, columns: list[str], stats: dict[str, dict[str, float]] | None = None, nonlinear: bool = False) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    result = frame[columns].astype(float).copy()
    if stats is None:
        stats = {}
        for column in columns:
            series = result[column].replace([np.inf, -np.inf], np.nan)
            mean = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std()) if series.notna().sum() > 1 else 1.0
            stats[column] = {"mean": mean, "std": std if std > 1e-9 else 1.0}
    normalized = []
    for column in columns:
        values = result[column].replace([np.inf, -np.inf], np.nan).fillna(stats[column]["mean"]).to_numpy(dtype=float)
        scaled = (values - stats[column]["mean"]) / stats[column]["std"]
        normalized.append(scaled)
        if nonlinear:
            normalized.append(np.clip(scaled, -8, 8) ** 2)
    matrix = np.column_stack([np.ones(len(frame)), *normalized])
    return matrix, stats


def _fit_ridge(train: pd.DataFrame, nonlinear: bool, alpha: float = 1.0) -> dict[str, Any]:
    columns = _feature_columns()
    x, stats = _prepare_matrix(train, columns, nonlinear=nonlinear)
    y = train["value_168h"].to_numpy(dtype=float)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    residuals = y - x @ coefficients
    return {
        "kind": "polynomial_ridge" if nonlinear else "linear_ridge",
        "columns": columns,
        "stats": stats,
        "coefficients": coefficients.tolist(),
        "residual_quantile_90": float(np.quantile(np.abs(residuals), 0.90)) if len(residuals) else 0.0,
        "residual_std": float(np.std(residuals)) if len(residuals) else 0.0,
    }


def _forecast(model: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, _ = _prepare_matrix(frame, model["columns"], stats=model["stats"], nonlinear=model["kind"] == "polynomial_ridge")
    prediction = x @ np.asarray(model["coefficients"], dtype=float)
    interval = np.full(len(frame), float(model["residual_quantile_90"] or model["residual_std"] or 0.0))
    return prediction, interval


def _metrics(actual: np.ndarray, predicted: np.ndarray, labels: np.ndarray | None = None, actions: np.ndarray | None = None) -> dict[str, float | int | None]:
    if not len(actual):
        return {"mae": None, "bias": None, "rmse": None, "recall": None, "false_negative_rate": None, "precision": None}
    errors = predicted - actual
    result: dict[str, float | int | None] = {
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "rmse": float(math.sqrt(np.mean(errors ** 2))),
        "recall": None,
        "false_negative_rate": None,
        "precision": None,
    }
    if labels is not None and actions is not None:
        defective = np.array([_is_defective(label) for label in labels])
        flagged = np.isin(actions, ["REVIEW", "REJECT"])
        positives = int(defective.sum())
        true_positive = int(np.logical_and(defective, flagged).sum())
        false_negative = int(np.logical_and(defective, ~flagged).sum())
        result.update({
            "recall": float(true_positive / positives) if positives else None,
            "false_negative_rate": float(false_negative / positives) if positives else None,
            "precision": float(true_positive / max(int(flagged.sum()), 1)),
            "false_negative_count": false_negative,
        })
    return result


def _default_policy() -> dict[str, Any]:
    """Return the conservative default policy before label-aware calibration.

    `safety_slope_mode` can be changed to `absolute` with explicit upper/lower
    slopes, or `peer_reference` with a peer-slope MAD margin. This makes the
    engineering policy explicit and versioned instead of embedded in code.
    """
    return {
        "peer_z_review": 3.5,
        "peer_z_reject": 5.0,
        "slope_z_review": 3.0,
        "slope_z_reject": 4.5,
        "review_risk": 0.55,
        "peer_weight": 0.55,
        "slope_weight": 0.45,
        "safety_slope_mode": "remaining_limit",
        "upper_safety_slope": None,
        "lower_safety_slope": None,
        "peer_slope_margin_mad": 3.0,
        "safety_slope_description": "remaining_limit: upper=(spec_max-value_24h)/144 and lower=(spec_min-value_24h)/144; absolute: supplied upper/lower slope values; peer_reference: peer slope median ± MAD margin.",
    }


def _merge_policy(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = _default_policy()
    if overrides:
        unknown = sorted(set(overrides) - set(policy))
        if unknown:
            raise ValueError(f"Unsupported policy option(s): {', '.join(unknown)}")
        policy.update(overrides)
    if policy["safety_slope_mode"] not in {"remaining_limit", "absolute", "peer_reference"}:
        raise ValueError("safety_slope_mode must be remaining_limit, absolute, or peer_reference.")
    if not math.isclose(float(policy["peer_weight"]) + float(policy["slope_weight"]), 1.0, abs_tol=1e-9):
        raise ValueError("peer_weight and slope_weight must add to 1.0.")
    return policy


def _allowed_safety_slopes(feature: pd.Series, policy: dict[str, Any]) -> tuple[float | None, float | None]:
    mode = policy["safety_slope_mode"]
    current = float(feature["value_24h"])
    spec_min = float(feature["spec_min"]) if pd.notna(feature["spec_min"]) else math.nan
    spec_max = float(feature["spec_max"]) if pd.notna(feature["spec_max"]) else math.nan
    if mode == "absolute":
        return policy.get("upper_safety_slope"), policy.get("lower_safety_slope")
    if mode == "peer_reference":
        median = float(feature.get("peer_slope_median", math.nan))
        mad = float(feature.get("peer_slope_mad", math.nan))
        if not np.isfinite(median) or not np.isfinite(mad):
            return None, None
        margin = float(policy["peer_slope_margin_mad"])
        return median + margin * mad, median - margin * mad
    upper = (spec_max - current) / (168 - 24) if np.isfinite(spec_max) else None
    lower = (spec_min - current) / (168 - 24) if np.isfinite(spec_min) else None
    return upper, lower


def _calibrate_policy(train_frame: pd.DataFrame, model: dict[str, Any], base_policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate conservative review thresholds using labels from training data only.

    A false negative is weighted 20 times more than a false positive. When the
    dataset has no usable labels, the supplied conservative policy is retained
    and marked as uncalibrated rather than claiming supervised calibration.
    """
    usable = train_frame[train_frame["label"].astype(str).str.lower().isin({"healthy", "defective", "fail", "failed", "reject", "bad", "pass"})]
    has_two_classes = len(usable) > 10 and usable["label"].map(_is_defective).nunique() == 2
    if not has_two_classes:
        return base_policy, {"mode": "conservative_default", "reason": "No sufficient two-class training labels were available.", "objective": None}
    predicted, interval = _forecast(model, usable)
    best_policy = base_policy.copy()
    best_summary: dict[str, Any] | None = None
    for peer_review in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        for slope_review in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
            for review_risk in (0.20, 0.30, 0.40, 0.50, 0.60):
                candidate = base_policy | {
                    "peer_z_review": peer_review,
                    "peer_z_reject": peer_review + 1.5,
                    "slope_z_review": slope_review,
                    "slope_z_reject": slope_review + 1.5,
                    "review_risk": review_risk,
                }
                temp_artifact = {"model": model, "policy": candidate}
                actions = np.array([_policy(row, float(pred), float(width), temp_artifact)["action"] for (_, row), pred, width in zip(usable.iterrows(), predicted, interval)])
                defects = usable["label"].map(_is_defective).to_numpy()
                flagged = np.isin(actions, ["REVIEW", "REJECT"])
                fn = int(np.logical_and(defects, ~flagged).sum())
                fp = int(np.logical_and(~defects, flagged).sum())
                reviews = int((actions == "REVIEW").sum())
                # Missing a component labelled defective is treated as far more
                # costly than routing a healthy component to human review.
                objective = 100 * fn + fp + 0.10 * reviews
                summary = {"objective": float(objective), "false_negatives": fn, "false_positives": fp, "reviews": reviews, "flagged": int(flagged.sum())}
                if best_summary is None or (summary["objective"], summary["false_negatives"], summary["reviews"]) < (best_summary["objective"], best_summary["false_negatives"], best_summary["reviews"]):
                    best_policy, best_summary = candidate, summary
    return best_policy, {"mode": "label_calibrated", "objective": best_summary, "false_negative_cost": 100, "fitted_on": "grouped training partition only"}


def _is_defective(label: object) -> bool:
    return str(label).strip().lower() in {"defective", "fail", "failed", "reject", "bad", "1", "true", "yes"}


def _policy(feature: pd.Series, predicted: float, interval: float, artifact: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic Module A/Module B rules using early-safe features only."""
    reasons: list[str] = []
    evidence: list[str] = []
    current = float(feature["value_24h"])
    value0 = float(feature["value_0h"])
    spec_min = float(feature["spec_min"]) if pd.notna(feature["spec_min"]) else math.nan
    spec_max = float(feature["spec_max"]) if pd.notna(feature["spec_max"]) else math.nan
    peer_z = abs(float(feature.get("peer_robust_z_24h", 0.0) or 0.0))
    slope_z = abs(float(feature.get("slope_robust_z", 0.0) or 0.0))
    slope = (current - value0) / 24.0
    policy = artifact["policy"]

    hard_fail = (np.isfinite(spec_min) and current < spec_min) or (np.isfinite(spec_max) and current > spec_max)
    if hard_fail:
        reasons.append("ABSOLUTE_LIMIT_EXCEEDED")
        evidence.append("The 24-hour measurement is outside the supplied specification limit.")
    if peer_z >= policy["peer_z_review"]:
        reasons.append("PEER_DEVIATION_HIGH")
        evidence.append(f"The 24-hour value is {peer_z:.2f} robust deviations from its lot/device/parameter peer median.")
    if slope_z >= policy["slope_z_review"]:
        reasons.append("EARLY_DRIFT_ABOVE_PEER_SLOPE")
        evidence.append(f"The 0h to 24h slope is {slope_z:.2f} robust deviations from the peer slope median.")

    upper_breach = np.isfinite(spec_max) and predicted + interval > spec_max
    lower_breach = np.isfinite(spec_min) and predicted - interval < spec_min
    if upper_breach or lower_breach:
        reasons.append("PREDICTED_168H_LIMIT_BREACH")
        evidence.append("The uncertainty-adjusted 168-hour forecast crosses a supplied specification limit.")
    upper_allowed_slope, lower_allowed_slope = _allowed_safety_slopes(feature, policy)
    if upper_allowed_slope is not None and slope > float(upper_allowed_slope):
            reasons.append("SAFETY_SLOPE_EXCEEDED")
            evidence.append(f"The early slope ({slope:.6g}/h) is above the configured upper safety slope ({float(upper_allowed_slope):.6g}/h) using the {policy['safety_slope_mode']} policy.")
    if lower_allowed_slope is not None and slope < float(lower_allowed_slope):
            reasons.append("SAFETY_SLOPE_EXCEEDED")
            evidence.append(f"The early slope ({slope:.6g}/h) is below the configured lower safety slope ({float(lower_allowed_slope):.6g}/h) using the {policy['safety_slope_mode']} policy.")
    if int(feature.get("measurement_quality_missing_count", 0)) > 2:
        reasons.append("DATA_QUALITY_REVIEW_REQUIRED")
        evidence.append("The record has insufficient supporting measurements for a high-confidence decision.")

    anomaly_risk = min(1.0, float(policy["peer_weight"]) * min(peer_z / policy["peer_z_reject"], 1.0) + float(policy["slope_weight"]) * min(slope_z / policy["slope_z_reject"], 1.0))
    model_risk = 1.0 if (upper_breach or lower_breach or "SAFETY_SLOPE_EXCEEDED" in reasons) else 0.0
    risk = max(anomaly_risk, model_risk)
    if hard_fail or (model_risk >= 1.0 and peer_z >= policy["peer_z_review"]):
        action = "REJECT"
    elif risk >= policy["review_risk"] or reasons:
        action = "REVIEW"
    else:
        action = "PASS"
    if not reasons:
        reasons = ["WITHIN_CONFIGURED_RISK_POLICY"]
        evidence = ["The early measurements, peer-relative values, forecast, and uncertainty interval are within the configured policy."]
    summary = (
        f"{action}: 24-hour value is {current:.6g}; predicted 168-hour value is {predicted:.6g} ± {interval:.6g}. "
        + " ".join(evidence)
    )
    return {"action": action, "risk_score": round(float(risk), 6), "reason_codes": reasons, "explanation": summary, "early_slope": slope}


def train(features: pd.DataFrame, seed: int = 26170, policy_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit and select the most accurate transparent/nonlinear candidate on grouped validation."""
    eligible = features[features["has_early_data"] & features["has_target_168h"]].copy()
    if len(eligible) < 12:
        raise ValueError("Training requires at least 12 component/parameter series with 0h, 24h, and 168h measurements.")
    train_frame, validation_frame = _split_groups(eligible)
    if len(train_frame) < 8 or len(validation_frame) < 2:
        raise ValueError("Grouped validation cannot be formed from the supplied lot/component groups.")
    base_policy = _merge_policy(policy_overrides)
    candidates = [_fit_ridge(train_frame, nonlinear=False), _fit_ridge(train_frame, nonlinear=True)]
    scores: list[dict[str, Any]] = []
    for candidate in candidates:
        calibrated_policy, calibration = _calibrate_policy(train_frame, candidate, base_policy)
        prediction, interval = _forecast(candidate, validation_frame)
        temp_artifact = {"model": candidate, "policy": calibrated_policy}
        actions = np.array([_policy(row, float(pred), float(width), temp_artifact)["action"] for (_, row), pred, width in zip(validation_frame.iterrows(), prediction, interval)])
        candidate_metrics = _metrics(validation_frame["value_168h"].to_numpy(dtype=float), prediction, validation_frame["label"].to_numpy(), actions)
        candidate_metrics["interval_coverage_90"] = float(np.mean(np.abs(validation_frame["value_168h"].to_numpy(dtype=float) - prediction) <= interval))
        candidate_metrics["threshold_policy"] = calibrated_policy
        candidate_metrics["policy_calibration"] = calibration
        scores.append({"model": candidate, "metrics": candidate_metrics, "policy": calibrated_policy, "calibration": calibration})
    selected = min(scores, key=lambda item: item["metrics"]["mae"] if item["metrics"]["mae"] is not None else float("inf"))["model"]
    final_model = _fit_ridge(eligible, nonlinear=selected["kind"] == "polynomial_ridge")
    final_policy, final_calibration = _calibrate_policy(eligible, final_model, base_policy)
    artifact = {
        "artifact_version": "1.0.0",
        "seed": seed,
        "model": final_model,
        "candidate_validation": [{"kind": item["model"]["kind"], **item["metrics"]} for item in scores],
        "training_data_hash": _stable_hash(eligible),
        "training_rows": int(len(eligible)),
        "feature_columns": _feature_columns(),
        "policy": final_policy,
        "policy_calibration": final_calibration,
    }
    return artifact


def score(features: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Score early data only and return auditable Module A + Module B outcomes."""
    records = features[features["has_early_data"]].copy()
    if records.empty:
        raise ValueError("Screening requires both a 0-hour and a 24-hour measurement for each series.")
    predicted, interval = _forecast(artifact["model"], records)
    decisions = [_policy(row, float(pred), float(width), artifact) for (_, row), pred, width in zip(records.iterrows(), predicted, interval)]
    records["predicted_168h"] = predicted
    records["prediction_interval_90"] = interval
    records["action"] = [decision["action"] for decision in decisions]
    records["risk_score"] = [decision["risk_score"] for decision in decisions]
    records["reason_codes"] = [json.dumps(decision["reason_codes"]) for decision in decisions]
    records["explanation"] = [decision["explanation"] for decision in decisions]
    records["early_slope"] = [decision["early_slope"] for decision in decisions]
    return records.sort_values(["action", "risk_score"], ascending=[True, False]).reset_index(drop=True)


def evaluate(features: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    eligible = features[features["has_early_data"] & features["has_target_168h"]].copy()
    predictions = score(eligible, artifact)
    actual = predictions["value_168h"].to_numpy(dtype=float)
    forecast = predictions["predicted_168h"].to_numpy(dtype=float)
    interval = predictions["prediction_interval_90"].to_numpy(dtype=float)
    metrics = _metrics(actual, forecast, predictions["label"].to_numpy(), predictions["action"].to_numpy())
    metrics["interval_coverage_90"] = float(np.mean(np.abs(actual - forecast) <= interval)) if len(actual) else None
    metrics["screened_series"] = int(len(predictions))
    metrics["action_counts"] = predictions["action"].value_counts().to_dict()
    metrics["early_decision_contract"] = "Only 0h and 24h values are used by the forecasting and screening decision path."
    return metrics


def generate_demo_data(seed: int = 26170, lots: int = 5, components_per_lot: int = 28) -> pd.DataFrame:
    """Generate a clearly synthetic dataset with latent drift and hard-failure cases."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    parameter_profiles = [
        {"parameter": "Iddq", "unit": "uA", "base": 10.0, "noise": 0.55, "spec_min": 0.0, "spec_max": 50.0},
        {"parameter": "LeakageCurrent", "unit": "uA", "base": 4.2, "noise": 0.32, "spec_min": 0.0, "spec_max": 20.0},
    ]
    for lot_index in range(1, lots + 1):
        lot_shift = rng.normal(0, 0.45)
        for component_index in range(1, components_per_lot + 1):
            defect = rng.random() < 0.14
            hard_defect = defect and rng.random() < 0.22
            for profile in parameter_profiles:
                base = profile["base"] + lot_shift + rng.normal(0, profile["noise"])
                early_slope = rng.normal(0.002, 0.002)
                late_acceleration = rng.normal(0.0002, 0.00015)
                if defect:
                    early_slope += rng.uniform(0.006, 0.018)
                    late_acceleration += rng.uniform(0.0007, 0.0020)
                if hard_defect:
                    base = profile["spec_max"] * rng.uniform(1.01, 1.15)
                for hour in SUPPORTED_TIMES:
                    value = base + early_slope * hour + late_acceleration * max(hour - 24, 0) ** 1.28 + rng.normal(0, profile["noise"] * 0.22)
                    rows.append({
                        "component_id": f"L{lot_index:02d}-C{component_index:03d}",
                        "lot_id": f"LOT-{lot_index:02d}",
                        "device_type": "SYNTH-SPACE-ASIC-A",
                        "parameter": profile["parameter"],
                        "time_hours": hour,
                        "value": round(float(value), 6),
                        "unit": profile["unit"],
                        "temperature_c": 125.0,
                        "voltage_v": 3.3,
                        "spec_min": profile["spec_min"],
                        "spec_max": profile["spec_max"],
                        "label": "defective" if defect else "healthy",
                        "data_origin": "SYNTHETIC_DEMONSTRATION_ONLY",
                    })
    return pd.DataFrame(rows)


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_html_report(scores: pd.DataFrame, metrics: dict[str, Any] | None = None, title: str = "Burn-In Screening Report") -> str:
    counts = scores["action"].value_counts().to_dict()
    rows = scores[["component_id", "lot_id", "parameter", "value_24h", "predicted_168h", "prediction_interval_90", "action", "reason_codes", "explanation"]].to_html(index=False, escape=True)
    metrics_html = "" if metrics is None else "<pre>" + json.dumps(metrics, indent=2) + "</pre>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#102a43}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d9e2ec;padding:8px;vertical-align:top}}th{{background:#243b53;color:white}}.note{{background:#fff8e1;padding:12px;border-left:4px solid #d69e2e}}</style></head><body><h1>{title}</h1><p class='note'><strong>Decision-support only.</strong> This report is deterministic and explainable, but it does not replace an approved engineering or flight-qualification decision.</p><p><strong>Action counts:</strong> PASS {counts.get('PASS', 0)} · REVIEW {counts.get('REVIEW', 0)} · REJECT {counts.get('REJECT', 0)}</p><h2>Evaluation summary</h2>{metrics_html}<h2>Component decisions</h2>{rows}</body></html>"""
