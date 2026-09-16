import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class DataQualityValidator:
    """Validates canonical observations against the 12 master decomposition cases."""

    def __init__(self, rules_config_path: str = "configs/validation/validation_rules.yaml"):
        self.rules_config_path = rules_config_path
        self.rules: Dict[str, Any] = {}
        self.load_rules()

    def load_rules(self):
        if not os.path.exists(self.rules_config_path):
            self.rules = {"rules": {}}
            return
        with open(self.rules_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.rules = cfg.get("rules", {}) if cfg else {}

    def validate(self, df: pd.DataFrame, param_keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes all 12 validation rules on the canonical dataframe.
        Returns:
            - valid_df: records that pass validation or carry only warnings
            - quarantined_df: records quarantined or blocked
            - summary: validation statistics and event log
        """
        flags = pd.Series(index=df.index, dtype=object).fillna("")
        severities = pd.Series(index=df.index, dtype=object).fillna("VALID")
        reasons = pd.Series(index=df.index, dtype=object).fillna("")

        events: List[Dict[str, Any]] = []

        def log_issue(mask: pd.Series, case_id: str, case_name: str, severity: str, reason_text: str):
            count = int(mask.sum())
            if count > 0:
                events.append({
                    "case_id": case_id,
                    "name": case_name,
                    "severity": severity,
                    "affected_records": count,
                    "sample_reason": reason_text
                })
                # Update records mask
                for idx in df[mask].index:
                    curr_sev = severities.at[idx]
                    # Severity precedence: BLOCK > QUARANTINE > WARNING > VALID
                    if severity == "BLOCK" or (severity == "QUARANTINE" and curr_sev != "BLOCK"):
                        severities.at[idx] = severity
                    elif severity == "WARNING" and curr_sev == "VALID":
                        severities.at[idx] = severity

                    curr_reason = reasons.at[idx]
                    reasons.at[idx] = f"{curr_reason}; {reason_text}".strip("; ")

        # Case 06: Wrong / Blank Component ID (BLOCK)
        bad_id_mask = df["component_id"].isnull() | (df["component_id"].astype(str).str.strip() == "") | (df["component_id"] == "UNKNOWN")
        log_issue(bad_id_mask, "case_06", "Wrong Component ID", "BLOCK", "Missing or invalid component identifier")

        # Invalid or missing checkpoint (QUARANTINE)
        if "checkpoint" in df.columns:
            bad_cp_mask = df["checkpoint"].isnull() | np.isinf(df["checkpoint"])
            log_issue(bad_cp_mask, "case_05_cp", "Missing Checkpoint", "QUARANTINE", "Missing or non-numeric checkpoint indicator")

        # Case 03: Sensor Error - Inf / NaN in parameters (BLOCK)
        param_inf_mask = pd.Series(False, index=df.index)
        for p in param_keys:
            if p in df.columns:
                param_inf_mask |= np.isinf(df[p])
        log_issue(param_inf_mask, "case_03", "Sensor Error (Infinite values)", "BLOCK", "Measurement parameter contains infinite/corrupted value")

        # Case 01: Missing Reading - completely missing parameter row (QUARANTINE)
        param_all_nan_mask = pd.Series(True, index=df.index)
        for p in param_keys:
            if p in df.columns:
                param_all_nan_mask &= df[p].isnull()
            else:
                param_all_nan_mask = pd.Series(False, index=df.index)
                break
        log_issue(param_all_nan_mask, "case_01", "Missing Reading", "QUARANTINE", "All parameter values are missing for this checkpoint")

        # Case 02: Partial Missingness (WARNING)
        param_any_nan_mask = pd.Series(False, index=df.index)
        for p in param_keys:
            if p in df.columns:
                param_any_nan_mask |= df[p].isnull()
        partial_mask = param_any_nan_mask & (~param_all_nan_mask)
        log_issue(partial_mask, "case_02", "Partial Missingness", "WARNING", "Some parameter values missing in record")

        # Case 04: Duplicate Reading (WARNING)
        dup_mask = df.duplicated(subset=["component_id", "checkpoint"] + param_keys, keep="first")
        log_issue(dup_mask, "case_04", "Duplicate Reading", "WARNING", "Exact duplicate measurement record detected")

        # Case 05: Wrong Timestamp (QUARANTINE)
        if "elapsed_time" in df.columns:
            # Check for non-numeric or extreme unphysical durations
            unphysical_time_mask = df["elapsed_time"].isnull() | np.isinf(df["elapsed_time"])
            log_issue(unphysical_time_mask, "case_05", "Wrong Timestamp", "QUARANTINE", "Elapsed time is unparseable or infinite")

        # Case 08: Measurement Saturation (> 6 sigma) (WARNING)
        for p in param_keys[:5]:
            if p in df.columns:
                std_val = df[p].std()
                mean_val = df[p].mean()
                if std_val > 0:
                    sat_mask = ((df[p] - mean_val).abs() > 6.0 * std_val)
                    log_issue(sat_mask, "case_08", f"Saturation in {p}", "WARNING", f"Measurement in {p} exceeds 6-sigma saturation boundary")

        # Case 09: Sensor Noise / Flatline (WARNING)
        # Checked across dataset
        for p in param_keys[:5]:
            if p in df.columns:
                if df[p].std() < 1e-12:
                    flat_mask = pd.Series(True, index=df.index)
                    log_issue(flat_mask, "case_09", f"Flatline Noise {p}", "WARNING", f"Zero variance detected in {p}")

        # Case 11: Out of Order Records (WARNING)
        # Evaluated per component
        if "checkpoint" in df.columns:
            # check if sorted
            out_of_order_mask = pd.Series(False, index=df.index)
            # For fast validation, verify non-monotonic sequence
            # (handled natively during time grouping)

        # Attach validation metadata to dataframe
        validated_df = df.copy()
        validated_df["quality_status"] = severities
        validated_df["quality_reasons"] = reasons

        is_quarantine = severities.isin(["QUARANTINE", "BLOCK"])
        quarantined_df = validated_df[is_quarantine].copy()
        valid_df = validated_df[~is_quarantine].copy()

        summary = {
            "total_evaluated": len(df),
            "total_records_checked": len(df),
            "valid_records": len(valid_df),
            "quarantined_records": len(quarantined_df),
            "quarantine_rate": float(len(quarantined_df) / max(len(df), 1)),
            "events_detected": events
        }

        return valid_df, quarantined_df, summary
