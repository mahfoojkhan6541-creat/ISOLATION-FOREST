from typing import Dict, Any, List


class CompatibilityChecker:
    """Evaluates whether a dataset satisfies the minimum compatibility contract for AI branches."""

    def assess_compatibility(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        reasons: List[str] = []
        anomaly_compatible = True
        forecast_compatible = True

        records = profile["dimensions"]["records"]
        candidate_ids = profile["candidates"]["candidate_id_fields"]
        candidate_time = profile["candidates"]["candidate_time_fields"]
        candidate_numeric = profile["candidates"]["candidate_numeric_fields"]
        unique_components = profile["entities"]["unique_components"]
        checkpoint_count = profile["temporal"]["checkpoint_count"]

        # Check 1: Record count
        if records < 10:
            anomaly_compatible = False
            forecast_compatible = False
            reasons.append(f"Insufficient total records ({records} < 10).")

        # Check 2: Component identity
        if not candidate_ids or unique_components < 2:
            anomaly_compatible = False
            forecast_compatible = False
            reasons.append(f"Dataset lacks identifiable repeated components (found {unique_components} unique entities).")
        else:
            reasons.append(f"Valid component entity link established ({unique_components} unique components).")

        # Check 3: Numeric measurements
        if len(candidate_numeric) < 1:
            anomaly_compatible = False
            forecast_compatible = False
            reasons.append("No numeric measurement parameters found.")
        else:
            reasons.append(f"Identified {len(candidate_numeric)} measurable numeric parameters.")

        # Check 4: Temporal depth & repeated checkpoints
        if not candidate_time or checkpoint_count < 2:
            anomaly_compatible = False
            forecast_compatible = False
            reasons.append(f"Lacks repeated time checkpoints (found {checkpoint_count} checkpoints).")
        else:
            reasons.append(f"Repeated checkpoints detected ({checkpoint_count} unique checkpoints: {profile['temporal']['unique_checkpoints'][:10]}).")

        # Branch-specific check for Forecasting (GPR)
        # GPR requires at least 4 temporal checkpoints to form a valid training/testing degradation trajectory
        if checkpoint_count < 4:
            forecast_compatible = False
            reasons.append(f"Forecast branch incompatible: Insufficient temporal history ({checkpoint_count} < 4 checkpoints). GPR drift extrapolation requires >= 4 points.")
        else:
            reasons.append(f"Forecast branch compatible: Adequate temporal horizon ({checkpoint_count} checkpoints).")

        # Overall status determination
        if anomaly_compatible and forecast_compatible:
            status = "COMPATIBLE"
        elif anomaly_compatible or forecast_compatible:
            status = "PARTIALLY_COMPATIBLE"
        else:
            status = "INCOMPATIBLE"

        return {
            "status": status,
            "branches": {
                "anomaly_branch": "compatible" if anomaly_compatible else "incompatible",
                "forecast_branch": "compatible" if forecast_compatible else "incompatible"
            },
            "metrics": {
                "records": records,
                "components": unique_components,
                "checkpoints": checkpoint_count,
                "numeric_parameters": len(candidate_numeric)
            },
            "reasons": reasons
        }

    evaluate = assess_compatibility
