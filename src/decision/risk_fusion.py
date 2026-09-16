from typing import Dict, Any


class RiskFusionEngine:
    """Combines independent anomaly, forecast, data-quality, and confounder evidence."""

    @staticmethod
    def fuse_evidence(evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        anomaly = evidence_pack.get("anomaly", {})
        forecast = evidence_pack.get("forecast", {})
        quality = evidence_pack.get("data_quality", {})
        confounder = evidence_pack.get("confounder", {})

        score = float(anomaly.get("score", 0.0))
        is_high_anomaly = score >= 0.85
        is_review_anomaly = (score >= 0.65) and (score < 0.85)

        is_quality_compromised = str(quality.get("status", "")).upper() in ["QUARANTINE", "BLOCK", "FAILED", "INVALID", "ERROR"]
        is_common_mode = confounder.get("common_mode_detected", False)

        has_forecast = forecast.get("status") == "available"
        is_forecast_risky = False
        wide_uncertainty = False

        if has_forecast and forecast.get("std") is not None:
            std_val = float(forecast["std"])
            if std_val > 1.0:
                wide_uncertainty = True
            # If forecast projected upper interval is high
            interval = forecast.get("interval", {})
            if interval and interval.get("upper") is not None and abs(float(interval["upper"])) > 3.5:
                is_forecast_risky = True

        return {
            "anomaly_score": score,
            "is_high_anomaly": is_high_anomaly,
            "is_review_anomaly": is_review_anomaly,
            "is_quality_compromised": is_quality_compromised,
            "is_common_mode": is_common_mode,
            "has_forecast": has_forecast,
            "is_forecast_risky": is_forecast_risky,
            "wide_uncertainty": wide_uncertainty
        }
