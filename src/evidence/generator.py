from typing import Dict, Any, Optional


class EvidenceGenerator:
    """Consolidates anomaly, forecast, data quality, and peer evidence into a structured Evidence Pack."""

    @staticmethod
    def build_evidence_pack(
        component_id: str,
        checkpoint: Any,
        data_quality_info: Dict[str, Any],
        anomaly_info: Dict[str, Any],
        forecast_info: Optional[Dict[str, Any]],
        peer_evidence: Dict[str, Any],
        confounder_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "component_id": str(component_id),
            "checkpoint": checkpoint,
            "data_quality": {
                "status": data_quality_info.get("status", "VALID"),
                "reasons": data_quality_info.get("reasons", "No quality issues detected")
            },
            "anomaly": {
                "score": float(anomaly_info.get("anomaly_score", 0.0)),
                "status": anomaly_info.get("anomaly_status", "normal"),
                "model_version": anomaly_info.get("model_version", "unknown")
            },
            "forecast": {
                "status": forecast_info.get("forecast_status", "unavailable") if forecast_info else "unavailable",
                "mean": forecast_info.get("forecast_mean") if forecast_info else None,
                "std": forecast_info.get("forecast_std") if forecast_info else None,
                "interval": forecast_info.get("interval") if forecast_info else None,
                "horizon": forecast_info.get("forecast_horizon") if forecast_info else None
            },
            "peer_comparison": peer_evidence,
            "confounder": confounder_info or {
                "common_mode_detected": False,
                "status": "nominal"
            }
        }
