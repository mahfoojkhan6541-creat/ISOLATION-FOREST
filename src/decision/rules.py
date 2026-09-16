from typing import Dict, Any
from src.decision.risk_fusion import RiskFusionEngine


class DecisionRuleEngine:
    """Evaluates fused evidence to produce controlled operational recommendations."""

    def __init__(self, review_threshold: float = 0.65, reject_threshold: float = 0.85):
        self.review_threshold = review_threshold
        self.reject_threshold = reject_threshold

    def evaluate(self, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
        fused = RiskFusionEngine.fuse_evidence(evidence_pack)

        # Rule 1: RETEST - Data quality failure takes priority to avoid false rejection
        if fused["is_quality_compromised"]:
            return {
                "recommendation": "RETEST",
                "triggered_rule": "RULE_DATA_INTEGRITY_FAIL",
                "confidence": "HIGH",
                "action": "Order re-test or inspection of measurement contact/channel. Do not reject hardware on untrusted data.",
                "fused_evidence": fused
            }

        # Rule 2: REJECT - Gross anomaly + forecast confirmation
        if fused["is_high_anomaly"]:
            return {
                "recommendation": "REJECT",
                "triggered_rule": "RULE_CRITICAL_ANOMALY",
                "confidence": "HIGH",
                "action": "Flag for engineering reject review. Trajectory severely deviates from validated peer population bounds.",
                "fused_evidence": fused
            }

        # Rule 3: REVIEW - Ambiguous / common mode / moderate anomaly / forecast risk
        if fused["is_common_mode"]:
            return {
                "recommendation": "REVIEW",
                "triggered_rule": "RULE_COMMON_MODE_CONFOUNDER",
                "confidence": "MODERATE",
                "action": "Common-mode drift detected across multiple peer components. Inspect chamber/instrument calibration before isolating component.",
                "fused_evidence": fused
            }

        if fused["is_review_anomaly"] or fused["is_forecast_risky"] or fused["wide_uncertainty"]:
            reasons = []
            if fused["is_review_anomaly"]:
                reasons.append(f"Anomaly score ({fused['anomaly_score']:.3f}) exceeds review threshold ({self.review_threshold})")
            if fused["is_forecast_risky"]:
                reasons.append("Forecast projects adverse trajectory towards risk boundary")
            if fused["wide_uncertainty"]:
                reasons.append("Wide predictive uncertainty due to sparse trajectory history")

            return {
                "recommendation": "REVIEW",
                "triggered_rule": "RULE_ELEVATED_WATCH_LIST",
                "confidence": "MODERATE",
                "action": f"Route to QA engineering disposition: {'; '.join(reasons)}.",
                "fused_evidence": fused
            }

        # Rule 4: PASS - Nominal screening
        return {
            "recommendation": "PASS",
            "triggered_rule": "RULE_NOMINAL_PASS",
            "confidence": "HIGH",
            "action": "Component behavior matches expected peer population baseline. Approved for screening milestone sign-off.",
            "fused_evidence": fused
        }
