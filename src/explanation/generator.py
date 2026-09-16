from typing import Dict, Any, List, Optional


class ExplanationGenerator:
    """Generates human-readable, evidence-grounded rationales and physical context hypotheses."""

    @staticmethod
    def generate_explanation(
        component_id: str,
        evidence_pack: Dict[str, Any],
        decision_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        recommendation = decision_result.get("recommendation", "REVIEW")
        triggered_rule = decision_result.get("triggered_rule", "UNKNOWN")
        fused = decision_result.get("fused_evidence", {})

        anomaly_score = fused.get("anomaly_score", 0.0)
        has_forecast = fused.get("has_forecast", False)

        # 1. Plain-English narrative following Section 33.2 template
        paragraphs = []

        if recommendation == "PASS":
            narrative = (
                f"Component {component_id} is operating nominally within expected screening parameters. "
                f"Its anomaly score ({anomaly_score:.3f}) is comfortably below the review threshold. "
                "Data quality is verified as trusted with no sensor or communication anomalies detected. "
                "The trajectory exhibits stability aligned with the approved peer reference population."
            )
        elif recommendation == "RETEST":
            quality_reasons = evidence_pack.get("data_quality", {}).get("reasons", "Data integrity check failed")
            narrative = (
                f"Component {component_id} has been routed for RETEST due to measurement integrity issues: {quality_reasons}. "
                "Because data reliability is compromised, an AI anomaly classification cannot be certified. "
                "Engineering policy requires re-measuring the component rather than condemning flight hardware on untrusted data."
            )
        elif recommendation == "REJECT":
            narrative = (
                f"Component {component_id} shows severe anomalous behavior with an anomaly score of {anomaly_score:.3f}, "
                "greatly exceeding acceptable screening limits. "
                "The trajectory significantly departs from comparable lot peers under identical test conditions. "
                "Hardware rejection is recommended subject to QA engineering confirmation."
            )
        else: # REVIEW
            reason_clauses = []
            if fused.get("is_review_anomaly"):
                reason_clauses.append(f"its anomaly score ({anomaly_score:.3f}) is elevated above the review threshold")
            if fused.get("is_common_mode"):
                reason_clauses.append("a common-mode response was detected across the peer population indicating possible chamber variation")
            if fused.get("wide_uncertainty"):
                reason_clauses.append("the GPR forecast interval is broad due to sparse trajectory history")

            narrative = (
                f"Component {component_id} requires QA Engineering REVIEW because "
                f"{'; '.join(reason_clauses) if reason_clauses else 'behavior warrants human engineering evaluation'}. "
                "Measurements are trusted, but trajectory features indicate watch-level deviation from nominal peer baselines."
            )

        # 2. Physical-Cause Context Hypotheses (Section 35)
        # Note: Interpretive hypotheses only, never stated as automated physical diagnoses.
        hypotheses = []
        if anomaly_score >= 0.65:
            hypotheses.append({
                "mechanism": "Threshold Voltage Shift / Gate-Oxide Wear (TDDB)",
                "rationale": "Gradual monotonic parameter drift over sequential screening steps is characteristic of dielectric charge trapping or gate degradation.",
                "disclaimer": "Interpretive engineering hypothesis only. Does not replace physical failure analysis."
            })
            hypotheses.append({
                "mechanism": "Interconnect Electromigration / Contact Resistance Drift",
                "rationale": "Localized step shifts under thermal/electrical stress may indicate voiding or contact resistance growth.",
                "disclaimer": "Interpretive engineering hypothesis only. Does not replace physical failure analysis."
            })

        return {
            "component_id": str(component_id),
            "recommendation": recommendation,
            "triggered_rule": triggered_rule,
            "plain_english_narrative": narrative,
            "plain_english_reason": narrative,
            "physical_hypotheses": hypotheses,
            "confidence_assessment": decision_result.get("confidence", "MODERATE"),
            "recommended_action": decision_result.get("action", "")
        }
