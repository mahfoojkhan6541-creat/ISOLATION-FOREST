import datetime
from typing import Dict, Any, List, Optional


class ProgressiveStateManager:
    """Manages progressive checkpoint arrival and state evolution without lookahead leakage."""

    def __init__(self):
        # Maps component_id -> state dictionary
        self.states: Dict[str, Dict[str, Any]] = {}

    def get_or_create_state(self, component_id: str) -> Dict[str, Any]:
        if component_id not in self.states:
            self.states[component_id] = {
                "component_id": str(component_id),
                "received_checkpoints": [],
                "latest_checkpoint": None,
                "history_timeline": [],
                "latest_anomaly_result": None,
                "latest_forecast_result": None,
                "latest_decision": None,
                "latest_explanation": None,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        return self.states[component_id]

    def update_checkpoint_state(
        self,
        component_id: str,
        checkpoint: Any,
        observation_values: Dict[str, float],
        anomaly_result: Dict[str, Any],
        forecast_result: Optional[Dict[str, Any]],
        decision_result: Dict[str, Any],
        explanation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates component state with newly arrived checkpoint data while preserving historical transition log.
        """
        state = self.get_or_create_state(component_id)

        if checkpoint not in state["received_checkpoints"]:
            state["received_checkpoints"].append(checkpoint)

        state["latest_checkpoint"] = checkpoint
        state["latest_anomaly_result"] = anomaly_result
        state["latest_forecast_result"] = forecast_result
        state["latest_decision"] = decision_result
        state["latest_explanation"] = explanation_result
        state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Append to historical transitions
        state["history_timeline"].append({
            "checkpoint": checkpoint,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "values": observation_values,
            "anomaly_score": anomaly_result.get("anomaly_score"),
            "anomaly_status": anomaly_result.get("anomaly_status"),
            "forecast_mean": forecast_result.get("forecast_mean") if forecast_result else None,
            "recommendation": decision_result.get("recommendation")
        })

        return state

    def get_component_history(self, component_id: str) -> List[Dict[str, Any]]:
        state = self.states.get(str(component_id))
        return state.get("history_timeline", []) if state else []

    def get_latest_state(self, component_id: str) -> Optional[Dict[str, Any]]:
        return self.states.get(str(component_id))

    def get_history(self, component_id: str) -> List[Dict[str, Any]]:
        return self.get_component_history(component_id)

    def update_component_state(
        self,
        component_id: str,
        checkpoint: Any,
        measurements: Dict[str, float],
        anomaly_result: Dict[str, Any],
        forecast_result: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        explanation: Any,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.update_checkpoint_state(
            component_id=component_id,
            checkpoint=checkpoint,
            observation_values=measurements,
            anomaly_result=anomaly_result,
            forecast_result=forecast_result,
            decision_result=decision,
            explanation_result=explanation
        )
