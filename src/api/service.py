import os
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.adapters import IntakeAdapterFactory
from src.ingestion.registry import DatasetRegistry, DatasetMetadata
from src.profiling.profiler import AutoProfiler
from src.compatibility.checker import CompatibilityChecker
from src.mapping.loader import MappingLoader
from src.mapping.canonical import CanonicalTransformer
from src.validation.validator import DataQualityValidator
from src.validation.quarantine import QuarantineManager
from src.grouping.population_selector import PopulationSelector
from src.features.engineering import FeatureEngineeringEngine
from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.forecasting.gpr_forecaster import GPRTrajectoryForecaster
from src.evidence.generator import EvidenceGenerator
from src.decision.rules import DecisionRuleEngine
from src.decision.risk_fusion import RiskFusionEngine
from src.explanation.generator import ExplanationGenerator
from src.state.progressive import ProgressiveStateManager
from src.audit.storage import AuditStorage

app = FastAPI(
    title="SIH26170 Generalized Burn-In Data Pipeline API",
    description="Configuration-driven, time-aware, multi-branch anomaly detection, forecasting & conservative decision API.",
    version="1.0.0"
)

import numpy as np
from fastapi.encoders import ENCODERS_BY_TYPE

# Register numpy types for clean JSON serialization
ENCODERS_BY_TYPE[np.int64] = int
ENCODERS_BY_TYPE[np.int32] = int
ENCODERS_BY_TYPE[np.int16] = int
ENCODERS_BY_TYPE[np.int8] = int
ENCODERS_BY_TYPE[np.uint64] = int
ENCODERS_BY_TYPE[np.uint32] = int
ENCODERS_BY_TYPE[np.uint16] = int
ENCODERS_BY_TYPE[np.uint8] = int
ENCODERS_BY_TYPE[np.float64] = float
ENCODERS_BY_TYPE[np.float32] = float
ENCODERS_BY_TYPE[np.bool_] = bool
ENCODERS_BY_TYPE[np.ndarray] = lambda arr: arr.tolist()

# Enable CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services & storage
registry = DatasetRegistry()
mapping_loader = MappingLoader()
canonical_transformer = CanonicalTransformer(mapping_loader=mapping_loader)
state_manager = ProgressiveStateManager()
audit_storage = AuditStorage()
rule_engine = DecisionRuleEngine()
evidence_generator = EvidenceGenerator()
explanation_generator = ExplanationGenerator()
quarantine_mgr = QuarantineManager()
feature_engine = FeatureEngineeringEngine()


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class DatasetRegisterRequest(BaseModel):
    dataset_id: str
    file_path: str
    domain: str = "burn_in"
    source_format: str = "csv"
    sampling_type: str = "progressive_checkpoints"
    intended_task: str = "trajectory_anomaly_and_forecast"


class ProfileRequest(BaseModel):
    dataset_id: str
    file_path: Optional[str] = None


class ValidateRequest(BaseModel):
    dataset_id: str
    config_path: str = "configs/validation/validation_rules.yaml"


class AnalyzeRequest(BaseModel):
    dataset_id: str
    mapping_version: str = "d1_v1"
    config_version: str = "default_v1"
    component_scope: str = "all"
    sample_limit: int = 150


class QAActionRequest(BaseModel):
    run_id: str
    component_id: str
    checkpoint: float
    ai_recommendation: str
    human_action: str  # e.g., "OVERRIDE_PASS", "CONFIRM_REJECT", "ESCALATE_RETEST"
    reviewer_id: str
    notes: Optional[str] = ""


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SIH26170 Generalized Data Pipeline",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


@app.post("/datasets/register")
def register_dataset(req: DatasetRegisterRequest):
    """Registers a new authentic burn-in dataset into the registry (Section 7)."""
    meta = DatasetMetadata(
        dataset_id=req.dataset_id,
        file_path=req.file_path,
        domain=req.domain,
        source_format=req.source_format,
        sampling_type=req.sampling_type,
        intended_task=req.intended_task
    )
    result = registry.register(meta)
    return {"status": "registered", "metadata": result}


@app.post("/datasets/profile")
def profile_dataset(req: ProfileRequest):
    """Performs automated profiling on a registered dataset (Section 8)."""
    reg_entry = registry.get(req.dataset_id)
    file_path = req.file_path or (reg_entry["file_path"] if reg_entry else None)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for dataset {req.dataset_id} not found.")

    adapter = IntakeAdapterFactory.get_adapter(file_path)
    df = adapter.load(file_path)
    profiler = AutoProfiler()
    profile = profiler.profile(df)
    return {"dataset_id": req.dataset_id, "profile": profile}


@app.post("/datasets/compatibility")
def check_compatibility(req: ProfileRequest):
    """Evaluates branch compatibility for IF and GPR (Section 9)."""
    reg_entry = registry.get(req.dataset_id)
    file_path = req.file_path or (reg_entry["file_path"] if reg_entry else None)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for dataset {req.dataset_id} not found.")

    adapter = IntakeAdapterFactory.get_adapter(file_path)
    df = adapter.load(file_path)
    profiler = AutoProfiler()
    profile = profiler.profile(df)

    checker = CompatibilityChecker()
    compat = checker.evaluate(profile)
    return {"dataset_id": req.dataset_id, "compatibility": compat}


@app.post("/datasets/validate")
def validate_dataset(req: ValidateRequest):
    """Executes the 12-case data quality validation gate and quarantine isolation (Section 14)."""
    reg_entry = registry.get(req.dataset_id)
    if not reg_entry:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not registered.")

    file_path = reg_entry["file_path"]
    adapter = IntakeAdapterFactory.get_adapter(file_path)
    df = adapter.load(file_path)

    mapping_cfg = mapping_loader.load_mapping(f"{req.dataset_id.lower()}_mapping")
    canonical_df, _ = canonical_transformer.transform(df, mapping_cfg, dataset_id=req.dataset_id)
    param_keys = [p["parameter_key"] for p in mapping_cfg["parameter_fields"]]

    validator = DataQualityValidator(rules_config_path=req.config_path)
    clean_df, quarantined_df, report = validator.validate(canonical_df, param_keys)

    return {
        "dataset_id": req.dataset_id,
        "validation_report": report,
        "quarantined_count": len(quarantined_df),
        "valid_count": len(clean_df)
    }


@app.post("/analyze")
def run_analysis(req: AnalyzeRequest):
    """
    Executes end-to-end analysis per Section 35.2 & 35.3:
    1. Ingestion & mapping
    2. Data quality gating
    3. Population grouping
    4. Behaviour feature engineering
    5. Isolation Forest anomaly scoring
    6. GPR trajectory forecasting & predictive uncertainty
    7. Evidence generation, risk fusion & conservative decision
    8. Plain-English explanation
    9. SQLite audit logging
    """
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    audit_storage.record_run_start(run_id, req.dataset_id, req.config_version)

    reg_entry = registry.get(req.dataset_id)
    if not reg_entry:
        candidate_path = f"data/{req.dataset_id}.csv"
        if os.path.exists(candidate_path):
            registry.register(DatasetMetadata(dataset_id=req.dataset_id, file_path=candidate_path))
            reg_entry = registry.get(req.dataset_id)
        else:
            raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not registered and file not found.")

    file_path = reg_entry["file_path"]
    adapter = IntakeAdapterFactory.get_adapter(file_path)
    raw_df = adapter.load(file_path)

    # 1. Profile & Compatibility
    profiler = AutoProfiler()
    profile = profiler.profile(raw_df)
    checker = CompatibilityChecker()
    compat = checker.evaluate(profile)

    # 2. Schema mapping
    mapping_name = f"{req.dataset_id.lower()}_mapping"
    mapping_cfg = mapping_loader.load_mapping(mapping_name)
    canonical_df, meta_info = canonical_transformer.transform(raw_df, mapping_cfg, dataset_id=req.dataset_id)
    param_keys = [p["parameter_key"] for p in mapping_cfg["parameter_fields"]]

    # 3. Data Quality Gate
    validator = DataQualityValidator()
    valid_df, quarantined_df, dq_report = validator.validate(canonical_df, param_keys)
    if not quarantined_df.empty:
        quarantine_mgr.quarantine_records(quarantined_df, run_id=run_id, reason="Data quality validation gate failure")

    # 4. Checkpoints & Feature Engineering
    checkpoints = sorted(valid_df["checkpoint"].dropna().unique())
    last_checkpoint = checkpoints[-1] if len(checkpoints) > 0 else 0

    # Limit to sample components for prompt response
    all_components = valid_df["component_id"].unique()
    target_components = all_components[:req.sample_limit]
    sub_df = valid_df[valid_df["component_id"].isin(target_components)]

    # Batch features at the latest checkpoint
    X_df, meta_df, eval_df = feature_engine.extract_feature_matrix_batch(sub_df, param_keys, last_checkpoint)

    # 5. Isolation Forest Anomaly Detection
    frozen_d2 = "models/D2_v2_no_acceleration_frozen_model.pkl"
    d2_cfg = "models/D2_v2_no_acceleration_config.pkl"
    if req.dataset_id == "D2" and os.path.exists(frozen_d2) and os.path.exists(d2_cfg):
        if_model = IsolationForestAnomalyDetector.load_frozen(config_path=d2_cfg, model_path=frozen_d2)
        score_df = if_model.score(X_df)
    else:
        if_model = IsolationForestAnomalyDetector(model_version=f"{req.dataset_id.lower()}_if_v1", n_estimators=200, contamination=0.01)
        if len(X_df) > 5:
            if_model.fit(X_df)
            score_df = if_model.score(X_df)
        else:
            score_df = pd.DataFrame({"anomaly_score": [0.0]*len(X_df), "anomaly_status": ["normal"]*len(X_df)}, index=X_df.index)

    # 6. GPR Forecaster
    gpr_forecaster = GPRTrajectoryForecaster(model_version="gpr_v1")
    gpr_compatible = compat.get("branches", {}).get("forecasting_branch", {}).get("status") == "COMPATIBLE"

    summary_decisions = {"PASS": 0, "RETEST": 0, "REVIEW": 0, "REJECT": 0}
    sample_components_results = []

    for cid in X_df.index:
        comp_traj = valid_df[valid_df["component_id"] == cid].sort_values("checkpoint")
        anom_score = float(score_df.loc[cid, "anomaly_score"])
        anom_status = str(score_df.loc[cid, "anomaly_status"])

        # Forecast
        forecast_info = None
        if gpr_compatible and len(comp_traj) >= 3:
            forecast_info = gpr_forecaster.fit_and_predict(
                trajectory=comp_traj,
                param_key=param_keys[0],
                future_checkpoints=[float(last_checkpoint) + 24.0]
            )

        # Evidence Pack
        top_feats = if_model.explain_component(X_df.loc[cid])
        evidence = evidence_generator.build_evidence_pack(
            component_id=cid,
            checkpoint=last_checkpoint,
            data_quality_info={"status": "VALID", "reasons": "Passed quality gate"},
            anomaly_info={
                "anomaly_score": anom_score,
                "anomaly_status": anom_status,
                "model_version": if_model.model_version,
                "top_features": top_feats
            },
            forecast_info=forecast_info,
            peer_evidence=meta_df.loc[cid].to_dict() if cid in meta_df.index else {}
        )

        # Conservative Decision & Narrative
        decision = rule_engine.evaluate(evidence)
        explanation_bundle = explanation_generator.generate_explanation(cid, evidence, decision)
        plain_reason = explanation_bundle.get("plain_english_reason", "")

        summary_decisions[decision.get("recommendation", "REVIEW")] = summary_decisions.get(decision.get("recommendation", "REVIEW"), 0) + 1

        # Audit logging
        audit_storage.record_anomaly_result(
            run_id=run_id,
            component_id=cid,
            checkpoint=last_checkpoint,
            score=anom_score,
            status=anom_status,
            model_version="if_v1",
            population_id=str(meta_df.loc[cid].get("population_id", "pop_default") if cid in meta_df.index else "pop_default")
        )

        if forecast_info and forecast_info.get("forecast_status") == "success":
            audit_storage.record_forecast_result(
                run_id=run_id,
                component_id=cid,
                param_key=param_keys[0],
                forecast_info=forecast_info
            )

        audit_storage.record_decision(
            run_id=run_id,
            component_id=cid,
            checkpoint=last_checkpoint,
            decision_result=decision,
            explanation=plain_reason,
            evidence_pack=evidence
        )

        # Update progressive state
        comp_measurements = comp_traj.iloc[-1][param_keys[:4]].to_dict() if not comp_traj.empty else {}
        state_manager.update_component_state(
            component_id=cid,
            checkpoint=last_checkpoint,
            measurements=comp_measurements,
            anomaly_result={"anomaly_score": anom_score, "anomaly_status": anom_status},
            forecast_result=forecast_info or {},
            decision=decision,
            explanation=plain_reason,
            evidence=evidence
        )

        if len(sample_components_results) < 15:
            sample_components_results.append({
                "component_id": cid,
                "checkpoint": last_checkpoint,
                "anomaly_score": round(anom_score, 4),
                "anomaly_status": anom_status,
                "recommendation": decision.get("recommendation"),
                "reason": plain_reason[:180] + "..." if len(plain_reason) > 180 else plain_reason
            })

    run_summary = {
        "dataset_id": req.dataset_id,
        "total_components_analyzed": len(X_df),
        "total_dataset_rows": len(valid_df),
        "decisions_breakdown": summary_decisions,
        "gpr_enabled": gpr_compatible,
        "quarantined_records": len(quarantined_df)
    }
    audit_storage.record_run_complete(run_id, run_summary, status="COMPLETED")

    return {
        "run_id": run_id,
        "status": "completed",
        "dataset_id": req.dataset_id,
        "compatibility": compat,
        "data_quality": dq_report,
        "decision_summary": summary_decisions,
        "sample_results": sample_components_results,
        "audit_reference": f"sqlite://data/audit_traceability.db#run_id={run_id}"
    }


@app.get("/components/{component_id}/state")
def get_component_state(component_id: str):
    """Retrieves current progressive state for a component (Section 36)."""
    state = state_manager.get_latest_state(component_id)
    if not state:
        audit_trail = audit_storage.query_audit_trail(component_id)
        if audit_trail.get("decisions"):
            return {"component_id": component_id, "latest_state": audit_trail}
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found.")
    return {"component_id": component_id, "latest_state": state}


@app.get("/components/{component_id}/history")
def get_component_history(component_id: str):
    """Retrieves chronological progression across all recorded checkpoints (Section 36)."""
    history = state_manager.get_history(component_id)
    audit_trail = audit_storage.query_audit_trail(component_id)
    return {
        "component_id": component_id,
        "checkpoints_history": history,
        "audit_trail": audit_trail
    }


@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    """Retrieves status and summary of a pipeline execution run."""
    with audit_storage.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id, dataset_id, started_at, completed_at, status, config_version, summary_json FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
        return {
            "run_id": row[0],
            "dataset_id": row[1],
            "started_at": row[2],
            "completed_at": row[3],
            "status": row[4],
            "config_version": row[5],
            "summary": json.loads(row[6]) if row[6] else {}
        }


@app.get("/audit/{run_id}")
def get_run_audit(run_id: str, limit: int = 50):
    """Returns audit trail records for a specific execution run (Section 38)."""
    with audit_storage.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT component_id, checkpoint, recommendation, triggered_rule, confidence, plain_english_reason, created_at
        FROM decisions WHERE run_id = ? LIMIT ?
        """, (run_id, limit))
        decisions = cursor.fetchall()
        
        cursor.execute("""
        SELECT component_id, checkpoint, anomaly_score, anomaly_status, model_version
        FROM anomaly_results WHERE run_id = ? LIMIT ?
        """, (run_id, limit))
        anomalies = cursor.fetchall()

        return {
            "run_id": run_id,
            "decisions_count": len(decisions),
            "decisions": decisions,
            "anomalies": anomalies
        }


@app.post("/audit/qa-action")
def record_qa_action(req: QAActionRequest):
    """
    Records human QA reviewer action and disposition, strictly separating
    AI recommendation from human authorization per Section 38.2.
    """
    audit_storage.record_human_qa_action(
        run_id=req.run_id,
        component_id=req.component_id,
        checkpoint=req.checkpoint,
        ai_recommendation=req.ai_recommendation,
        human_action=req.human_action,
        reviewer_id=req.reviewer_id,
        notes=req.notes or ""
    )
    return {
        "status": "recorded",
        "component_id": req.component_id,
        "human_action": req.human_action,
        "reviewer_id": req.reviewer_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
