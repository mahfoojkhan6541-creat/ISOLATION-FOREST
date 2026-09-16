import os
import pytest
import numpy as np
import pandas as pd

from src.ingestion.adapters import get_adapter, CSVSourceAdapter, compute_file_sha256
from src.ingestion.registry import DatasetRegistry, DatasetMetadata
from src.profiling.profiler import DatasetProfiler
from src.compatibility.checker import CompatibilityChecker
from src.mapping.loader import MappingLoader
from src.mapping.units import UnitConverter
from src.mapping.canonical import CanonicalTransformer
from src.validation.validator import DataQualityValidator
from src.validation.quarantine import QuarantineManager
from src.grouping.population_policy import PopulationPolicy
from src.grouping.population_validator import PopulationValidator
from src.grouping.population_selector import PopulationSelector
from src.features.baseline import BaselineFeatureExtractor
from src.features.drift import DriftFeatureExtractor
from src.features.peer_relative import PeerRelativeFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.forecasting.gpr_forecaster import GPRTrajectoryForecaster
from src.models.evaluation import ModelEvaluator
from src.evidence.generator import EvidenceGenerator
from src.decision.rules import DecisionRuleEngine
from src.explanation.generator import ExplanationGenerator
from src.state.progressive import ProgressiveStateManager
from src.audit.storage import AuditStorage


# -------------------------------------------------------------
# 1. Ingestion & Profiling Tests
# -------------------------------------------------------------
def test_intake_adapter_and_checksum():
    adapter = get_adapter("csv")
    assert isinstance(adapter, CSVSourceAdapter)
    df = adapter.read("data/D1.csv", nrows=50)
    assert len(df) == 50
    checksum = compute_file_sha256("data/D1.csv")
    assert len(checksum) == 64


def test_profiler_and_compatibility():
    df = pd.read_csv("data/D1.csv", nrows=200)
    profiler = DatasetProfiler()
    profile = profiler.profile(df, dataset_name="D1_sample")

    assert profile["dimensions"]["records"] == 200
    assert "MaterialID" in profile["candidates"]["candidate_id_fields"]
    assert profile["temporal"]["checkpoint_count"] > 0

    checker = CompatibilityChecker()
    compat = checker.assess_compatibility(profile)
    assert compat["status"] in ["COMPATIBLE", "PARTIALLY_COMPATIBLE"]


# -------------------------------------------------------------
# 2. Canonical Mapping & Units
# -------------------------------------------------------------
def test_canonical_mapping():
    loader = MappingLoader()
    cfg = loader.load_mapping("d1_mapping")
    assert cfg["mapping_version"] == "d1_v1"

    raw_df = pd.read_csv("data/D1.csv", nrows=100)
    transformer = CanonicalTransformer(mapping_loader=loader)
    canonical_df, meta = transformer.transform(raw_df, cfg, dataset_id="D1")

    assert "component_id" in canonical_df.columns
    assert "checkpoint" in canonical_df.columns
    assert "param_01" in canonical_df.columns
    assert meta["unique_components"] > 0


def test_unit_conversion():
    uc = UnitConverter()
    s = pd.Series([1.0, 2.5, 3.8])
    is_plausible, err = uc.check_plausible_bounds(s, "arb_norm")
    assert is_plausible is True
    assert err is None


# -------------------------------------------------------------
# 3. Data Quality Gate & Quarantine
# -------------------------------------------------------------
def test_data_quality_gate():
    # Construct synthetic test dataframe with intentional DQ issues
    test_df = pd.DataFrame({
        "component_id": ["comp_1", "", "comp_2", "comp_3", "comp_4"],
        "checkpoint": [0, 24, np.nan, 72, 96],
        "elapsed_time": [0.0, 24.0, 48.0, 72.0, 96.0],
        "param_01": [1.0, 1.2, 1.4, 999999.0, 1.5]
    })
    validator = DataQualityValidator()
    valid_df, quarantined_df, report = validator.validate(test_df, param_keys=["param_01"])

    # Blank component_id and NaN checkpoint should be quarantined
    assert len(quarantined_df) >= 2
    assert report["total_evaluated"] == 5


# -------------------------------------------------------------
# 4. Grouping & Population Target Exclusion
# -------------------------------------------------------------
def test_population_target_exclusion():
    canonical_df = pd.DataFrame({
        "component_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "checkpoint": [24, 24, 24, 24, 24, 24],
        "param_01": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]
    })
    selector = PopulationSelector()
    res = selector.select_reference_population(
        canonical_df=canonical_df,
        target_component_id="C1",
        checkpoint=24
    )
    # Target C1 must NOT be inside the peer population
    assert res["status"] == "VALID"
    members = res["members_df"]
    assert "C1" not in members["component_id"].values
    assert len(members) == 5


# -------------------------------------------------------------
# 5. Feature Engineering
# -------------------------------------------------------------
def test_feature_engineering():
    traj = pd.DataFrame({
        "checkpoint": [0, 24, 48, 72],
        "param_01": [1.0, 1.2, 1.4, 1.6]
    })
    base_feat = BaselineFeatureExtractor.extract(traj, "param_01")
    assert base_feat["param_01_baseline"] == 1.0
    assert base_feat["param_01_current"] == 1.6
    assert abs(base_feat["param_01_delta_from_baseline"] - 0.6) < 1e-6

    drift_feat = DriftFeatureExtractor.extract(traj, "param_01")
    assert drift_feat["param_01_drift_slope"] > 0
    assert drift_feat["drift_status"] == "computed"


# -------------------------------------------------------------
# 6. Model Branches (Isolation Forest & GPR)
# -------------------------------------------------------------
def test_isolation_forest_model():
    np.random.seed(42)
    # 50 nominal points + 2 gross outliers
    X_nominal = np.random.normal(loc=0.0, scale=0.1, size=(50, 4))
    X_outlier = np.array([[5.0, 5.0, 5.0, 5.0], [-6.0, -6.0, -6.0, -6.0]])
    X = np.vstack([X_nominal, X_outlier])
    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3", "feat_4"])

    detector = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)

    assert len(scores_df) == 52
    assert "anomaly_score" in scores_df.columns
    # Outliers should have higher scores than nominal median
    outlier_scores = scores_df.iloc[-2:]["anomaly_score"].values
    nominal_scores = scores_df.iloc[:50]["anomaly_score"].values
    assert np.mean(outlier_scores) > np.median(nominal_scores)


def test_gpr_trajectory_forecaster():
    comp_traj = pd.DataFrame({
        "checkpoint": [0.0, 24.0, 48.0, 72.0, 96.0],
        "leakage": [0.10, 0.12, 0.15, 0.18, 0.22]
    })
    gpr = GPRTrajectoryForecaster()
    res = gpr.fit_and_predict(comp_traj, "leakage", future_checkpoints=[120.0], min_history_points=3)

    assert res["forecast_status"] == "available"
    assert res["forecast_mean"] is not None
    assert res["interval"]["lower"] < res["forecast_mean"] < res["interval"]["upper"]


# -------------------------------------------------------------
# 7. Decision Rules & Plain-English Explanations
# -------------------------------------------------------------
def test_decision_and_explanation_pass():
    evidence_gen = EvidenceGenerator()
    evidence = evidence_gen.build_evidence_pack(
        component_id="COMP_TEST_01",
        checkpoint=168.0,
        data_quality_info={"status": "VALID", "reasons": "Nominal"},
        anomaly_info={"anomaly_score": 0.25, "anomaly_status": "normal", "model_version": "if_v1"},
        forecast_info=None,
        peer_evidence={"param_peer_zscore": 0.1}
    )

    decision_engine = DecisionRuleEngine()
    decision = decision_engine.evaluate(evidence)
    assert decision["recommendation"] == "PASS"

    exp_gen = ExplanationGenerator()
    narrative = exp_gen.generate_explanation("COMP_TEST_01", evidence, decision)
    assert "COMP_TEST_01" in narrative["plain_english_reason"]
    assert "PASS" in narrative["plain_english_reason"] or "nominally" in narrative["plain_english_reason"]


def test_decision_retest_on_data_quality_fail():
    evidence_gen = EvidenceGenerator()
    evidence = evidence_gen.build_evidence_pack(
        component_id="COMP_TEST_02",
        checkpoint=168.0,
        data_quality_info={"status": "FAILED", "reasons": "Sensor contact intermittent"},
        anomaly_info={"anomaly_score": 0.95, "anomaly_status": "high", "model_version": "if_v1"},
        forecast_info=None,
        peer_evidence={}
    )
    decision_engine = DecisionRuleEngine()
    decision = decision_engine.evaluate(evidence)
    # Rule priority: Data quality failure must trigger RETEST rather than false hardware condemnation
    assert decision["recommendation"] == "RETEST"


# -------------------------------------------------------------
# 8. Progressive State & Audit Storage Traceability
# -------------------------------------------------------------
def test_audit_storage_and_state(tmp_path):
    db_path = str(tmp_path / "test_audit.db")
    storage = AuditStorage(db_path=db_path)
    run_id = "test_run_101"

    storage.record_run_start(run_id, "D1", "v1.0")
    storage.record_anomaly_result(run_id, "C10", 168.0, 0.42, "normal", "if_v1", "pop_01")
    storage.record_decision(
        run_id, "C10", 168.0,
        {"recommendation": "PASS", "triggered_rule": "RULE_STABLE"},
        "Nominal component",
        {"anomaly_score": 0.42}
    )
    storage.record_human_qa_action(run_id, "C10", 168.0, "PASS", "CONFIRM_PASS", "QA_ENG_1", "Approved")
    storage.record_run_complete(run_id, {"components": 1}, "COMPLETED")

    trail = storage.query_audit_trail("C10")
    assert trail["component_id"] == "C10"
    assert len(trail["anomalies"]) == 1
    assert len(trail["decisions"]) == 1
    assert len(trail["human_qa_actions"]) == 1
