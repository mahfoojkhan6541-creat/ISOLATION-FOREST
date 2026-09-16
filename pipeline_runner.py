"""
SIH26170 Generalized Data Pipeline — Master Orchestration Runner
================================================================
Executes the end-to-end generalized burn-in screening pipeline per the
Implementation Guide:
  Phase 1: Dataset Intake, Auto-Profiling, Compatibility Gating,
           Canonical Mapping & 12-Case Data Quality Validation
  Phase 2: Temporal Normalization & Reference Population Grouping
  Phase 3: Behaviour Feature Representation (Baseline, Drift, Curvature, Peer-Z)
  Phase 4: Isolation Forest Anomaly Detection & GPR Trajectory Forecasting
  Phase 5: Conservative Risk Fusion, Decisions & Plain-English Explanations
  Phase 6: Progressive State Update & SQLite Traceability / Audit Logging
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import uuid
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.ingestion.adapters import IntakeAdapterFactory
from src.ingestion.registry import DatasetRegistry, DatasetMetadata
from src.profiling.profiler import DatasetProfiler
from src.compatibility.checker import CompatibilityChecker
from src.mapping.loader import MappingLoader
from src.mapping.canonical import CanonicalTransformer
from src.validation.validator import DataQualityValidator
from src.validation.quarantine import QuarantineManager
from src.grouping.population_selector import PopulationSelector
from src.features.engineering import FeatureEngineeringEngine
from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.forecasting.gpr_forecaster import GPRTrajectoryForecaster
from src.models.evaluation import ModelEvaluator
from src.evidence.generator import EvidenceGenerator
from src.decision.rules import DecisionRuleEngine
from src.explanation.generator import ExplanationGenerator
from src.state.progressive import ProgressiveStateManager
from src.audit.storage import AuditStorage


def run_pipeline(
    dataset_id: str,
    raw_file_path: str,
    mapping_version: str,
    max_components_detailed: int = 250
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"[*] RUNNING GENERALIZED DATA PIPELINE ON: {dataset_id}")
    print(f"    Source File: {raw_file_path}")
    print(f"    Mapping: {mapping_version}")
    print(f"{'='*75}")

    run_id = f"run_{dataset_id.lower()}_{uuid.uuid4().hex[:8]}"
    audit = AuditStorage()
    audit.record_run_start(run_id, dataset_id, config_version=mapping_version)

    # -------------------------------------------------------------
    # Step 1: Intake & Registration (Phase 1)
    # -------------------------------------------------------------
    print("\n[Step 1] Dataset Intake & Checksum Registration...")
    registry = DatasetRegistry()
    reg_meta = DatasetMetadata(
        dataset_id=dataset_id,
        file_path=raw_file_path,
        domain="burn_in",
        source_format="csv",
        sampling_type="checkpoints",
        intended_task="burn_in_anomaly_screening"
    )
    reg_info = registry.register(reg_meta)
    print(f"  ✓ Checksum SHA-256: {reg_info.get('sha256')[:16]}... (Recorded in configs/datasets/)")

    adapter = IntakeAdapterFactory.get_adapter(raw_file_path)
    raw_df = adapter.read(raw_file_path)
    print(f"  ✓ Ingested raw table: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

    # -------------------------------------------------------------
    # Step 2: Automated Profiling (Phase 1)
    # -------------------------------------------------------------
    print("\n[Step 2] Automated Dataset Profiling...")
    profiler = DatasetProfiler()
    profile = profiler.profile(raw_df, dataset_name=dataset_id)
    print(f"  ✓ Unique candidate IDs: {profile['entities']['unique_components']} components")
    print(f"  ✓ Candidate checkpoints: {profile['temporal']['checkpoint_count']} checkpoints ({profile['temporal']['unique_checkpoints'][:8]})")
    print(f"  ✓ Measurable parameters: {len(profile['candidates']['candidate_numeric_fields'])} fields")
    print(f"  ✓ Missing values: {profile['missingness'].get('total_missing_values', 0)} cells")

    # -------------------------------------------------------------
    # Step 3: Branch Compatibility Gating (Phase 1)
    # -------------------------------------------------------------
    print("\n[Step 3] Branch Compatibility Evaluation...")
    checker = CompatibilityChecker()
    compat = checker.assess_compatibility(profile)
    print(f"  ✓ Overall Status: {compat['status']}")
    print(f"  ✓ Anomaly Branch (Isolation Forest): {compat['branches']['anomaly_branch'].upper()}")
    print(f"  ✓ Forecast Branch (GPR): {compat['branches']['forecast_branch'].upper()}")
    for r in compat["reasons"]:
        print(f"    - {r}")

    # -------------------------------------------------------------
    # Step 4: Schema Mapping & Canonical Transformation (Phase 1)
    # -------------------------------------------------------------
    print("\n[Step 4] Canonical Schema & Unit Transformation...")
    loader = MappingLoader()
    mapping_cfg = loader.load_mapping(mapping_version)
    transformer = CanonicalTransformer(mapping_loader=loader)
    canonical_df, cat_meta = transformer.transform(raw_df, mapping_cfg, dataset_id=dataset_id)
    param_keys = [p["parameter_key"] for p in mapping_cfg["parameter_fields"]]
    print(f"  ✓ Transformed to Canonical Internal Representation: {canonical_df.shape[0]:,} records")
    print(f"  ✓ Standardized parameter keys: {param_keys}")

    # -------------------------------------------------------------
    # Step 5: 12-Case Data Quality Validation & Quarantine (Phase 1)
    # -------------------------------------------------------------
    print("\n[Step 5] 12-Case Data Quality Gate & Quarantine...")
    validator = DataQualityValidator()
    valid_df, quarantined_df, dq_summary = validator.validate(canonical_df, param_keys)
    quarantine_mgr = QuarantineManager()
    if not quarantined_df.empty:
        quarantine_mgr.quarantine_records(quarantined_df, run_id=run_id, reason="DQ Gate Failure")
        print(f"  ⚠ Quarantined {len(quarantined_df):,} records with provenance trail")
    else:
        print(f"  ✓ 100% records passed Data Quality Gate (0 quarantined)")
    print(f"  ✓ Trusted canonical records for screening: {len(valid_df):,}")

    # -------------------------------------------------------------
    # Step 6: Progressive Feature Representation (Phases 2 & 3)
    # -------------------------------------------------------------
    print("\n[Step 6] Progressive Behaviour Feature Engineering (No Leakage)...")
    checkpoints = sorted(valid_df["checkpoint"].dropna().unique())
    last_cp = checkpoints[-1] if len(checkpoints) > 0 else 0
    print(f"  ✓ Total checkpoints available: {len(checkpoints)} -> Final screening checkpoint: {last_cp}")

    unique_components = valid_df["component_id"].unique()
    sample_components = unique_components[:max_components_detailed]
    sub_canonical = valid_df[valid_df["component_id"].isin(sample_components)]

    feat_engine = FeatureEngineeringEngine(feature_version="feat_v1")
    X_df, meta_df, eval_df = feat_engine.extract_feature_matrix_batch(sub_canonical, param_keys, last_cp)
    print(f"  ✓ Computed behavior feature matrix: {X_df.shape[0]} components x {X_df.shape[1]} features")
    print(f"    (Includes: level, drift rate, trajectory curvature, and peer-relative z-scores)")

    # -------------------------------------------------------------
    # Step 7: Model Branch 1 — Isolation Forest (Phase 4)
    # -------------------------------------------------------------
    print("\n[Step 7] Model Branch 1: Isolation Forest Anomaly Scoring...")
    frozen_d2_path = "models/D2_v2_no_acceleration_frozen_model.pkl"
    d2_cfg_path = "models/D2_v2_no_acceleration_config.pkl"

    if dataset_id == "D2" and os.path.exists(frozen_d2_path) and os.path.exists(d2_cfg_path):
        if_detector = IsolationForestAnomalyDetector.load_frozen(
            config_path=d2_cfg_path,
            model_path=frozen_d2_path
        )
        print(f"  ✓ Loaded frozen validated Isolation Forest ({if_detector.model_version}, {len(if_detector.feature_names)} features, threshold={if_detector.frozen_threshold:.4f})")
        scores_df = if_detector.score(X_df)
    else:
        if_detector = IsolationForestAnomalyDetector(model_version=f"{dataset_id.lower()}_if_v1", n_estimators=200, contamination=0.01)
        if_detector.fit(X_df)
        scores_df = if_detector.score(X_df, watch_threshold=0.55, review_threshold=0.65)
    
    anom_counts = scores_df["anomaly_status"].value_counts().to_dict()
    print(f"  ✓ Calibrated Anomaly Scores Generated (0.0 to 1.0):")
    for status, cnt in anom_counts.items():
        print(f"    - {status.upper()}: {cnt} components ({cnt/len(scores_df)*100:.1f}%)")

    # Baseline Centroid Comparison (Section 24 & Section 27)
    lbl_series = eval_df["evaluation_label"] if ("evaluation_label" in eval_df.columns and eval_df["evaluation_label"].notnull().sum() > 0) else None
    base_comp = if_detector.compare_against_baseline(X_df, lbl_series)
    print(f"  ✓ Evaluated against Reference Centroid Baseline:")
    print(f"    - IF Score: {base_comp['if_score_mean']:.4f} ± {base_comp['if_score_std']:.4f}")
    print(f"    - Baseline Distance: {base_comp['baseline_distance_mean']:.4f} ± {base_comp['baseline_distance_std']:.4f}")
    if "if_roc_auc" in base_comp:
        print(f"    - IF ROC-AUC: {base_comp['if_roc_auc']:.4f} vs Baseline: {base_comp['baseline_roc_auc']:.4f}")

    # Evaluation against ground truth if labels exist
    if "evaluation_label" in eval_df.columns and eval_df["evaluation_label"].notnull().sum() > 0:
        eval_metrics = ModelEvaluator.evaluate_anomaly_branch(
            eval_df["evaluation_label"],
            scores_df["anomaly_score"],
            threshold=0.65
        )
        if eval_metrics.get("status") == "insufficient_classes":
            print(f"  ✓ Evaluation Labels: All {eval_metrics.get('total_evaluated')} sampled units are nominal screening population.")
        else:
            print(f"  ✓ Anomaly Branch Evaluation Metrics:")
            print(f"    - ROC-AUC: {eval_metrics.get('roc_auc', 0.0):.4f}")
            print(f"    - Average Precision: {eval_metrics.get('average_precision', 0.0):.4f}")
            print(f"    - Alert Rate: {eval_metrics.get('alert_rate', 0.0)*100:.2f}%")

    # -------------------------------------------------------------
    # Step 8: Model Branch 2 — GPR Trajectory Forecaster (Phase 4)
    # -------------------------------------------------------------
    is_gpr_compatible = compat["branches"]["forecast_branch"] == "compatible"
    forecast_results_map = {}
    gpr_forecaster = GPRTrajectoryForecaster(model_version="gpr_v1")

    print("\n[Step 8] Model Branch 2: GPR Trajectory Forecasting & Uncertainty...")
    if is_gpr_compatible:
        gpr_count = 0
        mae_list = []
        for cid in sample_components:
            comp_traj = valid_df[valid_df["component_id"] == cid].sort_values("checkpoint")
            if len(comp_traj) >= 3:
                # Horizon at last checkpoint + 24
                f_res = gpr_forecaster.fit_and_predict(
                    trajectory=comp_traj,
                    param_key=param_keys[0],
                    future_checkpoints=[float(last_cp) + 24.0],
                    min_history_points=3
                )
                if f_res.get("forecast_status") in ["success", "available"]:
                    forecast_results_map[cid] = f_res
                    gpr_count += 1
        print(f"  ✓ Fit GPR degradation trajectories for {gpr_count} components")
        print(f"  ✓ Evaluated Gaussian Process predictive mean and ±2σ uncertainty bounds")
    else:
        print("  ⊘ GPR Forecasting Branch bypassed: Dataset has insufficient checkpoints (< 4).")
        print("    (Adhering to Section 25.4 rule: no hallucinated trajectory forecasting)")

    # -------------------------------------------------------------
    # Step 9: Decision Layer, Risk Fusion & Explanations (Phase 5)
    # -------------------------------------------------------------
    print("\n[Step 9] Evidence Generation, Risk Fusion, Decisions & Explanations...")
    evidence_gen = EvidenceGenerator()
    decision_engine = DecisionRuleEngine(review_threshold=0.65, reject_threshold=0.85)
    explanation_gen = ExplanationGenerator()
    state_mgr = ProgressiveStateManager()

    decisions_breakdown = {"PASS": 0, "RETEST": 0, "REVIEW": 0, "REJECT": 0}
    sample_explanations = []

    for cid in X_df.index:
        comp_traj = valid_df[valid_df["component_id"] == cid].sort_values("checkpoint")
        anom_score = float(scores_df.loc[cid, "anomaly_score"])
        anom_status = str(scores_df.loc[cid, "anomaly_status"])
        f_info = forecast_results_map.get(cid)

        # 1. Evidence Pack
        top_anom_feats = if_detector.explain_component(X_df.loc[cid])
        evidence = evidence_gen.build_evidence_pack(
            component_id=cid,
            checkpoint=last_cp,
            data_quality_info={"status": "VALID", "reasons": "Passed quality gate"},
            anomaly_info={
                "anomaly_score": anom_score,
                "anomaly_status": anom_status,
                "model_version": if_detector.model_version,
                "top_features": top_anom_feats
            },
            forecast_info=f_info,
            peer_evidence=meta_df.loc[cid].to_dict() if cid in meta_df.index else {}
        )

        # 2. Operational Decision
        decision = decision_engine.evaluate(evidence)
        rec = decision.get("recommendation", "REVIEW")
        decisions_breakdown[rec] = decisions_breakdown.get(rec, 0) + 1

        # 3. Plain-English Narrative
        explanation = explanation_gen.generate_explanation(cid, evidence, decision)
        plain_reason = explanation.get("plain_english_reason", "")

        # 4. Audit Trail Persistence (Section 38 & 39)
        audit.record_anomaly_result(
            run_id=run_id,
            component_id=cid,
            checkpoint=last_cp,
            score=anom_score,
            status=anom_status,
            model_version="if_v1",
            population_id=str(meta_df.loc[cid].get("population_id", "peer_default") if cid in meta_df.index else "peer_default")
        )

        if f_info and f_info.get("forecast_status") in ["success", "available"]:
            audit.record_forecast_result(
                run_id=run_id,
                component_id=cid,
                param_key=param_keys[0],
                forecast_info=f_info
            )

        audit.record_decision(
            run_id=run_id,
            component_id=cid,
            checkpoint=last_cp,
            decision_result=decision,
            explanation=plain_reason,
            evidence_pack=evidence
        )

        # 5. Progressive State Update
        comp_measurements = comp_traj.iloc[-1][param_keys[:4]].to_dict() if not comp_traj.empty else {}
        state_mgr.update_component_state(
            component_id=cid,
            checkpoint=last_cp,
            measurements=comp_measurements,
            anomaly_result={"anomaly_score": anom_score, "anomaly_status": anom_status},
            forecast_result=f_info or {},
            decision=decision,
            explanation=plain_reason,
            evidence=evidence
        )

        if len(sample_explanations) < 3:
            sample_explanations.append({
                "component_id": cid,
                "anomaly_score": round(anom_score, 4),
                "recommendation": rec,
                "reason": plain_reason
            })

    print(f"  ✓ Operational Dispositions Breakdown:")
    for rec, cnt in sorted(decisions_breakdown.items()):
        print(f"    - {rec}: {cnt} components ({cnt/len(X_df)*100:.1f}%)")

    print("\n  Sample Plain-English Explanations:")
    for item in sample_explanations:
        print(f"    [Component {item['component_id']} - {item['recommendation']} (Score: {item['anomaly_score']})]:")
        print(f"      \"{item['reason']}\"")

    # -------------------------------------------------------------
    # Step 10: Run Summary & Audit Completion (Phase 6)
    # -------------------------------------------------------------
    run_summary = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "mapping_version": mapping_version,
        "total_records": len(valid_df),
        "total_components_screened": len(X_df),
        "quarantined_records": len(quarantined_df),
        "decisions": decisions_breakdown,
        "gpr_enabled": is_gpr_compatible
    }
    audit.record_run_complete(run_id, run_summary, status="COMPLETED")
    print(f"\n[✓] Pipeline execution finished successfully. Run ID: {run_id}")
    print(f"    Traceability stored in SQLite database: data/audit_traceability.db\n")

    return run_summary


def main():
    print("=" * 80)
    print("   SIH26170 GENERALIZED DATA PIPELINE — COMPLETE VERIFICATION RUNNER")
    print("=" * 80)

    # 1. Run Pipeline on Authentic Dataset D1
    d1_summary = run_pipeline(
        dataset_id="D1",
        raw_file_path="data/D1.csv",
        mapping_version="d1_mapping",
        max_components_detailed=300
    )

    # 2. Run Pipeline on Authentic Dataset D2
    d2_summary = run_pipeline(
        dataset_id="D2",
        raw_file_path="data/D2.csv",
        mapping_version="d2_mapping",
        max_components_detailed=200
    )

    print("\n" + "=" * 80)
    print("   FINAL PIPELINE SUMMARY ACROSS BOTH AUTHENTIC DATASETS")
    print("=" * 80)
    print(f"  Dataset D1: {d1_summary['total_components_screened']} components screened | Dispositions: {d1_summary['decisions']} | GPR: {d1_summary['gpr_enabled']}")
    print(f"  Dataset D2: {d2_summary['total_components_screened']} components screened | Dispositions: {d2_summary['decisions']} | GPR: {d2_summary['gpr_enabled']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
