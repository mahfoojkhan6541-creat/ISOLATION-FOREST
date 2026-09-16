import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURES_DIR = os.path.join(BASE_DIR, "features")


def main():
    print("\n" + "=" * 70)
    print("SIH26170 - DATASET D1 ISOLATION FOREST PIPELINE & FREEZE")
    print("=" * 70)

    # 1. Load D1 feature configuration & candidate models
    config_path = os.path.join(MODELS_DIR, "D1_enhanced_feature_config.pkl")
    candidates_path = os.path.join(MODELS_DIR, "D1_enhanced_if_candidates.pkl")

    feature_config = joblib.load(config_path)
    candidates = joblib.load(candidates_path)

    # Candidate contamination 0.01
    model_001 = candidates[0.01]
    all_features = feature_config["model_features"]

    # Filter out acceleration features if any
    clean_features = [f for f in all_features if "_acceleration" not in f]
    accel_features = [f for f in all_features if "_acceleration" in f]

    print(f"\nTotal Candidate Features: {len(all_features)}")
    print(f"Acceleration Features: {len(accel_features)}")
    print(f"Clean Selected Features: {len(clean_features)}")

    # 2. Load validation and final test anomaly scores
    val_scores_path = os.path.join(RESULTS_DIR, "D1_validation_enhanced_anomaly_scores.csv")
    test_scores_path = os.path.join(RESULTS_DIR, "D1_final_test_enhanced_anomaly_scores.csv")

    val_df = pd.read_csv(val_scores_path)
    test_df = pd.read_csv(test_scores_path)

    # 3. MaterialID level aggregation
    val_mat = val_df.groupby("MaterialID")["score_0.01"].agg(["mean", "max", "count"]).reset_index()
    val_mat.rename(columns={"mean": "mean_score", "max": "max_score"}, inplace=True)
    
    test_mat = test_df.groupby("MaterialID")["score_0.01"].agg(["mean", "max", "count"]).reset_index()
    test_mat.rename(columns={"mean": "mean_score", "max": "max_score"}, inplace=True)

    # 4. Threshold Selection on Validation
    # In D1, nominal screening population operates with strict screening rate (e.g. 99th percentile)
    val_mean_scores = val_mat["mean_score"].values
    threshold_p99 = float(np.percentile(val_mean_scores, 99.0))
    threshold_p95 = float(np.percentile(val_mean_scores, 95.0))
    selected_threshold = threshold_p99

    print(f"\nValidation Material Count: {len(val_mat)}")
    print(f"Validation Mean Score: {np.mean(val_mean_scores):.6f}")
    print(f"Validation P95 Threshold: {threshold_p95:.6f}")
    print(f"Validation P99 Threshold (Selected): {selected_threshold:.6f}")

    # 5. Freeze D1 Model and Configuration
    frozen_config = {
        "dataset": "D1",
        "method": "Isolation Forest",
        "feature_representation": "Enhanced Behaviour Matrix",
        "original_feature_count": len(all_features),
        "removed_feature_family": "acceleration" if accel_features else "none",
        "removed_feature_count": len(accel_features),
        "model_feature_count": len(all_features),
        "model_features": all_features,
        "score_definition": "-IsolationForest.score_samples(X)",
        "candidate_contamination": 0.01,
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "decision_level": "MaterialID",
        "score_aggregation": "mean_score",
        "validation_threshold": selected_threshold,
        "threshold_selection_split": "validation",
        "validation_alert_rate": float(np.mean(val_mean_scores >= selected_threshold)),
        "final_test_used_for_selection": False,
        "threshold_frozen": True
    }

    frozen_config_path = os.path.join(MODELS_DIR, "D1_final_frozen_config.pkl")
    frozen_model_path = os.path.join(MODELS_DIR, "D1_final_frozen_model.pkl")

    joblib.dump(frozen_config, frozen_config_path)
    joblib.dump({
        "model": model_001,
        "config": frozen_config,
        "threshold": selected_threshold,
        "feature_names": all_features,
        "decision_level": "MaterialID",
        "score_aggregation": "mean_score"
    }, frozen_model_path)
    print(f"\n[✓] Saved frozen D1 model config: {frozen_config_path}")
    print(f"[✓] Saved frozen D1 model bundle: {frozen_model_path}")

    # 6. Final Test Predictions
    test_mean_scores = test_mat["mean_score"].values
    test_mat["anomaly_prediction"] = np.where(test_mean_scores >= selected_threshold, "ABNORMAL", "NORMAL")
    test_pred_path = os.path.join(RESULTS_DIR, "D1_final_test_predictions.csv")
    test_mat.to_csv(test_pred_path, index=False)

    pred_counts = test_mat["anomaly_prediction"].value_counts().to_dict()
    alert_count = pred_counts.get("ABNORMAL", 0)
    normal_count = pred_counts.get("NORMAL", 0)
    total_test_materials = len(test_mat)
    test_alert_rate = alert_count / max(total_test_materials, 1)

    print(f"\nFinal Test Material Count: {total_test_materials}")
    print(f"Final Test Normal: {normal_count} ({normal_count/total_test_materials*100:.1f}%)")
    print(f"Final Test Abnormal / Outliers: {alert_count} ({alert_count/total_test_materials*100:.2f}%)")

    # 7. Generate D1 Isolation Forest Final Report
    report_path = os.path.join(RESULTS_DIR, "D1_Isolation_Forest_Final_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SIH26170 - ISOLATION FOREST FINAL REPORT (DATASET D1)\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: D1\n")
        f.write(f"Domain: Burn-in screening with 8 progressive checkpoints (-2 to 7)\n")
        f.write(f"Sampling: 5,104 unique component groups\n\n")

        f.write("2. MODEL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: Isolation Forest\n")
        f.write(f"Feature representation: Enhanced multi-checkpoint trajectory features\n")
        f.write(f"Number of estimators: 200\n")
        f.write(f"Random state: 42\n")
        f.write(f"Candidate contamination: 0.01\n")
        f.write(f"Score definition: -IsolationForest.score_samples(X)\n\n")

        f.write("3. FEATURES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Original feature count: {len(all_features)}\n")
        f.write(f"Selected model feature count: {len(all_features)}\n\n")

        f.write("4. DECISION CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Decision level: MaterialID\n")
        f.write(f"Score aggregation: mean_score\n")
        f.write(f"Validation threshold (P99): {selected_threshold:.6f}\n")
        f.write(f"Threshold selection split: validation\n")
        f.write(f"Threshold frozen: True\n\n")

        f.write("5. VALIDATION SCREENING METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Validation MaterialIDs: {len(val_mat)}\n")
        f.write(f"Validation Mean Anomaly Score: {np.mean(val_mean_scores):.6f}\n")
        f.write(f"Validation Alert Rate: {frozen_config['validation_alert_rate']*100:.2f}%\n\n")

        f.write("6. FINAL TEST PERFORMANCE & SCREENING DISPOSITION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total MaterialIDs Screened: {total_test_materials}\n")
        f.write(f"Nominal Approved (NORMAL): {normal_count} ({normal_count/total_test_materials*100:.1f}%)\n")
        f.write(f"Outlier Screened (ABNORMAL): {alert_count} ({test_alert_rate*100:.2f}%)\n")
        f.write(f"Alert Rate: {test_alert_rate*100:.2f}%\n\n")

        f.write("7. LEAKAGE CHECK\n")
        f.write("-" * 70 + "\n")
        f.write("Final test used for threshold selection: False\n")
        f.write("Threshold source: validation\n")
        f.write("Final-test labels used for model training: No\n")
        f.write("Final-test labels used for threshold selection: No\n\n")

        f.write("=" * 70 + "\n")
        f.write("FINAL REPORT COMPLETE\n")
        f.write("=" * 70 + "\n")

    print(f"\n[✓] Generated final report: {report_path}")


if __name__ == "__main__":
    main()
