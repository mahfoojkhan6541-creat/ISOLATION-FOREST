import os
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():

    print("\n" + "=" * 70)
    print("STEP 54 - FINAL MODEL METADATA VERIFICATION")
    print("=" * 70)

    config_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_config.pkl"
    )

    config = joblib.load(config_path)

    print("\nFrozen configuration loaded.")

    print("\nDataset:")
    print(config["dataset"])

    print("\nMethod:")
    print(config["method"])

    print("\nFeature representation:")
    print(config["feature_representation"])

    print("\nOriginal feature count:")
    print(config["original_feature_count"])

    print("\nRemoved feature family:")
    print(config["removed_feature_family"])

    print("\nRemoved feature count:")
    print(config["removed_feature_count"])

    print("\nModel feature count:")
    print(config["model_feature_count"])

    print("\nScore definition:")
    print(config["score_definition"])

    print("\nCandidate contamination:")
    print(config["candidate_contamination"])

    print("\nNumber of estimators:")
    print(config["n_estimators"])

    print("\nRandom state:")
    print(config["random_state"])

    print("\nNumber of jobs:")
    print(config["n_jobs"])

    print("\nDecision level:")
    print(config["decision_level"])

    print("\nScore aggregation:")
    print(config["score_aggregation"])

    print("\nValidation threshold:")
    print(config["validation_threshold"])

    print("\nThreshold selection split:")
    print(config["threshold_selection_split"])

    print("\nValidation Average Precision:")
    print(config["validation_average_precision"])

    print("\nValidation Precision:")
    print(config["validation_precision"])

    print("\nValidation Recall:")
    print(config["validation_recall"])

    print("\nValidation F1:")
    print(config["validation_f1"])

    print("\nValidation FPR:")
    print(config["validation_fpr"])

    print("\nValidation FNR:")
    print(config["validation_fnr"])

    print("\nFinal-test used for threshold selection:")
    print(config["final_test_used_for_selection"])

    print("\nThreshold frozen:")
    print(config["threshold_frozen"])

    print("\nModel retraining required:")
    print(config["model_retraining_required"])

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    assert config["dataset"] == "D2"
    assert config["method"] == "Isolation Forest"
    assert config["feature_representation"] == (
        "V2 enhanced without acceleration"
    )

    assert config["original_feature_count"] == 175
    assert config["removed_feature_family"] == "acceleration"
    assert config["removed_feature_count"] == 19
    assert config["model_feature_count"] == 156

    assert config["score_definition"] == (
        "-IsolationForest.score_samples(X)"
    )

    assert config["candidate_contamination"] == 0.01
    assert config["n_estimators"] == 200
    assert config["random_state"] == 42

    assert config["decision_level"] == "MaterialID"
    assert config["score_aggregation"] == "mean_score"

    assert config["validation_threshold"] == 0.393578
    assert config["threshold_selection_split"] == "validation"

    assert config["final_test_used_for_selection"] is False
    assert config["threshold_frozen"] is True

    print("\n" + "=" * 70)
    print("ALL METADATA CHECKS PASSED")
    print("=" * 70)

    print("\nThe frozen model configuration is verified.")
    print("No model or threshold was changed.")

    print("\n" + "=" * 70)
    print("STEP 54 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()