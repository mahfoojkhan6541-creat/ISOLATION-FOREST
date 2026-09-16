import os
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


def main():

    print("\n" + "=" * 70)
    print("STEP 48 - FREEZE V2 WITHOUT ACCELERATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load original V2 model
    # ---------------------------------------------------------

    candidate_file = os.path.join(
        MODELS_DIR,
        "D2_enhanced_if_candidates.pkl"
    )

    candidates = joblib.load(
        candidate_file
    )

    # Original V2 selected score uses contamination=0.01
    model = candidates[0.01]

    # ---------------------------------------------------------
    # Load exact V2 feature configuration
    # ---------------------------------------------------------

    feature_config = joblib.load(
        os.path.join(
            MODELS_DIR,
            "D2_enhanced_feature_config.pkl"
        )
    )

    original_features = feature_config[
        "model_features"
    ]

    # Remove acceleration features
    selected_features = [
        feature
        for feature in original_features
        if "_acceleration" not in feature
    ]

    acceleration_features = [
        feature
        for feature in original_features
        if "_acceleration" in feature
    ]

    print("\nOriginal features:", len(original_features))
    print(
        "Acceleration features removed:",
        len(acceleration_features)
    )
    print(
        "Final frozen features:",
        len(selected_features)
    )

    # ---------------------------------------------------------
    # IMPORTANT
    # ---------------------------------------------------------
    #
    # The existing trained model uses 175 features.
    # Therefore we cannot simply reuse that model after
    # removing 19 features.
    #
    # Step 48 stores the configuration only.
    # A new 156-feature model will be trained in the next
    # controlled step using TRAIN ONLY.
    #

    frozen_config = {
        "dataset": "D2",
        "method": "Isolation Forest",
        "feature_representation": "V2 enhanced without acceleration",
        "original_feature_count": len(original_features),
        "removed_feature_family": "acceleration",
        "removed_feature_count": len(acceleration_features),
        "model_feature_count": len(selected_features),
        "model_features": selected_features,
        "score_definition": "-IsolationForest.score_samples(X)",
        "candidate_contamination": 0.01,
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "decision_level": "MaterialID",
        "score_aggregation": "mean_score",
        "validation_threshold": 0.393578,
        "threshold_selection_split": "validation",
        "validation_average_precision": 0.834371,
        "validation_precision": 0.753846,
        "validation_recall": 0.924528,
        "validation_f1": 0.830508,
        "validation_fpr": 0.133333,
        "validation_fnr": 0.075472,
        "final_test_used_for_selection": False,
        "threshold_frozen": True,
        "model_retraining_required": True
    }

    output_file = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_config.pkl"
    )

    joblib.dump(
        frozen_config,
        output_file
    )

    print(
        "\nFrozen configuration saved:",
        output_file
    )

    print("\nThreshold:", 0.393578)
    print("Validation AP:", 0.834371)
    print("Final test used for selection: NO")

    print("\n" + "=" * 70)
    print("STEP 48 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()