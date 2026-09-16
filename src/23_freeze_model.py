import os
import joblib
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    threshold_file = os.path.join(
        RESULTS_DIR,
        "D2_selected_threshold.csv"
    )

    if_file = os.path.join(
        MODELS_DIR,
        "D2_enhanced_if_candidates.pkl"
    )

    feature_file = os.path.join(
        MODELS_DIR,
        "D2_enhanced_feature_config.pkl"
    )

    preprocessing_file = os.path.join(
        MODELS_DIR,
        "D2_enhanced_preprocessing.pkl"
    )

    # ---------------------------------------------------------
    # Check required files
    # ---------------------------------------------------------

    required_files = [
        threshold_file,
        if_file,
        feature_file,
        preprocessing_file
    ]

    for file in required_files:

        if not os.path.exists(file):

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    # ---------------------------------------------------------
    # Read selected threshold
    # ---------------------------------------------------------

    threshold_df = pd.read_csv(
        threshold_file
    )

    threshold = float(
        threshold_df.loc[
            0,
            "selected_threshold"
        ]
    )

    # ---------------------------------------------------------
    # Load Isolation Forest candidates
    # ---------------------------------------------------------

    candidate_models = joblib.load(
        if_file
    )

    # We use the 0.01 model because the anomaly score
    # score_samples is independent of contamination.
    #
    # The final operating decision is controlled by the
    # separately frozen threshold.

    selected_model = candidate_models[0.01]

    # ---------------------------------------------------------
    # Create frozen configuration
    # ---------------------------------------------------------

    frozen_configuration = {

        "dataset": "D2",

        "method": "Isolation Forest",

        "feature_representation":
            "Enhanced",

        "feature_configuration_file":
            "models/D2_enhanced_feature_config.pkl",

        "preprocessing_file":
            "models/D2_enhanced_preprocessing.pkl",

        "model": selected_model,

        "score_definition":
            "anomaly_score = -IsolationForest.score_samples(X)",

        "threshold": threshold,

        "decision_rule":
            "score >= threshold -> ABNORMAL; "
            "score < threshold -> NORMAL",

        "threshold_selection_split":
            "D2 validation",

        "final_test_used_for_selection":
            False,

        "threshold_frozen":
            True
    }

    output_file = os.path.join(
        MODELS_DIR,
        "D2_final_frozen_model.pkl"
    )

    joblib.dump(
        frozen_configuration,
        output_file
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL CONFIGURATION FROZEN")
    print("=" * 70)

    print(
        "\nDataset:",
        frozen_configuration["dataset"]
    )

    print(
        "Method:",
        frozen_configuration["method"]
    )

    print(
        "Feature representation:",
        frozen_configuration[
            "feature_representation"
        ]
    )

    print(
        "Selected threshold:",
        threshold
    )

    print(
        "\nDecision rule:"
    )

    print(
        "Score >= {:.6f} -> ABNORMAL".format(
            threshold
        )
    )

    print(
        "Score < {:.6f} -> NORMAL".format(
            threshold
        )
    )

    print(
        "\nFinal test used for selection:",
        frozen_configuration[
            "final_test_used_for_selection"
        ]
    )

    print(
        "\nFrozen model saved:"
    )

    print(output_file)

    print("\n" + "=" * 70)
    print("STEP 23 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()