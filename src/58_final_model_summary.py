import os
import joblib
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    print("\n" + "=" * 70)
    print("STEP 58 - FINAL ISOLATION FOREST MODEL SUMMARY")
    print("=" * 70)

    # --------------------------------------------------
    # Load frozen configuration
    # --------------------------------------------------

    config_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_config.pkl"
    )

    config = joblib.load(config_path)

    # --------------------------------------------------
    # Load final evaluation
    # --------------------------------------------------

    evaluation_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_evaluation.csv"
    )

    evaluation = pd.read_csv(evaluation_path).iloc[0]

    # --------------------------------------------------
    # Model summary
    # --------------------------------------------------

    print("\nDATASET")
    print("Dataset:", config["dataset"])

    print("\nMODEL")
    print("Method:", config["method"])
    print("Number of estimators:", config["n_estimators"])
    print("Random state:", config["random_state"])
    print("Candidate contamination:", config["candidate_contamination"])

    print("\nFEATURE REPRESENTATION")
    print("Representation:", config["feature_representation"])
    print("Original feature count:", config["original_feature_count"])
    print("Removed feature family:", config["removed_feature_family"])
    print("Removed feature count:", config["removed_feature_count"])
    print("Final feature count:", config["model_feature_count"])

    print("\nTHRESHOLD SELECTION")
    print("Selection split:", config["threshold_selection_split"])
    print("Frozen threshold:", config["validation_threshold"])
    print("Validation Average Precision:",
          round(config["validation_average_precision"], 6))
    print("Validation Precision:",
          round(config["validation_precision"], 6))
    print("Validation Recall:",
          round(config["validation_recall"], 6))
    print("Validation F1:",
          round(config["validation_f1"], 6))
    print("Validation FPR:",
          round(config["validation_fpr"], 6))
    print("Validation FNR:",
          round(config["validation_fnr"], 6))

    print("\nFINAL TEST EVALUATION")
    print("Decision level:", evaluation["decision_level"])
    print("Score aggregation:", evaluation["score_aggregation"])
    print("Accuracy:", f"{evaluation['accuracy'] * 100:.2f}%")
    print("Precision:", f"{evaluation['precision'] * 100:.2f}%")
    print("Recall:", f"{evaluation['recall'] * 100:.2f}%")
    print("F1 Score:", f"{evaluation['f1'] * 100:.2f}%")
    print("FPR:", f"{evaluation['fpr'] * 100:.2f}%")
    print("FNR:", f"{evaluation['fnr'] * 100:.2f}%")

    print("\nCONFUSION MATRIX")
    print("TN:", int(evaluation["TN"]))
    print("FP:", int(evaluation["FP"]))
    print("FN:", int(evaluation["FN"]))
    print("TP:", int(evaluation["TP"]))
    print("Total MaterialIDs:",
          int(evaluation["total_materials"]))

    print("\nDATA LEAKAGE CHECK")
    print(
        "Final test used for threshold selection:",
        config["final_test_used_for_selection"]
    )
    print(
        "Threshold frozen:",
        config["threshold_frozen"]
    )

    print("\nFINAL CONCLUSION")
    print(
        "The final Isolation Forest model uses "
        f"{config['model_feature_count']} engineered features."
    )
    print(
        f"The frozen decision threshold is "
        f"{config['validation_threshold']}."
    )
    print(
        f"Final-test F1 score is "
        f"{evaluation['f1'] * 100:.2f}%."
    )

    print("\n" + "=" * 70)
    print("STEP 58 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()