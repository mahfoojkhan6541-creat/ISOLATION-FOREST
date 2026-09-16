import os
import joblib
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


def main():

    threshold_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_threshold_summary.csv"
    )

    if not os.path.exists(threshold_file):
        raise FileNotFoundError(
            f"Threshold summary not found:\n{threshold_file}"
        )

    threshold_df = pd.read_csv(
        threshold_file
    )

    # Select the best-F1 validation operating point
    selected = threshold_df[
        threshold_df["criterion"] == "Best F1"
    ]

    if selected.empty:
        raise ValueError(
            "Best F1 threshold was not found."
        )

    selected = selected.iloc[0]

    threshold = float(
        selected["threshold"]
    )

    # ---------------------------------------------------------
    # Freeze configuration
    # ---------------------------------------------------------

    frozen_config = {

        "dataset": "D2",

        "model": "IsolationForest",

        "feature_pipeline": "enhanced",

        "decision_level": "MaterialID",

        "score_aggregation": "mean_score",

        "threshold": threshold,

        "decision_rule":
            "mean_score >= threshold -> ABNORMAL",

        "validation_groups": 173,

        "threshold_selection":
            "Best F1 on validation only",

        "final_test_used_for_selection":
            False
    }

    # ---------------------------------------------------------
    # Save frozen configuration
    # ---------------------------------------------------------

    output_file = os.path.join(
        MODELS_DIR,
        "D2_material_level_frozen_config.pkl"
    )

    joblib.dump(
        frozen_config,
        output_file
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MATERIALID-LEVEL V2 MODEL FROZEN")
    print("=" * 70)

    print(
        f"\nAggregation : {frozen_config['score_aggregation']}"
    )

    print(
        f"Threshold   : {threshold:.6f}"
    )

    print(
        "\nDecision rule:"
    )

    print(
        f"mean_score >= {threshold:.6f} -> ABNORMAL"
    )

    print(
        f"mean_score <  {threshold:.6f} -> NORMAL"
    )

    print(
        "\nValidation performance:"
    )

    print(
        f"Precision   : {selected['precision']:.4f}"
    )

    print(
        f"Recall      : {selected['recall']:.4f}"
    )

    print(
        f"F1          : {selected['f1']:.4f}"
    )

    print(
        f"FPR         : {selected['fpr']:.4f}"
    )

    print(
        f"FNR         : {selected['fnr']:.4f}"
    )

    print(
        f"TP          : {int(selected['tp'])}"
    )

    print(
        f"TN          : {int(selected['tn'])}"
    )

    print(
        f"FP          : {int(selected['fp'])}"
    )

    print(
        f"FN          : {int(selected['fn'])}"
    )

    print(
        "\nFrozen configuration saved:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 28 COMPLETE")
    print("=" * 70)

    print(
        "\nFINAL TEST HAS NOT BEEN USED."
    )


if __name__ == "__main__":
    main()