import os
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


def main():

    evaluation_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_evaluation.csv"
    )

    threshold_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_threshold_summary.csv"
    )

    if not os.path.exists(evaluation_file):
        raise FileNotFoundError(
            f"Evaluation file not found:\n{evaluation_file}"
        )

    if not os.path.exists(threshold_file):
        raise FileNotFoundError(
            f"Threshold file not found:\n{threshold_file}"
        )

    evaluation = pd.read_csv(
        evaluation_file
    )

    threshold = pd.read_csv(
        threshold_file
    )

    best_f1 = threshold[
        threshold["criterion"] == "Best F1"
    ]

    if best_f1.empty:
        raise ValueError(
            "Best F1 validation result not found."
        )

    best_f1 = best_f1.iloc[0]
    final = evaluation.iloc[0]

    summary = pd.DataFrame([
        {
            "dataset": "D2",
            "decision_level": "MaterialID",
            "score_aggregation": "mean_score",

            "validation_threshold":
                best_f1["threshold"],

            "validation_precision":
                best_f1["precision"],

            "validation_recall":
                best_f1["recall"],

            "validation_f1":
                best_f1["f1"],

            "validation_fpr":
                best_f1["fpr"],

            "validation_fnr":
                best_f1["fnr"],

            "final_accuracy":
                final["accuracy"],

            "final_precision":
                final["precision"],

            "final_recall":
                final["recall"],

            "final_f1":
                final["f1"],

            "final_fpr":
                final["fpr"],

            "final_fnr":
                final["fnr"],

            "final_average_precision":
                final["average_precision"],

            "final_tn":
                final["tn"],

            "final_fp":
                final["fp"],

            "final_fn":
                final["fn"],

            "final_tp":
                final["tp"]
        }
    ])

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL V2 SUMMARY")
    print("=" * 70)

    print("\nModel:")
    print("Isolation Forest")

    print("Dataset:")
    print("D2")

    print("Decision level:")
    print("MaterialID")

    print("Score aggregation:")
    print("mean_score")

    print(
        f"\nFrozen threshold: "
        f"{best_f1['threshold']:.6f}"
    )

    print("\nVALIDATION:")
    print(
        f"Precision : {best_f1['precision']:.4f}"
    )
    print(
        f"Recall    : {best_f1['recall']:.4f}"
    )
    print(
        f"F1        : {best_f1['f1']:.4f}"
    )
    print(
        f"FPR       : {best_f1['fpr']:.4f}"
    )
    print(
        f"FNR       : {best_f1['fnr']:.4f}"
    )

    print("\nFINAL TEST:")
    print(
        f"Accuracy  : {final['accuracy']:.4f}"
    )
    print(
        f"Precision : {final['precision']:.4f}"
    )
    print(
        f"Recall    : {final['recall']:.4f}"
    )
    print(
        f"F1        : {final['f1']:.4f}"
    )
    print(
        f"FPR       : {final['fpr']:.4f}"
    )
    print(
        f"FNR       : {final['fnr']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        f"TN = {int(final['tn'])}"
    )
    print(
        f"FP = {int(final['fp'])}"
    )
    print(
        f"FN = {int(final['fn'])}"
    )
    print(
        f"TP = {int(final['tp'])}"
    )

    print("\nSaved:")
    print(output_file)

    print("\n" + "=" * 70)
    print("STEP 31 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()