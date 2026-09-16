import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


def evaluate_dataset(dataset_name):

    print("\n" + "=" * 70)
    print(f"ENHANCED VALIDATION ANALYSIS - {dataset_name}")
    print("=" * 70)

    score_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_validation_enhanced_anomaly_scores.csv"
    )

    df = pd.read_csv(score_file)

    y_true = df["target"].astype(int)

    # Same continuous score for every contamination setting.
    # Use the 0.01 column as the reference score.
    scores = df["score_0.01"].to_numpy()

    print("\nValidation samples:", len(df))

    print("\nActual labels:")
    print(y_true.value_counts().sort_index())

    # ---------------------------------------------------------
    # Average Precision
    # ---------------------------------------------------------

    ap = average_precision_score(
        y_true,
        scores
    )

    print(
        f"\nAverage Precision: {ap:.6f}"
    )

    # ---------------------------------------------------------
    # Threshold search
    # ---------------------------------------------------------

    thresholds = np.unique(
        np.percentile(
            scores,
            np.arange(
                50,
                99.01,
                0.25
            )
        )
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\nBest F1 configuration:")
    print(
        best_f1.to_string()
    )

    # ---------------------------------------------------------
    # Best F1 with lower FPR
    # ---------------------------------------------------------

    near_best = results_df[
        results_df["f1"] >= (
            best_f1["f1"] * 0.95
        )
    ]

    if len(near_best) > 0:

        low_fpr = near_best.loc[
            near_best["fpr"].idxmin()
        ]

        print(
            "\nLowest-FPR configuration "
            "within 95% of best F1:"
        )

        print(
            low_fpr.to_string()
        )

    # ---------------------------------------------------------
    # Save validation results
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_enhanced_validation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:",
        output_file
    )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "average_precision": ap,
        "best_f1": best_f1["f1"],
        "best_f1_threshold": best_f1["threshold"],
        "best_f1_precision": best_f1["precision"],
        "best_f1_recall": best_f1["recall"],
        "best_f1_fpr": best_f1["fpr"],
        "best_f1_fnr": best_f1["fnr"],
        "best_f1_tp": best_f1["tp"],
        "best_f1_tn": best_f1["tn"],
        "best_f1_fp": best_f1["fp"],
        "best_f1_fn": best_f1["fn"]
    }])

    summary_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_enhanced_validation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "Saved:",
        summary_file
    )


if __name__ == "__main__":

    evaluate_dataset("D1")
    evaluate_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 20 COMPLETE")
    print("=" * 70)