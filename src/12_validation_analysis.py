import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def analyze_dataset(dataset_name):

    print("\n" + "=" * 70)
    print(f"VALIDATION ANALYSIS - {dataset_name}")
    print("=" * 70)

    file_path = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_validation_anomaly_scores.csv"
    )

    df = pd.read_csv(file_path)

    y_true = df["target"].astype(int).values

    print("\nValidation target distribution:")
    print(pd.Series(y_true).value_counts().to_dict())

    # ---------------------------------------------------------
    # D1 CHECK
    # ---------------------------------------------------------

    if len(np.unique(y_true)) < 2:

        print("\nWARNING:")
        print(
            "Validation set contains only one class."
        )
        print(
            "A supervised anomaly threshold cannot be selected "
            "reliably for this dataset."
        )

        return

    # ---------------------------------------------------------
    # D2 VALIDATION ANALYSIS
    # ---------------------------------------------------------

    score_columns = [
        col for col in df.columns
        if col.startswith("score_")
    ]

    results = []

    for score_column in score_columns:

        scores = df[score_column].values

        # Candidate thresholds based only on validation scores
        thresholds = np.percentile(
            scores,
            np.arange(80, 100, 1)
        )

        best_f1 = -1
        best_threshold = None
        best_precision = None
        best_recall = None

        for threshold in thresholds:

            y_pred = (
                scores >= threshold
            ).astype(int)

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

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        y_pred = (
            scores >= best_threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        pr_auc = average_precision_score(
            y_true,
            scores
        )

        results.append({
            "model": score_column.replace(
                "score_", ""
            ),
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "fpr": fpr,
            "fnr": fnr,
            "average_precision": pr_auc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "f1",
        ascending=False
    )

    output_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_validation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nValidation results:")
    print(
        results_df.to_string(index=False)
    )

    print("\nBest validation configuration:")

    best = results_df.iloc[0]

    print(
        "Contamination:",
        best["model"]
    )

    print(
        "Threshold:",
        best["threshold"]
    )

    print(
        "Precision:",
        f"{best['precision']:.4f}"
    )

    print(
        "Recall:",
        f"{best['recall']:.4f}"
    )

    print(
        "F1:",
        f"{best['f1']:.4f}"
    )

    print(
        "FPR:",
        f"{best['fpr']:.4f}"
    )

    print(
        "FNR:",
        f"{best['fnr']:.4f}"
    )

    print(
        "Average Precision:",
        f"{best['average_precision']:.4f}"
    )

    print("\nSaved:", output_file)


if __name__ == "__main__":

    analyze_dataset("D1")
    analyze_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 12 COMPLETE")
    print("=" * 70)