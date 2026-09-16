import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def analyze_d2():

    print("\n" + "=" * 70)
    print("STEP 13 - D2 VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    file_path = os.path.join(
        RESULTS_DIR,
        "D2_validation_anomaly_scores.csv"
    )

    df = pd.read_csv(file_path)

    y_true = df["target"].astype(int).values

    # Use the single continuous anomaly score.
    # All contamination models have the same score_samples().
    score_column = "score_0.01"

    scores = df[score_column].values

    print("\nUsing anomaly score:", score_column)
    print("Samples:", len(scores))

    print("\nTarget distribution:")
    print(pd.Series(y_true).value_counts().sort_index())

    # ---------------------------------------------------------
    # PRECISION-RECALL CURVE
    # ---------------------------------------------------------

    precision, recall, pr_thresholds = precision_recall_curve(
        y_true,
        scores
    )

    average_precision = average_precision_score(
        y_true,
        scores
    )

    # ---------------------------------------------------------
    # THRESHOLD SEARCH
    # ---------------------------------------------------------

    thresholds = np.unique(
        np.percentile(
            scores,
            np.arange(50, 100, 0.25)
        )
    )

    rows = []

    for threshold in thresholds:

        y_pred = (
            scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision_value = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall_value = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1_value = f1_score(
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

        rows.append({
            "threshold": threshold,
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "fpr": fpr,
            "fnr": fnr,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    results = pd.DataFrame(rows)

    # Best F1
    best_f1_row = results.loc[
        results["f1"].idxmax()
    ]

    # Best balanced F1 with FPR consideration.
    # We first keep thresholds with F1 >= 95% of maximum F1,
    # then select the one with lowest FPR.
    f1_max = results["f1"].max()

    candidates = results[
        results["f1"] >= 0.95 * f1_max
    ]

    best_balanced_row = candidates.sort_values(
        ["fpr", "fnr"]
    ).iloc[0]

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_threshold_analysis.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\nAverage Precision:",
          f"{average_precision:.4f}")

    print("\nBEST F1 THRESHOLD")
    print("-" * 40)

    for key, value in best_f1_row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nLOW-FPR THRESHOLD AMONG NEAR-BEST F1")
    print("-" * 40)

    for key, value in best_balanced_row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # ---------------------------------------------------------
    # GRAPH 1 — SCORE DISTRIBUTION
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        scores[y_true == 0],
        bins=80,
        alpha=0.6,
        label="NORMAL"
    )

    plt.hist(
        scores[y_true == 1],
        bins=80,
        alpha=0.6,
        label="ABNORMAL"
    )

    plt.axvline(
        best_f1_row["threshold"],
        linestyle="--",
        label="Best F1 Threshold"
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of Samples")
    plt.title("D2 Validation - Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "D2_validation_score_distribution.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # GRAPH 2 — PRECISION RECALL
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"D2 Validation Precision-Recall Curve "
        f"(AP = {average_precision:.4f})"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "D2_validation_precision_recall.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # GRAPH 3 — F1 vs THRESHOLD
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["f1"]
    )

    plt.axvline(
        best_f1_row["threshold"],
        linestyle="--"
    )

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("D2 Validation - F1 vs Threshold")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "D2_validation_f1_threshold.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # GRAPH 4 — PRECISION AND RECALL
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["precision"],
        label="Precision"
    )

    plt.plot(
        results["threshold"],
        results["recall"],
        label="Recall"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("D2 Validation - Precision and Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "D2_validation_precision_recall_threshold.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # GRAPH 5 — FPR AND FNR
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["fpr"],
        label="FPR"
    )

    plt.plot(
        results["threshold"],
        results["fnr"],
        label="FNR"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("D2 Validation - FPR and FNR vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "D2_validation_error_rates.png"
        ),
        dpi=150
    )

    plt.close()

    print("\nSaved threshold table:")
    print(output_file)

    print("\nSaved graphs:")
    print("D2_validation_score_distribution.png")
    print("D2_validation_precision_recall.png")
    print("D2_validation_f1_threshold.png")
    print("D2_validation_precision_recall_threshold.png")
    print("D2_validation_error_rates.png")

    print("\n" + "=" * 70)
    print("STEP 13 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    analyze_d2()