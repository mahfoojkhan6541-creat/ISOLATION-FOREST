import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(
    BASE_DIR,
    "features"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_scores(y_true, scores, method):

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

        rows.append({
            "method": method,
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

    return pd.DataFrame(rows)


def main():

    dataset_name = "D2"

    # ---------------------------------------------------------
    # Load processed TRAIN and VALIDATION data
    # ---------------------------------------------------------

    X_train = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_train_processed.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_validation_processed.npy"
        )
    )

    validation_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_validation_enhanced_anomaly_scores.csv"
    )

    validation_df = pd.read_csv(
        validation_file
    )

    y_true = validation_df["target"].astype(int)

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    training_center = X_train.mean(axis=0)

    baseline_train_distance = np.linalg.norm(
        X_train - training_center,
        axis=1
    )

    baseline_validation_distance = np.linalg.norm(
        X_validation - training_center,
        axis=1
    )

    # ---------------------------------------------------------
    # Isolation Forest scores
    # ---------------------------------------------------------

    isolation_forest_scores = (
        validation_df["score_0.01"]
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Evaluate baseline
    # ---------------------------------------------------------

    baseline_results = evaluate_scores(
        y_true,
        baseline_validation_distance,
        "Baseline_Distance"
    )

    # ---------------------------------------------------------
    # Evaluate Isolation Forest
    # ---------------------------------------------------------

    if_results = evaluate_scores(
        y_true,
        isolation_forest_scores,
        "Isolation_Forest"
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    all_results = pd.concat(
        [
            baseline_results,
            if_results
        ],
        ignore_index=True
    )

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_baseline_vs_isolation_forest.csv"
    )

    all_results.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Average Precision
    # ---------------------------------------------------------

    baseline_ap = average_precision_score(
        y_true,
        baseline_validation_distance
    )

    if_ap = average_precision_score(
        y_true,
        isolation_forest_scores
    )

    print("\n" + "=" * 70)
    print("BASELINE VS ISOLATION FOREST")
    print("=" * 70)

    print(
        f"\nBaseline Average Precision: "
        f"{baseline_ap:.6f}"
    )

    print(
        f"Isolation Forest Average Precision: "
        f"{if_ap:.6f}"
    )

    # ---------------------------------------------------------
    # Best F1 for each method
    # ---------------------------------------------------------

    for method in [
        "Baseline_Distance",
        "Isolation_Forest"
    ]:

        method_results = all_results[
            all_results["method"] == method
        ]

        best = method_results.loc[
            method_results["f1"].idxmax()
        ]

        print(
            f"\n{method} - Best F1:"
        )

        print(
            best.to_string()
        )

    print(
        "\nSaved:",
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 21 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()