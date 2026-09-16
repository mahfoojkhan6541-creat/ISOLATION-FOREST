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


def calculate_metrics(y_true, y_pred):

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def main():

    input_file = os.path.join(
        RESULTS_DIR,
        "D2_validation_enhanced_anomaly_scores.csv"
    )

    if not os.path.exists(input_file):

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required = [
        "MaterialID",
        "target",
        "score_0.01"
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # ---------------------------------------------------------
    # Verify target is constant within MaterialID
    # ---------------------------------------------------------

    target_counts = (
        df.groupby("MaterialID")["target"]
        .nunique()
    )

    if (target_counts > 1).any():

        raise ValueError(
            "Some MaterialID groups contain multiple target values."
        )

    print("\n" + "=" * 70)
    print("MATERIALID-LEVEL ANOMALY ANALYSIS")
    print("=" * 70)

    print(
        "\nValidation rows:",
        len(df)
    )

    print(
        "Validation MaterialID groups:",
        df["MaterialID"].nunique()
    )

    # ---------------------------------------------------------
    # Aggregate row-level scores to MaterialID level
    # ---------------------------------------------------------

    grouped = (
        df.groupby("MaterialID")
        .agg(
            target=("target", "first"),

            max_score=(
                "score_0.01",
                "max"
            ),

            p95_score=(
                "score_0.01",
                lambda x: np.percentile(x, 95)
            ),

            mean_score=(
                "score_0.01",
                "mean"
            ),

            median_score=(
                "score_0.01",
                "median"
            ),

            row_count=(
                "score_0.01",
                "size"
            )
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Top 10% score mean
    # ---------------------------------------------------------

    def top_10_percent_mean(values):

        values = np.sort(
            np.asarray(values)
        )[::-1]

        k = max(
            1,
            int(np.ceil(len(values) * 0.10))
        )

        return values[:k].mean()

    top10 = (
        df.groupby("MaterialID")["score_0.01"]
        .apply(top_10_percent_mean)
        .reset_index(
            name="top10_mean_score"
        )
    )

    grouped = grouped.merge(
        top10,
        on="MaterialID",
        how="left"
    )

    # ---------------------------------------------------------
    # Save group-level scores
    # ---------------------------------------------------------

    group_score_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_validation_scores.csv"
    )

    grouped.to_csv(
        group_score_file,
        index=False
    )

    print(
        "\nMaterialID-level scores saved:"
    )

    print(group_score_file)

    # ---------------------------------------------------------
    # Evaluate each aggregation method
    # ---------------------------------------------------------

    aggregation_methods = [
        "max_score",
        "p95_score",
        "top10_mean_score",
        "mean_score",
        "median_score"
    ]

    all_results = []

    for method in aggregation_methods:

        y_true = grouped["target"].astype(int)

        scores = grouped[method].astype(float)

        ap = average_precision_score(
            y_true,
            scores
        )

        # -----------------------------------------------------
        # Threshold search
        # -----------------------------------------------------

        thresholds = np.unique(
            np.percentile(
                scores,
                np.arange(
                    1,
                    100,
                    0.25
                )
            )
        )

        best_result = None

        for threshold in thresholds:

            y_pred = (
                scores >= threshold
            ).astype(int)

            metrics = calculate_metrics(
                y_true,
                y_pred
            )

            result = {
                "aggregation": method,
                "threshold": threshold,
                "average_precision": ap,
                **metrics
            }

            if (
                best_result is None
                or result["f1"] > best_result["f1"]
            ):

                best_result = result

        all_results.append(
            best_result
        )

    results = pd.DataFrame(
        all_results
    )

    results = results.sort_values(
        "f1",
        ascending=False
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "BEST RESULT FOR EACH AGGREGATION METHOD"
    )

    print(
        "-" * 70
    )

    print(
        results[
            [
                "aggregation",
                "threshold",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fnr",
                "tp",
                "tn",
                "fp",
                "fn"
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    comparison_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_comparison.csv"
    )

    results.to_csv(
        comparison_file,
        index=False
    )

    print(
        "\nComparison saved:"
    )

    print(comparison_file)

    # ---------------------------------------------------------
    # Best method
    # ---------------------------------------------------------

    best = results.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST MATERIALID-LEVEL DEVELOPMENT RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAggregation : {best['aggregation']}"
    )

    print(
        f"Threshold   : {best['threshold']:.6f}"
    )

    print(
        f"AP          : {best['average_precision']:.4f}"
    )

    print(
        f"Precision   : {best['precision']:.4f}"
    )

    print(
        f"Recall      : {best['recall']:.4f}"
    )

    print(
        f"F1          : {best['f1']:.4f}"
    )

    print(
        f"FPR         : {best['fpr']:.4f}"
    )

    print(
        f"FNR         : {best['fnr']:.4f}"
    )

    print("\n" + "=" * 70)
    print("STEP 26 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()