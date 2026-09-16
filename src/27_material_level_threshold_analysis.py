import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def main():

    input_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_validation_scores.csv"
    )

    if not os.path.exists(input_file):

        raise FileNotFoundError(
            f"Required file not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required_columns = [
        "MaterialID",
        "target",
        "mean_score"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column not found: {column}"
            )

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------

    y_true = df["target"].astype(int)

    scores = df["mean_score"].astype(float)

    # ---------------------------------------------------------
    # Average Precision
    # ---------------------------------------------------------

    average_precision = average_precision_score(
        y_true,
        scores
    )

    # ---------------------------------------------------------
    # Threshold range
    # ---------------------------------------------------------

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

    results = []

    for threshold in thresholds:

        y_pred = (
            scores >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        results.append({

            "threshold": threshold,

            "average_precision":
                average_precision,

            **metrics
        })

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    # ---------------------------------------------------------
    # Best F1 with FPR <= 20%
    # ---------------------------------------------------------

    fpr_20 = results_df[
        results_df["fpr"] <= 0.20
    ].copy()

    if not fpr_20.empty:

        best_f1_fpr20 = fpr_20.loc[
            fpr_20["f1"].idxmax()
        ]

    else:

        best_f1_fpr20 = None

    # ---------------------------------------------------------
    # Best F1 with FPR <= 15%
    # ---------------------------------------------------------

    fpr_15 = results_df[
        results_df["fpr"] <= 0.15
    ].copy()

    if not fpr_15.empty:

        best_f1_fpr15 = fpr_15.loc[
            fpr_15["f1"].idxmax()
        ]

    else:

        best_f1_fpr15 = None

    # ---------------------------------------------------------
    # Best F1 with FPR <= 10%
    # ---------------------------------------------------------

    fpr_10 = results_df[
        results_df["fpr"] <= 0.10
    ].copy()

    if not fpr_10.empty:

        best_f1_fpr10 = fpr_10.loc[
            fpr_10["f1"].idxmax()
        ]

    else:

        best_f1_fpr10 = None

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MATERIALID-LEVEL THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        "\nValidation MaterialID groups:",
        len(df)
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print("\n" + "-" * 70)
    print("BEST F1")
    print("-" * 70)

    print(
        best_f1.to_string()
    )

    if best_f1_fpr20 is not None:

        print("\n" + "-" * 70)
        print("BEST F1 WITH FPR <= 20%")
        print("-" * 70)

        print(
            best_f1_fpr20.to_string()
        )

    if best_f1_fpr15 is not None:

        print("\n" + "-" * 70)
        print("BEST F1 WITH FPR <= 15%")
        print("-" * 70)

        print(
            best_f1_fpr15.to_string()
        )

    if best_f1_fpr10 is not None:

        print("\n" + "-" * 70)
        print("BEST F1 WITH FPR <= 10%")
        print("-" * 70)

        print(
            best_f1_fpr10.to_string()
        )

    # ---------------------------------------------------------
    # Top candidates
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP 15 THRESHOLD CANDIDATES BY F1")
    print("-" * 70)

    top15 = (
        results_df
        .sort_values(
            "f1",
            ascending=False
        )
        .head(15)
    )

    print(
        top15[
            [
                "threshold",
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
    # Save all threshold results
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_threshold_analysis.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nAll threshold results saved:"
    )

    print(
        output_file
    )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary = []

    summary.append({
        "criterion":
            "Best F1",
        **best_f1.to_dict()
    })

    if best_f1_fpr20 is not None:

        summary.append({
            "criterion":
                "Best F1 with FPR <= 20%",
            **best_f1_fpr20.to_dict()
        })

    if best_f1_fpr15 is not None:

        summary.append({
            "criterion":
                "Best F1 with FPR <= 15%",
            **best_f1_fpr15.to_dict()
        })

    if best_f1_fpr10 is not None:

        summary.append({
            "criterion":
                "Best F1 with FPR <= 10%",
            **best_f1_fpr10.to_dict()
        })

    summary_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_threshold_summary.csv"
    )

    pd.DataFrame(
        summary
    ).to_csv(
        summary_file,
        index=False
    )

    print(
        "\nSummary saved:"
    )

    print(
        summary_file
    )

    print("\n" + "=" * 70)
    print("STEP 27 COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Only D2 VALIDATION data was used "
        "for threshold analysis."
    )

    print(
        "Final Test was NOT used."
    )


if __name__ == "__main__":
    main()