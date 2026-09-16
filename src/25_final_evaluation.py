import os
import pandas as pd
import numpy as np
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


def main():

    input_file = os.path.join(
        RESULTS_DIR,
        "D2_final_test_predictions.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Prediction file not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    # ---------------------------------------------------------
    # Verify required columns
    # ---------------------------------------------------------

    required_columns = [
        "target",
        "anomaly_score",
        "prediction"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column not found: {column}"
            )

    # ---------------------------------------------------------
    # Actual labels
    # ---------------------------------------------------------

    y_true = df["target"].astype(int)

    y_pred = df["prediction"].astype(int)

    scores = df["anomaly_score"].astype(float)

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

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

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
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

    average_precision = average_precision_score(
        y_true,
        scores
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nTotal samples:",
        len(df)
    )

    print("\nActual class distribution:")

    print(
        y_true
        .map({
            0: "NORMAL",
            1: "ABNORMAL"
        })
        .value_counts()
    )

    print("\nPredicted class distribution:")

    print(
        y_pred
        .map({
            0: "NORMAL",
            1: "ABNORMAL"
        })
        .value_counts()
    )

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        f"TN (Normal → Normal)       : {tn}"
    )

    print(
        f"FP (Normal → Abnormal)     : {fp}"
    )

    print(
        f"FN (Abnormal → Normal)     : {fn}"
    )

    print(
        f"TP (Abnormal → Abnormal)   : {tp}"
    )

    print("\n" + "-" * 70)
    print("FINAL METRICS")
    print("-" * 70)

    print(
        f"Accuracy          : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision         : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall / Sensitivity : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score          : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"FPR               : {fpr:.4f} "
        f"({fpr * 100:.2f}%)"
    )

    print(
        f"FNR               : {fnr:.4f} "
        f"({fnr * 100:.2f}%)"
    )

    print(
        f"Average Precision : {average_precision:.4f}"
    )

    # ---------------------------------------------------------
    # Save final evaluation
    # ---------------------------------------------------------

    metrics = {
        "dataset": "D2",
        "split": "final_test",
        "method": "Isolation Forest",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "average_precision": average_precision,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": int(len(df))
    }

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_final_evaluation.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_file,
        index=False
    )

    print(
        "\nFinal evaluation saved:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 25 COMPLETE")
    print("=" * 70)

    print(
        "\nNOTE:"
    )

    print(
        "Final-test labels were used only for evaluation."
    )

    print(
        "The model and threshold were NOT changed."
    )


if __name__ == "__main__":
    main()