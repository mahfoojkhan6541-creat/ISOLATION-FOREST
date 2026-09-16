import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


def main():

    input_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_test_predictions.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Prediction file not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    required_columns = [
        "MaterialID",
        "target",
        "prediction",
        "mean_score"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column not found: {column}"
            )

    # ---------------------------------------------------------
    # Actual and predicted labels
    # ---------------------------------------------------------

    y_true = df["target"].astype(int)
    y_pred = df["prediction"].astype(int)

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
        df["mean_score"]
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MATERIALID-LEVEL FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        f"\nTotal MaterialID groups: {len(df)}"
    )

    print("\nActual:")
    print(
        df["target"]
        .map({
            0: "NORMAL",
            1: "ABNORMAL"
        })
        .value_counts()
    )

    print("\nPredicted:")
    print(
        df["prediction"]
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
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
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
        f"Recall            : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score          : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"False Positive Rate: {fpr:.4f} "
        f"({fpr * 100:.2f}%)"
    )

    print(
        f"False Negative Rate: {fnr:.4f} "
        f"({fnr * 100:.2f}%)"
    )

    print(
        f"Average Precision : {average_precision:.4f}"
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "decision_level": "MaterialID",
            "aggregation": "mean_score",
            "total_groups": len(df),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "average_precision": average_precision,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    ])

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_evaluation.csv"
    )

    metrics_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nEvaluation saved:"
    )

    print(output_file)

    # ---------------------------------------------------------
    # Confusion matrix plot
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "ABNORMAL"]
    )

    display.plot()

    plt.title(
        "D2 MaterialID-Level Final Test Confusion Matrix"
    )

    plt.tight_layout()

    plot_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_confusion_matrix.png"
    )

    plt.savefig(
        plot_file,
        dpi=200
    )

    plt.close()

    print(
        "\nConfusion matrix saved:"
    )

    print(plot_file)

    print("\n" + "=" * 70)
    print("STEP 30 COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Final Test was used ONLY for final evaluation."
    )

    print(
        "The threshold was NOT changed."
    )


if __name__ == "__main__":
    main()